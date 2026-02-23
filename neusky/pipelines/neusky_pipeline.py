# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from time import time
from typing import Optional, Type, Union, Dict, List
import yaml
from pathlib import Path

import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.nn import Parameter

from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.cameras.cameras import Cameras, CameraType

from neusky.data.datamanagers.neusky_datamanager import NeuSkyDataManagerConfig, NeuSkyDataManager
from neusky.models.ddf_model import DDFModelConfig
from neusky.models.neusky_model import NeuSkyFactoModelConfig
from neusky.model_components.ddf_sampler import DDFSamplerConfig
from neusky.model_components.illumination_samplers import IcosahedronSamplerConfig
from neusky.utils.utils import look_at_target


@dataclass
class NeuSkyPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeuSkyPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=NeuSkyDataManagerConfig)
    """specifies the datamanager config"""
    model: NeuSkyFactoModelConfig = field(default_factory=NeuSkyFactoModelConfig)
    """specifies the model config"""
    visibility_field: Union[DDFModelConfig, None] = field(default_factory=DDFModelConfig)
    """Visibility field"""
    visibility_ckpt_path: Union[Path, None] = None
    """Path to visibility checkpoint"""
    visibility_ckpt_step: int = 0
    """Step of the visibility checkpoint"""
    visibility_field_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""
    num_visibility_field_train_rays_per_batch: int = 256
    """Number of rays to sample of the scene for training the visibility field"""
    visibility_train_sampler: DDFSamplerConfig = field(default_factory=DDFSamplerConfig)
    """Visibility field sampler for training"""
    visibility_accumulation_mask_threshold: float = 0.7
    """Threshold for visibility accumulation mask, 0.0 means no mask as mask = accum > threshold"""
    neusky_ckpt_path: Union[Path, None] = None
    """Path to neusky checkpoint"""
    neusky_ckpt_step: int = 0
    """Step of the neusky checkpoint"""
    eval_using_gt_envmaps: bool = False
    """Whether to use ground truth envmaps for evaluation"""
    test_mode: Union[Literal["test", "val", "inference"], None] = None
    """overwrite test mode"""
    least_squares_global_scale: bool = False
    """Whether to use least squares to find the global scale"""
    stop_sdf_gradients: bool = True
    """Whether to stop gradients to the SDF"""


class NeuSkyPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: NeuSkyPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config
        self.test_mode = test_mode if self.config.test_mode is None else self.config.test_mode
        self.datamanager: NeuSkyDataManager = config.datamanager.setup(
            device=device,
            test_mode=self.test_mode,
            world_size=world_size,
            local_rank=local_rank,
            eval_latent_optimise_method=config.model.eval_latent_optimise_method,
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if test_mode in ["val", "test"]:
            assert self.datamanager.eval_dataset is not None, "Missing validation dataset"

        self.eval_image_num = 0
        self.max_eval_num = self.datamanager.num_val if test_mode == "val" else self.datamanager.num_test

        # register buffers
        # this is so they are available if training DDF seperately
        self.register_buffer("num_train_data", torch.tensor(len(self.datamanager.train_dataset)))
        self.register_buffer("num_test_data", torch.tensor(self.datamanager.num_test))
        self.register_buffer("num_val_data", torch.tensor(self.datamanager.num_val))

        self.scene_box = self.datamanager.train_dataset.scene_box

        visibility_field = None
        if self.config.visibility_field is not None:
            # TODO god knows why I put the vis field here, move into model
            visibility_field = self._setup_visibility_field(device=device)
            self.visibility_train_sampler = self.config.visibility_train_sampler.setup(device=device)
            test_time_sampler_config = IcosahedronSamplerConfig(
                icosphere_order=2, apply_random_rotation=False, remove_lower_hemisphere=True
            )
            self.visibility_test_time_sampler = test_time_sampler_config.setup()

        self._model = config.model.setup(
            scene_box=self.scene_box,
            num_train_data=self.num_train_data.item(),
            num_val_data=self.num_val_data.item(),
            num_test_data=self.num_test_data.item(),
            visibility_field=visibility_field,
            test_mode=test_mode,
            train_metadata=self.datamanager.train_dataset.metadata,
            eval_metadata=self.datamanager.eval_dataset.metadata,
            grad_scaler=grad_scaler,
        )

        if self.config.neusky_ckpt_path is not None:
            assert self.config.neusky_ckpt_step is not None, "Invalid neusky_ckpt_step"
            ckpt = torch.load(
                self.config.neusky_ckpt_path
                / "nerfstudio_models"
                / f"step-{self.config.neusky_ckpt_step:09d}.ckpt",
                map_location=device,
                weights_only=False,
            )
            model_dict = {}
            for key in ckpt["pipeline"].keys():
                if self.config.visibility_ckpt_path is not None:
                    if key.startswith("_model.") and not key.startswith("_model.visibility_field."):
                        model_dict[key[7:]] = ckpt["pipeline"][key]
                else:
                    if key.startswith("_model."):
                        model_dict[key[7:]] = ckpt["pipeline"][key]

            self.model.load_state_dict(
                model_dict, strict=False
            )  # false as it will be loading the visibility field weights too # TODO is there a better way to share visibility field?
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.step_of_last_latent_optimisation = 0

    def _optimise_evaluation_latents(self, step):
        if self.step_of_last_latent_optimisation != step:
            self.model.fit_latent_codes_for_eval(
                datamanager=self.datamanager,
                global_step=step,
            )
            self.step_of_last_latent_optimisation = step

    def global_scale(self, pred_img, gt_img):
        pred_img_flat = pred_img.view(-1)
        gt_img_flat = gt_img.view(-1)

        # ensure both on same divice
        gt_img_flat = gt_img_flat.to(pred_img_flat.device)

        # Compute the optimal scale
        alpha = torch.dot(gt_img_flat, pred_img_flat) / torch.dot(pred_img_flat, pred_img_flat)

        # Scale the predicted image
        scaled_pred_img = alpha * pred_img

        return scaled_pred_img

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        if self.model.visibility_field is not None:
            visibility_params = self.model.visibility_field.get_param_groups()
            model_params = {**model_params, **visibility_params}
        return {**datamanager_params, **model_params}

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self._model.visibility_field and not self.model.config.fit_visibility_field:
            self._model.visibility_field.eval()

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(
            ray_bundle, batch=batch, step=step
        )  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # we now need to fit the visibility field to the scene
        if self.model.config.fit_visibility_field:
            vis_batch = (
                self.generate_ddf_samples()
            )  # we sample from the 3D scene, we want the visibility (ddf) to be consistent with the scene
            ray_bundle = vis_batch["ray_bundle"]
            # we stop gradients here as we are just fitting to the scene
            vis_outputs = self._model.visibility_field(
                ray_bundle=ray_bundle, batch=vis_batch, neusky=self.model, stop_gradients=self.config.stop_sdf_gradients
            )
            vis_metrics_dict = self.model.visibility_field.get_metrics_dict(vis_outputs, vis_batch)
            vis_loss_dict = self.model.visibility_field.get_loss_dict(vis_outputs, vis_batch, vis_metrics_dict)

            # TODO add the loss computed from model_outputs['sdf_at_termination] which is shape [N, 1] to the vis_loss_dict['sdf_l2_loss']
            # Currently this is done in the model.

            model_outputs = {**model_outputs, **vis_outputs}
            loss_dict = {**loss_dict, **vis_loss_dict}
            metrics_dict = {**metrics_dict, **vis_metrics_dict}

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        # if we haven't already fit the eval latents this step, do it now
        self._optimise_evaluation_latents(step)
        self.model.eval()
        if self.model.visibility_field is not None:
            self.model.visibility_field.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, step=step)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.model.train()
        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            self.model.visibility_field.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self._optimise_evaluation_latents(step)
        self.model.eval()
        if self.model.visibility_field is not None:
            self.model.visibility_field.eval()
        self.eval_image_num = self.eval_image_num % self.max_eval_num
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(self.eval_image_num)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, show_progress=True, step=step)
        if self.config.least_squares_global_scale:
            outputs['rgb'] = self.global_scale(outputs['rgb'], batch['image'])
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            positions = self.visibility_test_time_sampler()[:12]  # shape [12, 3]

            fx = self.datamanager.eval_dataset.cameras.fx[image_idx]
            fy = self.datamanager.eval_dataset.cameras.fy[image_idx]
            cx = self.datamanager.eval_dataset.cameras.cx[image_idx]
            cy = self.datamanager.eval_dataset.cameras.cy[image_idx]

            all_visibility_images = []
            all_accumulation_images = []

            for position_on_sphere in positions:
                position_on_sphere = position_on_sphere.unsqueeze(0)  # [1, 3]
                c2w = look_at_target(position_on_sphere, torch.zeros_like(position_on_sphere))[..., :3, :4]  # (3, 4)

                # update self.camera.camera_to_worlds
                camera = Cameras(
                    camera_to_worlds=c2w,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    camera_type=CameraType.PERSPECTIVE,
                )

                visibility_ray_bundle = camera.generate_rays(0)
                visibility_ray_bundle = visibility_ray_bundle.to(self.device)
                vis_outputs = self.model.visibility_field.get_outputs_for_camera_ray_bundle(
                    visibility_ray_bundle, neusky=None, show_progress=False
                )

                vis_images_dict = self.model.visibility_field.get_image_dict(vis_outputs)

                # Assuming the main visibility image is stored with a key 'visibility_image' in vis_images_dict
                all_visibility_images.append(vis_images_dict["ddf_depth"].permute(2, 0, 1))  # [3, H, W]

                if "ddf_accumulation" in vis_images_dict:
                    all_accumulation_images.append(vis_images_dict["ddf_accumulation"].permute(2, 0, 1))  # [3, H, W]

            # Combine all visibility images into a grid
            grid_image = make_grid(all_visibility_images, nrow=4).permute(1, 2, 0)  # [H, W, 3]
            images_dict["ddf_depth_grid"] = grid_image

            if len(all_accumulation_images) > 0:
                # Combine all accumulation images into a grid
                grid_image = make_grid(all_accumulation_images, nrow=4).permute(1, 2, 0)
                images_dict["ddf_accumulation_grid"] = grid_image

        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.model.train()
        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            self.model.visibility_field.train()
        self.eval_image_num += 1
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self._optimise_evaluation_latents(step)
        self.model.eval()
        if self.model.visibility_field is not None:
            self.model.visibility_field.eval()
        metrics_dict_list = []
        if hasattr(self.datamanager.eval_dataloader, 'image_indices'):
            num_images = len(self.datamanager.eval_dataloader.image_indices)
        else:
            num_images = len(self.datamanager.eval_dataloader)
        eval_image_num = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i in range(num_images):
                image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(eval_image_num)
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, step=step)
                if self.config.least_squares_global_scale:
                    outputs['rgb'] = self.global_scale(outputs['rgb'], batch['image'])
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                eval_image_num += 1
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.model.train()
        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            self.model.visibility_field.train()
        return metrics_dict

    def _setup_visibility_field(self, device):
        # setting up visibility field
        if self.config.visibility_field_radius == "AABB":
            ddf_radius = torch.abs(self.scene_box.aabb[0, 0]).item()
        else:
            ddf_radius = self.config.visibility_field_radius

        if self.config.visibility_ckpt_path is None:
            return self.config.visibility_field.setup(
                scene_box=None,
                num_train_data=self.num_train_data,
                ddf_radius=ddf_radius,
            )
        else:
            ckpt_path = (
                self.config.visibility_ckpt_path
                / "nerfstudio_models"
                / f"step-{self.config.visibility_ckpt_step:09d}.ckpt"
            )
            ckpt = torch.load(str(ckpt_path), weights_only=False)

            model_dict = {}
            for key in ckpt["pipeline"].keys():
                if key.startswith("_model."):
                    model_dict[key[7:]] = ckpt["pipeline"][key]

            # load yaml checkpoint config
            visibility_config = Path(self.config.visibility_ckpt_path) / "config.yml"
            visibility_config = yaml.load(visibility_config.open(), Loader=yaml.Loader)

            visibility_field = visibility_config.pipeline.model.setup(
                scene_box=None,
                num_train_data=-1,
                ddf_radius=ddf_radius,
            )

            visibility_field.load_state_dict(model_dict)

        visibility_field.to(device)

        if self.config.model.fit_visibility_field:
            visibility_field.train()
        else:
            visibility_field.eval()

        return visibility_field

    def generate_ddf_samples(self):
        """Generate samples for fitting the visibility field to the scene."""
        if self.config.stop_sdf_gradients:
            with torch.no_grad():
                ray_bundle = self.visibility_train_sampler()
        else:
            ray_bundle = self.visibility_train_sampler()

        data = self.model.generate_ddf_ground_truth(ray_bundle, self.config.visibility_accumulation_mask_threshold)

        data["sky_ray_bundle"] = self.datamanager.get_sky_ray_bundle(256)

        if self.config.stop_sdf_gradients:
            data["ray_bundle"].origins.requires_grad = False
            data["ray_bundle"].directions.requires_grad = False
            data["sky_ray_bundle"].origins.requires_grad = False
            data["sky_ray_bundle"].directions.requires_grad = False
            data["accumulations"] = data["accumulations"].detach()
            data["mask"] = data["mask"].detach()
            data["termination_dist"] = data["termination_dist"].detach()
            data["normals"] = data["normals"].detach()

        return data
