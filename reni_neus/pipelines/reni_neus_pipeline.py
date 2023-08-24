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

from reni_neus.data.datamanagers.reni_neus_datamanager import RENINeuSDataManagerConfig, RENINeuSDataManager
from reni_neus.models.ddf_model import DDFModelConfig
from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig
from reni_neus.model_components.ddf_sampler import DDFSamplerConfig


@dataclass
class RENINeuSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: RENINeuSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = RENINeuSDataManagerConfig()
    """specifies the datamanager config"""
    model: RENINeuSFactoModelConfig = RENINeuSFactoModelConfig()
    """specifies the model config"""
    eval_latent_optimisation_source: Literal["none", "envmap", "image_half", "image_full"] = "image_half"
    """Source for latent optimisation during eval"""
    eval_latent_optimisation_epochs: int = 100
    """Number of epochs to optimise latent during eval"""
    eval_latent_optimisation_lr: float = 0.1
    """Learning rate for latent optimisation during eval"""
    visibility_field: Union[DDFModelConfig, None] = DDFModelConfig()
    """Visibility field"""
    visibility_ckpt_path: Union[Path, None] = None
    """Path to visibility checkpoint"""
    visibility_ckpt_step: int = 0
    """Step of the visibility checkpoint"""
    visibility_field_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""
    num_visibility_field_train_rays_per_batch: int = 256
    """Number of rays to sample of the scene for training the visibility field"""
    visibility_train_sampler: DDFSamplerConfig = DDFSamplerConfig()
    """Visibility field sampler for training"""
    visibility_accumulation_mask_threshold: float = 0.7
    """Threshold for visibility accumulation mask"""
    reni_neus_ckpt_path: Union[Path, None] = None
    """Path to reni_neus checkpoint"""
    reni_neus_ckpt_step: int = 0
    """Step of the reni_neus checkpoint"""


class RENINeuSPipeline(VanillaPipeline):
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
        config: RENINeuSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config
        self.test_mode = test_mode
        self.datamanager: RENINeuSDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if test_mode in ["val", "test"]:
            assert self.datamanager.eval_dataset is not None, "Missing validation dataset"

        # register buffers
        # this is for access when training DDF seperately
        self.register_buffer("num_train_data", torch.tensor(len(self.datamanager.train_dataset)))
        self.register_buffer("num_test_data", torch.tensor(self.datamanager.num_test))
        self.register_buffer("num_val_data", torch.tensor(self.datamanager.num_val))

        self.scene_box = self.datamanager.train_dataset.scene_box

        visibility_field = None
        if self.config.visibility_field is not None:
            visibility_field = self._setup_visibility_field(device=device)
            self.visibility_train_sampler = self.config.visibility_train_sampler.setup(device=device)

        self._model = config.model.setup(
            scene_box=self.scene_box,
            num_train_data=self.num_train_data.item(),
            num_val_data=self.num_val_data.item(),
            num_test_data=self.num_test_data.item(),
            visibility_field=visibility_field,
            test_mode=test_mode,
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
        )

        if self.config.reni_neus_ckpt_path is not None:
            assert self.config.reni_neus_ckpt_step is not None, "Invalid reni_neus_ckpt_step"
            ckpt = torch.load(
                self.config.reni_neus_ckpt_path
                / "nerfstudio_models"
                / f"step-{self.config.reni_neus_ckpt_step:09d}.ckpt",
                map_location=device,
            )
            model_dict = {}
            for key in ckpt["pipeline"].keys():
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

    def _optimise_evaluation_latents(self, step):
        # If we are optimising per eval image latents then we need to do that first
        if self.config.eval_latent_optimisation_source in ["envmap", "image_half", "image_full"]:
            self.model.fit_latent_codes_for_eval(
                datamanager=self.datamanager,
                gt_source=self.config.eval_latent_optimisation_source,
                epochs=self.config.eval_latent_optimisation_epochs,
                learning_rate=self.config.eval_latent_optimisation_lr,
                step=step,
            )

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
                ray_bundle=ray_bundle, batch=vis_batch, reni_neus=self.model, stop_gradients=True
            )
            vis_metrics_dict = self.model.visibility_field.get_metrics_dict(vis_outputs, vis_batch)
            vis_loss_dict = self.model.visibility_field.get_loss_dict(vis_outputs, vis_batch, vis_metrics_dict)

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
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, show_progress=True, step=step)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            # we need to place some cameras on the sphere and sample the visibility field
            pass

        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.model.train()
        if self.model.visibility_field is not None and self.model.config.fit_visibility_field:
            self.model.visibility_field.train()
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
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, step=step)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
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
                scene_box=self.scene_box,
                num_train_data=self.num_train_data,
                ddf_radius=ddf_radius,
            )
        else:
            ckpt_path = (
                self.config.visibility_ckpt_path
                / "nerfstudio_models"
                / f"step-{self.config.visibility_ckpt_step:09d}.ckpt"
            )
            ckpt = torch.load(str(ckpt_path))

            model_dict = {}
            for key in ckpt["pipeline"].keys():
                if key.startswith("_model."):
                    model_dict[key[7:]] = ckpt["pipeline"][key]

            # load yaml checkpoint config
            visibility_config = Path(self.config.visibility_ckpt_path) / "config.yml"
            visibility_config = yaml.load(visibility_config.open(), Loader=yaml.Loader)

            visibility_field = visibility_config.pipeline.model.setup(
                scene_box=self.scene_box,
                num_train_data=-1,
                ddf_radius=ddf_radius,
            )

            visibility_field.load_state_dict(model_dict)

        visibility_field.to(device)
        if self.model.config.fit_visibility_field:
            visibility_field.train()
        else:
            visibility_field.eval()
        return visibility_field

    def generate_ddf_samples(self):
        """Generate samples for fitting the visibility field to the scene."""
        ray_bundle = self.visibility_train_sampler()

        data = self.model.generate_ddf_ground_truth(ray_bundle, self.config.visibility_accumulation_mask_threshold)

        data["sky_ray_bundle"] = self.datamanager.get_sky_ray_bundle(256)

        return data
