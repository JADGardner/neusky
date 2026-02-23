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
from typing import Optional, Type, Union
import yaml
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.data.scene_box import SceneBox

from neusky.data.datamanagers.neusky_datamanager import NeuSkyDataManagerConfig, NeuSkyDataManager
from neusky.data.datamanagers.ddf_datamanager import DDFDataManagerConfig, DDFDataManager
from neusky.utils.utils import find_nerfstudio_project_root


@dataclass
class DDFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DDFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=NeuSkyDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""
    eval_latent_optimisation_source: Literal["none", "envmap", "image_half", "image_full"] = "image_half"
    """Source for latent optimisation during eval"""
    eval_latent_optimisation_epochs: int = 100
    """Number of epochs to optimise latent during eval"""
    eval_latent_optimisation_lr: float = 0.1
    """Learning rate for latent optimisation during eval"""
    neusky_ckpt_path: Path = Path("path_to_neusky_checkpoint")
    """Path to neusky checkpoint"""
    neusky_ckpt_step: int = 10000
    """Step of neusky checkpoint"""
    ddf_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""


class DDFPipeline(VanillaPipeline):
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
        config: DDFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config
        self.test_mode = test_mode

        scene_box = self._setup_neusky_model(device)

        if self.config.ddf_radius == "AABB":
            self.ddf_radius = torch.abs(scene_box.aabb[0, 0]).item()
        else:
            self.ddf_radius = self.config.ddf_radius

        self.datamanager: DDFDataManager = config.datamanager.setup(
            device=device,
            world_size=world_size,
            local_rank=local_rank,
            neusky=self.neusky,
            neusky_ckpt_path=self.config.neusky_ckpt_path,
            scene_box=scene_box,
            ddf_radius=self.ddf_radius,
            log_depth=self.config.model.log_depth,
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if test_mode in ["val", "test"]:
            assert self.datamanager.eval_dataset is not None, "Missing validation dataset"

        self._model = config.model.setup(
            scene_box=scene_box,
            metadata=self.datamanager.train_dataset.metadata,
            ddf_radius=self.ddf_radius,
            num_train_data=-1,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def _setup_neusky_model(self, device):
        # Now you can use this to construct paths:
        project_root = find_nerfstudio_project_root(Path(__file__))
        relative_path = (
            self.config.neusky_ckpt_path / "nerfstudio_models" / f"step-{self.config.neusky_ckpt_step:09d}.ckpt"
        )
        ckpt_path = project_root / relative_path

        if not ckpt_path.exists():
            raise ValueError(f"Could not find illumination field checkpoint at {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), weights_only=False)

        model_dict = {}
        for key in ckpt["pipeline"].keys():
            if key.startswith("_model."):
                model_dict[key[7:]] = ckpt["pipeline"][key]

        scene_box = SceneBox(aabb=model_dict["field.aabb"])

        num_train_data = ckpt["pipeline"]["num_train_data"].item()
        num_val_data = ckpt["pipeline"]["num_val_data"].item()
        num_test_data = ckpt["pipeline"]["num_test_data"].item()

        # load yaml checkpoint config
        neusky_config = Path(self.config.neusky_ckpt_path) / "config.yml"
        neusky_config = yaml.load(neusky_config.open(), Loader=yaml.Loader)

        self.neusky = neusky_config.pipeline.model.setup(
            scene_box=scene_box,
            num_train_data=num_train_data,
            num_val_data=num_val_data,
            num_test_data=num_test_data,
            test_mode="train",
            visibility_field=None,
        )

        self.neusky.load_state_dict(model_dict, strict=False)  # no visiblity field
        self.neusky.eval()
        self.neusky.to(device)
        # things we don't need to render
        self.neusky.render_rgb_flag = False
        self.neusky.render_albedo_flag = False

        return scene_box

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(
            ray_bundle, batch, self.neusky
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

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, None, self.neusky)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        self.neusky.eval()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, self.neusky, show_progress=True)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        self.neusky.eval()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = self.datamanager.config.num_test_images_to_generate
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i in range(num_images):
                _, camera_ray_bundle, batch = self.datamanager.next_eval_image(i)
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle, self.neusky, show_progress=False
                )
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
        self.train()
        self.neusky.eval()
        return metrics_dict
