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
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union
import yaml

import torch
from rich.progress import Console
from torch.nn import Parameter
from torch.utils.data import Dataset
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, DataManager

from neusky.data.datasets.ddf_dataset import DDFDataset
from neusky.models.neusky_model import RENINeuSFactoModel
from neusky.model_components.ddf_sampler import DDFSamplerConfig

CONSOLE = Console(width=120)


@dataclass
class DDFDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: DDFDataManager)
    """Target class to instantiate."""
    num_test_images_to_generate: int = 1
    """Number of test images to generate"""
    test_image_cache_dir: Path = Path("test_images")
    """Directory to cache test images"""
    accumulation_mask_threshold: float = 0.7
    """Threshold for accumulation mask"""
    training_data_type: Literal["rand_pnts_on_sphere", "single_camera", "all_cameras"] = "rand_pnts_on_sphere"
    """Type of training data to use"""
    train_data_idx: int = 0
    """Index of training data to use if using single_camera"""
    ddf_sampler: DDFSamplerConfig = DDFSamplerConfig()
    """DDF sampler config"""
    num_of_sky_ray_samples: int = 256
    """Number of sky ray samples"""


class DDFDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DDFDataManagerConfig
    train_dataset: Dataset
    eval_dataset: Dataset

    def __init__(
        self,
        config: DDFDataManagerConfig,
        neusky: RENINeuSFactoModel,
        neusky_ckpt_path: Path,
        scene_box,
        ddf_radius: float,
        log_depth: bool = False,
        device: Union[torch.device, str] = "cpu",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        super().__init__()
        self.neusky = neusky
        self.neusky_ckpt_path = neusky_ckpt_path
        self.scene_box = scene_box
        self.ddf_radius = ddf_radius
        self.log_depth = log_depth
        self.ddf_sampler = self.config.ddf_sampler.setup(device=self.device)

        config = Path(self.neusky_ckpt_path) / "config.yml"
        config = yaml.load(config.open(), Loader=yaml.Loader)

        self.old_datamanager: VanillaDataManager = config.pipeline.datamanager.setup(
            device=self.device,
            test_mode="val",
            world_size=world_size,
            local_rank=local_rank,
        )

        # get average camera position and convert to normalised direction
        c2w = self.old_datamanager.train_dataset.cameras.camera_to_worlds
        positions = c2w[:, :3, 3]
        average_camera_position = torch.mean(positions, dim=0)
        self.dir_to_average_cam_pos = average_camera_position / torch.norm(average_camera_position)

        self.train_dataset = self.create_dataset()
        self.eval_dataset = self.train_dataset

        # not used just to get rid of error
        self.train_dataparser_outputs = self.train_dataset.dataparser_outputs

    def create_dataset(self) -> DDFDataset:
        """Create a single dataset for both train and eval."""

        return DDFDataset(
            neusky=self.neusky,
            neusky_ckpt_path=self.neusky_ckpt_path,
            scene_box=self.scene_box,
            sampler=self.ddf_sampler,
            training_data_type=self.config.training_data_type,
            num_generated_imgs=self.config.num_test_images_to_generate,
            cache_dir=self.config.test_image_cache_dir,
            ddf_sphere_radius=self.ddf_radius,
            log_depth=self.log_depth,
            accumulation_mask_threshold=self.config.accumulation_mask_threshold,
            num_sky_ray_samples=self.config.num_of_sky_ray_samples,
            old_datamanager=self.old_datamanager,
            dir_to_average_cam_pos=self.dir_to_average_cam_pos,
            device=self.device,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        if self.config.training_data_type == "rand_pnts_on_sphere":
            batch = self.train_dataset[
                (0, False)
            ]  # False is for is_viewer flag, viewer need to call train dataset and still get an image
        elif self.config.training_data_type == "single_camera":
            batch = self.train_dataset[(self.config.train_data_idx, False)]
        elif self.config.training_data_type == "all_cameras":
            batch = self.train_dataset[(self.train_count % len(self.train_dataset), False)]
        ray_bundle = batch["ray_bundle"]
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        idx = self.eval_count % self.config.num_test_images_to_generate
        if self.config.training_data_type == "rand_pnts_on_sphere":
            batch = self.eval_dataset[(idx, False)]
        elif self.config.training_data_type == "single_camera":
            batch = self.eval_dataset[(self.config.train_data_idx, False)]
        elif self.config.training_data_type == "all_cameras":
            batch = self.eval_dataset[(idx, False)]
        ray_bundle = batch["ray_bundle"]
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        idx = self.eval_count % self.config.num_test_images_to_generate
        if self.config.training_data_type == "rand_pnts_on_sphere":
            batch = self.eval_dataset[idx]
        elif self.config.training_data_type == "single_camera":
            batch = self.eval_dataset[self.config.train_data_idx]
        elif self.config.training_data_type == "all_cameras":
            batch = self.eval_dataset[idx]
        ray_bundle = batch["ray_bundle"]
        return idx, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.ddf_sampler.num_rays

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return Path("no_datapath")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
