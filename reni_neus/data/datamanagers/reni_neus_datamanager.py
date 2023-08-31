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
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from rich.progress import Console
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    variable_res_collate,
)

from reni_neus.data.reni_neus_pixel_sampler import RENINeuSPixelSampler
from reni_neus.data.datasets.reni_neus_dataset import RENINeuSDataset

CONSOLE = Console(width=120)


@dataclass
class RENINeuSDataManagerConfig(VanillaDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: RENINeuSDataManager)
    """Target class to instantiate."""


class RENINeuSDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RENINeuSDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[RENINeuSPixelSampler] = None
    eval_pixel_sampler: Optional[RENINeuSPixelSampler] = None

    def __init__(
        self,
        config: RENINeuSDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break

        super(VanillaDataManager, self).__init__()  # Call grandparent class constructor ignoring parent class

    def create_train_dataset(self) -> RENINeuSDataset:
        return RENINeuSDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> RENINeuSDataset:
        test_outputs = self.dataparser.get_dataparser_outputs("test")
        val_outputs = self.dataparser.get_dataparser_outputs("val")
        self.num_test = len(test_outputs.image_filenames)
        self.num_val = len(val_outputs.image_filenames)
        return RENINeuSDataset(
            dataparser_outputs=test_outputs if self.test_mode == "test" else val_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_sky_ray_bundle(self, number_of_rays: int) -> Tuple[RayBundle, Dict]:
        """Returns a sky ray bundle for the given step."""
        # choose random
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.collate_sky_ray_batch(image_batch, num_rays_per_batch=number_of_rays)
        ray_indices = batch["indices"].cpu()
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle

    def get_eval_image_half_bundle(
        self, sample_region: Literal["left_image_half", "right_image_half", "full_image"]
    ) -> Tuple[RayBundle, Dict]:
        """Returns a ray bundle of rays within the left image half and within standard mask."""
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.collate_image_half(
            batch=image_batch, num_rays_per_batch=self.config.eval_num_rays_per_batch, sample_region=sample_region
        )
        ray_indices = batch["indices"].cpu()
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch
