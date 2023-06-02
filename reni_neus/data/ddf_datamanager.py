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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from rich.progress import Console
from torch.nn import Parameter
from torch.utils.data import Dataset
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, DataManager, AnnotatedDataParserUnion

from reni_neus.data.reni_neus_pixel_sampler import RENINeuSPixelSampler
from reni_neus.data.reni_neus_dataset import RENINeuSDataset
from reni_neus.data.ddf_dataset import DDFDataset
from reni_neus.reni_neus_model import RENINeuSFactoModel

CONSOLE = Console(width=120)


@dataclass
class DDFDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: DDFDataManager)
    """Target class to instantiate."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    num_test_images_to_generate: int = 1
    """Number of test images to generate"""
    test_image_cache_dir: Path = Path("test_images")
    """Directory to cache test images"""
    ddf_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""
    accumulation_mask_threshold: float = 0.7
    """Threshold for accumulation mask"""


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
        reni_neus: RENINeuSFactoModel,
        reni_neus_ckpt_path: Path,
        scene_box,
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

        super().__init__()
        self.reni_neus = reni_neus
        self.reni_neus_ckpt_path = reni_neus_ckpt_path

        if self.config.ddf_radius == "AABB":
            self.ddf_radius = torch.abs(scene_box.aabb[0])
        else:
            self.ddf_radius = self.config.ddf_radius
        
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        # not used just to get rid of error
        self.train_dataparser_outputs = self.eval_dataset.old_datamanager.train_dataparser_outputs

    def create_train_dataset(self) -> DDFDataset:
        return DDFDataset(
            reni_neus=self.reni_neus,
            reni_neus_ckpt_path=self.reni_neus_ckpt_path,
            test_mode="train",
            num_generated_imgs=1,
            cache_dir=None,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            ddf_sphere_radius=self.ddf_radius,
            accumulation_mask_threshold=self.config.accumulation_mask_threshold,
            device=self.device,
        )

    def create_eval_dataset(self) -> DDFDataset:
        return DDFDataset(
            reni_neus=self.reni_neus,
            reni_neus_ckpt_path=self.reni_neus_ckpt_path,
            test_mode=self.test_mode,
            num_generated_imgs=self.config.num_test_images_to_generate,
            cache_dir=self.config.test_image_cache_dir,
            num_rays_per_batch=self.config.eval_num_rays_per_batch,
            ddf_sphere_radius=self.ddf_radius,
            accumulation_mask_threshold=self.config.accumulation_mask_threshold,
            device=self.device,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return RENINeuSPixelSampler(*args, **kwargs)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = self.train_dataset[len(self.train_dataset) + 1] # idx + 1 larger than len as flag, viewer needs img and will use actual idx
        ray_bundle = batch["ray_bundle"]
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        idx = self.eval_count % self.config.num_test_images_to_generate
        batch = self.eval_dataset[idx]
        ray_bundle = batch["ray_bundle"]
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

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
