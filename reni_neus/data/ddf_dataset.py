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
DDF dataset from trained RENI-NeuS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import yaml
from pathlib import Path
import os

from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions, look_at_target
from reni_neus.reni_neus_model import RENINeuSFactoModel


class DDFDataset(Dataset):
    """Dataset that returns images.

    Args:
        reni_neus_checkpoint_path: The path to the Reni Neus checkpoint.
        test_mode: The test mode to use. One of "train", "test", or "val".
        num_generated_imgs: The number of images to generate.
        cache_dir: The directory to cache/load the generated images.
    """

    cameras: Cameras

    def __init__(
        self,
        reni_neus: RENINeuSFactoModel,
        reni_neus_ckpt_path: Path,
        scene_box: SceneBox,
        test_mode: Literal["train", "test", "val"] = "train",
        num_generated_imgs: int = 10,
        cache_dir: Path = Path("path_to_img_cache"),
        num_rays_per_batch: int = 1024,
        ddf_sphere_radius: Union[Literal["AABB"], float] = "AABB",
        accumulation_mask_threshold: float = 0.7,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.reni_neus = reni_neus
        self.reni_neus_ckpt_path = reni_neus_ckpt_path
        self.test_mode = test_mode
        self.num_generated_imgs = num_generated_imgs
        self.cache_dir = cache_dir
        self.num_rays_per_batch = num_rays_per_batch
        self.ddf_sphere_radius = ddf_sphere_radius
        self.accumulation_mask_threshold = accumulation_mask_threshold
        self.device = device
        self.metadata = {}
        self.scale_factor = 1.0
        self.scene_box = scene_box
        self.old_datamanager = None

        config = Path(self.reni_neus_ckpt_path) / "config.yml"
        config = yaml.load(config.open(), Loader=yaml.Loader)
        scene_name = config.pipeline.datamanager.dataparser.scene

        data_file = str(self.cache_dir / f"{scene_name}_data.pt")

        if os.path.exists(data_file):
            self.cached_images = torch.load(data_file)
        else:
            self._setup_previous_datamanager(config)
            self.cached_images = self._generate_images()
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(
                self.cached_images, str(self.cache_dir / f"{self.old_datamanager.dataparser.config.scene}_data.pt")
            )

        camera_to_worlds = self.cached_images[0]["c2w"].unsqueeze(0)
        intrinsics = self.cached_images[0]["intrinsics"]

        self.cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )

        if self.test_mode in ["test", "val"] and self.old_datamanager is None:
            self._setup_previous_datamanager(config)

    def __len__(self):
        return self.num_generated_imgs

    def _setup_previous_datamanager(self, config):
        pipeline_config = config.pipeline

        self.old_datamanager: VanillaDataManager = pipeline_config.datamanager.setup(
            device=self.device,
            test_mode="test",
            world_size=1,
            local_rank=1,
        )

    def _generate_images(self):
        # setup old data
        original_data_c2w = self.old_datamanager.eval_dataloader.cameras.camera_to_worlds
        intrinsics = torch.zeros((1, 3, 3), device=self.device)
        intrinsics[0, 0, 0] = self.old_datamanager.eval_dataloader.cameras.fx[0]
        intrinsics[0, 1, 1] = self.old_datamanager.eval_dataloader.cameras.fy[0]
        intrinsics[0, 0, 2] = self.old_datamanager.eval_dataloader.cameras.cx[0]
        intrinsics[0, 1, 2] = self.old_datamanager.eval_dataloader.cameras.cy[0]

        min_x = torch.min(original_data_c2w[:, 0, 3])
        max_x = torch.max(original_data_c2w[:, 0, 3])
        min_y = torch.min(original_data_c2w[:, 1, 3])
        max_y = torch.max(original_data_c2w[:, 1, 3])

        batch_list = []

        for _ in range(self.num_generated_imgs):
            # generate random camera positions between min and max x and y and z > 0
            random_x = torch.empty(1).uniform_(min_x, max_x).type_as(min_x)
            random_y = torch.empty(1).uniform_(min_y, max_y).type_as(min_x)
            random_z = torch.empty(1).uniform_(0.1, 0.3).type_as(min_x)

            # combine x, y, z into a single tensor
            random_position = torch.stack([random_x, random_y, random_z], dim=1)

            # normalize the positions
            position = F.normalize(random_position, p=2, dim=1)

            # generate c2w looking at the origin
            c2w = look_at_target(position, torch.zeros_like(position).type_as(position))[..., :3, :4]  # (3, 4)

            # update c2w in dataloader.cameras use index 0
            original_data_c2w[0] = c2w

            # use index 0 (new c2w) to generate new camera ray bundle
            camera_ray_bundle, _ = self.old_datamanager.eval_dataloader.get_data_from_image_idx(0)

            outputs = self.reni_neus.get_outputs_for_camera_ray_bundle(camera_ray_bundle, show_progress=True)

            H, W = camera_ray_bundle.origins.shape[:2]
            positions = position.unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)  # [H, W, 1, 3]
            # positions = positions.reshape(-1, 3)  # [N, 3]
            directions = camera_ray_bundle.directions  # [H, W, 1, 3]
            # directions = directions.reshape(-1, 3)  # [N, 3]
            # accumulations = outputs["accumulation"].reshape(-1, 1).squeeze()  # [N]
            accumulations = outputs["accumulation"]  # [H, W, 1, 1]
            # termination_dist = outputs["p2p_dist"].reshape(-1, 1).squeeze()  # [N]
            termination_dist = outputs["p2p_dist"]  # [H, W, 1, 1]
            # normals = outputs["normal"].reshape(-1, 3).squeeze()  # [N, 3]
            normals = outputs["normal"]  # [H, W, 1, 3]
            mask = (accumulations > self.accumulation_mask_threshold).float()
            # pixel_area = camera_ray_bundle.pixel_area.reshape(-1, 1).squeeze()  # [N]
            pixel_area = camera_ray_bundle.pixel_area  # [H, W, 1, 1]
            metadata = camera_ray_bundle.metadata
            # metadata["directions_norm"] = metadata["directions_norm"].reshape(-1, 1).squeeze()  # [N]

            ray_bundle = RayBundle(
                origins=positions,
                directions=directions,
                pixel_area=pixel_area,
                camera_indices=torch.zeros_like(pixel_area),
                metadata=metadata,
            )

            data = {
                "c2w": c2w,
                "intrinsics": intrinsics,
                "image": outputs["rgb"],
                "ray_bundle": ray_bundle,
                "accumulations": accumulations,
                "mask": mask,
                "termination_dist": termination_dist,
                "normals": normals,
                "H": H,
                "W": W,
            }

            batch_list.append(data)

        return batch_list

    def _ddf_rays(self):
        num_samples = self.num_rays_per_batch

        positions = random_points_on_unit_sphere(1, cartesian=True)  # (1, 3)
        directions = random_inward_facing_directions(num_samples, normals=-positions)  # (1, num_directions, 3)

        positions = positions * self.ddf_sphere_radius.type_as(positions)

        pos_ray = positions.repeat(num_samples, 1).to(self.device)
        dir_ray = directions.reshape(-1, 3).to(self.device)
        pixel_area = torch.ones(num_samples, 1, device=self.device)
        camera_indices = torch.zeros(num_samples, 1, device=self.device, dtype=torch.int64)
        metadata = {"directions_norm": torch.ones(num_samples, 1, device=self.device)}

        ray_bundle = RayBundle(
            origins=pos_ray,
            directions=dir_ray,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )

        accumulations = None
        termination_dist = None
        normals = None

        field_outputs = self.reni_neus(ray_bundle)
        accumulations = field_outputs["accumulation"].reshape(-1, 1).squeeze()
        termination_dist = field_outputs["p2p_dist"].reshape(-1, 1).squeeze()
        normals = field_outputs["normal"].reshape(-1, 3).squeeze()
        mask = (accumulations > self.accumulation_mask_threshold).float()

        data = {
            "ray_bundle": ray_bundle,
            "accumulations": accumulations,
            "mask": mask,
            "termination_dist": termination_dist,
            "normals": normals,
        }

        return data

    def _get_generated_image(self, image_idx: int):
        data = self.cached_images[image_idx]
        return data

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if self.test_mode == "train":
            if image_idx == len(self) + 1:
                data = self._ddf_rays()
            else:
                data = self._get_generated_image(image_idx)
        else:
            data = self._get_generated_image(image_idx)
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data
