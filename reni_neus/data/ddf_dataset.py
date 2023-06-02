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
from nerfstudio.cameras.cameras import Cameras
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
        self.device = device

        # self._setup_reni()
        
        if self.test_mode in ["test", "val"]:
          self._setup_previous_datamanager()

          data_file = str(self.cache_dir / f"{self.old_datamanager.dataparser.config.scene}_data.pt")

          if os.path.exists(data_file):
              self.cached_images = torch.load(data_file)
          else:
              self.cached_images = self._generate_images()

    def __len__(self):
        return self.num_generated_imgs

    # def _setup_reni(self):
    #     # setting up reni_neus for pseudo ground truth
    #     ckpt_path = self.reni_neus_ckpt_path / "nerfstudio_models" / f"step-{self.reni_neus_ckpt_step:09d}.ckpt"
    #     ckpt = torch.load(str(ckpt_path))

    #     model_dict = {}
    #     for key in ckpt["pipeline"].keys():
    #         if key.startswith("_model."):
    #             model_dict[key[7:]] = ckpt["pipeline"][key]

    #     scene_box = SceneBox(aabb=model_dict["aabb"])
    #     num_train_data = model_dict["illumination_field_train.reni.mu"].shape[0]
    #     num_val_data = model_dict["illumination_field_val.reni.mu"].shape[0]
    #     num_test_data = model_dict["illumination_field_test.reni.mu"].shape[0]

    #     # load yaml checkpoint config
    #     reni_neus_config = Path(self.reni_neus_ckpt_path) / "config.yml"
    #     reni_neus_config = yaml.load(reni_neus_config.open(), Loader=yaml.Loader)

    #     self.reni_neus = reni_neus_config.pipeline.model.setup(
    #         scene_box=scene_box,
    #         num_train_data=num_train_data,
    #         num_val_data=num_val_data,
    #         num_test_data=num_test_data,
    #         test_mode="train",
    #     )

    #     self.reni_neus.load_state_dict(model_dict)
    #     self.reni_neus.eval()

    def _setup_previous_datamanager(self):
        # load config.yaml
        config = Path(self.reni_neus_ckpt_path) / "config.yml"
        config = yaml.load(config.open(), Loader=yaml.Loader)

        pipeline_config = config.pipeline

        self.old_datamanager: VanillaDataManager = pipeline_config.datamanager.setup(
            device=self.device,
            test_mode="test",
            world_size=1,
            local_rank=1,
        )

    def _generate_images(self):
        original_data_c2w = self.old_datamanager.eval_dataloader.cameras.camera_to_worlds
        min_x = torch.min(original_data_c2w[:, 0, 3])
        max_x = torch.max(original_data_c2w[:, 0, 3])
        min_y = torch.min(original_data_c2w[:, 1, 3])
        max_y = torch.max(original_data_c2w[:, 1, 3])

        batch_list = []

        for _ in range(self.num_generated_imgs):
            # generate random camera positions between min and max x and y and z > 0
            random_x = torch.rand(1).type_as(min_x) * (max_x - min_x) + min_x
            random_y = torch.rand(1).type_as(min_y) * (max_y - min_y) + min_y
            random_z = torch.rand(1).type_as(min_x)  # positive z so in upper hemisphere

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
            positions = position.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)  # [H, W, 1, 3]
            positions = positions.reshape(-1, 3)  # [N, 3]
            directions = camera_ray_bundle.directions  # [H, W, 1, 3]
            directions = directions.reshape(-1, 3)  # [N, 3]
            accumulations = outputs["accumulation"].reshape(-1, 1).squeeze()  # [N]
            termination_dist = outputs["p2p_dist"].reshape(-1, 1).squeeze()  # [N]
            normals = outputs["normal"].reshape(-1, 3).squeeze()  # [N, 3]
            mask = (accumulations > self.accumulation_mask_threshold).float()

            ray_bundle = RayBundle(
                origins=positions,
                directions=directions,
                pixel_area=camera_ray_bundle.pixel_area,
                camera_indices=camera_ray_bundle.camera_indices,
                metadata=camera_ray_bundle.metadata,
            )

            data = {
                "ray_bundle": ray_bundle,
                "accumulations": accumulations,
                "mask": mask,
                "termination_dist": termination_dist,
                "normals": normals,
            }

            batch_list.append(data)

        # save data to cache
        torch.save(batch_list, str(self.cache_dir / f"{self.old_datamanager.dataparser.scene}_data.pt"))
        return batch_list

    def _ddf_rays(self):
        num_samples = self.num_rays_per_batch

        positions = random_points_on_unit_sphere(1, cartesian=True)  # (1, 3)
        directions = random_inward_facing_directions(num_samples, normals=-positions)  # (1, num_directions, 3)

        positions = positions * self.ddf_sphere_radius

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

        field_outputs = self.reni_neus.field(ray_bundle)
        accumulations = field_outputs["accumulation"].reshape(-1, 1).squeeze()
        termination_dist = field_outputs["p2p_dist"].reshape(-1, 1).squeeze()
        normals = field_outputs["normal"].reshape(-1, 3).squeeze()
        mask = (accumulations > self.accumulation_mask_threshold).float()

        data = {
            "accumulations": accumulations,
            "mask": mask,
            "termination_dist": termination_dist,
            "normals": normals,
        }

        return ray_bundle, data

    def _get_generated_image(self, image_idx: int):
        if self.cached_images is None:
            self.cached_images = self._generate_images()
        else:
            data = self.cached_images[image_idx]
            ray_bundle = data["ray_bundle"]
            return ray_bundle, data

    def get_data(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if self.test_mode == "train":
            ray_bundle, data = self._ddf_rays()
        else:
            ray_bundle, data = self._get_generated_image(image_idx)
        return ray_bundle, data

    def __getitem__(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, data = self.get_data(image_idx)
        return ray_bundle, data
