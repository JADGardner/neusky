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
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import yaml
from pathlib import Path
import os
import sys
import random
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

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
from reni_neus.models.reni_neus_model import RENINeuSFactoModel
from reni_neus.model_components.ddf_sampler import DDFSampler
from reni_neus.model_components.illumination_samplers import IcosahedronSamplerConfig
from reni_neus.utils.utils import find_nerfstudio_project_root

CONSOLE = Console(width=120)


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
        sampler: DDFSampler,
        training_data_type: Literal["rand_pnts_on_sphere", "single_camera", "all_cameras"] = "rand_pnts_on_sphere",
        num_generated_imgs: int = 10,
        cache_dir: Path = Path("path_to_img_cache"),
        ddf_sphere_radius: float = 1.0,
        log_depth: bool = False,
        accumulation_mask_threshold: float = 0.7,
        num_sky_ray_samples: int = 256,
        old_datamanager: VanillaDataManager = None,
        dir_to_average_cam_pos: torch.Tensor = None,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.reni_neus = reni_neus
        self.reni_neus_ckpt_path = reni_neus_ckpt_path
        self.training_data_type = training_data_type
        self.sampler = sampler
        self.num_generated_imgs = num_generated_imgs
        self.cache_dir = cache_dir
        self.ddf_sphere_radius = ddf_sphere_radius
        self.log_depth = log_depth
        self.accumulation_mask_threshold = accumulation_mask_threshold
        self.device = device
        self.metadata = {}
        self.scale_factor = 1.0
        self.scene_box = scene_box
        self.num_sky_ray_samples = num_sky_ray_samples
        self.old_datamanager = old_datamanager
        self.dir_to_average_cam_pos = dir_to_average_cam_pos

        # for creating eval images
        camera_sampler_config = IcosahedronSamplerConfig(
            icosphere_order=1, apply_random_rotation=True, remove_lower_hemisphere=True
        )
        self.camera_sampler = camera_sampler_config.setup()

        config = Path(self.reni_neus_ckpt_path) / "config.yml"
        config = yaml.load(config.open(), Loader=yaml.Loader)
        scene_name = config.pipeline.datamanager.dataparser.scene

        project_root = find_nerfstudio_project_root()
        self.cache_dir = project_root / self.cache_dir

        data_file = str(self.cache_dir / f"{scene_name}_data.pt")

        if os.path.exists(data_file):
            self.cached_images = torch.load(data_file)
        else:
            self.camera = Cameras(
                camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :4].to(self.device),
                fx=torch.tensor([1007]).unsqueeze(0).to(self.device),
                fy=torch.tensor([1007]).unsqueeze(0).to(self.device),
                cx=torch.tensor([640]).unsqueeze(0).to(self.device),
                cy=torch.tensor([411.5]).unsqueeze(0).to(self.device),
                camera_type=CameraType.PERSPECTIVE,
            )
            self.cached_images = self._generate_images()
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self.cached_images, str(self.cache_dir / f"{scene_name}_data.pt"))

        # get all the c2w from self.cached_images and concatenate them, same with intrinsics
        camera_to_worlds = torch.cat([img["c2w"] for img in self.cached_images], dim=0)
        intrinsics = torch.cat([img["intrinsics"] for img in self.cached_images], dim=0)

        self.cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )

        self.dataparser_outputs = DataparserOutputs(
            image_filenames=["filename"] * len(self.cached_images), cameras=self.cameras
        )

    def __len__(self):
        return self.num_generated_imgs

    def _generate_images(self):
        # setup old data
        intrinsics = torch.zeros((1, 3, 3), device=self.device)
        intrinsics[0, 0, 0] = self.camera.fx[0]
        intrinsics[0, 1, 1] = self.camera.fy[0]
        intrinsics[0, 0, 2] = self.camera.cx[0]
        intrinsics[0, 1, 2] = self.camera.cy[0]

        batch_list = []

        # TODO this is a bodge due to illumination sampler not being refactored
        # positions is an empyty tensor
        ddf_sample_positions = torch.empty((0, 3), device=self.device)

        # keep sampling and concatenating positions until > num_generated_imgs
        while ddf_sample_positions.shape[0] < self.num_generated_imgs:
            position_sample = self.camera_sampler()
            position_sample = position_sample.type_as(ddf_sample_positions)
            ddf_sample_positions = torch.cat([ddf_sample_positions, position_sample], dim=0)

        # select only the first num_generated_imgs positions if we've gone over
        ddf_sample_positions = ddf_sample_positions[: self.num_generated_imgs]

        for _, position in enumerate(ddf_sample_positions):
            position = position.unsqueeze(0)  # [1, 3]

            # generate c2w looking at the origin
            c2w = look_at_target(position, torch.zeros_like(position).type_as(position))[..., :3, :4]  # (3, 4)

            # update self.camera.camera_to_worlds
            self.camera.camera_to_worlds[0] = c2w.type_as(self.camera.camera_to_worlds)

            camera_ray_bundle = self.camera.generate_rays(0)

            num_rays_per_chunk = 256
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            outputs_lists = defaultdict(list)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("[green]Generating eval images...", total=num_rays, extra="")
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = self.reni_neus.generate_ddf_ground_truth(
                        ray_bundle, self.accumulation_mask_threshold, self.log_depth
                    )
                    for output_name, output in outputs.items():  # type: ignore
                        outputs_lists[output_name].append(output)
                    progress.update(task, completed=i)

            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                if output_name != "ray_bundle":
                    outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

            H, W = camera_ray_bundle.origins.shape[:2]
            positions = position.unsqueeze(0).repeat(H, W, 1)  # [H, W, 3]
            directions = camera_ray_bundle.directions  # [H, W, 3]
            pixel_area = camera_ray_bundle.pixel_area  # [H, W, 1]
            metadata = camera_ray_bundle.metadata

            ray_bundle = RayBundle(
                origins=positions,
                directions=directions,
                pixel_area=pixel_area,
                camera_indices=torch.zeros_like(pixel_area),
                metadata=metadata,
            )

            outputs["c2w"] = c2w
            outputs["intrinsics"] = intrinsics
            outputs["termination_dist"] = torch.clamp(outputs["termination_dist"], max=2 * self.ddf_sphere_radius)
            outputs["image"] = outputs["termination_dist"]
            outputs["ray_bundle"] = ray_bundle
            outputs["H"] = image_height
            outputs["W"] = image_width

            batch_list.append(outputs)

        return batch_list

    def _ddf_rays(self):
        ray_bundle = self.sampler()

        outputs = self.reni_neus.generate_ddf_ground_truth(
            ray_bundle, mask_threshold=self.accumulation_mask_threshold, log_depth=self.log_depth
        )
        outputs["termination_dist"] = torch.clamp(outputs["termination_dist"], max=2 * self.ddf_sphere_radius)

        # this is so we can use the fact that sky rays don't intersect anything
        # so we can go from the DDF boundary to the known camera position as
        # ground truth distance for DDF.
        sky_ray_bundle = self.old_datamanager.get_sky_ray_bundle(number_of_rays=self.num_sky_ray_samples)

        outputs["sky_ray_bundle"] = sky_ray_bundle

        return outputs

    def _get_generated_image(self, image_idx: int, is_viewer: bool = False) -> Dict:
        data = self.cached_images[image_idx]
        if is_viewer:
            # return full image data
            return data
        else:
            # we want to select self.num_rays_per_batch rays from the data
            # and return that
            num_samples = self.num_rays_per_batch
            indices = random.sample(range(data["ray_bundle"].origins.reshape(-1, 3).shape[0]), k=num_samples)
            ray_bundle = RayBundle(
                origins=data["ray_bundle"].origins.reshape(-1, 3)[indices].to(self.device),
                directions=data["ray_bundle"].directions.reshape(-1, 3)[indices].to(self.device),
                pixel_area=data["ray_bundle"].pixel_area.reshape(-1, 1)[indices].to(self.device),
            )
            new_data = {
                "ray_bundle": ray_bundle,
                "accumulations": data["accumulations"].reshape(-1, 1)[indices].to(self.device),
                "mask": data["mask"].reshape(-1, 1)[indices].to(self.device),
                "termination_dist": data["termination_dist"].reshape(-1, 1)[indices].to(self.device),
                "normals": data["normals"].reshape(-1, 3)[indices].to(self.device),
            }
            return new_data

    def get_data(self, image_idx: int, is_viewer: bool) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if is_viewer:
            data = self._get_generated_image(image_idx, is_viewer=True)
        else:
            if self.training_data_type == "rand_pnts_on_sphere":
                data = self._ddf_rays()
            else:
                data = self._get_generated_image(image_idx, is_viewer=False)
        return data

    def __getitem__(self, idx_flag) -> Dict:
        if isinstance(idx_flag, tuple):
            image_idx, flag = idx_flag
        else:
            image_idx = idx_flag
            flag = True  # the viewer needs to have a full image returned

        data = self.get_data(image_idx, is_viewer=flag)
        return data
