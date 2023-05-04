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
""" Data parser for NeRF-OSR datasets

    Presented in the paper: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/

"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from PIL import Image
import numpy as np
import torch
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSRDataParserConfig

CONSOLE = Console(width=120)


def _find_files(directory: str, exts: List[str]):
    """Find all files in a directory that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []


def _parse_osm_txt(filename: str):
    """Parse a text file containing numbers and return a 4x4 numpy array of float32 values.

    Args:
        filename : a file containing numbers in a 4x4 matrix.

    Returns:
        A numpy array of shape [4, 4] containing the numbers from the file.
    """
    assert os.path.isfile(filename)
    with open(filename, encoding="UTF-8") as f:
        nums = f.read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)


def get_camera_params(
    scene_dir: str, split: Literal["train", "validation", "test"]
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load camera intrinsic and extrinsic parameters for a given scene split.

    Args"
      scene_dir : The directory containing the scene data.
      split : The split for which to load the camera parameters.

    Returns
        A tuple containing the intrinsic parameters (as a torch.Tensor of shape [N, 4, 4]),
        the camera-to-world matrices (as a torch.Tensor of shape [N, 4, 4]), and the number of cameras (N).
    """
    split_dir = f"{scene_dir}/{split}"

    # camera parameters files
    intrinsics_files = _find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
    pose_files = _find_files(f"{split_dir}/pose", exts=["*.txt"])

    num_cams = len(pose_files)

    intrinsics = []
    camera_to_worlds = []
    for i in range(num_cams):
        intrinsics.append(_parse_osm_txt(intrinsics_files[i]))

        pose = _parse_osm_txt(pose_files[i])

        # convert from COLMAP/OpenCV to nerfstudio camera (OpenGL/Blender)
        pose[0:3, 1:3] *= -1

        camera_to_worlds.append(pose)

    intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))  # [N, 4, 4]
    camera_to_worlds = torch.from_numpy(np.stack(camera_to_worlds).astype(np.float32))  # [N, 4, 4]

    return intrinsics, camera_to_worlds, num_cams


@dataclass
class NeRFOSRCityScapesDataParserConfig(NeRFOSRDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NeRFOSRCityScapes)
    """target class to instantiate"""
    mask_source: Literal["none", "original", "cityscapes"] = "cityscapes"
    """Source of masks, can be none, cityscapes not provided in original dataset."""


@dataclass
class NeRFOSRCityScapes(DataParser):
    """NeRFOSR Dataparser
    Presented in the paper: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/

    Some of this code comes from https://github.com/r00tman/NeRF-OSR/blob/main/data_loader_split.py

    Source data convention is:
      camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
      poses is camera-to-world
      masks are 0 for dynamic content, 255 for static content
    """

    config: NeRFOSRCityScapesDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        data = self.config.data
        scene = self.config.scene
        split = "validation" if split == "val" else split

        if scene == "trevi":
            scene_dir = f"{data}/{scene}/final_clean"
            split_dir = f"{data}/{scene}/final_clean/{split}"
        else:
            scene_dir = f"{data}/{scene}/final"
            split_dir = f"{data}/{scene}/final/{split}"

        # get all split cam params
        intrinsics_train, camera_to_worlds_train, n_train = get_camera_params(scene_dir, "train")
        intrinsics_val, camera_to_worlds_val, n_val = get_camera_params(scene_dir, "validation")
        intrinsics_test, camera_to_worlds_test, _ = get_camera_params(scene_dir, "test")

        # combine all cam params
        intrinsics = torch.cat([intrinsics_train, intrinsics_val, intrinsics_test], dim=0)
        camera_to_worlds = torch.cat([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], dim=0)

        camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # get average z component of camera-to-world translation and shift all cameras by that amount towards the origin
        # just to move the cameras onto the z=0 plane
        camera_to_worlds[:, 2, 3] -= torch.mean(camera_to_worlds[:, 2, 3], dim=0)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        if split == "train":
            camera_to_worlds = camera_to_worlds[:n_train]
            intrinsics = intrinsics[:n_train]
        elif split == "validation":
            camera_to_worlds = camera_to_worlds[n_train : n_train + n_val]
            intrinsics = intrinsics[n_train : n_train + n_val]
        elif split == "test":
            camera_to_worlds = camera_to_worlds[n_train + n_val :]
            intrinsics = intrinsics[n_train + n_val :]

        # c2w_colmap = camera_to_worlds.detach().clone()
        # # convert to COLMAP/OpenCV convention from NeRFStudio
        # c2w_colmap[:, 0:3, 1:3] *= -1

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # --- images ---
        image_filenames = _find_files(f"{split_dir}/rgb", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        # --- masks ---
        mask_filenames = []
        semantics = None
        if self.config.mask_source == "original":
            mask_filenames = _find_files(f"{split_dir}/mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
        elif self.config.mask_source == "cityscapes":
            panoptic_classes = load_from_json(Path(data) / "cityscapes_classes.json")
            classes = panoptic_classes["classes"]
            colors = torch.tensor(panoptic_classes["colours"], dtype=torch.uint8)
            segmentation_filenames = _find_files(
                f"{split_dir}/cityscapes_mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"]
            )
            semantics = Semantics(
                filenames=segmentation_filenames,
                classes=classes,
                colors=colors,
            )
            masks = []
            fg_masks = []
            for i, _ in enumerate(segmentation_filenames):
                mask = self.get_mask_from_semantics(
                    idx=i,
                    semantics=semantics,
                    mask_classes=["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
                )
                # mask shape is (H, W) we want H, W
                mask = (~mask).float()

                # handle foreground mask
                fg_mask = self.get_mask_from_semantics(idx=i, semantics=semantics, mask_classes=["sky"])
                fg_mask = fg_mask.unsqueeze(-1).float()

                masks.append(mask)
                fg_masks.append(fg_mask)

        metadata = {
            "semantics": None,
            # "depth_filenames": None,
            # "normal_filenames": None,
            # "include_mono_prior": False,
            "mask": masks,
            "fg_mask": fg_masks,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs

    def get_mask_from_semantics(self, idx, semantics, mask_classes):
        """function to get mask from semantics"""
        filepath = semantics.filenames[idx]
        pil_image = Image.open(filepath)
        semantic_img = torch.from_numpy(np.array(pil_image, dtype="int32"))
        mask = torch.zeros_like(semantic_img[:, :, 0])
        combined_mask = torch.zeros_like(semantic_img[:, :, 0])

        for mask_class in mask_classes:
            class_colour = semantics.colors[semantics.classes.index(mask_class)].type_as(semantic_img)
            class_mask = torch.where(
                torch.all(torch.eq(semantic_img, class_colour), dim=2), torch.ones_like(mask), mask
            )
            combined_mask += class_mask
        combined_mask = combined_mask.bool()
        return combined_mask
