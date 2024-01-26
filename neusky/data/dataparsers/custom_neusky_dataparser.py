# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from typing import List, Literal, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox

CITYSCAPE_CLASSES = {
    "classes": [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ],
    "colours": [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
}

@dataclass
class CustomNeuskyDataparserConfig(DataParserConfig):
    """Neusky Dataparser Config"""

    _target: Type = field(default_factory=lambda: CustomNeuskyDataparser)
    """target class to instantiate"""
    data: Path = Path("path/to/data")
    """Directory specifying location of data."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical" # Setting scene to algin with vertical axis, needed for RENI++
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "focus" # Setting focus of cemras as scene centre
    """The method to use for centering."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    mask_vegetation: bool = False
    """Include vegetation in transient masks"""

@dataclass
class CustomNeuskyDataparser(DataParser):
    """Custom Neusky Dataparser
    """

    config: CustomNeuskyDataparserConfig

    def _generate_dataparser_outputs(self, split="train"):
        data = self.config.data

        # TODO: Get camera_to_worlds for each image in the split
        # Ensure they are in the correct coordinate system as per https://docs.nerf.studio/quickstart/data_conventions.html
        camera_to_worlds = None

        # align scene with vertical axis and center at origin
        camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # TODO: Get intrinsics for split
        intrinsics = None

        # get average z component of camera-to-world translation and shift all cameras by that amount towards the origin
        # just to move the cameras onto the z=0 plane, this assumes cameras are all taken from roughly the same height
        camera_to_worlds[:, 2, 3] -= torch.mean(camera_to_worlds[:, 2, 3], dim=0)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))
        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        # TODO: Set up Cameras
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
        # TODO Get image filenames
        image_filenames = None

        # --- get cityscapes segmentations ---
        # TODO: Get segmentation filenames
        segmentation_filenames = None
        panoptic_classes = CITYSCAPE_CLASSES
        classes = panoptic_classes["classes"]
        colors = torch.tensor(panoptic_classes["colours"], dtype=torch.uint8)

        semantics = Semantics(
            filenames=segmentation_filenames,
            classes=classes,
            colors=colors,
        )
 
        metadata = {
            "semantics": semantics,
            "mask_vegetation": self.config.mask_vegetation,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs