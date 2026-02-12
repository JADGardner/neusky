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
"""Custom dataparser for synthetic multi-illumination datasets.

Reads a transforms.json file (instant-ngp / BlenderNeRF format) and
PNG images organised into train/validation/test splits.

Expected directory layout (produced by scripts/prepare_synthetic_data.py):

    <data>/
        transforms.json          # camera poses for ALL frames
        train/
            rgb/*.png
            cityscapes_mask/*.png
        validation/
            rgb/*.png
            cityscapes_mask/*.png
        test/
            rgb/*.png
            cityscapes_mask/*.png
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Type

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

_FRAME_NUM_RE = re.compile(r"(\d+)")


def _extract_frame_num(filename: str) -> int:
    """Extract the last group of digits from a filename."""
    matches = _FRAME_NUM_RE.findall(Path(filename).stem)
    if not matches:
        raise ValueError(f"Cannot extract frame number from {filename}")
    return int(matches[-1])


def _find_files(directory: str, exts: List[str]) -> List[str]:
    """Find files with given extensions in a directory, sorted."""
    if not os.path.isdir(directory):
        return []
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


@dataclass
class CustomNeuskyDataparserConfig(DataParserConfig):
    """Neusky Dataparser Config for synthetic datasets"""

    _target: Type = field(default_factory=lambda: CustomNeuskyDataparser)
    """target class to instantiate"""
    data: Path = Path("path/to/data")
    """Directory specifying location of data."""
    transforms_filename: str = "transforms.json"
    """Name of the transforms file relative to data root."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical"
    """The method to use for orientation. 'vertical' needed for RENI++."""
    center_method: Literal["poses", "focus", "none"] = "focus"
    """The method to use for centering."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    mask_vegetation: bool = False
    """Include vegetation in transient masks"""
    include_sidewalk_in_ground_mask: bool = True
    """Include sidewalk in ground mask"""


@dataclass
class CustomNeuskyDataparser(DataParser):
    """Custom Neusky Dataparser for synthetic multi-illumination datasets.

    Reads transforms.json (instant-ngp / BlenderNeRF format) and PNG images
    split into train/validation/test directories.

    Camera poses in the transforms file are assumed to be in the nerfstudio /
    OpenGL / Blender convention (x-right, y-up, -z-forward).
    """

    config: CustomNeuskyDataparserConfig

    def _load_transforms(self):
        """Load the transforms JSON and build a frame_num -> pose/intrinsics map."""
        transforms_path = self.config.data / self.config.transforms_filename
        with open(transforms_path, "r") as f:
            meta = json.load(f)

        fl_x = float(meta["fl_x"])
        fl_y = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])

        frame_data = {}
        for frame in meta["frames"]:
            frame_num = _extract_frame_num(frame["file_path"])
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            frame_data[frame_num] = {
                "c2w": c2w,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
            }
        return frame_data

    def _get_split_files(self, split: str, subdir: str = "rgb"):
        """Find files for a split. Returns sorted list of paths."""
        split_name = "validation" if split == "val" else split
        d = self.config.data / split_name / subdir
        return _find_files(str(d), ["*.png", "*.jpg", "*.PNG", "*.JPG"])

    def _generate_dataparser_outputs(self, split="train"):
        frame_data = self._load_transforms()

        # --- Collect matched images + poses for ALL splits (consistent normalisation) ---
        all_c2w = []
        all_fx, all_fy, all_cx, all_cy = [], [], [], []
        per_split_images = {}    # split_name -> [image_path, ...]
        per_split_masks = {}     # split_name -> [mask_path, ...]
        split_counts = {}        # split_name -> count of matched frames

        for s in ["train", "val", "test"]:
            rgb_files = self._get_split_files(s, "rgb")
            mask_files = self._get_split_files(s, "cityscapes_mask")

            # Build frame_num -> mask_path lookup
            mask_by_num = {}
            for mf in mask_files:
                mask_by_num[_extract_frame_num(mf)] = mf

            matched_images = []
            matched_masks = []
            count = 0
            for img_path in rgb_files:
                fnum = _extract_frame_num(img_path)
                if fnum not in frame_data:
                    continue
                fd = frame_data[fnum]
                all_c2w.append(fd["c2w"])
                all_fx.append(fd["fl_x"])
                all_fy.append(fd["fl_y"])
                all_cx.append(fd["cx"])
                all_cy.append(fd["cy"])
                matched_images.append(img_path)
                matched_masks.append(mask_by_num.get(fnum))
                count += 1

            per_split_images[s] = matched_images
            per_split_masks[s] = matched_masks
            split_counts[s] = count

        total = sum(split_counts.values())
        if total == 0:
            raise ValueError(
                f"No images found matching transforms in {self.config.data}. "
                "Run scripts/prepare_synthetic_data.py first."
            )

        camera_to_worlds = torch.from_numpy(np.stack(all_c2w))  # [N, 4, 4]

        # --- Normalise poses across all splits ---
        camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Shift cameras so mean z is at origin
        camera_to_worlds[:, 2, 3] -= torch.mean(camera_to_worlds[:, 2, 3], dim=0)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))
        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        # --- Slice out the requested split ---
        query = split if split != "val" else "val"
        # Build cumulative offsets
        offset = 0
        for s in ["train", "val", "test"]:
            if s == query:
                break
            offset += split_counts[s]
        count = split_counts.get(query, 0)

        # Fallback to train if requested split is empty
        if count == 0:
            query = "train"
            offset = 0
            count = split_counts["train"]

        sl = slice(offset, offset + count)
        c2w_split = camera_to_worlds[sl]
        fx = torch.tensor(all_fx[sl], dtype=torch.float32)
        fy = torch.tensor(all_fy[sl], dtype=torch.float32)
        cx_t = torch.tensor(all_cx[sl], dtype=torch.float32)
        cy_t = torch.tensor(all_cy[sl], dtype=torch.float32)

        cameras = Cameras(
            camera_to_worlds=c2w_split[:, :3, :4],
            fx=fx,
            fy=fy,
            cx=cx_t,
            cy=cy_t,
            camera_type=CameraType.PERSPECTIVE,
        )

        # Scene box
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale],
                 [aabb_scale, aabb_scale, aabb_scale]],
                dtype=torch.float32,
            )
        )

        # Image and mask filenames for this split
        image_filenames = per_split_images[query]
        mask_filenames_list = per_split_masks[query]

        # Cityscapes segmentation
        segmentation_filenames = [m for m in mask_filenames_list if m is not None]
        panoptic_classes = CITYSCAPE_CLASSES
        classes = panoptic_classes["classes"]
        colors = torch.tensor(panoptic_classes["colours"], dtype=torch.uint8)

        semantics = None
        if segmentation_filenames and len(segmentation_filenames) == len(image_filenames):
            semantics = Semantics(
                filenames=segmentation_filenames,
                classes=classes,
                colors=colors,
            )

        metadata = {
            "semantics": semantics,
            "session_to_indices": None,
            "indices_to_session": None,
            "session_holdout_indices": [],
            "envmap_filenames": [],
            "envmap_cameras": None,
            "depth_filenames": None,
            "normal_filenames": None,
            "include_mono_prior": False,
            "c2w_colmap": None,
            "crop_to_equal_size": False,
            "pad_to_equal_size": False,
            "width_height": [],
            "mask_vegetation": self.config.mask_vegetation,
            "test_eval_mask_dict": {},
            "out_of_view_frustum_objects_masks": [None] * len(image_filenames),
            "include_sidewalk_in_ground_mask": self.config.include_sidewalk_in_ground_mask,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs
