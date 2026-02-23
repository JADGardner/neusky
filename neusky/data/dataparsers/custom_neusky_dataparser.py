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
        points3d.ply             # (optional) SfM point cloud for centering
        train/
            rgb/*.png
            cityscapes_mask/*.png
        validation/
            rgb/*.png
            cityscapes_mask/*.png
            albedo/*.exr          # GT layers (val/test only)
            normal/*.exr
            depth/*.exr
            roughness/*.exr
            metallic/*.exr
            ior/*.exr
            transmission/*.exr
        test/
            rgb/*.png
            cityscapes_mask/*.png
            albedo/*.exr
            ...
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type

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
from nerfstudio.utils.rich_utils import CONSOLE

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

GT_LAYER_NAMES = ["albedo", "normal", "depth", "roughness", "metallic", "ior", "transmission"]


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
    center_method_sfm: bool = False
    """Use SfM point cloud for scene centering instead of camera positions."""
    sfm_outlier_percentile: float = 95.0
    """Keep closest N% of SfM points when computing center (filters outliers)."""
    points3d_filename: str = "points3d.ply"
    """Name of the SfM point cloud file relative to data root."""


@dataclass
class CustomNeuskyDataparser(DataParser):
    """Custom Neusky Dataparser for synthetic multi-illumination datasets.

    Reads transforms.json (instant-ngp / BlenderNeRF format) and PNG images
    split into train/validation/test directories.

    Camera poses in the transforms file are assumed to be in the nerfstudio /
    OpenGL / Blender convention (x-right, y-up, -z-forward).
    """

    config: CustomNeuskyDataparserConfig

    def _load_transforms(self) -> Dict[str, dict]:
        """Load the transforms JSON and build a file_path -> pose/intrinsics map.

        Uses per-frame intrinsics with fallback to global defaults.
        Keys are the file_path strings from the JSON (e.g. 'train/rgb/0000.png').
        """
        transforms_path = self.config.data / self.config.transforms_filename
        with open(transforms_path, "r") as f:
            meta = json.load(f)

        # Global defaults
        default_fl_x = float(meta["fl_x"])
        default_fl_y = float(meta["fl_y"])
        default_cx = float(meta["cx"])
        default_cy = float(meta["cy"])

        frame_data = {}
        for frame in meta["frames"]:
            file_path = frame["file_path"]
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            frame_data[file_path] = {
                "c2w": c2w,
                "fl_x": float(frame.get("fl_x", default_fl_x)),
                "fl_y": float(frame.get("fl_y", default_fl_y)),
                "cx": float(frame.get("cx", default_cx)),
                "cy": float(frame.get("cy", default_cy)),
            }
        return frame_data

    def _get_split_files(self, split: str, subdir: str = "rgb"):
        """Find files for a split. Returns sorted list of paths."""
        split_name = "validation" if split == "val" else split
        d = self.config.data / split_name / subdir
        return _find_files(str(d), ["*.png", "*.jpg", "*.PNG", "*.JPG"])

    def _discover_gt_layers(self, split: str, image_filenames: List[str]) -> Dict[str, List[str]]:
        """Discover ground-truth EXR layers for a split.

        Checks for {split}/albedo/*.exr, {split}/normal/*.exr, etc.
        Returns a dict mapping layer names to aligned lists of file paths.
        Only returns layers where every image has a matching EXR file.
        """
        split_name = "validation" if split == "val" else split
        gt_layers = {}

        # Build stem -> index lookup from image filenames
        stem_to_idx = {}
        for i, img_path in enumerate(image_filenames):
            stem_to_idx[Path(img_path).stem] = i

        for layer_name in GT_LAYER_NAMES:
            layer_dir = self.config.data / split_name / layer_name
            if not layer_dir.is_dir():
                continue

            exr_files = _find_files(str(layer_dir), ["*.exr", "*.EXR"])
            if not exr_files:
                continue

            # Build stem -> exr_path lookup
            exr_by_stem = {}
            for ef in exr_files:
                exr_by_stem[Path(ef).stem] = ef

            # Try to match every image to an EXR
            aligned = [None] * len(image_filenames)
            all_matched = True
            for stem, idx in stem_to_idx.items():
                if stem in exr_by_stem:
                    aligned[idx] = exr_by_stem[stem]
                else:
                    all_matched = False
                    break

            if all_matched:
                gt_layers[f"gt_{layer_name}_filenames"] = aligned
                CONSOLE.log(f"  Found GT layer '{layer_name}': {len(aligned)} files")

        return gt_layers

    def _load_sfm_points(self) -> Optional[np.ndarray]:
        """Load SfM point cloud from PLY file. Returns (N, 3) array or None."""
        ply_path = self.config.data / self.config.points3d_filename
        if not ply_path.exists():
            CONSOLE.log(f"[yellow]SfM point cloud not found: {ply_path}")
            return None

        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(ply_path))
            vertex = plydata["vertex"]
            points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
            CONSOLE.log(f"Loaded {len(points)} SfM points from {ply_path}")
            return points
        except ImportError:
            CONSOLE.log("[yellow]plyfile not installed, falling back to numpy PLY loading")
            # Simple binary PLY fallback using numpy
            return self._load_ply_numpy(ply_path)
        except Exception as e:
            CONSOLE.log(f"[yellow]Failed to load SfM points: {e}")
            return None

    def _load_ply_numpy(self, ply_path: Path) -> Optional[np.ndarray]:
        """Fallback PLY loader using numpy for binary_little_endian format."""
        try:
            with open(ply_path, "rb") as f:
                # Read header
                header_lines = []
                while True:
                    line = f.readline().decode("ascii").strip()
                    header_lines.append(line)
                    if line == "end_header":
                        break

                # Parse header for vertex count and format
                n_vertices = 0
                is_binary = False
                for line in header_lines:
                    if line.startswith("element vertex"):
                        n_vertices = int(line.split()[-1])
                    if "binary_little_endian" in line:
                        is_binary = True

                if n_vertices == 0:
                    return None

                if is_binary:
                    # Assume x,y,z float32 + r,g,b uint8 (common format)
                    dtype = np.dtype([
                        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                    ])
                    data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)
                    points = np.stack([data["x"], data["y"], data["z"]], axis=-1)
                    return points
                else:
                    # ASCII format
                    data = np.loadtxt(f, max_rows=n_vertices)
                    return data[:, :3].astype(np.float32)
        except Exception as e:
            CONSOLE.log(f"[yellow]Numpy PLY loading failed: {e}")
            return None

    def _compute_sfm_centering(self, points: np.ndarray):
        """Compute scene center and scale from SfM points.

        Filters outliers (keeps closest sfm_outlier_percentile% of points),
        then computes mean center and scale to normalize mean distance to 1.

        Returns (center, scale) as (np.ndarray[3], float).
        """
        # Initial center estimate
        median = np.median(points, axis=0)
        dists = np.linalg.norm(points - median, axis=1)

        # Filter outliers
        threshold = np.percentile(dists, self.config.sfm_outlier_percentile)
        inliers = points[dists <= threshold]
        CONSOLE.log(
            f"SfM centering: {len(inliers)}/{len(points)} inlier points "
            f"(percentile={self.config.sfm_outlier_percentile}%)"
        )

        # Compute center and scale from inliers
        center = np.mean(inliers, axis=0)
        mean_dist = np.mean(np.linalg.norm(inliers - center, axis=1))
        scale = 1.0 / max(mean_dist, 1e-6)

        CONSOLE.log(f"SfM center: {center}, scale: {scale:.4f}")
        return center, scale

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

            # Build stem -> mask_path lookup (stems are unique within a split)
            mask_by_stem = {}
            for mf in mask_files:
                mask_by_stem[Path(mf).stem] = mf

            matched_images = []
            matched_masks = []
            count = 0
            for img_path in rgb_files:
                # Compute relative path from data root to match transforms.json keys
                rel_path = str(Path(img_path).relative_to(self.config.data))
                if rel_path not in frame_data:
                    continue
                fd = frame_data[rel_path]
                all_c2w.append(fd["c2w"])
                all_fx.append(fd["fl_x"])
                all_fy.append(fd["fl_y"])
                all_cx.append(fd["cx"])
                all_cy.append(fd["cy"])
                matched_images.append(img_path)
                stem = Path(img_path).stem
                matched_masks.append(mask_by_stem.get(stem))
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

        CONSOLE.log(
            f"Matched frames: train={split_counts['train']}, "
            f"val={split_counts['val']}, test={split_counts['test']}"
        )

        camera_to_worlds = torch.from_numpy(np.stack(all_c2w))  # [N, 4, 4]

        # --- Normalise poses across all splits ---
        if self.config.center_method_sfm:
            # SfM centering: use auto_orient for up-vector alignment only
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=self.config.orientation_method,
                center_method="none",
            )

            sfm_points = self._load_sfm_points()
            if sfm_points is not None:
                # Apply the same orientation transform to the SfM points
                # transform is a 3x4 matrix: R|t
                R = transform[:3, :3].numpy()
                t = transform[:3, 3].numpy()
                sfm_points = (R @ sfm_points.T).T + t

                center, scale = self._compute_sfm_centering(sfm_points)

                # Apply SfM-derived centering to camera positions
                camera_to_worlds[:, :3, 3] -= torch.from_numpy(center).float()
                camera_to_worlds[:, :3, 3] *= scale * self.config.scale_factor
            else:
                CONSOLE.log("[yellow]SfM centering requested but no points found, falling back to pose centering")
                camera_to_worlds[:, 2, 3] -= torch.mean(camera_to_worlds[:, 2, 3], dim=0)
                if self.config.auto_scale_poses:
                    scale_factor = 1.0 / torch.max(torch.abs(camera_to_worlds[:, :3, 3]))
                    camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor
        else:
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

        # Discover ground-truth layers for this split
        gt_layers = self._discover_gt_layers(query, image_filenames)

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
        metadata.update(gt_layers)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs
