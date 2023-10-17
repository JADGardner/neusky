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
from collections import defaultdict
from typing import List, Tuple, Type
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from PIL import Image
import numpy as np
import torch
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
from typing_extensions import Literal
from mmseg.apis import MMSegInferencer

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


def _find_files(directory: str, exts: List[str], recursive: bool = False) -> List[str]:
    """Find all files in a directory (and its subdirectories based on the recursive flag) that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.
        recursive : Whether to search in subdirectories as well. Defaults to False.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        files_grabbed = []
        for ext in exts:
            pattern = os.path.join(directory, "**", ext) if recursive else os.path.join(directory, ext)
            files_grabbed.extend(glob.glob(pattern, recursive=recursive))

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
    crop_to_equal_size: bool = False
    """Crop images to equal size"""
    pad_to_equal_size: bool = False
    """Pad images to equal size"""
    run_segmentation_inference: bool = False
    """Run segmentation inference on images if none are provided"""
    segmentation_model: str = "ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024"
    """Segmentation model to use for inference"""
    mask_vegetation: bool = False
    """Include vegetation in transient masks"""
    session_holdout_indices: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    """Indices of images within sessions (idx are relative to session) to use for evaluation"""
    session_env_map_scaling: float = 1.0
    """Scaling factor for session environment maps as per NeRF-OSR relighting benchmark"""
    session_env_map_scaling_threshold: float = 0.0
    """Threshold for pixels scaled by session_env_map_scaling as per NeRF-OSR relighting benchmark"""


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

    def __post_init__(self):
        assert not (
            self.config.crop_to_equal_size and self.config.pad_to_equal_size
        ), "Cannot crop and pad at the same time"

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

        self.width_height = []
        if self.config.crop_to_equal_size:
            min_cx = torch.min(intrinsics[:, 0, 2])
            min_cy = torch.min(intrinsics[:, 1, 2])
            self.width_height = [int(min_cx.item() * 2), int(min_cy.item() * 2)]
            intrinsics[:, 0, 2] = min_cx
            intrinsics[:, 1, 2] = min_cy

        if self.config.pad_to_equal_size:
            max_cx = torch.max(intrinsics[:, 0, 2])
            max_cy = torch.max(intrinsics[:, 1, 2])
            self.width_height = [int(max_cx.item() * 2), int(max_cy.item() * 2)]
            intrinsics[:, 0, 2] = max_cx
            intrinsics[:, 1, 2] = max_cy

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

        # --- environment maps ---
        envmap_filenames = _find_files(
            f"{scene_dir}/ENV_MAP_CC", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"], recursive=True
        )

        # load a single envmap to get its size
        envmap = Image.open(envmap_filenames[0])
        envmap_width, envmap_height = envmap.size

        c2w = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=torch.float32).repeat(
            len(envmap_filenames), 1, 1
        )

        envmap_cameras = Cameras(
            fx=torch.tensor(envmap_height, dtype=torch.float32).repeat(len(envmap_filenames)),
            fy=torch.tensor(envmap_height, dtype=torch.float32).repeat(len(envmap_filenames)),
            cx=torch.tensor(envmap_width // 2, dtype=torch.float32).repeat(len(envmap_filenames)),
            cy=torch.tensor(envmap_height // 2, dtype=torch.float32).repeat(len(envmap_filenames)),
            camera_to_worlds=c2w,
            camera_type=CameraType.EQUIRECTANGULAR,
        )

        # --- session IDs ---
        # names of sessions are the folders within scene_dir/ENV_MAP
        sessions = [os.path.basename(x) for x in glob.glob(f"{scene_dir}/ENV_MAP_CC/*")]
        session_to_indices = defaultdict(list)

        for idx, filename in enumerate(image_filenames):
            # if filename contains a session name, use that as the session ID
            # if no match just skip so as to not have sessions with no images
            for session in sessions:
                if session in filename:
                    session_to_indices[session].append(int(idx))

        # update keys from strings to integers from 0 to len(session_to_indices) - 1
        session_to_indices = {i: session_to_indices[k] for i, k in enumerate(session_to_indices.keys())}

        # also create mapping from indices to sessions
        indices_to_session = defaultdict(list)
        for session_idx, indices in session_to_indices.items():
            for idx in indices:
                indices_to_session[idx] = session_idx

        if split in ["validation", "test"]:
            session_to_indices = dict(session_to_indices)
            assert len(self.config.session_holdout_indices) == len(
                session_to_indices
            ), "number of relative eval indicies must match number of unique sessions"

        # --- masks ---
        mask_filenames = None
        segmentation_filenames = None
        semantics = None
        if self.config.mask_source == "original":
            mask_filenames = _find_files(f"{split_dir}/mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
        elif self.config.mask_source == "cityscapes":
            panoptic_classes = CITYSCAPE_CLASSES
            classes = panoptic_classes["classes"]
            colors = torch.tensor(panoptic_classes["colours"], dtype=torch.uint8)
            segmentation_folder = f"{split_dir}/cityscapes_mask"

            if not os.path.exists(segmentation_folder):
                if not self.config.run_segmentation_inference:
                    raise ValueError(
                        f"Cityscapes segmentation folder {segmentation_folder} does not exist and run inference is False"
                    )
                else:
                    self.run_segmentation_inference(image_filenames=image_filenames, output_folder=segmentation_folder)

            segmentation_filenames = _find_files(
                f"{split_dir}/cityscapes_mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"]
            )

            semantics = Semantics(
                filenames=segmentation_filenames,
                classes=classes,
                colors=colors,
            )

        metadata = {
            "semantics": semantics,
            "session_to_indices": session_to_indices,
            "indices_to_session": indices_to_session,
            "session_holdout_indices": self.config.session_holdout_indices,
            "envmap_filenames": envmap_filenames,
            "envmap_cameras": envmap_cameras,
            "depth_filenames": None,
            "normal_filenames": None,
            "include_mono_prior": False,
            "c2w_colmap": None,
            "crop_to_equal_size": self.config.crop_to_equal_size,
            "pad_to_equal_size": self.config.pad_to_equal_size,
            "width_height": self.width_height,
            "mask_vegetation": self.config.mask_vegetation,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs

    def run_segmentation_inference(self, image_filenames, output_folder):
        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        inferencer = MMSegInferencer(model=self.config.segmentation_model)
        target_size = (1024, 1024)

        def load_and_resize_image(img_path, target_size=(1024, 1024)):
            """Load and resize an image to the given target size."""
            original_img = Image.open(img_path)
            original_shape = original_img.size
            resized_img = original_img.resize(target_size)
            return np.array(resized_img), original_shape

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Running segmentation inference... ", total=len(image_filenames))
            for image_filename in tqdm(image_filenames):
                _, resized_img, original_shape = load_and_resize_image(image_filename)
                out = inferencer(resized_img)
                predictions = out["predictions"]  # [1024, 1024]
                predictions = predictions.astype(np.uint8)
                predictions = Image.fromarray(predictions).resize(original_shape, Image.NEAREST)
                # image_filename will end in .jpg or .JPG we want to save as .png
                # and save in the output folder
                output_filename = os.path.join(
                    output_folder, os.path.basename(image_filename).replace(".jpg", ".png").replace(".JPG", ".png")
                )
                predictions.save(output_filename)
                progress.update(task, advance=1)
