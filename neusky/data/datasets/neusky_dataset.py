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
NeuSky dataset with ground-truth EXR layer loading for evaluation.
"""
from copy import deepcopy
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataparserOutputs,
    Semantics,
)

GT_LAYER_NAMES = ["albedo", "normal", "depth", "roughness", "metallic", "ior", "transmission"]

# Number of channels per GT layer
GT_LAYER_CHANNELS = {
    "albedo": 3,
    "normal": 3,
    "depth": 1,
    "roughness": 1,
    "metallic": 1,
    "ior": 1,
    "transmission": 1,
}


def _load_exr(filepath: str, num_channels: int = 3) -> Optional[np.ndarray]:
    """Load an EXR file using OpenCV. Returns float32 array (H, W, C) or None on failure."""
    img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    if num_channels >= 3 and img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img.astype(np.float32)

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


class NeuSkyDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device: List[str] = [
        "image", "mask", "fg_mask", "ground_mask",
        "gt_albedo", "gt_normal", "gt_depth", "gt_roughness",
        "gt_metallic", "gt_ior", "gt_transmission",
    ]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, split: str = "train"):
        super().__init__(dataparser_outputs, scale_factor)

        # can be none if monoprior not included
        self.semantics = self.metadata["semantics"]
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.c2w_colmap = self.metadata["c2w_colmap"]
        self.include_mono_prior = self.metadata["include_mono_prior"]
        self.crop_to_equal_size = self.metadata["crop_to_equal_size"]
        self.pad_to_equal_size = self.metadata["pad_to_equal_size"]
        if self.crop_to_equal_size:
            self.min_width = self.metadata["width_height"][0]
            self.min_height = self.metadata["width_height"][1]
        if self.pad_to_equal_size:
            self.max_width = self.metadata["width_height"][0]
            self.max_height = self.metadata["width_height"][1]

        self.metadata["c2w"] = dataparser_outputs.cameras.camera_to_worlds
        self.envmap_cameras = deepcopy(self.metadata["envmap_cameras"])
        if dataparser_outputs.metadata["session_to_indices"] is not None:
            self.metadata["num_sessions"] = len(dataparser_outputs.metadata["session_to_indices"])
        self.test_eval_mask_dict = dataparser_outputs.metadata["test_eval_mask_dict"]
        self.out_of_view_frustum_objects_masks = dataparser_outputs.metadata["out_of_view_frustum_objects_masks"]
        self.split = split

        # Extract GT layer filenames from metadata (populated by dataparser for val/test)
        self.gt_layer_filenames = {}
        for layer_name in GT_LAYER_NAMES:
            key = f"gt_{layer_name}_filenames"
            if key in self.metadata and self.metadata[key] is not None:
                self.gt_layer_filenames[layer_name] = self.metadata[key]

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.crop_to_equal_size:
            # Crop the image to the new size around the center point
            width, height = pil_image.size
            left = max((width - self.min_width) // 2, 0)
            top = max((height - self.min_height) // 2, 0)
            right = min((width + self.min_width) // 2, width)
            bottom = min((height + self.min_height) // 2, height)
            pil_image = pil_image.crop((left, top, right, bottom))
        if self.pad_to_equal_size:
            width, height = pil_image.size
            left_pad = (self.max_width - width) // 2
            top_pad = (self.max_height - height) // 2
            new_image = Image.new("RGB", (self.max_width, self.max_height), (0, 0, 0))
            # Paste the original image at its center
            new_image.paste(pil_image, (left_pad, top_pad))
            pil_image = new_image
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}

        metadata["mask"] = self.get_mask(data["image_idx"])

        # Load GT EXR layers for evaluation (val/test splits only)
        idx = data["image_idx"]
        for layer_name, filenames in self.gt_layer_filenames.items():
            filepath = filenames[idx]
            if filepath is None:
                continue
            num_ch = GT_LAYER_CHANNELS.get(layer_name, 3)
            arr = _load_exr(filepath, num_channels=num_ch)
            if arr is None:
                continue
            # Apply same spatial downscale as images
            if self.scale_factor != 1.0:
                h, w = arr.shape[:2]
                new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
                arr_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                arr_t = torch.nn.functional.interpolate(arr_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                arr = arr_t.squeeze(0).permute(1, 2, 0).numpy()
            metadata[f"gt_{layer_name}"] = torch.from_numpy(arr)

        return metadata

    def get_mask(self, idx):
        mask = None
        if self.split == "test":
            if idx in self.test_eval_mask_dict.keys():
                mask_filename = self.test_eval_mask_dict[idx]
                mask = torch.from_numpy(np.array(Image.open(mask_filename), dtype="uint8")) # Shape (H, W)
                # set between 0 and 1
                mask = mask.float() / 255.0
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(-1)
                # if the final dimension is > 1, then just take the first channel
                if mask.shape[-1] > 1:
                    mask = mask[:, :, 0:1]
                mask = mask.float()  # 1 is static, 0 is transient # Shape (H, W, 1)

        transient_mask_classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        fg_mask_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "terrain",
        ]

        if self.metadata["mask_vegetation"]:
            transient_mask_classes.append("vegetation")
        else:
            fg_mask_classes.append("vegetation")

        if mask is None:
            mask = self.get_mask_from_semantics(
                idx=idx,
                semantics=self.semantics,
                mask_classes=transient_mask_classes,
            )

            mask = (~mask).unsqueeze(-1).float()  # 1 is static, 0 is transient           

        # get_foreground_mask
        fg_mask = self.get_mask_from_semantics(idx=idx, semantics=self.semantics, mask_classes=fg_mask_classes)
        fg_mask = fg_mask.unsqueeze(-1).float()  # 1 is foreground + statics, 0 is background + transients

        # get_ground_mask
        ground_mask_classes = ["road"]
        if self.metadata["include_sidewalk_in_ground_mask"]:
            ground_mask_classes.append("sidewalk")
        ground_mask = self.get_mask_from_semantics(idx=idx, semantics=self.semantics, mask_classes=ground_mask_classes)
        ground_mask = (ground_mask).unsqueeze(-1).float()  # 1 is ground, 0 is not ground

        # sky_mask
        sky_mask = self.get_mask_from_semantics(idx=idx, semantics=self.semantics, mask_classes=["sky"])
        sky_mask = sky_mask.unsqueeze(-1).float()  # 1 is sky, 0 is not sky

        if self.out_of_view_frustum_objects_masks[idx] is not None:
            object_mask = torch.from_numpy(np.array(Image.open(self.out_of_view_frustum_objects_masks[idx]), dtype="uint8"))[:, :, 0] # Shape (H, W) With 1 as Tree and 0 and Not Tree
            # convert to bool and invert
            object_mask = object_mask / 255.0
            object_mask = object_mask.bool()
            object_mask = ~object_mask
            object_mask = object_mask.unsqueeze(-1).float() # 1 is not tree, 0 is tree
            # now AND the tree mask and the mask
            mask = mask * object_mask
            fg_mask = fg_mask * object_mask

        # stack masks to shape H, W, 3
        mask = torch.cat([mask, fg_mask, ground_mask, sky_mask], dim=-1)

        if self.crop_to_equal_size:
            height, width = mask.shape[:2]
            left = max((width - self.min_width) // 2, 0)
            top = max((height - self.min_height) // 2, 0)
            right = min((width + self.min_width) // 2, width)
            bottom = min((height + self.min_height) // 2, height)
            mask = mask[top:bottom, left:right, :]

        if self.pad_to_equal_size:
            height, width = mask.shape[:2]
            # compute padding
            pad_left = (self.max_width - width) // 2
            pad_right = self.max_width - width - pad_left
            pad_top = (self.max_height - height) // 2
            pad_bottom = self.max_height - height - pad_top
            # Pad the mask to place it in the center
            mask = mask.permute(2, 0, 1)
            mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
            mask = mask.permute(1, 2, 0)

        if self.scale_factor != 1.0:
            h, w = mask.shape[:2]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            mask = mask.permute(2, 0, 1).unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=(new_h, new_w), mode='nearest')
            mask = mask.squeeze(0).permute(1, 2, 0)

        return mask

    def get_mask_from_semantics(self, idx, semantics, mask_classes):
        """function to get mask from semantics"""
        filepath = semantics.filenames[idx]
        pil_image = Image.open(filepath)

        semantic_img = torch.from_numpy(np.array(pil_image, dtype="int32"))[:, :, :3]

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

    def get_envmap(self, idx):
        """Returns the environment map of shape (3, H, W)."""
        envmap_filename = self._dataparser_outputs.envmap_filenames[idx]
        envmap = torch.from_numpy(np.array(Image.open(envmap_filename), dtype="float32") / 255.0).permute(2, 0, 1)
        return envmap
