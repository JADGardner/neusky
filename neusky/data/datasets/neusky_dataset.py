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

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
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


def fg_boundary_confidence(fg_mask: torch.Tensor, erode_pixels: int, soften_kernel: int) -> torch.Tensor:
    """Per-pixel confidence for mask-based density supervision.

    Semantic masks are unreliable at silhouette boundaries: a pixel labelled
    foreground there may partially cover sky, so binary cross-view density
    supervision is contradictory and produces SDF boundary floaters. Returns
    1 far from any foreground/background transition, ~0 within
    ``erode_pixels`` of one, optionally gaussian-softened for a smooth
    falloff (IDS fixed the same failure on NeRF-OSR with erode 30 /
    kernel 51 at full resolution).

    Args:
        fg_mask: (H, W, 1) float tensor, 1 = foreground.
        erode_pixels: half-width of the ignore band; <=0 returns all ones.
        soften_kernel: odd gaussian kernel size; <=1 keeps the hard band.
    """
    if erode_pixels <= 0:
        return torch.ones_like(fg_mask)
    m = fg_mask.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
    k = 2 * erode_pixels + 1
    dilated_fg = F.max_pool2d(m, kernel_size=k, stride=1, padding=erode_pixels)
    dilated_bg = F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=erode_pixels)
    confidence = 1.0 - dilated_fg * dilated_bg  # 0 within the transition band
    if soften_kernel and soften_kernel > 1:
        ks = soften_kernel + 1 if soften_kernel % 2 == 0 else soften_kernel
        sigma = ks / 6.0
        x = torch.arange(ks, dtype=confidence.dtype) - (ks - 1) / 2.0
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = (g / g.sum()).to(confidence)
        pad = ks // 2
        confidence = F.conv2d(F.pad(confidence, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, ks))
        confidence = F.conv2d(F.pad(confidence, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, ks, 1))
    return confidence.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def _fg_transition_mask(fg_mask: torch.Tensor) -> torch.Tensor:
    """(H, W) bool mask of pixels whose 3x3 neighbourhood contains both
    foreground and background (both sides of every fg/background transition)."""
    m = fg_mask.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
    dilated_fg = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    dilated_bg = F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    return (dilated_fg * dilated_bg)[0, 0] > 0.5


def _distance_to_transition_cv2(transition: torch.Tensor, clamp_px: float) -> torch.Tensor:
    """Chebyshev distance to the nearest transition pixel via cv2.distanceTransform.

    DIST_C with a 3x3 mask is exact for the Chebyshev metric, matching both the
    max-pool band geometry of ``fg_boundary_confidence`` and the iterative
    torch fallback.
    """
    import cv2

    src = (~transition).numpy().astype(np.uint8)  # 0 at transitions, 1 elsewhere
    dist = cv2.distanceTransform(src, cv2.DIST_C, 3)
    return torch.from_numpy(dist.astype(np.float32)).clamp(max=clamp_px)


def _distance_to_transition_torch(transition: torch.Tensor, clamp_px: float) -> torch.Tensor:
    """Pure-torch fallback: iterative 3x3 max-pool dilation of the transition
    mask. The first iteration at which a pixel is covered is exactly its
    Chebyshev distance, so this agrees with the cv2 DIST_C path."""
    dist = torch.full(transition.shape, float(clamp_px), dtype=torch.float32)
    dist[transition] = 0.0
    covered = transition.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    for i in range(1, int(clamp_px)):
        if bool((covered > 0.5).all()):
            break
        new_covered = F.max_pool2d(covered, kernel_size=3, stride=1, padding=1)
        newly = (new_covered[0, 0] > 0.5) & (covered[0, 0] <= 0.5)
        dist[newly] = float(i)
        covered = new_covered
    return dist


def fg_boundary_distance(fg_mask: torch.Tensor, clamp_px: float = 255.0) -> torch.Tensor:
    """Distance (px) from each pixel to the nearest fg/background transition.

    Written as mask channel 4 instead of the static confidence when the
    dataparser sets ``fg_boundary_distance_channel``. The model converts it to
    a boundary confidence whose ignore-band width is annealed to zero over
    training, so silhouettes re-sharpen once geometry has settled (the static
    band suppresses boundary floaters but leaves edges noisy/blurry).

    Chebyshev (L-inf) metric, matching the max-pool band of
    ``fg_boundary_confidence`` so a band of ``erode_pixels`` reproduces the
    static ignore band. Uses cv2.distanceTransform when available with a
    pure-torch iterative fallback; distances are clamped at ``clamp_px``.

    Args:
        fg_mask: (H, W, 1) float tensor, 1 = foreground.
        clamp_px: maximum distance value stored in the channel.
    """
    transition = _fg_transition_mask(fg_mask)
    if not bool(transition.any()):
        return torch.full_like(fg_mask, clamp_px)
    try:
        dist = _distance_to_transition_cv2(transition, clamp_px)
    except Exception:
        dist = _distance_to_transition_torch(transition, clamp_px)
    return dist.unsqueeze(-1)

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
    """Load an EXR file using pyexr. Returns float32 array (H, W, C) or None on failure."""
    try:
        import pyexr
        img = pyexr.open(filepath).get()  # float32, (H, W, C) or (H, W, 1)
    except Exception:
        return None
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    # Trim to expected channels (e.g. RGBA -> RGB)
    if img.shape[2] > num_channels:
        img = img[:, :, :num_channels]
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

        # Extract GT layer filenames from metadata. Validation/test load full images
        # for metrics; train loads only sampled pixels lazily to avoid caching all
        # full-resolution EXR layers in RAM.
        self.gt_layer_filenames = {}
        for layer_name in GT_LAYER_NAMES:
            key = f"gt_{layer_name}_filenames"
            if key in self.metadata and self.metadata[key] is not None:
                self.gt_layer_filenames[layer_name] = self.metadata[key]
        self._gt_layer_cache = {}
        self._gt_layer_cache_order = []
        self._gt_layer_cache_max_items = 24

    def set_gt_layer_cache_max_items(self, max_items: int) -> None:
        self._gt_layer_cache_max_items = max(1, int(max_items))
        while len(self._gt_layer_cache_order) > self._gt_layer_cache_max_items:
            old_key = self._gt_layer_cache_order.pop(0)
            self._gt_layer_cache.pop(old_key, None)

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

        # Load full GT EXR layers only for evaluation. For train, loading full
        # albedo/normal/depth tensors here would make CacheDataloader keep all
        # full-resolution GT layers in RAM. Train GT pixels are added lazily in
        # NeuSkyDataManager.next_train after ray sampling.
        if self.split != "train":
            idx = data["image_idx"]
            for layer_name in self.gt_layer_filenames:
                arr = self._load_gt_layer(layer_name, idx)
                if arr is not None:
                    metadata[f"gt_{layer_name}"] = arr

        return metadata

    def _load_gt_layer(self, layer_name: str, idx: int) -> Optional[torch.Tensor]:
        filenames = self.gt_layer_filenames.get(layer_name)
        if filenames is None:
            return None
        filepath = filenames[idx]
        if filepath is None:
            return None
        num_ch = GT_LAYER_CHANNELS.get(layer_name, 3)
        arr = _load_exr(filepath, num_channels=num_ch)
        if arr is None:
            return None
        if self.scale_factor != 1.0:
            h, w = arr.shape[:2]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            arr_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            arr_t = torch.nn.functional.interpolate(arr_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            arr = arr_t.squeeze(0).permute(1, 2, 0).numpy()
        return torch.from_numpy(arr)

    def _get_cached_gt_layer(self, layer_name: str, idx: int) -> Optional[torch.Tensor]:
        key = (layer_name, int(idx))
        if key in self._gt_layer_cache:
            self._gt_layer_cache_order.remove(key)
            self._gt_layer_cache_order.append(key)
            return self._gt_layer_cache[key]

        tensor = self._load_gt_layer(layer_name, int(idx))
        if tensor is None:
            return None
        self._gt_layer_cache[key] = tensor
        self._gt_layer_cache_order.append(key)
        while len(self._gt_layer_cache_order) > self._gt_layer_cache_max_items:
            old_key = self._gt_layer_cache_order.pop(0)
            self._gt_layer_cache.pop(old_key, None)
        return tensor

    def add_lazy_train_gt_to_batch(self, batch: Dict) -> Dict:
        if self.split != "train" or not self.gt_layer_filenames or "indices" not in batch:
            return batch

        indices = batch["indices"].cpu().long()
        image_indices = indices[:, 0]
        ys = indices[:, 1]
        xs = indices[:, 2]
        unique_image_indices = torch.unique(image_indices).tolist()

        for layer_name in ("albedo", "normal", "depth"):
            if layer_name not in self.gt_layer_filenames:
                continue
            num_ch = GT_LAYER_CHANNELS.get(layer_name, 3)
            values = torch.empty((indices.shape[0], num_ch), dtype=torch.float32)
            have_values = torch.zeros((indices.shape[0],), dtype=torch.bool)
            for img_idx in unique_image_indices:
                layer = self._get_cached_gt_layer(layer_name, int(img_idx))
                if layer is None:
                    continue
                selector = image_indices == int(img_idx)
                values[selector] = layer[ys[selector], xs[selector]]
                have_values[selector] = True
            if have_values.all():
                batch[f"gt_{layer_name}"] = values

        return batch

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

        # Boundary channel for the fg-mask density loss (channel 4). Two modes:
        # - static confidence (default): 1 far from foreground/background
        #   transitions, ->0 within the configured band, so mixed silhouette
        #   pixels stop injecting contradictory cross-view density supervision
        #   (SDF boundary floaters). All ones when disabled (erode 0), which
        #   keeps the loss identical.
        # - distance map (fg_boundary_distance_channel): distance in pixels to
        #   the nearest transition; the model derives an annealed confidence
        #   from it, shrinking the ignore band to zero over training so
        #   silhouettes re-sharpen after geometry settles.
        if self.metadata.get("fg_boundary_distance_channel", False):
            fg_boundary_conf = fg_boundary_distance(fg_mask)
        else:
            fg_boundary_conf = fg_boundary_confidence(
                fg_mask,
                int(self.metadata.get("fg_boundary_erode_pixels", 0) or 0),
                int(self.metadata.get("fg_boundary_soften_kernel", 0) or 0),
            )

        # stack masks to shape H, W, 5: [static, fg, ground, sky, fg_boundary_conf]
        mask = torch.cat([mask, fg_mask, ground_mask, sky_mask, fg_boundary_conf], dim=-1)

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
