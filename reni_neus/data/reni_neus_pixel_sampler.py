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
Code for sampling pixels.
"""

import torch
from typing import Dict, Optional, Union
from torchtyping import TensorType

from nerfstudio.data.pixel_samplers import PixelSampler


class RENINeuSPixelSampler(PixelSampler):
    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        super().__init__(num_rays_per_batch=num_rays_per_batch, keep_full_image=keep_full_image, **kwargs)

    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = torch.randint(0, len(nonzero_indices), (batch_size,))
            height_width_indices = nonzero_indices[chosen_indices]  # shape (batch_size, 2)
            image_index = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # shape (batch_size, 1)
            indices = torch.cat((image_index, height_width_indices), dim=1)  # shape (batch_size, 3)
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_fg_masks = []
        all_transient_masks = []
        all_semantic_masks = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )

                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

                all_transient_masks.append(batch["mask"][i][indices[:, 1], indices[:, 2]])

                if "fg_mask" in batch:
                    all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])
                if "semantic" in batch:
                    all_semantic_masks.append(batch["semantic"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx"
            and key != "image"
            and key != "mask"
            and key != "fg_mask"
            and key != "semantic"
            and key != "normal"
            and key != "depth"
            and key != "sparse_pts"
            and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        if len(all_fg_masks) > 0:
            collated_batch["fg_mask"] = torch.cat(all_fg_masks, dim=0)

        if len(all_transient_masks) > 0:
            collated_batch["mask"] = torch.cat(all_transient_masks, dim=0)

        if len(all_semantic_masks) > 0:
            collated_batch["semantic"] = torch.cat(all_semantic_masks, dim=0)

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
