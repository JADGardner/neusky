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
from typing import Dict, Optional, Union, Type
from torchtyping import TensorType
from dataclasses import dataclass, field

import random

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig

@dataclass
class RENINeuSPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: RENINeuSPixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""

class RENINeuSPixelSampler(PixelSampler):

    config: RENINeuSPixelSamplerConfig

    def __init__(self, config: RENINeuSPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

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
          all_masks = []

          if "mask" in batch:
              num_rays_in_batch = num_rays_per_batch // num_images
              for i in range(num_images):
                  image_height, image_width, _ = batch["image"][i].shape

                  if i == num_images - 1:
                      num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                  indices = self.sample_method(
                      num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                  )
                  indices[:, 0] = i
                  all_indices.append(indices)
                  all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                  all_masks.append(batch["mask"][i][indices[:, 1], indices[:, 2]])
          else:
              num_rays_in_batch = num_rays_per_batch // num_images
              for i in range(num_images):
                  image_height, image_width, _ = batch["image"][i].shape
                  if i == num_images - 1:
                      num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                  if self.config.is_equirectangular:
                      indices = self.sample_method_equirectangular(
                          num_rays_in_batch, 1, image_height, image_width, device=device
                      )
                  else:
                      indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                  indices[:, 0] = i
                  all_indices.append(indices)
                  all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

          indices = torch.cat(all_indices, dim=0)

          c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
          collated_batch = {
              key: value[c, y, x]
              for key, value in batch.items()
              if key != "image_idx" and key != "image" and key != "mask" and value is not None
          }

          collated_batch["image"] = torch.cat(all_images, dim=0)

          if "mask" in batch:
              collated_batch["mask"] = torch.cat(all_masks, dim=0)

          assert collated_batch["image"].shape[0] == num_rays_per_batch

          # Needed to correct the random indices to their actual camera idx locations.
          indices[:, 0] = batch["image_idx"][c]
          collated_batch["indices"] = indices  # with the abs camera indices

          if keep_full_image:
              collated_batch["full_image"] = batch["image"]

          return collated_batch

    def collate_sky_ray_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'sky_mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        assert "fg_mask" in batch, "fg_mask must be in batch"

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        mask = 1.0 - batch["mask"][..., 1] # mask[..., 1] for fg_mask, 1 - fg_mask is the sky mask

        indices = self.sample_method(
            num_rays_per_batch, num_images, image_height, image_width, mask=mask, device=device
        )

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch