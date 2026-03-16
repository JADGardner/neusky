"""
Pixel sampling for NeuSky.

Extends nerfstudio's PixelSampler with sky-region and image-half sampling
methods used during evaluation. Standard training sampling is inherited
from the parent PixelSampler.
"""

from __future__ import annotations

import torch
from typing import Dict, Type, Literal
from dataclasses import dataclass, field

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class NeuSkyPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for NeuSky pixel sampler."""

    _target: Type = field(default_factory=lambda: NeuSkyPixelSampler)
    num_rays_per_batch: int = 4096
    keep_full_image: bool = False
    is_equirectangular: bool = False


class NeuSkyPixelSampler(PixelSampler):
    """Pixel sampler with sky-region and image-half sampling for NeuSky evaluation.

    Training uses the parent PixelSampler.sample() directly.
    """

    config: NeuSkyPixelSamplerConfig

    def sample_method(self, *args, mask=None, **kwargs):
        """Override to slice multi-channel masks to channel 0 for pixel selection.

        NeuSky masks have 4 channels [static, fg, ground, sky]. Nerfstudio's
        rejection_sample_mask expects [N, H, W, 1]. We extract channel 0
        (static mask) for sampling, but the full mask is still used by the
        parent's collate method for indexing into the output batch.
        """
        if mask is not None and mask.shape[-1] > 1:
            mask = mask[..., 0:1]
        return super().sample_method(*args, mask=mask, **kwargs)

    def sample(self, image_batch: Dict) -> Dict:
        """Override to ensure indices are on CPU for ray generator compatibility."""
        result = super().sample(image_batch)

        # Ensure indices are on CPU — nerfstudio's ray_generator.image_coords is on CPU
        if "indices" in result and result["indices"].device.type != "cpu":
            result["indices"] = result["indices"].cpu()
        return result

    def collate_sky_ray_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """Sample pixels from sky regions (inverse of foreground mask, channel 1)."""
        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        # Invert fg_mask (channel 1) to get sky mask: [N, H, W, 1]
        mask = 1.0 - batch["mask"][..., 1:2]

        indices = self.sample_method(
            num_rays_per_batch, num_images, image_height, image_width, mask=mask, device=device
        )

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_sky_ray_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """Sample pixels from sky regions — ragged image list version."""
        device = batch["image"][0].device
        num_images = len(batch["image"])

        all_indices = []
        all_images = []
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            # Invert fg_mask (channel 1) to get sky mask: [H, W, 1]
            mask = 1.0 - batch["mask"][i][..., 1:2]
            image_height, image_width, _ = batch["image"][i].shape

            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

            indices = self.sample_method(
                num_rays_in_batch, 1, image_height, image_width, mask=mask.unsqueeze(0), device=device
            )
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_half(self, batch: Dict, num_rays_per_batch: int, sample_region: Literal['left_image_half', 'right_image_half', 'full_image']):
        """Sample pixels from a region of the image, masked by static mask (channel 0)."""
        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        # Static mask (channel 0): 1 = valid, 0 = transient. Shape [N, H, W, 1]
        mask = batch["mask"][..., 0:1].clone()

        if sample_region == 'left_image_half':
            mask[:, :, image_width // 2:, :] = 0
        elif sample_region == 'right_image_half':
            mask[:, :, :image_width // 2, :] = 0

        indices = self.sample_method(
            num_rays_per_batch, num_images, image_height, image_width, mask=mask, device=device
        )

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        return collated_batch

    def collate_image_half_list(self, batch: Dict, num_rays_per_batch: int, sample_region: Literal['left_image_half', 'right_image_half', 'full_image']):
        """Sample pixels from a region — ragged image list version."""
        device = batch["image"][0].device
        num_images = len(batch["image"])

        all_indices = []
        all_images = []
        all_masks = []
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            # Static mask (channel 0): [H, W, 1]
            mask = batch["mask"][i][..., 0:1].clone()
            image_height, image_width, _ = batch["image"][i].shape

            if sample_region == 'left_image_half':
                mask[:, image_width // 2:, :] = 0
            elif sample_region == 'right_image_half':
                mask[:, :image_width // 2, :] = 0

            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

            indices = self.sample_method(
                num_rays_in_batch, 1, image_height, image_width, mask=mask.unsqueeze(0), device=device
            )
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            all_masks.append(batch["mask"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)
        collated_batch["mask"] = torch.cat(all_masks, dim=0)

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        return collated_batch
