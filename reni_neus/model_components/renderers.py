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
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import contextlib
import math
from typing import Generator, Optional, Union

import nerfacc
import torch
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize

from reni_neus.utils.utils import sRGB

BACKGROUND_COLOR_OVERRIDE: Optional[TensorType[3]] = None


@contextlib.contextmanager
def background_color_override_context(mode: TensorType[3]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE  # pylint: disable=global-statement
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


class RGBLambertianRendererWithVisibility(nn.Module):
    """Renderer for RGB Lambertian field with visibility."""

    @classmethod
    def render_and_combine_rgb(
        cls,
        albedos: TensorType["bs":..., "num_samples", 3],
        normals: TensorType["bs":..., "num_samples", 3],
        light_directions: TensorType["bs":..., "num_samples", 3],
        light_colors: TensorType["bs":..., "num_samples", 3],
        visibility: TensorType["bs":..., "num_samples", 1],
        background_illumination: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            albedo: Albedo for each sample
            normal: Normal for each sample
            light_directions: Light directions for each sample
            light_colors: Light colors for each sample
            visibility: Visibility of illumination for each sample
            weights: Weights for each sample
            background_illumination: Background color if ray does not hit anything
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """

        albedos = albedos.view(-1, 3)
        normals = normals.view(-1, 3)

        # compute dot product between normals [num_rays * samples_per_ray, 3] and light directions [num_rays * samples_per_ray, num_reni_directions, 3]
        dot_prod = torch.einsum(
            "bi,bji->bj", normals, light_directions
        )  # [num_rays * samples_per_ray, num_reni_directions]

        # clamp dot product values to be between 0 and 1
        dot_prod = torch.clamp(dot_prod, 0, 1)

        # count the number of elements in dot product that are greater than 0
        count = torch.sum((dot_prod > 0).float(), dim=1, keepdim=True)

        # replace all 0 values with 1 to avoid division by 0
        count = torch.where(count > 0, count, torch.ones_like(count))

        dot_prod = dot_prod / count

        if visibility is not None:
            # Apply the visibility mask to the dot product
            dot_prod = dot_prod * visibility

        # compute final color by multiplying dot product with albedo color and light color
        color = torch.einsum("bi,bj,bji->bi", albedos, dot_prod, light_colors)  # [num_rays * samples_per_ray, 3]

        radiance = color.view(*weights.shape[:-1], 3)

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, radiance, num_rays)
            accumulated_weight = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            comp_rgb = torch.sum(weights * radiance, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        assert isinstance(background_illumination, torch.Tensor)

        comp_rgb = sRGB(comp_rgb) # background_illumination is already sRGB
        comp_rgb = comp_rgb + background_illumination.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb

    def forward(
        self,
        albedos: TensorType["bs":..., "num_samples", 3],
        normals: TensorType["bs":..., "num_samples", 3],
        light_directions: TensorType["bs":..., "num_samples", 3],
        light_colors: TensorType["bs":..., "num_samples", 3],
        visibility: TensorType["bs":..., "num_samples", 1],
        background_illumination: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            albedo: Albedo for each sample
            normal: Normal for each sample
            light_directions: Light directions for each sample
            light_colors: Light colors for each sample
            visibility: Visibility of illumination for each sample
            weights: Weights for each sample
            background_illumination: Background color if ray does not hit anything
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        rgb = self.render_and_combine_rgb(
                albedos=albedos,
                normals=normals,
                light_directions=light_directions,
                light_colors=light_colors,
                visibility=visibility,
                background_illumination=background_illumination,
                weights=weights,
                ray_indices=ray_indices,
                num_rays=num_rays,
              )
        
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
            
        return rgb
