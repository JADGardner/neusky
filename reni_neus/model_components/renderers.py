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
from typing import Generator, Optional

import nerfacc
import torch
from torch import nn
from torchtyping import TensorType
import torch.nn.functional as F

from reni_neus.utils.utils import linear_to_sRGB

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

        # compute dot product between normals [num_rays * samples_per_ray, 3] and light directions [num_rays * samples_per_ray, num_illumination_directions, 3]
        dot_prod = torch.einsum(
            "bi,bji->bj", normals, light_directions
        )  # [num_rays * samples_per_ray, num_reni_directions]

        # clamp dot product values to be between 0 and 1
        dot_prod.clamp_(min=0.0, max=1.0)

        # count the number of elements in dot product that are greater than 0
        count = torch.sum((dot_prod > 0).float(), dim=1, keepdim=True)

        # replace all 0 values with 1 to avoid division by 0
        count = torch.where(count > 0, count, torch.ones_like(count))

        dot_prod = dot_prod / count

        if visibility is not None:
            # Apply the visibility mask to the dot product
            dot_prod = dot_prod * visibility.squeeze()

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

        comp_rgb = comp_rgb + background_illumination.to(weights.device) * (1.0 - accumulated_weight)
        comp_rgb = linear_to_sRGB(comp_rgb)

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


class RGBBlinnPhongRendererWithVisibility(nn.Module):
    """Renderer for RGB Blinn-Phong field with visibility based on shininess only."""

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
        shininess: TensorType["bs":..., "num_samples", 1],  # Shininess tensor
        c2w_matrices: TensorType["bs":..., "num_samples", 3, 4],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image using Blinn-Phong shading."""

        albedos = albedos.view(-1, 3)  # [num_rays * samples_per_ray, 3]
        normals = normals.view(-1, 3)  # [num_rays * samples_per_ray, 3]
        shininess = shininess.view(-1, 1)  # [num_rays * samples_per_ray, 1]
        c2w_matrices = c2w_matrices.view(-1, 3, 4)  # [num_rays * samples_per_ray, 3, 4]

        # Compute view direction in world space using c2w matrix
        canonical_view_dir = torch.tensor([0.0, 0.0, -1.0, 1.0], device=c2w_matrices.device)  # Extend to 4D
        view_dir_world_homogeneous = torch.einsum("bij,j->bi", c2w_matrices, canonical_view_dir)
        view_dir_world = view_dir_world_homogeneous[
            :, :3
        ]  # Discard the fourth component [num_rays * samples_per_ray, 3]
        view_dir_world = view_dir_world.type_as(normals)

        # Compute half-vector
        half_vector = (light_directions + view_dir_world.unsqueeze(1)) / torch.norm(
            light_directions + view_dir_world.unsqueeze(1), dim=-1, keepdim=True
        )  # [num_rays * samples_per_ray, num_illumination_directions, 3]

        # Compute dot products
        dot_nl = torch.einsum("bi,bji->bj", normals, light_directions)  # N.L
        dot_nh = torch.einsum("bi,bji->bj", normals, half_vector)  # N.H

        # Clamp values
        dot_nl = torch.clamp(dot_nl, 0, 1)  # [num_rays * samples_per_ray, num_illumination_directions]
        dot_nh = torch.clamp(dot_nh, 0, 1)  # [num_rays * samples_per_ray, num_illumination_directions]

        if visibility is not None:
            # Adjust light colors using visibility
            light_colors = light_colors * visibility  # [num_rays * samples_per_ray, num_illumination_directions, 3]

        # Compute Blinn-Phong reflection model
        diffuse = albedos.unsqueeze(1) * dot_nl.unsqueeze(
            -1
        )  # [num_rays * samples_per_ray, num_illumination_directions, 3]
        specular = dot_nh.unsqueeze(-1) ** shininess.unsqueeze(
            1
        )  # [num_rays * samples_per_ray, num_illumination_directions, 1]

        color = light_colors * (diffuse + specular)

        color = torch.sum(color, dim=1)  # Sum over the num_illumination_directions

        radiance = color.view(*weights.shape[:-1], 3)

        if ray_indices is not None and num_rays is not None:
            comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, radiance, num_rays)
            accumulated_weight = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            comp_rgb = torch.sum(weights * radiance, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        comp_rgb = comp_rgb + background_illumination.to(weights.device) * (1.0 - accumulated_weight)
        comp_rgb = linear_to_sRGB(comp_rgb)

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
        shininess: TensorType["bs":..., "num_samples", 1],  # Shininess tensor
        c2w_matrices: TensorType["bs":..., "num_samples", 3, 4],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image using Blinn-Phong shading."""

        rgb = self.render_and_combine_rgb(
            albedos=albedos,
            normals=normals,
            light_directions=light_directions,
            light_colors=light_colors,
            visibility=visibility,
            background_illumination=background_illumination,
            weights=weights,
            shininess=shininess,
            c2w_matrices=c2w_matrices,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb
