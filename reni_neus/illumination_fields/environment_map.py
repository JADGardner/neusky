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
Base class for the graphs.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Type, Tuple
from pathlib import Path
import math
import imageio

import torch
from torch import nn

from reni_neus.illumination_fields.base_illumination_field import IlluminationFieldConfig, IlluminationField
from reni_neus.utils.utils import sRGB
# Field related configs
@dataclass
class EnvironmentMapConfig(IlluminationFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: EnvironmentMapField)
    """target class to instantiate"""
    path: Path = Path('path/to/environment_map/s.pt')
    """path to environment map"""
    resolution: Tuple[int, int] = (512, 256)
    """resolution of environment map"""
    trainable: bool = False
    """whether to train the environment map or not"""


class EnvironmentMapField(IlluminationField):
    """Environment map illumination fields."""

    def __init__(
        self,
        config: EnvironmentMapConfig,
        num_latent_codes: int,
    ) -> None:
        super().__init__()
        self.config = config
        if self.config.trainable:
            self.environment_maps = nn.Parameter(torch.randn(num_latent_codes, 3, self.config.resolution[0], self.config.resolution[1]))
        else:
            # if the path ends in .pt load it
            if self.config.path.suffix == '.pt':
                self.environment_maps = torch.load(self.config.path)
            elif self.config.path.suffix == '.exr':
                self.environment_maps = torch.from_numpy(imageio.imread(self.config.path)).permute(2, 0, 1)
            # if its a folder load all images in the folder
            elif self.config.path.is_dir():
                self.environment_maps = []
                for file in self.config.path.iterdir():
                    if file.suffix == '.exr':
                        self.environment_maps.append(torch.from_numpy(imageio.imread(file)).permute(2, 0, 1))
                self.environment_maps = torch.stack(self.environment_maps)

            # check if envmap is BCWH
            if not self.environment_maps.shape[1] == 4:
                self.environment_maps.unsqueeze(1)

            if self.environment_maps.shape[0] != num_latent_codes:
                # just repeat the first environment map
                self.environment_maps = self.environment_maps.repeat(num_latent_codes, 1, 1, 1)

    def sample_envmaps(self, envmaps, directions):
        """Sample colors from the environment maps given a set of 3D directions.

        Args:
            envmaps: Environment maps of shape [unique_indices, 3, H, W].
            directions: Directions of shape [num_directions, 3].

        Returns:
            Sampled colors from environment maps.
        """
        num_directions = directions.shape[0]

        # Convert 3D directions to 2D coordinates in the environment map.
        # Note that we assume directions are already normalized to unit length.
        # We consider that the environment map's up is the positive Z direction.
        phi = torch.atan2(directions[:,1], directions[:,0])  # azimuthal angle
        theta = torch.acos(directions[:,2])  # polar angle

        # Convert spherical to pixel coordinates. We assume the environment map spans 360° horizontally and 180° vertically.
        u = (phi + math.pi) / (2 * math.pi)  # horizontal coordinate between 0 and 1
        v = theta / math.pi  # vertical coordinate between 0 and 1

        # Rescale and shift coordinates to match the grid_sample convention.
        u = 2 * u - 1  # horizontal coordinate between -1 and 1
        v = 2 * v - 1  # vertical coordinate between -1 and 1

        # Repeat coordinates for each environment map.
        u = u[None, :].repeat(envmaps.shape[0], 1)  # [unique_indices, num_directions]
        v = v[None, :].repeat(envmaps.shape[0], 1)  # [unique_indices, num_directions]

        # Concatenate u and v coordinates to form the sampling grid.
        grid = torch.stack([u, v], dim=-1)  # [num_latent_codes, num_directions, 2]

        # Convert the grid to a 4D tensor of shape [B, H, W, 2] as expected by grid_sample.
        grid = grid.view(envmaps.shape[0], num_directions, 1, 2)

        # Sample colors from the environment maps using bilinear interpolation.
        colors = torch.nn.functional.grid_sample(envmaps, grid, align_corners=False, mode='bilinear') # [unique_indices, 3, num_directions, 1]

        return colors.squeeze(-1).permute(0, 2, 1) # [unique_indices, num_directions, 3]

    @abstractmethod
    def get_outputs(self, unique_indices, inverse_indices, directions, rotation, illumination_type):
        """Computes and returns the colors. Returns output field values.

        Args:
            unique_indices: [unique_indices]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [num_directions, 3]
        """

        temp_device = unique_indices.device
        unique_indices = unique_indices.to(self.environment_maps.device)
        directions = directions.to(self.environment_maps.device)
        inverse_indices = inverse_indices.to(self.environment_maps.device)

        if illumination_type == "illumination":
            envmaps = self.environment_maps[unique_indices]  # [unique_indices, 3, H, W]
            light_directions = directions.type_as(envmaps)  # [num_directions, 3]

            light_colours = self.sample_envmaps(envmaps, light_directions) # [unique_indices, 3, 3]
            light_directions = light_directions.unsqueeze(0).repeat(envmaps.shape[0], 1, 1)  # [unique_indices, num_directions, 3]

            light_colours = light_colours[inverse_indices]  # [rays_per_batch, samples_per_ray, 3, num_directions]
            # light_colours = light_colours.permute(0, 1, 3, 2)  # Desired shape: [rays_per_batch, samples_per_ray, num_directions, 3]
            light_colours = light_colours.reshape(-1, directions.shape[0], 3)  # [rays_per_batch * samples_per_ray, num_directions, 3]

            light_directions = light_directions[inverse_indices]  # [rays_per_batch, samples_per_ray, num_directions, 3]
            light_directions = light_directions.reshape(
                -1, directions.shape[0], 3
            )  # [rays_per_batch * samples_per_ray, num_directions, 3]

            light_colours = light_colours.to(temp_device) # [rays_per_batch * samples_per_ray, num_directions, 3]
            light_directions = light_directions.to(temp_device) # [rays_per_batch * samples_per_ray, num_directions, 3]

            return light_colours, light_directions
        if illumination_type == "background":
            envmaps = self.environment_maps[unique_indices]  # [unique_indices, 3, H, W]
            light_directions = directions.type_as(envmaps)  # [num_directions, 3]

            light_colours = self.sample_envmaps(envmaps, light_directions) # [unique_indices, num_directions, 3]
            light_directions = light_directions.unsqueeze(0).repeat(envmaps.shape[0], 1, 1)  # [unique_indices, num_directions, 3]

            # light_colours = light_colours[inverse_indices]  # [rays_per_batch, samples_per_ray, num_directions, 3]
            # light_colours = light_colours[:, 0, 0, :]  # [rays_per_batch, num_directions, 3]
            light_colours = light_colours[0, :, :]  # [num_directions, 3]
            light_colours = sRGB(light_colours) # [num_directions, 3]
            light_colours = light_colours.to(temp_device) # [num_directions, 3]
            return light_colours, None
        if illumination_type == "envmap":
            envmaps = self.environment_maps[unique_indices]  # [unique_indices, 3, H, W]
            light_directions = torch.stack(
                [-directions[:, :, 0], directions[:, :, 2], directions[:, :, 1]], dim=2
            )
            light_directions = light_directions.type_as(envmaps)  # [num_directions, 3]
            light_colours = self.sample_envmaps(envmaps, light_directions) # [unique_indices, num_directions, 3]
            light_colours = light_colours[inverse_indices]  # [rays_per_batch, samples_per_ray, 3, num_directions]
            light_colours = light_colours.permute(0, 1, 3, 2)  # Desired shape: [rays_per_batch, samples_per_ray, num_directions, 3]
            light_colours = light_colours.reshape(-1, directions.shape[0], 3)  # [rays_per_batch * samples_per_ray, num_directions, 3]
            light_colours = sRGB(light_colours)
            light_colours = light_colours.to(temp_device) # [rays_per_batch * samples_per_ray, num_directions, 3]
            return light_colours, None
            


    @abstractmethod
    def get_latents(self):
        """Returns the latents of the field."""
        raise NotImplementedError

    @abstractmethod
    def set_no_grad(self):
        """Sets the latents of the field to no_grad."""
        self.environment_maps.requires_grad = False
