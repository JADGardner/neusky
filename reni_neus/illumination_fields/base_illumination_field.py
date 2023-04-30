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
from typing import Any, Literal, Type

import torch
from torch import nn

from nerfstudio.configs.base_config import InstantiateConfig


# Field related configs
@dataclass
class IlluminationFieldConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IlluminationField)
    """target class to instantiate"""


class IlluminationField(nn.Module):
    """Base class for illumination fields."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.split = 'train'

    def set_split(self, split):
        assert split in ['train', 'val', 'test']
        self.split = split

    @abstractmethod
    def get_outputs(self, unique_indices, inverse_indices, directions, illumination_type):
        """Computes and returns the colors. Returns output field values.

        Args:
            unique_indices: [rays_per_batch]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [rays_per_batch, samples_per_ray, num_directions, 3]
        """

    @abstractmethod
    def get_latents(self):
        """Returns the latents of the field."""

    @abstractmethod
    def set_no_grad(self):
        """Sets the latents of the field to no_grad."""

    def forward(self, camera_indices, positions, directions, illumination_type=Literal["background", "illumination"]):
        """Evaluates illumination field for cameras and directions.

        Args:
            camera_indicies: [rays_per_batch, samples_per_ray]
            positions: [rays_per_batch, samples_per_ray, 3]
            directions: [rays_per_batch, samples_per_ray, num_directions, 3]
        """
        unique_indices, inverse_indices = torch.unique(camera_indices, return_inverse=True)
        illumination_colours, illumination_directions = self.get_outputs(
            unique_indices, inverse_indices, directions, illumination_type
        )
        return illumination_colours, illumination_directions
