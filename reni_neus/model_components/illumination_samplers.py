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
from typing import Optional, Type
from dataclasses import dataclass, field

import icosphere
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from nerfstudio.configs.base_config import InstantiateConfig


# Field related configs
@dataclass
class IlluminationSamplerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IlluminationSampler)
    """target class to instantiate"""


class IlluminationSampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        config: IlluminationSamplerConfig,
    ) -> None:
        super().__init__()

    @abstractmethod
    def generate_direction_samples(self, num_directions: Optional[int] = None) -> torch.Tensor:
        """Generate Direction Samples"""

    def forward(self, num_directions: Optional[int] = None) -> torch.Tensor:
        """Returns directions for each position.

        Args:
            num_directions: number of directions to sample

        Returns:
            directions: [num_directions, 3]
        """

        return self.generate_direction_samples(num_directions)


# Field related configs
@dataclass
class IcosahedronSamplerConfig(IlluminationSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IcosahedronSampler)
    """target class to instantiate"""
    icosphere_order: int = 2
    """order of the icosphere"""
    apply_random_rotation: bool = False
    """apply random rotation to the icosphere"""
    remove_lower_hemisphere: bool = False
    """remove lower hemisphere"""


class IcosahedronSampler(IlluminationSampler):
    """For sampling directions from an icosahedron."""

    def __init__(
        self,
        config: IcosahedronSamplerConfig,
    ):
        super().__init__(config)
        self.icosphere_order = config.icosphere_order
        self.apply_random_rotation = config.apply_random_rotation
        self.remove_lower_hemisphere = config.remove_lower_hemisphere

        vertices, _ = icosphere.icosphere(self.icosphere_order)
        self.directions = torch.from_numpy(vertices).float()  # [N, 3], # Z is up

    def set_icosphere_order(self, icosphere_order: int):
        self.icosphere_order = icosphere_order
        vertices, _ = icosphere.icosphere(self.icosphere_order)
        self.directions = torch.from_numpy(vertices).float()

    def generate_direction_samples(self, num_directions=None) -> torch.Tensor:
        # generate N random rotations
        directions = self.directions
        if self.apply_random_rotation:
            R = torch.from_numpy(Rotation.random(1).as_matrix())[0].float()
            directions = directions @ R

        if self.remove_lower_hemisphere:
            directions = directions[directions[:, 2] > 0]

        return directions
