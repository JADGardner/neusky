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
from typing import Optional

import icosphere
import torch
from scipy.spatial.transform import Rotation
from torch import nn


class IlluminationSampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_direction_samples(self, num_directions) -> torch.Tensor:
        """Generate Direction Samples"""

    def forward(self, num_directions):
        """Returns directions for each position.

        Args:
            num_directions: number of directions to sample

        Returns:
            directions: [num_directions, 3]
        """

        return self.generate_direction_samples(num_directions)


class IcosahedronSampler(IlluminationSampler):
    """For sampling directions from an icosahedron."""

    def __init__(
        self, icosphere_order: int = 2, apply_random_rotation: bool = False, remove_lower_hemisphere: bool = False
    ):
        super().__init__()
        self.icosphere_order = icosphere_order
        self.apply_random_rotation = apply_random_rotation
        self.remove_lower_hemisphere = remove_lower_hemisphere

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
