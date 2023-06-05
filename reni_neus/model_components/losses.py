# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Collection of Losses.
"""
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0


import torch
from torch import nn
from typing import Literal, Tuple

class NormalLoss(nn.Module):
    """
    Custom loss function for vector alignment
    """

    def __init__(self):
        super().__init__()
        # the scalar factor, modify as required
        self.xi = 1.0

    def forward(
        self,
        n: Tuple[torch.Tensor, torch.Tensor], 
        n_hat: Tuple[torch.Tensor, torch.Tensor],
        i_star: int, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            n: predicted normal vector
            n_hat: ground truth normal vector
            i_star: index for selecting specific element in n_hat
            mask: mask of valid pixels
        Returns:
            custom loss based on reduction function
        """
        loss = -self.xi * torch.abs(torch.sum(n * n_hat[i_star], dim=-1))

        return image_loss
