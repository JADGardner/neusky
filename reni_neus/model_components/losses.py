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

class RENISkyPixelLoss(object):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss(reduction="mean")
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)

    def __call__(self, inputs, targets, mask):
        inputs = inputs * mask
        targets = targets * mask
        mse = self.mse(inputs, targets)
        similarity = self.cosine_similarity(inputs, targets)
        cosine_loss = 1 - similarity.mean()
        loss = mse + self.alpha * cosine_loss
        return loss

