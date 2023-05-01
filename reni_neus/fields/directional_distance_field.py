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
Field for SDF based model, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with extracting high fidelity surfaces
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class DirectionalDistanceFieldConfig(FieldConfig):
    """DD Field Config"""

    _target: Type = field(default_factory=lambda: DirectionalDistanceField)
    use_encoding: bool = False
    """Whether to use encoding"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "periodic"
    """Type of encoding to use"""
    network_type: Literal["fused_mlp", "siren"] = "siren"
    """Type of network to use"""
    num_layers: int = 8
    """Number of layers for geometric network"""


class DirectionalDistanceField(Field):
    """
    A field for Directional Distance Functions (DDF).

    Args:
        config: The configuration for the SDF field.
    """

    config: DirectionalDistanceFieldConfig

    def __init__(
        self,
        config: DirectionalDistanceFieldConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: TensorType | None = None
    ) -> Dict[FieldHeadNames, TensorType]:
        return super().get_outputs(ray_samples, density_embedding)
