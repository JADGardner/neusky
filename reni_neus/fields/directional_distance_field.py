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
from typing import Dict, Tuple, Type

import torch
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, FieldConfig

from reni_neus.utils.siren import Siren
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class DirectionalDistanceFieldConfig(FieldConfig):
    """DD Field Config"""

    _target: Type = field(default_factory=lambda: DirectionalDistanceField)
    position_encoding_type: Literal["hash", "nerf", "sh", "none"] = "none"
    """Type of encoding to use for position"""
    direction_encoding_type: Literal["hash", "nerf", "sh", "none"] = "none"
    """Type of encoding to use for direction"""
    network_type: Literal["fused_mlp", "siren"] = "siren"
    """Type of network to use"""
    hidden_layers: int = 8
    """Number of hidden layers for ddf network"""
    hidden_features: int = 256
    """Number of features for ddf network"""
    predict_probability_of_hit: bool = False
    """Whether to predict probability of hit"""


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

        encoding_dim = self._setup_encoding()

        if self.config.network_type == "siren":
            self.ddf = Siren(
                in_features=6 + encoding_dim,
                hidden_features=self.config.hidden_features,
                hidden_layers=self.config.hidden_layers,
                out_features=1 if not self.config.predict_probability_of_hit else 2,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30,
            )
        elif self.config.network_type == "fused_mlp":
            self.ddf = tcnn.Network(
                n_input_dims=6 + encoding_dim,
                n_output_dims=1 if not self.config.predict_probability_of_hit else 2,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_features,
                    "n_hidden_layers": self.config.hidden_layers,
                },
            )

    def _setup_encoding(self):
        encoding_dim = 0
        self.position_encoding = None
        self.direction_encoding = None

        position_encoding_type = self.config.position_encoding_type
        direction_encoding_type = self.config.direction_encoding_type

        if position_encoding_type == "hash" or direction_encoding_type == "hash":
            raise NotImplementedError

        if position_encoding_type == "nerf":
            self.position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
            )
            encoding_dim += self.position_encoding.get_out_dim()

        if direction_encoding_type == "nerf":
            self.direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
            )
            encoding_dim += self.direction_encoding.get_out_dim()

        if position_encoding_type == "sh":
            self.position_encoding = SHEncoding(4)
            encoding_dim += self.position_encoding.get_out_dim()

        if direction_encoding_type == "sh":
            self.direction_encoding = SHEncoding(4)
            encoding_dim += self.direction_encoding.get_out_dim()

        return encoding_dim

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: TensorType | None = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}

        origins = ray_samples.frustums.origins
        directions = ray_samples.frustums.directions

        if self.position_encoding is not None:
            origins = torch.cat([origins, self.position_encoding(origins)], dim=-1)

        if self.direction_encoding is not None:
            directions = torch.cat([directions, self.direction_encoding(directions)], dim=-1)

        inputs = torch.cat([origins, directions], dim=-1)

        output = self.ddf(inputs)

        expected_termination_dist = torch.relu(output[..., 0])  # [N] # Only want distances ahead of us

        outputs.update({RENINeuSFieldHeadNames.TERMINATION_DISTANCE: expected_termination_dist})

        if self.config.predict_probability_of_hit:
            probability_of_hit = torch.sigmoid(output[..., 1])  # We want a probability between 0 and 1 [N]
            outputs.update({RENINeuSFieldHeadNames.PROBABILITY_OF_HIT: probability_of_hit})

        return outputs
