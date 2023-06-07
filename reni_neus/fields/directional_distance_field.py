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
import torch.nn.functional as F

import numpy as np

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, FieldConfig

from reni_neus.utils.siren import Siren, DDFFiLMSiren
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
    network_type: Literal["fused_mlp", "siren", "film_siren"] = "siren"
    """Type of network to use"""
    termination_output_activation: Literal["sigmoid", "tanh", "relu"] = "sigmoid"
    """Activation function for termination network"""
    probability_of_hit_output_activation: Literal["sigmoid", "tanh", "relu"] = "sigmoid"
    """Activation function for probability of hit network"""
    hidden_layers: int = 8
    """Number of hidden layers for ddf network"""
    hidden_features: int = 256
    """Number of features for ddf network"""
    predict_probability_of_hit: bool = False
    """Whether to predict probability of hit"""
    ddf_type: Literal["ddf", "pddf"] = "ddf"
    """Type of ddf to use, ddf or probibalisitic ddf"""
    num_dirac_components: int = 2
    """Dirac delta functions num K components"""
    eta_T: float = 1.0
    """The temperature parameter."""
    epsilon_s: float = 1e-5
    """The maximum inverse depth scale."""


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
        ddf_radius: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.ddf_radius = ddf_radius

        encoding_dim = self._setup_encoding()

        self.num_depth_components = self.config.num_dirac_components
        self.num_weight_components = self.config.num_dirac_components - 1
        depth_out_features = 1 if self.config.ddf_type == "ddf" else self.num_depth_components + self.num_weight_components
        out_features = depth_out_features + 1 if self.config.predict_probability_of_hit else depth_out_features

        if self.config.network_type == "siren":
            self.ddf = Siren(
                in_features=6 + encoding_dim,
                hidden_features=self.config.hidden_features,
                hidden_layers=self.config.hidden_layers,
                out_features=out_features,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30,
            )
        elif self.config.network_type == "film_siren":
            self.ddf = DDFFiLMSiren(
              input_dim=3 + encoding_dim,
              mapping_network_input_dim=3 + encoding_dim,
              siren_hidden_features=self.config.hidden_features,
              siren_hidden_layers=self.config.hidden_layers,
              mapping_network_features=self.config.hidden_features,
              mapping_network_layers=self.config.hidden_layers,
              out_features=out_features
            )
        elif self.config.network_type == "fused_mlp":
            self.ddf = tcnn.Network(
                n_input_dims=6 + encoding_dim,
                n_output_dims=out_features,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_features,
                    "n_hidden_layers": self.config.hidden_layers,
                },
            )

        self.termination_output_activation = self._get_activation(self.config.termination_output_activation)
        self.probability_of_hit_output_activation = self._get_activation(
            self.config.probability_of_hit_output_activation
        )

    def _setup_encoding(self):
        encoding_dim = 0
        self.position_encoding = None
        self.direction_encoding = None

        position_encoding_type = self.config.position_encoding_type
        direction_encoding_type = self.config.direction_encoding_type

        if position_encoding_type == "hash":
            num_levels: int = 16
            base_res: int = 16
            features_per_level: int = 2
            log2_hashmap_size: int = 19
            max_res: int = 2048
            growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
            self.position_encoding = tcnn.Encoding(n_input_dims=3,
                                                   encoding_config={"otype": "HashGrid",
                                                                    "n_levels": num_levels,
                                                                    "n_features_per_level": features_per_level,
                                                                    "log2_hashmap_size": log2_hashmap_size,
                                                                    "base_resolution": base_res,
                                                                    "per_level_scale": growth_factor}
            )

        if direction_encoding_type == "hash":
            num_levels: int = 16
            base_res: int = 16
            features_per_level: int = 2
            log2_hashmap_size: int = 19
            max_res: int = 2048
            growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
            self.direction_encoding = tcnn.Encoding(n_input_dims=3, 
                                                   encoding_config={"otype": "HashGrid",
                                                                    "n_levels": num_levels,
                                                                    "n_features_per_level": features_per_level,
                                                                    "log2_hashmap_size": log2_hashmap_size,
                                                                    "base_resolution": base_res,
                                                                    "per_level_scale": growth_factor}
            )

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

    def _get_activation(self, activation: Literal["sigmoid", "tanh", "relu"]):
        if activation == "sigmoid":
            return torch.sigmoid
        elif activation == "tanh":
            return torch.tanh
        elif activation == "relu":
            return torch.relu
        else:
            raise NotImplementedError

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: TensorType | None = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}

        origins = ray_samples.frustums.origins
        directions = ray_samples.frustums.directions

        # TODO I think SH would need to be scaled between 0 and 1
        if self.position_encoding is not None:
            origins = torch.cat([origins, self.position_encoding(origins)], dim=-1)

        if self.direction_encoding is not None:
            directions = torch.cat([directions, self.direction_encoding(directions)], dim=-1)

        inputs = torch.cat([origins, directions], dim=-1)

        output = self.ddf(inputs)

        if self.config.ddf_type == "pddf":
            expected_termination_distances = self.termination_output_activation(output[..., :self.num_depth_components])
            expected_termination_distances = self.termination_output_activation(expected_termination_distances)

            expected_termination_weights = output[..., self.num_depth_components:self.num_depth_components+self.num_weight_components]
            weights = torch.cat([expected_termination_weights, 1 - expected_termination_weights], dim=-1)

            # Apply the visibility and depth adjustment to the logits
            adjusted_logits = self.config.eta_T * weights / (self.config.epsilon_s + expected_termination_distances)

            # Compute the weighted sum of the potential depths
            expected_termination_dist = torch.sum(F.softmax(adjusted_logits, dim=1) * expected_termination_distances, dim=1, keepdim=True)
        else:
            expected_termination_dist = self.termination_output_activation(output[..., 0])

        expected_termination_dist = expected_termination_dist * (2 * self.ddf_radius)
        outputs.update({RENINeuSFieldHeadNames.TERMINATION_DISTANCE: expected_termination_dist})

        if self.config.predict_probability_of_hit:
            probability_of_hit = self.probability_of_hit_output_activation(output[..., -1])
            outputs.update({RENINeuSFieldHeadNames.PROBABILITY_OF_HIT: probability_of_hit})

        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples)
        return field_outputs
