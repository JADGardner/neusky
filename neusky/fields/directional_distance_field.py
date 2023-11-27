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
import torch.nn.functional as F
from torchtyping import TensorType
from typing_extensions import Literal

import numpy as np

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, FieldConfig

from neusky.field_components.neusky_fieldheadnames import NeuSkyFieldHeadNames
from reni.field_components.siren import Siren
from reni.field_components.film_siren import FiLMSiren
from reni.field_components.transformer_decoder import Decoder

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class DirectionalDistanceFieldConfig(FieldConfig):
    """DDF Field Config"""

    _target: Type = field(default_factory=lambda: DirectionalDistanceField)
    position_encoding_type: Literal["hash", "nerf", "sh", "icosphere_hash", "none"] = "none"
    """Type of encoding to use for position"""
    direction_encoding_type: Literal["hash", "nerf", "sh", "icosphere_hash", "none"] = "none"
    """Type of encoding to use for direction"""
    conditioning: Literal["FiLM", "Concat", "Attention"] = "Concat"
    """Type of conditioning to use"""
    termination_output_activation: Literal["sigmoid", "tanh", "relu"] = "sigmoid"
    """Activation function for termination network"""
    probability_of_hit_output_activation: Literal["sigmoid", "tanh", "relu"] = "sigmoid"
    """Activation function for probability of hit network"""
    hidden_layers: int = 3
    """Number of hidden layers"""
    hidden_features: int = 128
    """Number of hidden features"""
    mapping_layers: int = 3
    """Number of mapping layers"""
    mapping_features: int = 128
    """Number of mapping features"""
    num_attention_heads: int = 8
    """Number of attention heads"""
    num_attention_layers: int = 3
    """Number of attention layers"""
    out_features: int = 3  # RGB
    """Number of output features"""
    last_layer_linear: bool = True
    """Whether to use a linear layer as the last layer"""
    first_omega_0: float = 30.0
    """Omega_0 for first layer"""
    hidden_omega_0: float = 30.0
    """Omega_0 for hidden layers"""
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

        pos_encoding_dim, dir_encoding_dim = self._setup_encoding()

        self.num_depth_components = self.config.num_dirac_components
        self.num_weight_components = self.config.num_dirac_components - 1
        depth_out_features = (
            1 if self.config.ddf_type == "ddf" else self.num_depth_components + self.num_weight_components
        )
        out_features = depth_out_features + 1 if self.config.predict_probability_of_hit else depth_out_features

        self.ddf = self._setup_network(
            pos_encoding_dim=pos_encoding_dim, dir_encoding_dim=dir_encoding_dim, out_features=out_features
        )

        self.termination_output_activation = self._get_activation(self.config.termination_output_activation)
        self.probability_of_hit_output_activation = self._get_activation(
            self.config.probability_of_hit_output_activation
        )

    def _setup_encoding(self):
        pos_encoding_dim, dir_encoding_dim = 0, 0
        self.position_encoding = None
        self.direction_encoding = None

        position_encoding_type = self.config.position_encoding_type
        direction_encoding_type = self.config.direction_encoding_type

        if position_encoding_type == "hash":
            num_levels: int = 16
            log2_hashmap_size: int = 19
            features_per_level: int = 2
            base_res: int = 16
            max_res: int = 2048
            growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
            self.position_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
            )

        if direction_encoding_type == "hash":
            num_levels: int = 16
            log2_hashmap_size: int = 19
            features_per_level: int = 2
            base_res: int = 16
            max_res: int = 2048
            growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                },
            )

        if position_encoding_type == "icosphere_hash":
            raise NotImplementedError("Icosphere hash encoding not implemented yet")

        if direction_encoding_type == "icosphere_hash":
            raise NotImplementedError("Icosphere hash encoding not implemented yet")

        if position_encoding_type == "nerf":
            self.position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=False
            )

        if direction_encoding_type == "nerf":
            self.direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=False
            )

        if position_encoding_type == "sh":
            self.position_encoding = SHEncoding(4)

        if direction_encoding_type == "sh":
            self.direction_encoding = SHEncoding(4)

        if position_encoding_type == "hash":
            pos_encoding_dim += self.position_encoding.n_output_dims
        else:
            pos_encoding_dim += self.position_encoding.get_out_dim() if self.position_encoding is not None else 0
        if direction_encoding_type == "hash":
            dir_encoding_dim += self.direction_encoding.n_output_dims
        else:
            dir_encoding_dim += self.direction_encoding.get_out_dim() if self.direction_encoding is not None else 0

        return pos_encoding_dim, dir_encoding_dim

    def _get_activation(self, activation: Literal["sigmoid", "tanh", "relu"]):
        if activation == "sigmoid":
            return torch.sigmoid
        elif activation == "tanh":
            return torch.tanh
        elif activation == "relu":
            return torch.relu
        else:
            raise NotImplementedError

    def _setup_network(self, pos_encoding_dim, dir_encoding_dim, out_features):
        if self.config.conditioning == "Concat":
            ddf = Siren(
                in_dim=6 + pos_encoding_dim + dir_encoding_dim,
                hidden_layers=self.config.hidden_layers,
                hidden_features=self.config.hidden_features,
                out_dim=out_features,
                outermost_linear=self.config.last_layer_linear,
                first_omega_0=self.config.first_omega_0,
                hidden_omega_0=self.config.hidden_omega_0,
                out_activation=None,
            )
        elif self.config.conditioning == "FiLM":
            ddf = FiLMSiren(
                in_dim=3 + dir_encoding_dim,
                hidden_layers=self.config.hidden_layers,
                hidden_features=self.config.hidden_features,
                mapping_network_in_dim=3 + pos_encoding_dim,
                mapping_network_layers=self.config.mapping_layers,
                mapping_network_features=self.config.mapping_features,
                out_dim=out_features,
                outermost_linear=self.config.last_layer_linear,
                out_activation=None,
            )
        elif self.config.conditioning == "Attention":
            # transformer where K, V is from conditioning input and Q is from directional input
            ddf = Decoder(
                in_dim=3 + dir_encoding_dim,
                conditioning_input_dim=3 + pos_encoding_dim,
                hidden_features=self.config.hidden_features,
                num_heads=self.config.num_attention_heads,
                num_layers=self.config.num_attention_layers,
                out_activation=None,
            )
        else:
            raise NotImplementedError
        return ddf

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError

    def get_outputs(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}

        origins = ray_samples.frustums.origins
        directions = ray_samples.frustums.directions

        if self.position_encoding is not None:
            origins = torch.cat([origins, self.position_encoding(origins)], dim=-1)

        if self.direction_encoding is not None:
            directions = torch.cat([directions, self.direction_encoding(directions)], dim=-1)

        if self.config.conditioning == "Concat":
            model_outputs = self.ddf(torch.cat((directions, origins), dim=1))  # [num_rays, 3]
        elif self.config.conditioning == "FiLM" or self.config.conditioning == "Attention":
            model_outputs = self.ddf(x=directions, conditioning_input=origins)

        if self.config.ddf_type == "pddf":
            expected_termination_distances = self.termination_output_activation(
                model_outputs[..., : self.num_depth_components]
            )
            expected_termination_distances = self.termination_output_activation(expected_termination_distances)

            expected_termination_weights = model_outputs[
                ..., self.num_depth_components : self.num_depth_components + self.num_weight_components
            ]
            weights = torch.cat([expected_termination_weights, 1 - expected_termination_weights], dim=-1)

            # Apply the visibility and depth adjustment to the logits
            adjusted_logits = self.config.eta_T * weights / (self.config.epsilon_s + expected_termination_distances)

            # Compute the weighted sum of the potential depths
            expected_termination_dist = torch.sum(
                F.softmax(adjusted_logits, dim=1) * expected_termination_distances, dim=1, keepdim=True
            )
        else:
            expected_termination_dist = self.termination_output_activation(model_outputs[..., 0])

        expected_termination_dist = expected_termination_dist * (2 * self.ddf_radius)
        outputs.update({NeuSkyFieldHeadNames.TERMINATION_DISTANCE: expected_termination_dist})

        if self.config.predict_probability_of_hit:
            probability_of_hit = self.probability_of_hit_output_activation(model_outputs[..., -1])
            outputs.update({NeuSkyFieldHeadNames.PROBABILITY_OF_HIT: probability_of_hit})

        return outputs

    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples)
        return field_outputs
