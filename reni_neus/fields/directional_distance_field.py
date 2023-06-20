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
import icosphere

import torch
from torchtyping import TensorType
from typing_extensions import Literal
import torch.nn.functional as F
import torch.nn as nn

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
    position_encoding_type: Literal["hash", "nerf", "sh", "icosphere_hash", "none"] = "none"
    """Type of encoding to use for position"""
    direction_encoding_type: Literal["hash", "nerf", "sh", "icosphere_hash", "none"] = "none"
    """Type of encoding to use for direction"""
    network_type: Literal["fused_mlp", "siren", "film_siren", "siren_grid"] = "siren"
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
    icosphere_level: int = 4
    """The level of the icosphere to use for the grid network"""


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
        depth_out_features = 1 if self.config.ddf_type == "ddf" else self.num_depth_components + self.num_weight_components
        out_features = depth_out_features + 1 if self.config.predict_probability_of_hit else depth_out_features

        if self.config.network_type == "siren":
            self.ddf = Siren(
                in_features=6 + pos_encoding_dim + dir_encoding_dim,
                hidden_features=self.config.hidden_features,
                hidden_layers=self.config.hidden_layers,
                out_features=out_features,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30,
            )
        elif self.config.network_type == "film_siren":
            self.ddf = DDFFiLMSiren(
              input_dim=3 + pos_encoding_dim,
              mapping_network_input_dim=3 + dir_encoding_dim,
              siren_hidden_features=self.config.hidden_features,
              siren_hidden_layers=self.config.hidden_layers,
              mapping_network_features=self.config.hidden_features,
              mapping_network_layers=self.config.hidden_layers,
              out_features=out_features
            )
        elif self.config.network_type == "fused_mlp":
            self.ddf = tcnn.Network(
                n_input_dims=6 + pos_encoding_dim + dir_encoding_dim,
                n_output_dims=out_features,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_features,
                    "n_hidden_layers": self.config.hidden_layers,
                },
            )
        elif self.config.network_type == "siren_grid":
            vertices, _ = icosphere.icosphere(self.config.icosphere_level)
            self.vertices = torch.tensor(vertices, dtype=torch.float32) * self.ddf_radius
            net = []
            for _ in range(vertices.shape[0]):
                net.append(
                    Siren(
                        in_features=6 + pos_encoding_dim + dir_encoding_dim,
                        hidden_features=self.config.hidden_features,
                        hidden_layers=self.config.hidden_layers,
                        out_features=out_features,
                        outermost_linear=True,
                        first_omega_0=30,
                        hidden_omega_0=30,
                    )
                )
            self.ddf = nn.ModuleList(net)


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
            log2_hashmap_size: int = 19
            features_per_level: int = 2
            base_res: int = 16
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
        
        if position_encoding_type == "icosphere_hash":
            raise NotImplementedError("Icosphere hash encoding not implemented yet")

        if direction_encoding_type == "icosphere_hash":
            raise NotImplementedError("Icosphere hash encoding not implemented yet")
        
        if position_encoding_type == "nerf":
            self.position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=True
            )

        if direction_encoding_type == "nerf":
            self.direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=True
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
        
    def find_nearest_vertices(self, positions):
        """Find indices of the three nearest vertices to each position."""
        
        # Calculate squared Euclidean distance
        self.vertices = self.vertices.type_as(positions)
        dists = torch.sum((positions - self.vertices)**2, dim=-1)  # Output shape: (B, N)
        
        # Find indices of the smallest three distances
        _, indices = torch.topk(dists, 3, dim=-1, largest=False)

        # Get three smallest distances
        nearest_dists = dists[indices]
        
        return indices, nearest_dists


    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: TensorType | None = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}

        origins = ray_samples.frustums.origins
        directions = ray_samples.frustums.directions

        if self.config.network_type == "siren_grid":
            # Find the nearest vertices for each origin
            nearest_vertex_indices, nearest_dists = self.find_nearest_vertices(origins[0, :])

            # Calculate weights based on inverse distances
            weights = 1.0 / nearest_dists
            weights /= torch.sum(weights)  # Normalize so that the weights sum to 1

            # if self.position_encoding is not None:
            #     origins = torch.cat([origins, self.position_encoding(origins)], dim=-1)

            # if self.direction_encoding is not None:
            #     directions = torch.cat([directions, self.direction_encoding(directions)], dim=-1)
            
            # inputs = torch.cat([origins, directions], dim=-1)

            # # Store the outputs from the SIREN networks associated with these vertices
            # siren_outputs = [self.ddf[idx](inputs) for idx in nearest_vertex_indices]

            # Store the outputs from the SIREN networks associated with these vertices
            siren_outputs = []
            for idx in nearest_vertex_indices:
                # Get the position of the nearest vertex
                nearest_vertex = self.vertices[idx]
                
                # Normalize origin such that it represents the offset from the vertex
                offset_origin = origins - nearest_vertex

                if self.position_encoding is not None:
                    offset_origin = torch.cat([offset_origin, self.position_encoding(offset_origin)], dim=-1)

                if self.direction_encoding is not None:
                    directions = torch.cat([directions, self.direction_encoding(directions)], dim=-1)

                inputs = torch.cat([offset_origin, directions], dim=-1)

                siren_outputs.append(self.ddf[idx](inputs))

            # Stack the outputs along a new dimension
            stacked_outputs = torch.stack(siren_outputs, dim=0)  # shape: [3, N, 2]

            # Weighted interpolation
            output = torch.sum(stacked_outputs * weights.unsqueeze(-1).unsqueeze(-1), dim=0)  # shape: [N, 2]

        else:
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
