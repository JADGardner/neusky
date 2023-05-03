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
from typing import Dict, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.sdf_field import SDFFieldConfig, SDFField, LearnedVariance

from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class SDFAlbedoFieldConfig(SDFFieldConfig):
    """SDF Albedo Field Config"""

    _target: Type = field(default_factory=lambda: SDFAlbedoField)


class SDFAlbedoField(SDFField):
    """
    A field for Signed Distance Functions (SDF).

    Args:
        config: The configuration for the SDF field.
        aabb: An axis-aligned bounding box for the SDF field.
        num_images: The number of images for embedding appearance.
        use_average_appearance_embedding: Whether to use average appearance embedding. Defaults to False.
        spatial_distortion: The spatial distortion. Defaults to None.
    """

    config: SDFFieldConfig

    def __init__(
        self,
        config: SDFAlbedoFieldConfig,
        aabb: TensorType[2, 3],
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super(SDFField, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config

        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        growth_factor = np.exp((np.log(config.max_res) - np.log(config.base_res)) / (config.num_levels - 1))

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if config.use_hash else "DenseGrid",
                    "n_levels": config.num_levels,
                    "n_features_per_level": config.features_per_level,
                    "log2_hashmap_size": config.log2_hashmap_size,
                    "base_resolution": config.base_res,
                    "per_level_scale": growth_factor,
                    "interpolation": "Smoothstep" if config.smoothstep else "Linear",
                },
            )

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, include_input=False
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # initialize geometric network
        self.initialize_geo_layers()

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = LearnedVariance(init_val=self.config.beta_init)

        # albedo network, only depends on position and geo features
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        # point, feature
        in_dim = 3 + self.position_encoding.get_out_dim() + self.config.geo_feat_dim
        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "clin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0

    def get_sdf_at_pos(self, positions):
        """get sdf at provided positions"""
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat)
        sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def get_density(self, ray_samples: RaySamples):
        raise NotImplementedError

    def get_colors(
        self,
        points: TensorType[..., 3],
        geo_features: TensorType[..., "geo-feat-dim"],
    ) -> TensorType[..., 3]:
        """compute colors"""

        encoded_points = self.position_encoding(points)

        hidden_input = torch.cat(
            [points, encoded_points, geo_features.view(-1, self.config.geo_feat_dim)],
            dim=-1,
        )

        for layer in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(layer))

            hidden_input = lin(hidden_input)

            if layer < self.num_layers_color - 2:
                hidden_input = self.relu(hidden_input)

        rgb = self.sigmoid(hidden_input)

        return rgb

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[TensorType] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, TensorType]:
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # doesn't actually use directions or normals or camera_indices
        albedo = self.get_colors(points=inputs, geo_features=geo_feature)

        albedo = albedo.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        outputs.update(
            {
                RENINeuSFieldHeadNames.ALBEDO: albedo,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False
    ) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas)
        return field_outputs
