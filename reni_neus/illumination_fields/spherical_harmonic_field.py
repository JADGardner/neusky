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

"""Classic NeRF field"""

from reni_neus.illumination_fields.base_illumination_field import IlluminationField, IlluminationFieldConfig

import torch
import torch.nn as nn
from scipy.special import sph_harm


# Field related configs
@dataclass
class SphericalHarmonicsIlluminationFieldConfig(IlluminationFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SphericalHarmonicsIlluminationField)
    """target class to instantiate"""


class SphericalHarmonicsIlluminationField(IlluminationField):
    """Spherical Harmonics illumination field class."""

    def __init__(
        self,
        config: SphericalHarmonicsIlluminationFieldConfig,
        num_unique_indices: int,
        num_sph_harmonics: int,
    ) -> None:
        super().__init__()
        self.num_unique_indices = num_unique_indices
        self.num_sph_harmonics = num_sph_harmonics
        self.coefficients = nn.Parameter(torch.randn(num_unique_indices, num_sph_harmonics, 3))

    def get_outputs(self, unique_indices, inverse_indices, directions, illumination_type):
        """Computes and returns the colors. Returns output field values.

        Args:
            unique_indices: [rays_per_batch]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [rays_per_batch, samples_per_ray, num_directions, 3]
        """
        # Compute spherical coordinates for directions
        r = torch.norm(directions, dim=-1)
        theta = torch.acos(directions[..., 2] / r)
        phi = torch.atan2(directions[..., 1], directions[..., 0])

        # Calculate spherical harmonics for each direction
        sph_harmonics = torch.stack(
            [torch.tensor(sph_harm(m, n, phi, theta).real, device=directions.device)
             for n in range(self.num_sph_harmonics)
             for m in range(-n, n+1)], dim=-1)

        # Retrieve the corresponding coefficients
        coefficients = self.coefficients[unique_indices]

        # Compute the colors
        illumination_colours = torch.einsum("...d,...d->...", sph_harmonics, coefficients)
        illumination_colours = torch.clamp(illumination_colours, min=0)

        # No specific illumination directions are returned for spherical harmonics
        illumination_directions = None

        return illumination_colours, illumination_directions

    def get_latents(self):
        """Returns the latents of the field."""
        return self.coefficients

    def set_no_grad(self):
        """Sets the latents of the field to no_grad."""
        self.coefficients.requires_grad_(False)
