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
import numpy as np


def factorial(x):
    if x == 0:
        return 1.0
    return x * factorial(x - 1)


def P(l, m, x, device):
    pmm = 1.0
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x)).to(device)
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm * torch.ones(x.shape).to(device)

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if l == m + 1:
        return pmmp1

    pll = torch.zeros(x.shape).to(device)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def shTerms(lmax):
    return (lmax + 1) * (lmax + 1)


def K(l, m, device):
    return torch.sqrt(
        torch.tensor(
            ((2 * l + 1) * factorial(l - m))
            / (4 * torch.pi * factorial(l + m))
        )
    ).to(device)


def shIndex(l, m):
    return l * l + l + m


def SH(l, m, theta, phi, device):
    sqrt2 = np.sqrt(2.0)
    if m == 0:
        return (
            K(l, m, device)
            * P(l, m, torch.cos(theta), device)
            * torch.ones(phi.shape).to(device)
        )
    elif m > 0:
        return (
            sqrt2
            * K(l, m, device)
            * torch.cos(m * phi)
            * P(l, m, torch.cos(theta), device)
        )
    else:
        return (
            sqrt2
            * K(l, -m, device)
            * torch.sin(-m * phi)
            * P(l, -m, torch.cos(theta), device)
        )


def shEvaluate(theta, phi, lmax, device):
    coeffsMatrix = torch.zeros((theta.shape[0], phi.shape[0], shTerms(lmax))).to(
        device
    )

    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            coeffsMatrix[:, :, index] = SH(l, m, theta, phi, device)
    return coeffsMatrix


def xy2ll(x, y, width, height):
    def yLocToLat(yLoc, height):
        return yLoc / (float(height) / torch.pi)

    def xLocToLon(xLoc, width):
        return xLoc / (float(width) / (torch.pi * 2))

    return yLocToLat(y, height), xLocToLon(x, width)


def getCoefficientsMatrix(xres, lmax, device):
    yres = int(xres / 2)
    # setup fast vectorisation
    x = torch.arange(0, xres).to(device)
    y = torch.arange(0, yres).reshape(yres, 1).to(device)

    # Setup polar coordinates
    lat, lon = xy2ll(x, y, xres, yres)

    # Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
    Ylm = shEvaluate(lat, lon, lmax, device)
    return Ylm


def sh_lmax_from_terms(terms):
    return int(torch.sqrt(terms) - 1)


def shReconstructSignal(coeffs, width, device):
    lmax = sh_lmax_from_terms(torch.tensor(coeffs.shape[0]).to(device))
    sh_basis_matrix = getCoefficientsMatrix(width, lmax, device)
    return torch.einsum("ijk,kl->ijl", sh_basis_matrix, coeffs)  # (H, W, 3)

def calc_num_sh_coeffs(order):
    coeffs = 0
    for i in range(order + 1):
        coeffs += 2 * i + 1
    return coeffs

def get_sh_order(ndims):
    order = 0
    while calc_num_sh_coeffs(order) < ndims:
        order += 1
    return order

def get_spherical_harmonic_representation(img, nBands):
    # img: (H, W, 3), nBands: int
    iblCoeffs = getCoefficientsFromImage(img, nBands)
    sh_radiance_map = shReconstructSignal(
        iblCoeffs, width=img.shape[1]
    )
    sh_radiance_map = torch.from_numpy(sh_radiance_map)
    return sh_radiance_map


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
            [
                torch.tensor(sph_harm(m, n, phi, theta).real, device=directions.device)
                for n in range(self.num_sph_harmonics)
                for m in range(-n, n + 1)
            ],
            dim=-1,
        )

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
