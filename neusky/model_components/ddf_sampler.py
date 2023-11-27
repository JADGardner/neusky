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
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union, Type

import torch
from nerfacc import OccGridEstimator
from dataclasses import dataclass, field
from torch import nn
from torchtyping import TensorType
from torch.distributions import von_mises

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import Sampler

from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions, sph2cart


from nerfstudio.configs.base_config import InstantiateConfig


# Field related configs
@dataclass
class DDFSamplerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: DDFSampler)
    """target class to instantiate"""
    num_samples_on_sphere: int = 1,
    """Number of samples on sphere"""
    num_rays_per_sample: int = 1024
    """Number of inward facing rays per sample"""
    only_sample_upper_hemisphere: bool = False
    """Only sample upper hemisphere for positions"""


class DDFSampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        config: DDFSamplerConfig,
        ddf_sphere_radius: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.config = config
        self.ddf_sphere_radius = ddf_sphere_radius
        self.device = device
        self.num_rays = self.config.num_samples_on_sphere * self.config.num_rays_per_sample

    def random_points_on_unit_sphere(self, num_points, cartesian=True):
        """
        Generate a random set of points on a unit sphere.

        :param num_points: number of points to generate
        :param cartesian: if True, return points in cartesian coordinates
        :return: (num_points, 2 or 3) tensor of points
        """
        # get random points in spherical coordinates
        theta = 2 * torch.pi * torch.rand(num_points)
        phi = torch.acos(2 * torch.rand(num_points) - 1)
        if cartesian:
            return torch.stack(sph2cart(theta, phi), dim=1)
        return torch.stack([theta, phi], dim=1)

    @abstractmethod
    def generate_ddf_samples(self, num_positions, num_directions, positions) -> RayBundle:
        """Generate Direction Samples"""
        raise NotImplementedError

    def forward(self, num_positions: Optional[int] = None, num_directions: Optional[int] = None, positions: Optional[torch.Tensor] = None) -> RayBundle:
        """Returns directions for each position.

        Args:
            num_positions: Optional[int] number of positions on ddf sphere to sample
            num_directions: Optional[int] number of directions to sample for each position
            positions: Optional[torch.Tensor] positions on ddf sphere to sample

        Returns:
            RayBundle: ray bundle with origins and directions
        """

        if num_positions is None:
            num_positions = self.config.num_samples_on_sphere
        if num_directions is None:
            num_directions = self.config.num_rays_per_sample

        return self.generate_ddf_samples(num_positions, num_directions, positions)

# Field related configs
@dataclass
class UniformDDFSamplerConfig(DDFSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: UniformDDFSampler)
    """target class to instantiate"""

class UniformDDFSampler(DDFSampler):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        config: DDFSamplerConfig,
        ddf_sphere_radius: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(config=config, ddf_sphere_radius=ddf_sphere_radius, device=device)

    def random_inward_facing_directions(self, num_directions, normals):
        # num_directions = scalar
        # normals = (N, 3)
        # returns (N, num_directions, 3)

        # For each normal get a random set of directions
        directions = self.random_points_on_unit_sphere(num_directions * normals.shape[0], cartesian=True)
        directions = directions.reshape(normals.shape[0], num_directions, 3)

        # identify any directions that are not in the hemisphere of the associated normal
        dot_products = torch.sum(normals.unsqueeze(1) * directions, dim=2)
        mask = dot_products < 0

        # negate the directions that are not in the hemisphere
        directions[mask] = -directions[mask]

        return directions

    def generate_ddf_samples(self, num_positions, num_directions, positions=None) -> RayBundle:
        """Generate Direction Samples"""

        if positions is None:
            positions = self.random_points_on_unit_sphere(num_positions, cartesian=True)  # (num_positions, 3)

        if self.config.only_sample_upper_hemisphere:
            # negate any z values that are negative
            positions[positions[:, 2] < 0] = -positions[positions[:, 2] < 0]

        directions = self.random_inward_facing_directions(num_directions, normals=-positions)  # (num_positions, num_directions, 3)

        positions = positions * self.ddf_sphere_radius

        pos_ray = positions.repeat(num_directions, 1).to(self.device)
        dir_ray = directions.reshape(-1, 3).to(self.device)
        pixel_area = torch.ones(num_directions, 1, device=self.device)
        camera_indices = torch.zeros(num_directions, 1, device=self.device, dtype=torch.int64)
        metadata = {"directions_norm": torch.ones(num_directions, 1, device=self.device)}

        ray_bundle = RayBundle(
            origins=pos_ray,
            directions=dir_ray,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )

        return ray_bundle
    
# Field related configs
@dataclass
class VMFDDFSamplerConfig(DDFSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: VMFDDFSampler)
    """target class to instantiate"""
    concentration: float = 1.0
    """concentration parameter for von Mises-Fisher distribution"""
    

class VMFDDFSampler(DDFSampler):
    """Generate Samples using von Mises-Fisher distribution"""

    def __init__(
        self,
        config: DDFSamplerConfig,
        ddf_sphere_radius: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(config, ddf_sphere_radius=ddf_sphere_radius, device=device)
        self.concentration = self.config.concentration  # von Mises-Fisher parameter

    def _random_vmf_cos(self, d, kappa, n):
        """
        Generate n iid samples t with density function given by
        p(t) = some constant * (1 - t**2)**((d - 2)/2) * exp(kappa*t)
        """
        b = (d - 1) / (2 * kappa + (4 * kappa ** 2 + (d - 1) ** 2) ** 0.5)
        x0 = torch.tensor((1 - b) / (1 + b))
        c = kappa * x0 + (d - 1) * torch.log(1 - x0 ** 2)
        found = 0
        out = []
        while found < n:
            m = min(n, int((n - found) * 1.5))
            z = torch.distributions.beta.Beta((d - 1) / 2, (d - 1) / 2).sample((m,))
            t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            test = kappa * t + (d - 1) * torch.log(1 - x0 * t) - c
            accept = test >= -torch.exp(torch.ones(m))
            out.append(t[accept])
            found += len(out[-1])
        return torch.cat(out)[:n]
    
    def random_vmf(self, normals: torch.Tensor, kappa: int, num_samples: int = 10):
        """
        Von Mises - Fisher distribution sampler with
        mean direction mu and concentration kappa.
        Source : https://hal.science/hal-04004568
        """
        normals = normals / torch.norm(normals, dim=-1, keepdim=True) # shape (N, d) # ensure normalised
        (N, d) = normals.shape # N = number of normals, d = dimension of normals
        # z component : radial samples perpendicular to mu
        z = torch.normal(0, 1, (N, num_samples, d)) # shape (N, num_samples, d)
        z = z / torch.norm(z, dim=-1, keepdim=True) # normalise
        # project z onto the plane perpendicular to normals
        z = z - (torch.einsum('nij,nj->ni', z, normals))[..., None] * normals[:, None, :]
        z = z / torch.norm(z, dim=-1, keepdim=True)
        # sample angles (in cos and sin form)
        cos = self._random_vmf_cos(d, kappa, N * num_samples) # shape (N * num_samples)
        cos = cos.reshape(N, num_samples)
        sin = torch.sqrt(1 - cos ** 2) # shape (N, num_samples)
        # combine angles with the z component
        x = z * sin[..., None] + cos[..., None] * normals[:, None, :] # shape (N, num_samples, d)
        # ensure normalised
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x

    def generate_ddf_samples(self, num_positions, num_directions, positions=None) -> RayBundle:
        """Generate Direction Samples"""

        if positions is None:
            positions = self.random_points_on_unit_sphere(num_positions, cartesian=True)  # (num_positions, 3)

        if self.config.only_sample_upper_hemisphere:
            # negate any z values that are negative
            positions[positions[:, 2] < 0] = -positions[positions[:, 2] < 0]
            
        directions = self.random_vmf(normals=-positions, kappa=self.concentration, num_samples=num_directions) # (num_positions, num_directions, 3)

        # # identify any directions that are not in the hemisphere of the associated normal
        dot_products = torch.einsum('nij,nj->ni', directions, -positions) # (num_positions, num_directions)
        mask = dot_products < 0

        # negate the directions that are not in the hemisphere
        directions[mask] = -directions[mask] # (num_positions, num_directions, 3)

        positions = positions * self.ddf_sphere_radius # (num_positions, 3)

        positions = positions.unsqueeze(1).repeat(1, num_directions, 1).reshape(-1, 3) # (num_positions * num_directions, 3)
        directions = directions.reshape(-1, 3) # (num_positions * num_directions, 3)
        positions = positions.to(self.device)
        directions = directions.to(self.device)
        pixel_area = torch.ones(positions.shape[0], 1, device=self.device)
        camera_indices = torch.zeros(positions.shape[0], 1, device=self.device, dtype=torch.int64)
        metadata = {"directions_norm": torch.ones(positions.shape[0], 1, device=self.device)}

        ray_bundle = RayBundle(
            origins=positions,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )

        return ray_bundle