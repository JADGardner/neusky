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
from typing import Callable, List, Optional, Tuple, Union

import torch
from nerfacc import OccGridEstimator
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import Sampler

from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions


class DDFSDFSampler(Sampler):
    def __init__(self, num_samples, ddf_sphere_radius, sdf_function):
        super().__init__(num_samples=num_samples)
        self.sdf_function = sdf_function
        self.ddf_sphere_radius = ddf_sphere_radius

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        return_gt: bool = True,
    ):
        device = self.sdf_function.device
        if ray_bundle is None:
            num_samples = num_samples or self.num_samples

            positions = random_points_on_unit_sphere(1, cartesian=True)  # (1, 3)
            directions = random_inward_facing_directions(num_samples, normals=-positions)  # (1, num_directions, 3)

            positions = positions * self.ddf_sphere_radius

            pos_ray = positions.repeat(num_samples, 1).to(device)
            dir_ray = directions.reshape(-1, 3).to(device)
            pixel_area = torch.ones(num_samples, 1, device=device)
            camera_indices = torch.zeros(num_samples, 1, device=device, dtype=torch.int64)
            metadata = {"directions_norm": torch.ones(num_samples, 1, device=device)}

            ray_bundle = RayBundle(
                origins=pos_ray,
                directions=dir_ray,
                pixel_area=pixel_area,
                camera_indices=camera_indices,
                metadata=metadata,
            )

        accumulations = None
        termination_dist = None
        normals = None
        if return_gt:
            field_outputs = self.sdf_function(ray_bundle)
            accumulations = field_outputs["accumulation"].reshape(-1, 1).squeeze()
            termination_dist = field_outputs["p2p_dist"].reshape(-1, 1).squeeze()
            normals = field_outputs["normal"].reshape(-1, 3).squeeze()

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=ray_bundle.origins.reshape(-1, 3),
                directions=ray_bundle.directions.reshape(-1, 3),
                starts=torch.zeros_like(ray_bundle.origins),
                ends=torch.zeros_like(ray_bundle.origins),
                pixel_area=torch.ones_like(ray_bundle.origins),
            ),
        )

        return ray_samples, accumulations, termination_dist, normals
