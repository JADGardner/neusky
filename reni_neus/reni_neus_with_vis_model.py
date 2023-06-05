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
Implementation of NeuS similar to nerfacto where proposal sampler is used.
Based on SDFStudio https://github.com/autonomousvision/sdfstudio/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal
from collections import defaultdict
import random
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

import nerfacc
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

from reni_neus.illumination_fields.base_illumination_field import IlluminationFieldConfig
from reni_neus.model_components.renderers import RGBLambertianRendererWithVisibility
from reni_neus.model_components.illumination_samplers import IlluminationSamplerConfig
from reni_neus.utils.utils import RENITestLossMask, get_directions
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames
from reni_neus.ddf_model import DDFModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler

from reni_neus.reni_neus_model import RENINeuSFactoModelConfig, RENINeuSFactoModel

CONSOLE = Console(width=120)


@dataclass
class RENINeuSFactoWithVisibilityModelConfig(RENINeuSFactoModelConfig):
    """RENINeuSFacto With Visibility Model Config"""

    _target: Type = field(default_factory=lambda: RENINeuSFactoWithVisibilityModel)
    illumination_field: IlluminationFieldConfig = IlluminationFieldConfig()
    """Illumination Field"""
    illumination_sampler: IlluminationSamplerConfig = IlluminationSamplerConfig()
    """Illumination sampler to use"""
    illumination_field_prior_loss_weight: float = 1e-7
    """Weight for the prior loss"""
    illumination_field_cosine_loss_weight: float = 1e-1
    """Weight for the reni cosine loss"""
    illumination_field_loss_weight: float = 1.0
    """Weight for the reni loss"""
    visibility_loss_mse_multi: float = 0.01
    """Weight for the visibility mse loss"""
    render_only_albedo: bool = False
    """Render only albedo"""
    include_occupancy_network: bool = False
    """Include occupancy network in the model"""
    occupancy_grid_resolution: int = 64
    """Resolution of the occupancy grid"""
    occupancy_grid_levels: int = 4
    """Levels of the occupancy grid"""
    hashgrid_density_loss_weight: float = 0.0
    """Weight for the hashgrid density loss"""
    visibility_field: DDFModelConfig = DDFModelConfig()
    """Visibility field"""
    ddf_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""


class RENINeuSFactoWithVisibilityModel(RENINeuSFactoModel):
    """NeuSFactoModel extends NeuSModel for a more efficient sampling strategy.

    The model improves the rendering speed and quality by incorporating a learning-based
    proposal distribution to guide the sampling process.(similar to mipnerf-360)

    Args:
        config: NeuS configuration to instantiate model
    """

    config: RENINeuSFactoModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        num_val_data: int,
        num_test_data: int,
        test_mode: str,
        **kwargs,
    ) -> None:
        self.num_val_data = num_val_data
        self.num_test_data = num_test_data
        self.test_mode = test_mode
        self.fitting_eval_latents = False
        super().__init__(config, scene_box, num_train_data, **kwargs)

    def populate_modules(self):
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        if self.config.ddf_radius == "AABB":
            self.ddf_radius = torch.abs(self.scene_box.aabb[0, 0]).item()
        else:
            self.ddf_radius = self.config.ddf_radius

        self.visibility_field = self.config.visibility_field.setup(reni_neus=self, ddf_radius=self.ddf_radius)

    def ray_sphere_intersection(self, positions, directions, radius):
        """Ray sphere intersection"""
        # ray-sphere intersection
        # positions is the origins of the rays
        # directions is the directions of the rays
        # radius is the radius of the sphere

        sphere_origin = torch.zeros_like(positions)
        radius = torch.ones_like(positions[..., 0]) * radius

        a = 1 # direction is normalized
        b = 2 * torch.einsum("ij,ij->i", directions, positions - sphere_origin)
        c = torch.einsum("ij,ij->i", positions - sphere_origin, positions - sphere_origin) - radius**2

        discriminant = b**2 - 4 * a * c

        t0 = (-b - torch.sqrt(discriminant)) / (2 * a)
        t1 = (-b + torch.sqrt(discriminant)) / (2 * a)

        # since we are inside the sphere we want the positive t
        t = torch.max(t0, t1)

        # now we need to point on the sphere that we intersected
        intersection_point = positions + t.unsqueeze(-1) * directions

        return intersection_point

    def compute_visibility(self, ray_samples, p2p_dist, illumination_directions, threshold_distance):
        """Compute visibility"""
        # ddf_model directional distance field model
        # positions is the origins of the rays from the surface of the object
        # directions is the directions of the rays from the surface of the object
        # sphere_intersection_points is the point on the sphere that we intersected
        
        # shortcuts
        num_light_directions = illumination_directions.shape[1]
        num_rays = ray_samples.frustums.origins.shape[0]

        # since we are only using a single sample, the sample we think has hit the object,
        # we can just use one of each of these values, they are all just copied for each
        # sample along the ray. So here I'm just taking the first one.
        origins = ray_samples.frustums.origins[:, 0:1, :]  # [num_rays, 1, 3]
        ray_directions = ray_samples.frustums.directions[:, 0:1, :]  # [num_rays, 1, 3]

        # get positions based on p2p distance (expected termination depth)
        # this is our sample on the surface of the SDF representing the scene
        positions = origins + ray_directions * p2p_dist.unsqueeze(-1)

        positions = positions.unsqueeze(1).repeat(
            1, num_light_directions, 1, 1
        )  # [num_rays, num_light_directions, 1, 3]
        directions = (
            illumination_directions[0:1, :, :].unsqueeze(2).repeat(num_rays, 1, 1, 1)
        )  # [num_rays, num_light_directions, 1, 3]

        positions = positions.reshape(-1, 3)  # [num_rays * num_light_directions, 3]
        directions = directions.reshape(-1, 3)  # [num_rays * num_light_directions, 3]

        positions = ray_samples.frustums.origins # [N, 3]

        sphere_intersection_points = self.ray_sphere_intersection(positions, directions, self.ddf_radius) # [num_rays * num_light_directions, 3]
        
        # we need directions from intersection points to ray origins
        directions = -directions 

        # build a ray_sample object to pass to the visibility_field
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=directions,
                starts=ray_samples.frustums.starts,
                ends=ray_samples.frustums.ends,
                pixel_area=ray_samples.frustums.pixel_area,
            ),
        )

        # Get output of visibility field (DDF)
        outputs = self.visibility_field(ray_samples) # [N, 2]

        # the distance from the point on the sphere to the point on the SDF
        dist_to_ray_origins = torch.norm(positions - sphere_intersection_points, dim=-1) # [N]

        # add threshold_distance extra to the expected_termination_dist (i.e slighly futher into the SDF) 
        # and get the difference between it and the distance from the point on the sphere to the point on the SDF
        difference = (outputs['expected_termination_dist'] + threshold_distance) - dist_to_ray_origins

        # if the difference is positive then the expected termination distance
        # is greater than the distance from the point on the sphere to the point on the SDF
        # so the point on the sphere is visible to it
        visibility = (difference > 0).float()

        visibility_dict = {
            "visibility": visibility,
            "expected_termination_dist": outputs['expected_termination_dist'],
            "sdf_at_termination": outputs['sdf_at_termination'],
        }

        return visibility_dict


    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        """Sample rays using proposal networks and compute the corresponding field outputs."""
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays, samples_per_ray]

        field_outputs = self.field(ray_samples, return_alphas=True)

        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        illumination_field = self.get_illumination_field()

        illumination_directions = self.illumination_sampler()
        illumination_directions = illumination_directions.to(self.device)

        # Get environment illumination for samples along the rays for each unique camera
        hdr_illumination_colours, illumination_directions = illumination_field(
            camera_indices=camera_indices,
            positions=None,
            directions=illumination_directions,
            illumination_type="illumination",
        )

        # Get LDR colour of the background for rays from the camera that don't hit the scene
        background_colours, _ = illumination_field(
            camera_indices=camera_indices,
            positions=None,
            directions=ray_samples.frustums.directions[:, 0, :],
            illumination_type="background",
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
            "illumination_directions": illumination_directions,
            "hdr_illumination_colours": hdr_illumination_colours,
            "background_colours": background_colours,
        }

        if self.config.hashgrid_density_loss_weight > 0.0:
            pass
            # generate a set of uniform samples in the scene within the aabb of the scene
            # and compute the density at those points

        albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)

        if not self.config.render_only_albedo:
            p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)

            visibility_dict = self.compute_visibility(ray_samples=ray_samples,
                                                      p2p_dist=p2p_dist,
                                                      illumination_directions=illumination_directions,
                                                      threshold_distance=0.5)

            rgb = self.lambertian_renderer(
                albedos=field_outputs[RENINeuSFieldHeadNames.ALBEDO],
                normals=field_outputs[FieldHeadNames.NORMALS],
                light_directions=illumination_directions,
                light_colors=hdr_illumination_colours,
                visibility=visibility_dict["visibility"],
                background_illumination=background_colours,
                weights=weights,
            )
            # the rendered depth is point-to-point distance and we should convert to depth
            depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            accumulation = self.renderer_accumulation(weights=weights)
        else:
            rgb = torch.zeros_like(albedo)
            p2p_dist = torch.zeros_like(albedo)[..., 0]
            depth = torch.zeros_like(albedo)[..., 0]
            normal = torch.zeros_like(albedo)
            accumulation = torch.zeros_like(albedo)[..., 0]

        samples_and_field_outputs["rgb"] = rgb
        samples_and_field_outputs["accumulation"] = accumulation
        samples_and_field_outputs["depth"] = depth
        samples_and_field_outputs["normal"] = normal
        samples_and_field_outputs["albedo"] = albedo
        samples_and_field_outputs["p2p_dist"] = p2p_dist

        return samples_and_field_outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # shortcuts
        field_outputs = samples_and_field_outputs["field_outputs"]

        weights = samples_and_field_outputs["weights"]
        rgb = samples_and_field_outputs["rgb"]
        accumulation = samples_and_field_outputs["accumulation"]
        depth = samples_and_field_outputs["depth"]
        normal = samples_and_field_outputs["normal"]
        p2p_dist = samples_and_field_outputs["p2p_dist"]
        background_colours = samples_and_field_outputs["background_colours"]
        albedo = samples_and_field_outputs["albedo"]
        normal = samples_and_field_outputs["normal"]

        outputs = {
            "rgb": rgb,
            "albedo": albedo,
            "accumulation": accumulation,
            "depth": depth,
            "p2p_dist": p2p_dist,
            "normal": normal,
            "weights": weights,
            "background_colours": background_colours,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs