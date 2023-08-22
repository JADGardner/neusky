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
from pathlib import Path
import numpy as np
import cv2
import functools

import nerfacc
import torch
from torch.nn import Parameter
from torch.functional import F

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.configs.config_utils import to_immutable_dict

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SphereCollider
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.utils import colormaps, misc
from nerfstudio.model_components.losses import (
    interlevel_loss,
    monosdf_normal_loss,
)

from nerfstudio.viewer.server.viewer_elements import *

from reni_neus.model_components.renderers import RGBLambertianRendererWithVisibility
from reni_neus.model_components.losses import RENISkyPixelLoss
from reni_neus.field_components.reni_neus_fieldheadnames import RENINeuSFieldHeadNames
from reni_neus.models.ddf_model import DDFModelConfig, DDFModel
from reni_neus.model_components.ddf_sampler import DDFSamplerConfig

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.model_components.illumination_samplers import IlluminationSamplerConfig, EquirectangularSamplerConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni_neus.utils.utils import find_nerfstudio_project_root, rot_z

CONSOLE = Console(width=120)


@dataclass
class RENINeuSFactoModelConfig(NeuSFactoModelConfig):
    """NeusFacto Model Config"""

    _target: Type = field(default_factory=lambda: RENINeuSFactoModel)
    illumination_field: SphericalFieldConfig = SphericalFieldConfig()
    """Illumination Field"""
    illumination_field_ckpt_path: Path = Path('/path/to/ckpt.pt')
    """Path of pretrained illumination field"""
    illumination_field_ckpt_step: int = 0
    """Step of pretrained illumination field"""
    illumination_sampler: IlluminationSamplerConfig = IlluminationSamplerConfig()
    """Illumination sampler to use"""
    illumination_field_cosine_loss_weight: float = 1e-1
    """Weight for the reni cosine loss"""
    illumination_field_loss_weight: float = 1.0
    """Weight for the reni loss"""
    visibility_threshold: Union[Literal["learnable"], Tuple[float, float], float] = "learnable"
    """Learnable visibility threshold"""
    steps_till_min_visibility_threshold: int = 15000
    """Number of steps till min visibility threshold"""
    only_upperhemisphere_visibility: bool = False
    """Lower hemisphere visibility will always be 1.0 if this is True"""
    fix_test_illumination_directions: bool = False
    """Fix the test illumination directions"""
    use_visibility: Union[int, bool] = False 
    """Use visibility field either bool or after int steps"""
    scene_contraction_order: Literal["Linf", "L2"] = "Linf"
    """Norm for scene contraction"""
    collider_shape: Literal["sphere", "box"] = "box"
    """Shape of the collider"""
    loss_inclusions: Dict[str, any] = to_immutable_dict({
        "rgb_l1_loss": True,
        "rgb_l2_loss": True,
        "eikonal loss": True,
        "fg_mask_loss": True,
        "normal_loss": True,
        "depth_loss": True,
        "interlevel_loss": True,
        "sky_pixel_loss": {
            "enabled": True,
            "cosine_weight": 0.1,
        },
        "hashgrid_density_loss": {
            "enabled": True,
            "grid_resolution": 256,
        },
        "ground_plane_loss": True,
    })
    """Which losses to include in the training"""


class RENINeuSFactoModel(NeuSFactoModel):
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
        visibility_field: Union[DDFModel, None],
        test_mode: str,
        **kwargs,
    ) -> None:
        self.num_val_data = num_val_data
        self.num_test_data = num_test_data
        self.test_mode = test_mode
        self.fitting_eval_latents = False
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.visibility_field = visibility_field

        if self.config.scene_contraction_order == "Linf":
            self.scene_contraction = SceneContraction(order=float("inf"))
        else:
            self.scene_contraction = SceneContraction()
        
        if self.config.collider_shape == "box":
            self.collider = AABBBoxCollider(self.scene_box, near_plane=0.05)
        else:
            self.collider = SphereCollider(center=torch.zeros(3), radius=1.0, near_plane=0.05)

        self.setup_gui()

        # if self.config.ddf_radius == "AABB":
        #     self.ddf_radius = torch.abs(self.scene_box.aabb[0, 0]).item()
        # else:
        #     self.ddf_radius = self.config.ddf_radius

        if self.visibility_field is not None:
            self.ddf_radius = self.visibility_field.ddf_radius
            # self.visibility_field = self._setup_visibility_field()

            self.visibility_threshold_end = None
            if self.config.visibility_threshold == "learnable":
                self.visibility_threshold = Parameter(torch.tensor(1.0))
            elif isinstance(self.config.visibility_threshold, tuple):
                # this is start and end and we decrease exponentially
                self.visibility_threshold_start = torch.tensor(self.config.visibility_threshold[0])
                self.visibility_threshold_end = torch.tensor(self.config.visibility_threshold[1])
            else:
                self.visibility_threshold = torch.tensor(self.config.visibility_threshold)
        

    def populate_modules(self):
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        # Three seperate illumination fields for train, val and test, we are using a fixed decoder so
        # train_data for the illumination field is none and we are optimising the eval latents instead.

        # TODO Get from checkpoint
        normalisations = {'min_max': None,
                          'log_domain': True}

        self.illumination_field_train = self.config.illumination_field.setup(num_train_data=None, num_eval_data=self.num_train_data, normalisations=normalisations)
        self.illumination_field_val = self.config.illumination_field.setup(num_train_data=None, num_eval_data=self.num_val_data, normalisations=normalisations)
        self.illumination_field_test = self.config.illumination_field.setup(num_train_data=None, num_eval_data=self.num_test_data, normalisations=normalisations)

        # if self.illumination_field_train is of type RENIFieldConfig
        if isinstance(self.illumination_field_train, RENIField):
            # Now you can use this to construct paths:
            project_root = find_nerfstudio_project_root(Path(__file__))
            relative_path = self.config.illumination_field_ckpt_path / 'nerfstudio_models' / f'step-{self.config.illumination_field_ckpt_step:09d}.ckpt'
            ckpt_path = project_root / relative_path
            
            if not ckpt_path.exists():
                raise ValueError(f'Could not find illumination field checkpoint at {ckpt_path}')
        
            ckpt = torch.load(str(ckpt_path))
            illumination_field_dict = {}
            match_str = '_model.field.network.'
            for key in ckpt['pipeline'].keys():
                if key.startswith(match_str):
                    illumination_field_dict[key[len(match_str):]] = ckpt['pipeline'][key]

            # load weights of the decoder
            self.illumination_field_train.network.load_state_dict(illumination_field_dict)
            self.illumination_field_val.network.load_state_dict(illumination_field_dict)
            self.illumination_field_test.network.load_state_dict(illumination_field_dict)

        self.illumination_sampler = self.config.illumination_sampler.setup()
        self.equirectangular_sampler = EquirectangularSamplerConfig(width=128).setup()

        self.field_background = None

        self.albedo_renderer = RGBRenderer(background_color=torch.tensor([1.0, 1.0, 1.0]))
        self.lambertian_renderer = RGBLambertianRendererWithVisibility()

        assert not (self.config.loss_inclusions['rgb_l1_loss'] and self.config.loss_inclusions['rgb_l2_loss']), "Cannot have both L1 and L2 loss"
        if self.config.loss_inclusions['rgb_l1_loss']:
            self.rgb_l1_loss = torch.nn.L1Loss()
        if self.config.loss_inclusions['rgb_l2_loss']:
            self.rgb_l2_loss = torch.nn.MSELoss()
        if self.config.loss_inclusions['sky_pixel_loss']['enabled']:
            self.sky_pixel_loss = RENISkyPixelLoss(
                alpha=self.config.loss_inclusions['sky_pixel_loss']['cosine_weight'],
            )
        if self.config.loss_inclusions['hashgrid_density_loss']['enabled']:
            self.hashgrid_density_loss = torch.nn.L1Loss()
        if self.config.loss_inclusions['ground_plane_loss']:
            self.ground_plane_loss = monosdf_normal_loss


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return a dictionary with the parameters of the proposal networks."""
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["illumination_field"] = list(self.illumination_field_train.parameters())
        return param_groups

    def get_illumination_field(self):
        """Return the illumination field for the current mode."""
        if self.training and not self.fitting_eval_latents:
            illumination_field = self.illumination_field_train
        else:
            illumination_field = (
                self.illumination_field_test if self.test_mode == "test" else self.illumination_field_val
            )

        return illumination_field
    
    def decay_threshold(self, step):
        """Decay the visibility threshold exponentially"""
        if step >= self.config.steps_till_min_visibility_threshold:
            return self.visibility_threshold_end
        else:
            decay_rate = -torch.log(self.visibility_threshold_end / self.visibility_threshold_start) / self.config.steps_till_min_visibility_threshold
            return self.visibility_threshold_start * torch.exp(-decay_rate * step)
    
    def forward(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None, step: Union[int, None] = None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            batch: batch needed for DDF: masks, etc.
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, batch=batch, rotation=rotation, step=step)
    
    def sample_illumination(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None] = None):
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays, samples_per_ray]

        illumination_field = self.get_illumination_field()

        if not self.training and self.config.fix_test_illumination_directions:
            illumination_ray_samples = self.illumination_sampler(apply_random_rotation=False) # [num_illumination_direction]
        else:
            illumination_ray_samples = self.illumination_sampler() # [num_illumination_direction]

        illumination_ray_samples = illumination_ray_samples.to(self.device)

        # we want unique camera indices and for each one to sample illumination directions
        unique_indices, inverse_indices = torch.unique(camera_indices, return_inverse=True) # [num_unique_camera_indices], [num_rays, samples_per_ray]
        num_unique_camera_indices = unique_indices.shape[0]
        num_illumination_directions = illumination_ray_samples.shape[0]
        unique_indices = unique_indices.unsqueeze(1).repeat(1, num_illumination_directions) # [num_unique_camera_indices, num_illumination_directions]
        # illumination_ray_samples.frustums.directions # shape [num_illumination_directions, 3] we need to repeat this for each unique camera index
        directions = illumination_ray_samples.frustums.directions.unsqueeze(0).repeat(num_unique_camera_indices, 1, 1) # [num_unique_camera_indices, num_illumination_directions, 3]
        # now reshape both and put into ray_samples
        illumination_ray_samples.camera_indices = unique_indices.reshape(-1) # [num_unique_camera_indices * num_illumination_directions]
        illumination_ray_samples.frustums.directions = directions.reshape(-1, 3) # [num_unique_camera_indices * num_illumination_directions, 3]
        illuination_field_outputs = illumination_field.forward(illumination_ray_samples, rotation=rotation) # [num_unique_camera_indices * num_illumination_directions, 3]
        hdr_illumination_colours = illuination_field_outputs[RENIFieldHeadNames.RGB] # [num_unique_camera_indices * num_illumination_directions, 3]
        hdr_illumination_colours = illumination_field.unnormalise(hdr_illumination_colours) # [num_unique_camera_indices * num_illumination_directions, 3]
        # so now we reshape back to [num_unique_camera_indices, num_illumination_directions, 3]
        hdr_illumination_colours = hdr_illumination_colours.reshape(num_unique_camera_indices, num_illumination_directions, 3)
        # and use inverse indices to get back to [num_rays, num_illumination_directions, 3]
        hdr_illumination_colours = hdr_illumination_colours[inverse_indices] # [num_rays, samples_per_ray, num_illumination_directions, 3]
        # and then unsqueeze, repeat and reshape to get to [num_rays * samples_per_ray, num_illumination_directions, 3]
        hdr_illumination_colours = hdr_illumination_colours.reshape(-1, num_illumination_directions, 3) # [num_rays * samples_per_ray, num_illumination_directions, 3]
        # same for illumination directions
        illumination_directions = directions[inverse_indices] # [num_rays, samples_per_ray, num_illumination_directions, 3]
        illumination_directions = illumination_directions.reshape(-1, num_illumination_directions, 3) # [num_rays * samples_per_ray, num_illumination_directions, 3]

        # now we need to get samples from distant illumination field for camera rays
        illumination_field_outputs = illumination_field.forward(ray_samples[:, 0], rotation=rotation)
        hdr_background_colours = illumination_field_outputs[RENIFieldHeadNames.RGB]
        hdr_background_colours = illumination_field.unnormalise(hdr_background_colours)

        return hdr_illumination_colours, illumination_directions, hdr_background_colours

    def sample_and_forward_field(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None, step: Union[int, None] = None) -> Dict[str, Any]:
        """Sample rays using proposal networks and compute the corresponding field outputs."""
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field(ray_samples, return_alphas=True)

        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        hdr_illumination_colours, illumination_directions, hdr_background_colours = self.sample_illumination(ray_samples, rotation)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
            "illumination_directions": illumination_directions,
            "hdr_illumination_colours": hdr_illumination_colours,
            "hdr_background_colours": hdr_background_colours,
        }

        if self.visibility_field is not None:
            # we need depth to compute visibility so render it here instead of in get_outputs()
            p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            # the rendered depth is point-to-point distance and we should convert to depth
            depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            # we need accumulation so we can ignore samples that don't hit the scene
            accumulation = self.renderer_accumulation(weights=weights)

            if self.config.use_visibility:
                # if we are decaying visibility threshold then compute it here
                if self.visibility_threshold_end is not None:
                    visibility_threshold = self.decay_threshold(step)
                else:
                    visibility_threshold = self.visibility_threshold

                visibility_dict = self.compute_visibility(ray_samples=ray_samples,
                                                          depth=depth,
                                                          illumination_directions=illumination_directions,
                                                          threshold_distance=visibility_threshold,
                                                          accumulation=accumulation,
                                                          batch=batch)
                
                samples_and_field_outputs["visibility_dict"] = visibility_dict
            
            samples_and_field_outputs["p2p_dist"] = p2p_dist
            samples_and_field_outputs["depth"] = depth
            samples_and_field_outputs["accumulation"] = accumulation

            if not self.training and self.render_shadow_map_static_flag:
                accumulation_mask = accumulation > self.accumulation_mask_threshold_static
                # convert self.shadow_map_azimuth_static and self.shadow_map_elevation_static to radians
                # from degrees to x, y, z direction with z being up
                azimuth = self.shadow_map_azimuth_static * np.pi / 180
                elevation = self.shadow_map_elevation_static * np.pi / 180
                shadow_map_direction = torch.tensor([np.cos(azimuth) * np.cos(elevation),
                                                      np.sin(azimuth) * np.cos(elevation),
                                                      np.sin(elevation)]).to(self.device)
                shadow_map_direction = shadow_map_direction.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 3]
                shadow_map_direction = shadow_map_direction.repeat(ray_samples.shape[0], 1, 1) # Shape: [num_rays, 1, 3]
                shadow_map_direction = shadow_map_direction.type_as(illumination_directions)

                # illumination_directions = [num_rays * num_samples, num_light_directions, xyz]
                shadow_map = self.compute_visibility(ray_samples=ray_samples[:, 0:1], # Shape: [num_rays, 1]
                                                     depth=p2p_dist,
                                                     illumination_directions=shadow_map_direction,
                                                     threshold_distance=self.shadow_map_threshold_static,
                                                     accumulation=accumulation,
                                                     batch=batch,
                                                     compute_shadow_map=True)
                
                # mask using accumulation_mask
                shadow_map['visibility'] = shadow_map['visibility'] * accumulation_mask.unsqueeze(1).type_as(shadow_map['visibility'])
                shadow_map['difference'] = shadow_map['difference'] * accumulation_mask.unsqueeze(1).type_as(shadow_map['difference'])
                
                samples_and_field_outputs["shadow_map"] = shadow_map

        if self.training and self.config.loss_inclusions['hashgrid_density_loss']['enabled']:
            # Get min and max coordinates
            min_coord, max_coord = self.scene_box.aabb

            # Create a linear space for each dimension
            x = torch.linspace(min_coord[0], max_coord[0], self.config.loss_inclusions['hashgrid_density_loss']['grid_resolution'])
            y = torch.linspace(min_coord[1], max_coord[1], self.config.loss_inclusions['hashgrid_density_loss']['grid_resolution'])
            z = torch.linspace(min_coord[2], max_coord[2], self.config.loss_inclusions['hashgrid_density_loss']['grid_resolution'])

            # Generate a 3D grid of points
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            positions = torch.stack((X, Y, Z), -1)  # size will be (resolution, resolution, resolution, 3)

            # Flatten and reshape
            positions = positions.reshape(-1, 3)

            # Calculate gaps between each sample
            gap = torch.tensor([(max_coord[i] - min_coord[i]) / self.config.loss_inclusions['hashgrid_density_loss']['grid_resolution'] for i in range(3)])

            # Generate random perturbations
            perturbations = torch.rand_like(positions) * gap - gap / 2

            # Apply perturbations
            positions += perturbations
            
            # generate random normalised directions of shape positions
            # these are needed for generating alphas
            directions = torch.randn_like(positions)
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)

            # Create ray_samples
            grid_samples = RaySamples(
                frustums=Frustums(
                origins=positions,
                directions=directions,
                starts=torch.zeros_like(positions),
                ends=torch.zeros_like(positions),
                pixel_area=torch.zeros_like(positions[:, 0]),
              ),
              deltas=gap,
            )

            grid_samples.frustums.origins = grid_samples.frustums.origins.to(self.device)
            grid_samples.frustums.directions = grid_samples.frustums.directions.to(self.device)
            grid_samples.frustums.starts = grid_samples.frustums.starts.to(self.device)
            grid_samples.deltas = grid_samples.deltas.to(self.device)

            # get density
            density = self.field.get_alpha(grid_samples)

            samples_and_field_outputs["grid_density"] = density

        return samples_and_field_outputs

    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None, step: Union[int, None] = None) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle, batch=batch, rotation=rotation, step=step)

        # shortcuts
        field_outputs = samples_and_field_outputs["field_outputs"]

        weights = samples_and_field_outputs["weights"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        illumination_directions = samples_and_field_outputs["illumination_directions"]
        hdr_illumination_colours = samples_and_field_outputs["hdr_illumination_colours"]
        hdr_background_colours = samples_and_field_outputs["hdr_background_colours"]

        visibility = None
        accumulation = None
        p2p_dist = None
        depth = None
        if 'visibility_dict' in samples_and_field_outputs:
            visibility = samples_and_field_outputs["visibility_dict"]["visibility"]
        if 'accumulation' in samples_and_field_outputs:
            accumulation = samples_and_field_outputs["accumulation"]
        if 'p2p_dist' in samples_and_field_outputs:
            p2p_dist = samples_and_field_outputs["p2p_dist"]
            depth = samples_and_field_outputs["depth"]
        
        if self.training:
            rgb = self.lambertian_renderer(
                albedos=field_outputs[RENINeuSFieldHeadNames.ALBEDO],
                normals=field_outputs[FieldHeadNames.NORMALS],
                light_directions=illumination_directions,
                light_colors=hdr_illumination_colours,
                visibility=visibility,
                background_illumination=hdr_background_colours,
                weights=weights,
            )
            if accumulation is None:
                accumulation = self.renderer_accumulation(weights=weights)
            if p2p_dist is None:
                p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                # the rendered depth is point-to-point distance and we should convert to depth
                depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)
        else:
            if self.render_rgb_static_flag:
                rgb = self.lambertian_renderer(
                    albedos=field_outputs[RENINeuSFieldHeadNames.ALBEDO],
                    normals=field_outputs[FieldHeadNames.NORMALS],
                    light_directions=illumination_directions,
                    light_colors=hdr_illumination_colours,
                    visibility=visibility,
                    background_illumination=hdr_background_colours,
                    weights=weights,
                )
            else:
                rgb = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)

            if accumulation is None and self.render_accumulation_static_flag:
                accumulation = self.renderer_accumulation(weights=weights)
            elif accumulation is None and not self.render_accumulation_static_flag:
                accumulation = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)

            if p2p_dist is None and self.render_depth_static_flag:
                p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                # the rendered depth is point-to-point distance and we should convert to depth
                depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            elif p2p_dist is None and not self.render_depth_static_flag:
                p2p_dist = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)
                depth = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)
                
            if self.render_normal_static_flag:
                normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            else:
                normal = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)

            if self.render_albedo_static_flag:
                albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)
            else:
                albedo = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)


        outputs = {
            "rgb": rgb,
            "albedo": albedo,
            "accumulation": accumulation,
            "depth": depth,
            "p2p_dist": p2p_dist,
            "normal": normal,
            "weights": weights,
            "hdr_background_colours": hdr_background_colours,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }

        if not self.training and self.render_shadow_map_static_flag:
            shadow_map = samples_and_field_outputs["shadow_map"]['visibility']
            outputs["shadow_map"] = shadow_map

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

        if 'grid_density' in samples_and_field_outputs:
            outputs['grid_density'] = samples_and_field_outputs['grid_density']

        if 'visibility_dict' in samples_and_field_outputs:
            outputs['visibility_batch'] = samples_and_field_outputs['visibility_dict']['visibility_batch']

        return outputs

    def get_loss_dict(
        self, outputs: Dict[str, Any], batch: Dict[str, Any], metrics_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the loss dictionary, including interlevel loss for proposal networks."""
        image = batch["image"].to(self.device)
        fg_mask = batch["mask"][..., 1].to(self.device) # [num_rays]
        ground_mask = batch["mask"][..., 2].to(self.device) # [num_rays]
        loss_dict = {}

        if self.config.loss_inclusions["rgb_l1_loss"]:
            loss_dict['rgb_l1_loss'] = self.rgb_l1_loss(input=outputs["rgb"], target=image)
        if self.config.loss_inclusions["rgb_l2_loss"]:
            loss_dict['rgb_l2_loss'] = self.rgb_l2_loss(input=outputs["rgb"], target=image)
            # image = batch["image"].to(self.device)
            # pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            #     pred_image=outputs["rgb"],
            #     pred_accumulation=outputs["accumulation"],
            #     gt_image=image,
            # )
            # loss_dict["rgb_mse_loss"] = self.rgb_loss(image, pred_image)
        
        if self.config.loss_inclusions["sky_pixel_loss"]["enabled"]:
            sky_colours = linear_to_sRGB(outputs["hdr_background_colours"])
            fg_label = fg_mask.float().unsqueeze(1).expand_as(sky_colours)
            sky_label = 1 - fg_label
            loss_dict["sky_pixel_loss"] = self.sky_pixel_loss(
                    inputs=linear_to_sRGB(outputs["hdr_background_colours"]),
                    targets=batch["image"].type_as(sky_label),
                    mask=sky_label,
                )

        if self.training:
            # eikonal loss
            if self.config.loss_inclusions["eikonal loss"]:
                grad_theta = outputs["eik_grad"]
                loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()

            # foreground mask loss
            if self.config.loss_inclusions["fg_mask_loss"]:
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3) # [num_rays, 1]
                fg_label = fg_mask.float().unsqueeze(1) # [num_rays, 1]
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label)
                )

            # monocular normal loss
            if self.config.loss_inclusions["normal_loss"]:
                assert "normal" in batch
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                    monosdf_normal_loss(normal_pred, normal_gt)
                )
            
            # mono depth loss
            if self.config.loss_inclusions["depth_loss"]:
                assert "depth" in batch
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                depth_mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), depth_mask)
                )

            if self.config.loss_inclusions["interlevel_loss"]:
                loss_dict["interlevel_loss"] = interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )
            
            if self.config.loss_inclusions["hashgrid_density_loss"]["enabled"]:
                loss_dict['hashgrid_density_loss'] = self.hashgrid_density_loss(outputs['grid_density'], torch.zeros_like(outputs['grid_density']))

            if self.config.loss_inclusions["ground_plane_loss"]:
                normal_pred = outputs["normal"]
                # ground plane should be facing up in z direction
                normal_gt = torch.tensor([0.0, 0.0, 1.0]).expand_as(normal_pred).to(self.device)
                ground_mask = ground_mask.unsqueeze(1).expand_as(normal_pred)
                loss_dict["ground_plane_loss"] = monosdf_normal_loss(normal_pred * ground_mask, normal_gt * ground_mask)
        
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images, including the proposal depth for each iteration."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        illumination_field = self.get_illumination_field()

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        images_dict["albedo"] = outputs["albedo"]
        images_dict["envmap_from_camera"] = linear_to_sRGB(outputs["hdr_background_colours"])

        with torch.no_grad():
            ray_samples = self.equirectangular_sampler.generate_direction_samples()
            ray_samples = ray_samples.to(self.device)
            ray_samples.camera_indices = torch.ones_like(ray_samples.camera_indices) * batch["image_idx"]
            illumination_field_outputs = illumination_field(ray_samples)
            hdr_envmap = illumination_field_outputs[RENIFieldHeadNames.RGB]
            hdr_envmap = illumination_field.unnormalise(hdr_envmap) # N, 3
            ldr_envmap = linear_to_sRGB(hdr_envmap) # N, 3
            # reshape to H, W, 3
            height = self.equirectangular_sampler.height
            width = self.equirectangular_sampler.width
            ldr_envmap = ldr_envmap.reshape(height, width, 3)
            images_dict["ldr_envmap"] = ldr_envmap

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, show_progress=False, rotation=None, to_cpu=False, step: Union[int, None] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)

        # This handles thread issues as viewer may change states during rendering of the whole frame
        self.render_rgb_static_flag = self.render_rgb_flag
        self.render_accumulation_static_flag = self.render_accumulation_flag
        self.render_depth_static_flag = self.render_depth_flag
        self.render_normal_static_flag = self.render_normal_flag
        self.render_albedo_static_flag = self.render_albedo_flag
        self.render_shadow_map_static_flag = self.render_shadow_map_flag
        self.shadow_map_threshold_static = self.shadow_map_threshold.value
        self.shadow_map_azimuth_static = self.shadow_map_azimuth.value
        self.shadow_map_elevation_static = self.shadow_map_elevation.value
        self.accumulation_mask_threshold_static = self.accumulation_mask_threshold.value
        self.render_shadow_envmap_overlay_static_flag = self.render_shadow_envmap_overlay_flag
        self.shadow_envmap_overlay_pos_static = self.shadow_envmap_overlay_pos
        self.shadow_envmap_overlay_dir_static = self.shadow_envmap_overlay_dir

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("[green]Generating eval images...", total=num_rays, extra="")
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = self.forward(ray_bundle=ray_bundle, rotation=rotation, step=step)
                    for output_name, output in outputs.items():  # type: ignore
                        if not torch.is_tensor(output):
                            # TODO: handle lists of tensors as well
                            continue
                        else:
                            if to_cpu:
                                output = output.cpu()
                        outputs_lists[output_name].append(output)
                    progress.update(task, completed=i)
        else:
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle, rotation=rotation)
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        # if not self.training and self.render_shadow_envmap_overlay_static:
        #         position = torch.tensor(self.shadow_envmap_overlay_pos_static, dtype=torch.float32, device=self.device)
        #         direction = torch.tensor(self.shadow_envmap_overlay_dir_static, dtype=torch.float32, device=self.device)
                
        #         # if these are None just set them to 000 and 010
        #         if position is None:
        #             position = torch.tensor([0,0,0], dtype=torch.float32, device=self.device)
        #         if direction is None:
        #             direction = torch.tensor([0,1,0], dtype=torch.float32, device=self.device)

        #         ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(0, 1)
        #         # update the ray bundle with the position and direction
        #         ray_bundle.origins = position.unsqueeze(0)
        #         ray_bundle.directions = direction.unsqueeze(0)

        #         if self.collider is not None:
        #             ray_bundle = self.collider(ray_bundle)
                
        #         ray_samples, _, _ = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        #         field_outputs = self.field(ray_samples, return_alphas=True)

        #         weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
        #             field_outputs[FieldHeadNames.ALPHA]
        #         )

        #         p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        #         depth = p2p_dist / ray_bundle.metadata["directions_norm"]

        #         # now we have the position on the surface we sample for a shadow map using equirectangular sampling
        #         side_length = 128
        #         directions = get_directions(side_length, 'Nerfstudio').type_as(position)

        #         # check sizes of everything
        #         vis_dict = self.compute_visibility(ray_samples=ray_samples[:, 0:1], # Shape: [1, 1]
        #                                              depth=depth, # Shape: [1, 1]
        #                                              illumination_directions=directions, # Shape: [1, side_length*(side_length//2), 3]
        #                                              threshold_distance=self.shadow_map_threshold_static,
        #                                              accumulation=None,
        #                                              batch=None,
        #                                              compute_shadow_map=True)
                
        #         shadow_map = vis_dict['visibility'] # Shape: [1, side_length*side_length//2, 1]
        #         # reshape to 1, 64, 32, 1
        #         shadow_map = shadow_map.reshape(1, 1, side_length, side_length//2)
        #         # repeat to rgb
        #         shadow_map = shadow_map.repeat(1, 3, 1, 1)

        #         if (image_width // 2) < side_length:
        #             new_width = image_width // 2
        #             new_height = image_width // 4
        #             shadow_map = torch.nn.functional.interpolate(shadow_map, size=(new_height, new_width), mode='bilinear')
        #         else:
        #             new_height = side_length // 2
        #             new_width = side_length

        #         shadow_map = shadow_map.squeeze(0).permute(1, 2, 0)  # Changes from (N, C, H, W) -> (H, W, C)
        #         outputs['rgb'][:new_height, new_width:, :] = shadow_map.squeeze(0)

        return outputs

    def fit_latent_codes_for_eval(self, datamanager, gt_source, epochs, learning_rate, step):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""

        # Make sure we are using eval RENI
        self.fitting_eval_latents = True

        # get the correct illumination field
        illumination_field = self.get_illumination_field()

        # Reset latents to zeros for fitting
        illumination_field.reset_eval_latents()

        opt = torch.optim.Adam(illumination_field.parameters(), lr=learning_rate)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[extra]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising eval latents... ", total=epochs, extra="")

            # Fit latents
            for _ in range(epochs):
                epoch_loss = 0.0
                for step in range(len(datamanager.eval_dataset)):
                    # Lots of admin to get the data in the right format depending on task
                    idx, ray_bundle, batch = datamanager.next_eval_image(step)
                    mask = batch['mask']

                    if gt_source == "envmap":
                        raise NotImplementedError
                    elif gt_source in ["image_half", "image_full"]:
                        divisor = 2 if gt_source == "image_half" else 1

                        rgb = batch["image"].to(self.device)  # [H, W, 3]
                        rgb = rgb[:, : rgb.shape[1] // divisor, :]  # [H, W//divisor, 3]
                        rgb = rgb.reshape(-1, 3)  # [H*W, 3]

                        # Use with the left half of the image or the full image, depending on divisor
                        ray_bundle = ray_bundle[:, : ray_bundle.shape[1] // divisor]

                        ray_bundle = ray_bundle.get_row_major_sliced_ray_bundle(
                            0, len(ray_bundle)
                        )  # [H * W//divisor, N]

                        if "mask" in batch:
                            mask = mask.to(self.device)  # [H, W, 3]
                            mask = mask[:, : mask.shape[1] // divisor, :].unsqueeze(-1)  # [H, W//divisor, 3]
                            mask = mask.reshape(-1, 3)  # [H*W, 3]
                            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                            chosen_indices = random.sample(range(len(nonzero_indices)), k=256)
                            indices = nonzero_indices[chosen_indices].squeeze()
                        else:
                            # Sample N rays and build a new ray_bundle
                            indices = random.sample(range(len(ray_bundle)), k=256)

                        ray_bundle = ray_bundle[indices]  # [N]

                        # Get GT RGB values for the sampled rays
                        rgb = rgb[indices, :]  # [N, 3]
                    
                    batch['image'] = rgb
                    batch['mask'] = mask
                    # Get model output
                    if gt_source == "envmap":
                        raise NotImplementedError
                    else:
                        model_output = self.forward(ray_bundle=ray_bundle, step=step)
                        # model_output = outputs["rgb"]  # [N, 3]

                    opt.zero_grad()
                    if gt_source in ["envmap", "image_half_sky"]:
                        raise NotImplementedError
                        # loss, _, _, _ = reni_test_loss(model_output, rgb, S, Z)
                    else:
                        loss_dict = self.get_loss_dict(model_output, batch)
                        loss = functools.reduce(torch.add, loss_dict.values())
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()

                progress.update(task, advance=1, extra=f"{epoch_loss:.4f}")

        # No longer using eval RENI
        self.fitting_eval_latents = False

    def setup_gui(self):
        """Setup the GUI."""
        self.viewer_control = ViewerControl()  # no arguments

        self.render_rgb_flag = True
        self.render_rgb_static_flag = True
        self.render_accumulation_flag = True
        self.render_accumulation_static_flag = True
        self.render_depth_flag = True
        self.render_depth_static_flag = True
        self.render_normal_flag = True
        self.render_normal_static_flag = True
        self.render_albedo_flag = True
        self.render_albedo_static_flag = True
        self.render_shadow_envmap_overlay_flag = False
        self.render_shadow_envmap_overlay_static_flag = False
        self.shadow_envmap_overlay_pos = [0,0,0]
        self.shadow_envmap_overlay_pos_static = None
        self.shadow_envmap_overlay_dir = [0,1,0]
        self.shadow_envmap_overlay_dir_static = None

        def click_cb(click: ViewerClick):
            self.shadow_envmap_overlay_pos = click.origin
            self.shadow_envmap_overlay_dir = click.direction

        self.viewer_control.register_click_cb(click_cb)

        self.render_shadow_map_flag = False
        self.render_shadow_map_static_flag = False
        self.shadow_map_threshold = ViewerSlider(name="Shadowmap Threshold", default_value=1.0, min_value=0.0, max_value=2.0)
        self.shadow_map_azimuth = ViewerSlider(name="Shadow Map Azimuth", default_value=0.0, min_value=-180.0, max_value=180.0)
        self.shadow_map_elevation = ViewerSlider(name="Shadow Map Elevation", default_value=0.0, min_value=-90.0, max_value=90.0)
        self.shadow_map_threshold_static = 0.5
        self.shadow_map_azimuth_static = 0.5
        self.shadow_map_elevation_static = 0.5

        self.accumulation_mask_threshold = ViewerSlider(name="Accumulation Mask Threshold", default_value=0.0, min_value=0.0, max_value=1.0)
        self.accumulation_mask_threshold_static = 0.0

        def render_rgb_callback(handle: ViewerCheckbox) -> None:
            self.render_rgb_flag = handle.value

        self.render_rgb_checkbox = ViewerCheckbox(name="Render RGB",
                                                     default_value=True,
                                                     cb_hook=render_rgb_callback)
        
        def render_accumulation_callback(handle: ViewerCheckbox) -> None:
            self.render_accumulation_flag = handle.value
        
        self.render_accumulation_checkbox = ViewerCheckbox(name="Render Accumulation",
                                                     default_value=True,
                                                     cb_hook=render_accumulation_callback)
        
        if self.visibility_field is None:
            def render_depth_callback(handle: ViewerCheckbox) -> None:
                self.render_depth_flag = handle.value

            self.render_depth_checkbox = ViewerCheckbox(name="Render Depth",
                                                        default_value=True,
                                                        cb_hook=render_depth_callback)
        
        def render_normal_callback(handle: ViewerCheckbox) -> None:
            self.render_normal_flag = handle.value

        self.render_normal_checkbox = ViewerCheckbox(name="Render Normal",
                                                    default_value=True,
                                                    cb_hook=render_normal_callback)
        
      
        def render_albedo_callback(handle: ViewerCheckbox) -> None:
            self.render_albedo_flag = handle.value
        
        self.render_albedo_checkbox = ViewerCheckbox(name="Render Albedo",
                                                     default_value=True,
                                                     cb_hook=render_albedo_callback)
        

        def render_shadow_map_callback(handle: ViewerCheckbox) -> None:
            self.render_shadow_map_flag = handle.value

        self.render_shadow_map_checkbox = ViewerCheckbox(name="Render Shadow Map",
                                                         default_value=False,
                                                         cb_hook=render_shadow_map_callback)
        
        def render_shadow_envmap_overlay_callback(handle: ViewerCheckbox) -> None:
            self.render_shadow_envmap_overlay_flag = handle.value

        self.render_shadow_envmap_overlay_checkbox = ViewerCheckbox(name="Render Shadow Envmap Overlay",
                                                          default_value=False,
                                                          cb_hook=render_shadow_envmap_overlay_callback)
        
        def on_sphere_look_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(0, 1, 0), look_at=(0,0,0), instant=False)
        
        self.viewer_button = ViewerButton(name="Camera on DDF",cb_hook=on_sphere_look_at_origin)


    def ray_sphere_intersection(self, positions, directions, radius):
        """Ray sphere intersection"""
        # ray-sphere intersection
        # positions is the origins of the rays
        # directions is the directions of the rays
        # radius is the radius of the sphere

        sphere_origin = torch.zeros_like(positions)
        radius = torch.ones_like(positions[..., 0]) * radius

        # ensure normalized directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
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

    def compute_visibility(self, ray_samples, depth, illumination_directions, threshold_distance, accumulation, batch, compute_shadow_map=False):
        """Compute visibility"""
        # ray_samples = [num_rays, num_samples]
        # depth = [num_rays, 1]
        # illumination_directions = [num_rays * num_samples, num_light_directions, xyz]
        # threshold_distance = scalar
        # accumulation = None
        # batch = dict

        # shortcuts for later
        num_rays = ray_samples.frustums.origins.shape[0]
        num_samples = ray_samples.frustums.origins.shape[1]
        original_num_light_directions = illumination_directions.shape[1]

        # illumination_directions = [num_rays * num_samples, num_light_directions, xyz]
        # we only want [num_light_directions, xyz]
        illumination_directions = illumination_directions[0, :, :]

        if self.config.only_upperhemisphere_visibility and not compute_shadow_map:
            # we dot product illumination_directions with the vertical z axis
            # and use it as mask to select only the upper hemisphere
            dot_products = torch.sum(torch.tensor([0, 0, 1]).type_as(illumination_directions) * illumination_directions, dim=1)
            mask = dot_products > 0
            directions = illumination_directions[mask] # [num_light_directions, xyz]
        else:
            directions = illumination_directions
        
        # more shortcuts
        num_light_directions = directions.shape[0]

        # since we are only using a single sample, the sample we think has hit the object,
        # we can just use one of each of these values, they are all just copied for each
        # sample along the ray. So here I'm just taking the first one.
        origins = ray_samples.frustums.origins[:, 0:1, :]  # [num_rays, 1, 3]
        ray_directions = ray_samples.frustums.directions[:, 0:1, :]  # [num_rays, 1, 3]

        # Now calculate the new positions
        positions = origins + ray_directions * depth.unsqueeze(-1) # [num_rays, 1, 3]

        # check if the new pos are within the sphere
        inside_sphere = positions.norm(dim=-1) < self.ddf_radius # [num_rays, 1]

        # for all positions not inside sphere get sphere intersection points for origins and directions
        # and just subract a small amount from the positions to make sure they are inside the sphere,
        # bit of a hack as we need a sphere intersection point for each light direction
        positions[~inside_sphere] = self.ray_sphere_intersection(origins[~inside_sphere], ray_directions[~inside_sphere], self.ddf_radius) * 0.01 * -ray_directions[~inside_sphere]

        positions = positions.unsqueeze(1).repeat(
            1, num_light_directions, 1, 1
        )  # [num_rays, num_light_directions, 1, 3]
        directions = (
            directions.unsqueeze(0).repeat(num_rays, 1, 1)
        )  # [num_rays, num_light_directions, 1, 3]

        positions = positions.reshape(-1, 3)  # [num_rays * num_light_directions, 3]
        directions = directions.reshape(-1, 3)  # [num_rays * num_light_directions, 3]

        sphere_intersection_points = self.ray_sphere_intersection(positions, directions, self.ddf_radius) # [num_rays * num_light_directions, 3]

        termination_dist = torch.norm(sphere_intersection_points - positions, dim=-1) # [num_rays * num_light_directions]

        # we need directions from intersection points to ray origins
        directions = -directions

        # build a ray_bundle object to pass to the visibility_field
        visibility_ray_bundle = RayBundle(
            origins=sphere_intersection_points,
            directions=directions,
            pixel_area=torch.ones_like(positions[..., 0]), # not used but required for class
        )

        # Get output of visibility field (DDF)
        outputs = self.visibility_field(visibility_ray_bundle, batch=None, reni_neus=self) # [N, 2]

        # the ground truth distance from the point on the sphere to the point on the SDF
        dist_to_ray_origins = torch.norm(positions - sphere_intersection_points, dim=-1) # [N]

        # as the DDF can only predict 2*its radius, we need to clamp gt to that
        dist_to_ray_origins = torch.clamp(dist_to_ray_origins, max=self.ddf_radius * 2.0)

        # add threshold_distance extra to the expected_termination_dist (i.e slighly futher into the SDF)
        # and get the difference between it and the distance from the point on the sphere to the point on the SDF
        difference = (outputs['expected_termination_dist'] + threshold_distance) - dist_to_ray_origins

        # if the difference is positive then the expected termination distance
        # is greater than the distance from the point on the sphere to the point on the SDF
        # so the point on the sphere is visible to it
        if compute_shadow_map:
            visibility = (difference > 0).float()
        else:
            # Use a large scale to make the transition steep
            scale = 50.0
            # Adjust the bias to control the point at which the transition happens
            bias = 0.0
            visibility = torch.sigmoid(scale * (difference - bias))

        if self.config.only_upperhemisphere_visibility and not compute_shadow_map:
            # we now need to use the mask we created earlier to select only the upper hemisphere
            # and then use the predicted visibility values there
            total_vis = torch.ones(num_rays, original_num_light_directions, 1).type_as(visibility)
            # reshape mask to match the total_vis dimensions
            mask = mask.reshape(1, original_num_light_directions, 1).expand_as(total_vis)
            # use the mask to replace the values
            total_vis[mask] = visibility
            visibility = total_vis

        visibility = visibility.unsqueeze(1).repeat(1, num_samples, 1, 1) # [num_rays, num_samples, num_light_directions, 1]
        # and reshape so that it is [num_rays * num_samples, original_num_light_directions, 1]
        visibility = visibility.reshape(-1, original_num_light_directions, 1)

        visibility_dict = outputs
        visibility_dict["visibility"] = visibility
        visibility_dict["expected_termination_dist"] = outputs["expected_termination_dist"]
        if compute_shadow_map:
            visibility_dict["difference"] = difference

        visibility_dict["visibility_batch"] = {
            "termination_dist": termination_dist,
            "mask": torch.ones_like(termination_dist),
        }

        return visibility_dict
        
    def render_illumination_animation(self, ray_bundle: RayBundle, batch: Dict, num_frames: int, fps: float, visibility_threshold: float, output_path: Path, render_final_animation: bool, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Render an animation rotating the illumination field around the scene."""
        temp_visibility_threshold = self.config.visibility_threshold
        self.visibility_threshold = visibility_threshold

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = num_frames

        project_root = find_nerfstudio_project_root()

        path = project_root / output_path / 'render_frames'
        # Creating a directory to save the intermediate .pt files
        if not path.exists():
            path.mkdir(parents=True)
        
        saved_data = []
        # Check if the render_sequence.pt file already exists
        render_sequence_path = path / 'render_sequence.pt'
        if render_sequence_path.exists():
            saved_data = torch.load(str(render_sequence_path))
        else:
          for i in range(start_frame, end_frame):
              angle = i * (360 / num_frames)  # angle in degrees
              rotation = rot_z(torch.tensor(np.deg2rad(angle)).float()).to(self.device)

              pt_file_path = path / f'frame_{i}.pt'

              if pt_file_path.exists():
                  # Load already computed frame
                  frame_data = torch.load(str(pt_file_path))
                  rgb = frame_data["rgb"]
              else:
                  print(f"Rendering frame {i}/{num_frames}")
                  outputs = self.get_outputs_for_camera_ray_bundle(ray_bundle, show_progress=True, rotation=rotation)
                  rgb = outputs['rgb']
                  # Saving the outputs and envmap to .pt file for each frame
                  torch.save({"rgb": rgb}, str(pt_file_path))

              # Storing the data in memory for final animation
              saved_data.append(rgb.detach().cpu().numpy())

        if render_final_animation:
            # Save entire sequence to a .pt file
            torch.save(saved_data, str(render_sequence_path))

            # Create the animation
            rgb_images = []

            for rgb in saved_data:
                # ensure no nan or inf values
                rgb = np.nan_to_num(rgb)
                rgb_images.append(rgb)

            # Assuming rgb_images are in range [0, 1] and have shape (height, width, channels)
            rgb_images = np.array(rgb_images)  # convert list to numpy array
            rgb_images = (rgb_images * 255).astype(np.uint8)  # scale to [0, 255] and convert to uint8

            height, width, channels = rgb_images[0].shape

            # Define the codec using VideoWriter_fourcc and creat7e a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_output_path = path / 'rgb_animation.mp4'
            video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

            for frame in rgb_images:
                # OpenCV uses BGR format, so we need to convert RGB to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()

        self.visibility_threshold = temp_visibility_threshold