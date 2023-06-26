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
import yaml
import os
import numpy as np
import cv2

import nerfacc
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils import colormaps

from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    ScaleAndShiftInvariantLoss,
    monosdf_normal_loss,
)

from nerfstudio.viewer.server.viewer_elements import *

from reni_neus.illumination_fields.base_illumination_field import IlluminationFieldConfig
from reni_neus.model_components.renderers import RGBLambertianRendererWithVisibility
from reni_neus.model_components.illumination_samplers import IlluminationSamplerConfig
from reni_neus.utils.utils import RENITestLossMask, get_directions, rotation_matrix
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames
from reni_neus.ddf_model import DDFModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler

CONSOLE = Console(width=120)


@dataclass
class RENINeuSFactoModelConfig(NeuSFactoModelConfig):
    """NeusFacto Model Config"""

    _target: Type = field(default_factory=lambda: RENINeuSFactoModel)
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
    render_only_albedo: bool = False # TODO remove for next full training run
    """Render only albedo"""
    include_occupancy_network: bool = False
    """Include occupancy network in the model"""
    occupancy_grid_resolution: int = 64
    """Resolution of the occupancy grid"""
    occupancy_grid_levels: int = 4
    """Levels of the occupancy grid"""
    include_hashgrid_density_loss: bool = False
    """Include hashgrid density loss"""
    hashgrid_density_loss_weight: float = 0.0
    """Weight for the hashgrid density loss"""
    hashgrid_density_loss_sample_resolution: int = 256
    """Resolution of the hashgrid density loss"""
    include_ground_plane_normal_alignment: bool = False
    """Align the ground plane normal to the z-axis"""
    ground_plane_normal_alignment_multi: float = 1.0
    """Weight for the ground plane normal alignment loss"""
    visibility_field: Union[DDFModelConfig, None] = None
    """Visibility field"""
    ddf_radius: Union[Literal["AABB"], float] = "AABB"
    """Radius of the DDF sphere"""
    visibility_threshold: Union[Literal["learnable"], float] = "learnable"
    """Learnable visibility threshold"""
    optimise_visibility: bool = False
    """Optimise visibility"""
    visibility_ckpt_path: Union[Path, None] = None
    """Path to visibility checkpoint"""
    visibility_ckpt_step: int = 0
    """Step of the visibility checkpoint"""
    only_upperhemisphere_visibility: bool = False
    """Lower hemisphere visibility will always be 1.0 if this is True"""
    fix_test_illumination_directions: bool = False
    """Fix the test illumination directions"""
    use_visibility: bool = False
    """Use visibility network output"""


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
        test_mode: str,
        **kwargs,
    ) -> None:
        self.num_val_data = num_val_data
        self.num_test_data = num_test_data
        self.test_mode = test_mode
        self.fitting_eval_latents = False
        self.rendering_animation = False
        super().__init__(config, scene_box, num_train_data, **kwargs)

        self.setup_gui()

        if self.config.ddf_radius == "AABB":
            self.ddf_radius = torch.abs(self.scene_box.aabb[0, 0]).item()
        else:
            self.ddf_radius = self.config.ddf_radius

        if self.config.visibility_field is not None:
            self.visibility_field = self._setup_visibility_field()

            if self.config.visibility_threshold == "learnable":
                self.visibility_threshold = Parameter(torch.tensor(1.0))
            else:
                self.visibility_threshold = torch.tensor(self.config.visibility_threshold)
        

    def populate_modules(self):
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        self.illumination_field_train = self.config.illumination_field.setup(num_latent_codes=self.num_train_data)
        self.illumination_field_val = self.config.illumination_field.setup(num_latent_codes=self.num_val_data)
        self.illumination_field_test = self.config.illumination_field.setup(num_latent_codes=self.num_test_data)

        self.illumination_sampler = self.config.illumination_sampler.setup()

        self.field_background = None

        if self.config.include_occupancy_network:
            # Occupancy Grid.
            self.occupancy_grid = nerfacc.OccGridEstimator(
                roi_aabb=self.scene_box.aabb,
                resolution=self.config.occupancy_grid_resolution,
                levels=self.config.occupancy_grid_levels,
            )
            # Volumetric sampler.
            self.volumetric_sampler = VolumetricSampler(
                occupancy_grid=self.occupancy_grid,
                density_fn=self.field.density_fn,
            )

        self.albedo_renderer = RGBRenderer(background_color=torch.tensor([1.0, 1.0, 1.0]))
        self.lambertian_renderer = RGBLambertianRendererWithVisibility()

        self.direct_illumination_loss = RENITestLossMask(
            alpha=self.config.illumination_field_prior_loss_weight,
            beta=self.config.illumination_field_cosine_loss_weight,
        )

        # l1 loss
        self.grid_density_loss = torch.nn.L1Loss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return a dictionary with the parameters of the proposal networks."""
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["illumination_field"] = list(self.illumination_field_train.parameters())
        if self.config.visibility_threshold == "learnable":
            param_groups["visibility_threshold"] = [self.visibility_threshold]
        if self.config.optimise_visibility:
            param_groups.update(self.visibility_field.get_param_groups())
        return param_groups

    def get_illumination_field(self):
        if self.training and not self.fitting_eval_latents:
            illumination_field = self.illumination_field_train
        else:
            illumination_field = (
                self.illumination_field_test if self.test_mode == "test" else self.illumination_field_val
            )

        return illumination_field
    
    def forward(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            batch: batch needed for DDF: masks, etc.
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, batch=batch, rotation=rotation)

    def sample_and_forward_field(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None) -> Dict[str, Any]:
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

        if not self.training and self.config.fix_test_illumination_directions:
            illumination_directions = self.illumination_sampler(apply_random_rotation=False)
        else:
            illumination_directions = self.illumination_sampler()

        illumination_directions = illumination_directions.to(self.device)

        # Get environment illumination for samples along the rays for each unique camera
        hdr_illumination_colours, illumination_directions = illumination_field(
            camera_indices=camera_indices,
            positions=None,
            directions=illumination_directions,
            rotation=rotation,
            illumination_type="illumination",
        )

        # Get LDR colour of the background for rays from the camera that don't hit the scene
        background_colours, _ = illumination_field(
            camera_indices=camera_indices,
            positions=None,
            directions=ray_samples.frustums.directions[:, 0, :],
            rotation=rotation,
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

        if self.config.visibility_field is not None:
            # we need depth to compute visibility so render it here instead of in get_outputs()
            p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            # the rendered depth is point-to-point distance and we should convert to depth
            depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            # we need accumulation so we can ignore samples that don't hit the scene
            accumulation = self.renderer_accumulation(weights=weights)

            if self.config.use_visibility:
                visibility_dict = self.compute_visibility(ray_samples=ray_samples,
                                                          p2p_dist=depth,
                                                          illumination_directions=illumination_directions,
                                                          threshold_distance=self.visibility_threshold,
                                                          accumulation=accumulation,
                                                          batch=batch)
                
                samples_and_field_outputs["visibility_dict"] = visibility_dict
            
            samples_and_field_outputs["p2p_dist"] = p2p_dist
            samples_and_field_outputs["depth"] = depth
            samples_and_field_outputs["accumulation"] = accumulation
            accumulation_mask = accumulation > self.accumulation_mask_threshold_static

            if self.render_shadow_map_static:
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
                                                     p2p_dist=p2p_dist,
                                                     illumination_directions=shadow_map_direction,
                                                     threshold_distance=self.shadow_map_threshold_static,
                                                     accumulation=accumulation,
                                                     batch=batch,
                                                     compute_shadow_map=True)
                
                # mask using accumulation_mask
                shadow_map['visibility'] = shadow_map['visibility'] * accumulation_mask.unsqueeze(1).type_as(shadow_map['visibility'])
                shadow_map['difference'] = shadow_map['difference'] * accumulation_mask.unsqueeze(1).type_as(shadow_map['difference'])
                
                samples_and_field_outputs["shadow_map"] = shadow_map
                

        if self.config.include_hashgrid_density_loss and self.training:
            # Get min and max coordinates
            min_coord, max_coord = self.scene_box.aabb

            # Create a linear space for each dimension
            x = torch.linspace(min_coord[0], max_coord[0], self.config.hashgrid_density_loss_sample_resolution)
            y = torch.linspace(min_coord[1], max_coord[1], self.config.hashgrid_density_loss_sample_resolution)
            z = torch.linspace(min_coord[2], max_coord[2], self.config.hashgrid_density_loss_sample_resolution)

            # Generate a 3D grid of points
            X, Y, Z = torch.meshgrid(x, y, z)
            positions = torch.stack((X, Y, Z), -1)  # size will be (resolution, resolution, resolution, 3)

            # Flatten and reshape
            positions = positions.reshape(-1, 3)

            # Calculate gaps between each sample
            gap = torch.tensor([(max_coord[i] - min_coord[i]) / self.config.hashgrid_density_loss_sample_resolution for i in range(3)])

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

    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None, rotation: Union[torch.Tensor, None]= None) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle, batch=batch, rotation=rotation)

        # shortcuts
        field_outputs = samples_and_field_outputs["field_outputs"]

        weights = samples_and_field_outputs["weights"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        illumination_directions = samples_and_field_outputs["illumination_directions"]
        hdr_illumination_colours = samples_and_field_outputs["hdr_illumination_colours"]
        background_colours = samples_and_field_outputs["background_colours"]

        visibility = None
        if 'visibility_dict' in samples_and_field_outputs:
            visibility = samples_and_field_outputs["visibility_dict"]["visibility"]
            
        if self.render_rgb_static:
            rgb = self.lambertian_renderer(
                albedos=field_outputs[RENINeuSFieldHeadNames.ALBEDO],
                normals=field_outputs[FieldHeadNames.NORMALS],
                light_directions=illumination_directions,
                light_colors=hdr_illumination_colours,
                visibility=visibility,
                background_illumination=background_colours,
                weights=weights,
            )
        else:
            rgb = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)

        if 'accumulation' in samples_and_field_outputs:
            accumulation = samples_and_field_outputs["accumulation"]
        else:
            if self.render_accumulation_static:
                accumulation = self.renderer_accumulation(weights=weights)
            else:
                accumulation = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)

        if 'p2p_dist' in samples_and_field_outputs:
            p2p_dist = samples_and_field_outputs["p2p_dist"]
            depth = samples_and_field_outputs["depth"]
        else:
            if self.render_depth_static:
                p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                # the rendered depth is point-to-point distance and we should convert to depth
                depth = p2p_dist / ray_bundle.metadata["directions_norm"]
            else:
                p2p_dist = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)
                depth = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)

        if self.render_normal_static:
            normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        else:
            normal = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)

        if self.render_albedo_static:
            albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)
        else:
            albedo = torch.zeros((ray_bundle.shape[0], 3)).to(self.device)

        if self.render_shadow_map_static:
            shadow_map = samples_and_field_outputs["shadow_map"]['visibility']
        else:
            shadow_map = torch.zeros((ray_bundle.shape[0], 1)).to(self.device)


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
            "shadow_map": shadow_map,
        }

        # if self.render_shadow_map_static:
        #     if 'difference' in samples_and_field_outputs["shadow_map"]:
        #         outputs["difference"] = samples_and_field_outputs["shadow_map"]['difference']

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

        if self.rendering_animation:
            outputs['render_albedos'] = field_outputs[RENINeuSFieldHeadNames.ALBEDO]
            outputs['render_normals'] = field_outputs[FieldHeadNames.NORMALS]
            outputs['render_visibility'] = visibility
            outputs['directions'] = ray_samples.frustums.directions[:, 0, :]
            outputs['camera_indices'] = ray_samples.camera_indices[0, 0]

        return outputs

    def get_loss_dict(
        self, outputs: Dict[str, Any], batch: Dict[str, Any], metrics_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the loss dictionary, including interlevel loss for proposal networks."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            if "fg_mask" in batch:
                fg_label = batch["fg_mask"].float()
                sky_label = 1 - fg_label
                loss_dict["illumination_loss"] = (
                    self.direct_illumination_loss(
                        inputs=outputs["background_colours"],
                        targets=batch["image"].type_as(sky_label),
                        mask=sky_label,
                        Z=self.illumination_field_train.get_latents(),
                    )
                    * self.config.illumination_field_loss_weight
                )

            if 'grid_density' in outputs:
                loss_dict['grid_density_loss'] = self.grid_density_loss(outputs['grid_density'], torch.zeros_like(outputs['grid_density'])) * self.config.hashgrid_density_loss_weight

            if self.config.include_ground_plane_normal_alignment:
                normal_pred = outputs["normal"]
                # ground plane should be facing up in z direction
                normal_gt = torch.tensor([0.0, 0.0, 1.0]).to(self.device).expand_as(normal_pred)
                loss_dict["ground_plane_alignment_loss"] = (
                    monosdf_normal_loss(normal_pred * batch["ground_mask"], normal_gt * batch["ground_mask"]) * self.config.ground_plane_normal_alignment_multi
                )

        if self.config.visibility_field is not None and self.config.use_visibility:
            # add the loss_dict from the visibility field
            loss_dict.update(self.visibility_field.get_loss_dict(outputs, outputs["visibility_batch"]))

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
        images_dict["background"] = outputs["background_colours"]

        with torch.no_grad():
            idx = torch.tensor(batch["image_idx"], device=self.device)
            W = 512
            H = W // 2
            D = get_directions(W).to(self.device)  # [B, H*W, 3]
            envmap, _ = illumination_field(idx, None, D, None, "envmap")
            envmap = envmap.reshape(1, H, W, 3).squeeze(0)
            images_dict["RENI"] = envmap

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, show_progress=False, rotation=None, to_cpu=False
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
        self.render_rgb_static = self.render_rgb
        self.render_accumulation_static = self.render_accumulation
        self.render_depth_static = self.render_depth
        self.render_normal_static = self.render_normal
        self.render_albedo_static = self.render_albedo
        self.render_shadow_map_static = self.render_shadow_map
        self.shadow_map_threshold_static = self.shadow_map_threshold.value
        self.shadow_map_azimuth_static = self.shadow_map_azimuth.value
        self.shadow_map_elevation_static = self.shadow_map_elevation.value
        self.accumulation_mask_threshold_static = self.accumulation_mask_threshold.value

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
                    outputs = self.forward(ray_bundle=ray_bundle, rotation=rotation)
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
        return outputs

    def fit_latent_codes_for_eval(self, datamanager, gt_source, epochs, learning_rate):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""

        # Make sure we are using eval RENI
        self.fitting_eval_latents = True

        # get the correct illumination field
        illumination_field = self.get_illumination_field()

        # Reset latents to zeros for fitting
        illumination_field.reset_latents()

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
                            mask = batch["mask"].to(self.device)  # [H, W]
                            mask = mask[:, : mask.shape[1] // divisor].unsqueeze(-1)  # [H, W//divisor]
                            mask = mask.reshape(-1, 1)  # [H*W, 1]
                            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                            chosen_indices = random.sample(range(len(nonzero_indices)), k=256)
                            indices = nonzero_indices[chosen_indices].squeeze()
                        else:
                            # Sample N rays and build a new ray_bundle
                            indices = random.sample(range(len(ray_bundle)), k=256)

                        ray_bundle = ray_bundle[indices]  # [N]

                        # Get GT RGB values for the sampled rays
                        rgb = rgb[indices, :]  # [N, 3]

                    # Get model output
                    if gt_source == "envmap":
                        raise NotImplementedError
                    else:
                        outputs = self.forward(ray_bundle=ray_bundle)
                        model_output = outputs["rgb"]  # [N, 3]

                    opt.zero_grad()
                    if gt_source in ["envmap", "image_half_sky"]:
                        raise NotImplementedError
                        # loss, _, _, _ = reni_test_loss(model_output, rgb, S, Z)
                    else:
                        loss = (
                            self.rgb_loss(rgb, model_output)
                            + self.config.illumination_field_prior_loss_weight
                            * torch.pow(illumination_field.get_latents(), 2).sum()
                        )
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()

                progress.update(task, advance=1, extra=f"{epoch_loss:.4f}")

        # No longer using eval RENI
        self.fitting_eval_latents = False

    def setup_gui(self):
        """Setup the GUI."""
        self.viewer_control = ViewerControl()  # no arguments

        self.render_rgb = True
        self.render_rgb_static = True
        self.render_accumulation = True
        self.render_accumulation_static = True
        self.render_depth = True
        self.render_depth_static = True
        self.render_normal = True
        self.render_normal_static = True
        self.render_albedo = True
        self.render_albedo_static = True

        self.render_shadow_map = True
        self.render_shadow_map_static = False
        self.shadow_map_threshold = ViewerSlider(name="Shadowmap Threshold", default_value=1.0, min_value=0.0, max_value=2.0)
        self.shadow_map_azimuth = ViewerSlider(name="Shadow Map Azimuth", default_value=0.0, min_value=-180.0, max_value=180.0)
        self.shadow_map_elevation = ViewerSlider(name="Shadow Map Elevation", default_value=0.0, min_value=-90.0, max_value=90.0)
        self.shadow_map_threshold_static = 0.5
        self.shadow_map_azimuth_static = 0.5
        self.shadow_map_elevation_static = 0.5

        self.accumulation_mask_threshold = ViewerSlider(name="Accumulation Mask Threshold", default_value=0.0, min_value=0.0, max_value=1.0)
        self.accumulation_mask_threshold_static = 0.0

        def render_rgb_callback(handle: ViewerCheckbox) -> None:
            self.render_rgb = handle.value

        self.render_rgb_checkbox = ViewerCheckbox(name="Render RGB",
                                                     default_value=True,
                                                     cb_hook=render_rgb_callback)
        
        def render_accumulation_callback(handle: ViewerCheckbox) -> None:
            self.render_accumulation = handle.value
        
        self.render_accumulation_checkbox = ViewerCheckbox(name="Render Accumulation",
                                                     default_value=True,
                                                     cb_hook=render_accumulation_callback)
        
        if self.config.visibility_field is None:
            def render_depth_callback(handle: ViewerCheckbox) -> None:
                self.render_depth = handle.value

            self.render_depth_checkbox = ViewerCheckbox(name="Render Depth",
                                                        default_value=True,
                                                        cb_hook=render_depth_callback)
        
        def render_normal_callback(handle: ViewerCheckbox) -> None:
            self.render_normal = handle.value

        self.render_normal_checkbox = ViewerCheckbox(name="Render Normal",
                                                    default_value=True,
                                                    cb_hook=render_normal_callback)
        
      
        def render_albedo_callback(handle: ViewerCheckbox) -> None:
            self.render_albedo = handle.value
        
        self.render_albedo_checkbox = ViewerCheckbox(name="Render Albedo",
                                                     default_value=True,
                                                     cb_hook=render_albedo_callback)
        

        def render_shadow_map_callback(handle: ViewerCheckbox) -> None:
            self.render_shadow_map = handle.value

        self.render_shadow_map_checkbox = ViewerCheckbox(name="Render Shadow Map",
                                                         default_value=True,
                                                         cb_hook=render_shadow_map_callback)
        
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

    def compute_visibility(self, ray_samples, p2p_dist, illumination_directions, threshold_distance, accumulation, batch, compute_shadow_map=False):
        """Compute visibility"""
        # ddf_model directional distance field model
        # positions is the origins of the rays from the surface of the object
        # directions is the directions of the rays from the surface of the object # [98304, 1212, 3] -> [number_of_rays * samples_per_ray, number_of_light_directions, xyz]
        # sphere_intersection_points is the point on the sphere that we intersected

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

        # # get positions based on p2p distance (expected termination depth)
        # # this is our sample on the surface of the SDF representing the scene
        # positions = origins + ray_directions * p2p_dist.unsqueeze(-1) # [num_rays, 1, 3]
        
        # Calculate the distance from each origin to the sphere's center
        dist_to_center = torch.norm(origins - torch.zeros_like(origins), dim=-1)

        # Check which rays would end up outside the sphere
        outside_sphere = dist_to_center + p2p_dist > self.ddf_radius

        # Set a small bias
        bias = 0.01

        # Adjust p2p_dist for those rays so they end up inside the sphere
        p2p_dist[outside_sphere] = self.ddf_radius - dist_to_center[outside_sphere] - bias

        # Now calculate the new positions
        positions = origins + ray_directions * p2p_dist.unsqueeze(-1) # [num_rays, 1, 3]

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
            origins=positions,
            directions=directions,
            pixel_area=torch.ones_like(positions[..., 0]), # not used but required for class
        )
        
        ddf_batch = {"termination_dist": termination_dist}
        
        if self.training and batch is not None:
            # we can use the fact that any rays that hit the sky we know
            fg_mask = batch['fg_mask'].detach().clone()
            sky_mask = (1.0 - fg_mask).bool().repeat(1, 3) # [num_sky_rays, 3]

            sky_origins = ray_samples.frustums.origins[:, 0, :][sky_mask].reshape(-1, 3) # [num_sky_rays, 3]
            sky_directions = ray_samples.frustums.directions[:, 0, :][sky_mask].reshape(-1, 3) # [num_sky_rays, 3]
            sky_pixel_ares = torch.ones_like(sky_origins[..., 0]).reshape(-1, 1) # [num_sky_rays, 1]

            sky_ray_bundle = RayBundle(
                origins=sky_origins,
                directions=sky_directions,
                pixel_area=sky_pixel_ares,
            )

            ddf_batch["sky_ray_bundle"] = sky_ray_bundle

        # Get output of visibility field (DDF)
        outputs = self.visibility_field(visibility_ray_bundle, batch=ddf_batch, reni_neus=self) # [N, 2]

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
        if compute_shadow_map:
            visibility_dict["difference"] = difference
        visibility_dict["visibility_batch"] = {
            "termination_dist": termination_dist,
            "mask": torch.ones_like(termination_dist),
        }
        
        return visibility_dict
    
    def _setup_visibility_field(self):
        # setting up visibility field
        if self.config.visibility_ckpt_path is None:
            return self.config.visibility_field.setup(scene_box=self.scene_box, num_train_data=self.num_train_data, ddf_radius=self.ddf_radius)
        else:
            ckpt_path = self.config.visibility_ckpt_path / "nerfstudio_models" / f"step-{self.config.visibility_ckpt_step:09d}.ckpt"
            ckpt = torch.load(str(ckpt_path))

            model_dict = {}
            for key in ckpt["pipeline"].keys():
                if key.startswith("_model."):
                    model_dict[key[7:]] = ckpt["pipeline"][key]

            # load yaml checkpoint config
            visibility_config = Path(self.config.visibility_ckpt_path) / "config.yml"
            visibility_config = yaml.load(visibility_config.open(), Loader=yaml.Loader)

            visibility_field = visibility_config.pipeline.model.setup(
                scene_box=self.scene_box,
                num_train_data=-1,
                ddf_radius=self.ddf_radius,
            )

            visibility_field.load_state_dict(model_dict)

            if not self.config.optimise_visibility:
              visibility_field.eval()

            visibility_field.to(self.device)

            return visibility_field
        
    def render_illumination_animation(self, ray_bundle, batch, num_frames, fps, visibility_threshold, output_path):
        """Render an animation rotating the illumination field around the scene."""
        temp_visibility_threshold = self.config.visibility_threshold
        self.visibility_threshold = visibility_threshold
        self.rendering_animation = False

        # there is some stuff we can reuse such as albedo and normals
        path = output_path + 'render_frames'
        # Creating a directory to save the intermediate .pt files
        if not os.path.exists(path):
            os.makedirs(path)
        
        saved_data = []
        # Check if the render_sequence.pt file already exists
        if os.path.exists(output_path + 'render_sequence.pt'):
            saved_data = torch.load(output_path + 'render_sequence.pt')
        else:
          #   with Progress(
          #     TextColumn("[progress.description]{task.description}"),
          #     BarColumn(),
          #     TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
          #     TimeRemainingColumn(),
          # ) as progress:
          # task = progress.add_task("[green]Rendering animation... ", total=num_frames, extra="")
          for i in range(num_frames):  # Wrap the loop with tqdm for progress bar
              angle = i * (360 / num_frames)  # angle in degrees
              rotation = rotation_matrix(axis=np.array([0, 1, 0]), angle=np.deg2rad(angle))  # RENI is Y-up

              pt_file_path = f'{path}/frame_{i}.pt'

              if os.path.exists(pt_file_path):
                  # Load already computed frame
                  frame_data = torch.load(pt_file_path)
                  rgb = frame_data["rgb"]
              else:
                  print(f"Rendering frame {i}/{num_frames}")
                  outputs = self.get_outputs_for_camera_ray_bundle(ray_bundle, show_progress=True, rotation=rotation)
                  rgb = outputs['rgb']
                  # Saving the outputs and envmap to .pt file for each frame
                  torch.save({"rgb": rgb}, pt_file_path)

              # Storing the data in memory for final animation
              saved_data.append(rgb.detach().cpu().numpy())

              # # Update the progress bar
              # progress.update(task, advance=1)

        # Save entire sequence to a .pt file
        torch.save(saved_data, output_path + 'render_sequence.pt')

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
        video = cv2.VideoWriter(output_path + 'rgb_animation.mp4', fourcc, fps, (width, height))

        for frame in rgb_images:
            # OpenCV uses BGR format, so we need to convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()

        self.visibility_threshold = temp_visibility_threshold
        self.rendering_animation = False