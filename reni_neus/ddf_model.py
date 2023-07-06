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
Implementation of mip-NeRF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union, Literal
from pathlib import Path
import yaml
from torch.utils.data import Dataset
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
from collections import defaultdict

import torch
from torch.nn import Parameter
from torch import nn
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

from reni_neus.fields.directional_distance_field import DirectionalDistanceField, DirectionalDistanceFieldConfig
from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions, ray_sphere_intersection, log_loss
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames

CONSOLE = Console(width=120)


@dataclass
class DDFModelConfig(ModelConfig):
    """DDF Model Config"""

    _target: Type = field(default_factory=lambda: DDFModel)
    ddf_field: DirectionalDistanceFieldConfig = DirectionalDistanceFieldConfig()
    """DDF field configuration"""
    sdf_loss: Literal["L1", "L2"] = "L2"
    """SDF loss type"""
    depth_loss: Literal["L1", "L2", "Log_Loss", "Inverse_Scaled_L2"] = "L2"
    """Depth loss type"""
    sdf_loss_mult: float = 1.0
    """Multiplier for the sdf loss"""
    depth_loss_mult: float = 1.0
    """Multiplier for the depth loss"""
    prob_hit_loss_mult: float = 1.0
    """Multiplier for the probability of hit loss"""
    normal_loss_mult: float = 1.0
    """Multiplier for the normal loss"""
    compute_normals: bool = False
    """Whether to compute normals"""
    include_multi_view_loss: bool = False
    """Whether to include multi-view loss"""
    multi_view_loss_mult: float = 1.0
    """Multiplier for the multi-view loss"""
    multi_view_loss_stop_gradient: bool = False
    """Whether to stop gradient for the multi-view loss"""
    include_sky_ray_loss: bool = False
    """Whether to include sky ray loss"""
    sky_ray_loss_mult: float = 1.0
    """Multiplier for the sky ray loss"""
    include_sdf_loss: bool = True # perhaps make all losses a union of literals with none as option
    """Whether to include sdf loss"""
    include_depth_loss_scene_center_weight: bool = False
    """Whether to include depth loss scene center weight"""
    scene_center_weight_exp: float = 1.0
    """Exponent for the scene center weight"""
    scene_center_use_xyz: bool = False
    """Whether to use xyz or xy for the scene center weight"""
    mask_depth_to_circumference: bool = False
    """Whether to set depth under mask to the circumference"""

class DDFModel(Model):
    """Directional Distance Field model

    Args:
        config: DDFModelConfig configuration to instantiate model
    """

    config: DDFModelConfig

    def __init__(self, config: DDFModelConfig, ddf_radius, **kwargs) -> None:
        self.ddf_radius = ddf_radius
        super().__init__(config=config, **kwargs)
        self.viewer_control = ViewerControl()  # no arguments

        def on_sphere_look_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(0, 1, 0), look_at=(0,0,0), instant=False)
        
        self.viewer_button = ViewerButton(name="Camera on DDF",cb_hook=on_sphere_look_at_origin)

    def populate_modules(self):
        """Set the fields and modules"""

        self.collider = SphereCollider(center=torch.tensor([0.0, 0.0, 0.0]), radius=self.ddf_radius)
        # self.collider = None

        self.field = self.config.ddf_field.setup(ddf_radius=self.ddf_radius)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.sdf_loss = MSELoss() if self.config.sdf_loss == "L2" else nn.L1Loss()
        
        if self.config.depth_loss == "L2":
            self.depth_loss = nn.MSELoss()
        elif self.config.depth_loss == "L1":
            self.depth_loss = nn.L1Loss()
        elif self.config.depth_loss == "Log_Loss":
            self.depth_loss = log_loss

        self.normal_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.probability_loss = torch.nn.BCELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["ddf_field"] = list(self.field.parameters())
        return param_groups

    def get_localised_transforms(self, positions):
        """Computes the local coordinate system for each point in the input positions"""
        up_vector = torch.tensor([0, 0, 1]).type_as(positions)  # Assuming world up-vector is along z-axis as is the case in nerfstudio
        up_vector = up_vector.expand_as(positions)  # Expand to match the shape of positions

        positions = -positions # negate to ensure [0, 1, 0] direction is facing origin

        # Calculate the cross product between the position vector and the world up-vector to obtain the x-axis of the local coordinate system
        x_local = torch.cross(up_vector, positions)
        x_local = x_local / x_local.norm(dim=-1, keepdim=True)  # Normalize

        # Compute the local z-axis by crossing position and x_local
        z_local = torch.cross(positions, x_local)
        z_local = z_local / z_local.norm(dim=-1, keepdim=True)  # Normalize

        # The y-axis is the position itself
        y_local = positions

        # Stack the local basis vectors to form the rotation matrices
        rotation_matrices = torch.stack((x_local, y_local, z_local), dim=-1)

        return rotation_matrices


    def get_outputs(self, ray_bundle: RayBundle, batch, reni_neus):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # get H, W from ray_bundle if it's shape is (H, W, 3) and not (N, 3)
        H, W = None, None
        if len(ray_bundle.origins.shape) in [3, 4]:
            H, W = ray_bundle.origins.shape[:2]

        positions = ray_bundle.origins.reshape(-1, 3) # (N, 3)
        directions = ray_bundle.directions.reshape(-1, 3) # (N, 3)

        rotation_matrices = self.get_localised_transforms(positions) # (N, 3, 3)

        transformed_directions = torch.einsum('ijl,ij->il', rotation_matrices, directions) # (N, 3)

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=transformed_directions,
                starts=torch.zeros_like(positions),
                ends=torch.zeros_like(positions),
                pixel_area=torch.ones_like(positions[..., 0]),
            ),
        )

        outputs = {}

        if self.config.compute_normals:
            ray_samples.frustums.origins.requires_grad = True

        field_outputs = self.field.forward(ray_samples)
        expected_termination_dist = field_outputs[RENINeuSFieldHeadNames.TERMINATION_DISTANCE]
        outputs["expected_termination_dist"] = expected_termination_dist

        if RENINeuSFieldHeadNames.PROBABILITY_OF_HIT in field_outputs:
            outputs["expected_probability_of_hit"] = field_outputs[RENINeuSFieldHeadNames.PROBABILITY_OF_HIT]

        if self.config.include_depth_loss_scene_center_weight and batch is not None:
            gt_termination_points = positions + directions * batch["termination_dist"].repeat(1, 3)
            
            if self.config.scene_center_use_xyz:
                # use XYZ
                distance_from_center = torch.norm(positions, dim=-1)
            else:
                # use only the XY plane, ignoring the Z coordinate
                distance_from_center = torch.norm(positions[..., :2], dim=-1)

            # normalize to [0, 1]
            distance_from_center = distance_from_center / self.ddf_radius
            # invert so that points closer to the center have higher weight
            distance_weight = 1.0 - distance_from_center**self.config.scene_center_weight_exp
            outputs['distance_weight'] = distance_weight

        # get sdf at expected termination distance for loss
        if self.config.include_sdf_loss and batch is not None:
          if reni_neus is not None:
              termination_points = (
                  positions + directions * expected_termination_dist.unsqueeze(-1)
              )
              sdf_at_termination = reni_neus.field.get_sdf_at_pos(termination_points)
              outputs['sdf_at_termination'] = sdf_at_termination
          elif batch is not None and 'sdf_at_termination' in batch:
              sdf_at_termination = batch['sdf_at_termination']
              outputs['sdf_at_termination'] = sdf_at_termination


        # # Compute the gradient of the depths with respect to the ray origins
        if self.config.compute_normals:
            d_output = torch.ones_like(expected_termination_dist, requires_grad=False, device=expected_termination_dist.device)
            gradients = torch.autograd.grad(
                outputs=expected_termination_dist, inputs=ray_samples.frustums.origins, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            # Normalize the gradient to obtain the predicted normal
            n_hat = gradients / torch.norm(gradients, dim=-1, keepdim=True)

            # Choose the sign of n_hat such that n_hat * direction < 0
            varsigma = torch.sign(-torch.sum(n_hat * ray_samples.frustums.directions, dim=-1, keepdim=True))
            n_hat = varsigma * n_hat

            outputs["predicted_normals"] = n_hat

        if self.config.include_multi_view_loss and self.training and batch is not None:
            # for every gt termination point we choose a random other position on the sphere
            # we the the ddf to predict the termination distance from the random
            # point to the termination point. This distance should be no greater than the distance
            # from the random point to the gt termination point.

            # get gt_termination_points using gt_termination_dist
            gt_termination_points = positions + directions * batch["termination_dist"].repeat(1, 3)

            # for every termination point we choose a random other position on the sphere
            points_on_sphere = random_points_on_unit_sphere(num_points=gt_termination_points.shape[0]).type_as(gt_termination_points)

            # get directions from points_on_sphere to termination_points
            direction_to_term_points = gt_termination_points - points_on_sphere

            # distance is the norm of the direction vector (this will be used in loss)
            distance_to_term_points = torch.norm(direction_to_term_points, dim=-1)

            # normalize the direction vector
            direction_to_term_points = direction_to_term_points / distance_to_term_points.unsqueeze(-1)

            # normalise directions such that [0, 1, 0] is facing the origin
            rotation_matrices = self.get_localised_transforms(points_on_sphere)
            transformed_directions = torch.einsum('ijl,ij->il', rotation_matrices, direction_to_term_points)

            ray_samples = RaySamples(
                frustums=Frustums(
                    origins=points_on_sphere,
                    directions=transformed_directions,
                    starts=torch.zeros_like(points_on_sphere),
                    ends=torch.zeros_like(points_on_sphere),
                    pixel_area=torch.ones_like(points_on_sphere[:, 0]),
                ),
            )
        
            field_outputs = self.field.forward(ray_samples)

            outputs["multi_view_termintation_dist"] = batch["termination_dist"]
            outputs["multi_view_expected_termination_dist"] = field_outputs[RENINeuSFieldHeadNames.TERMINATION_DISTANCE]

        if self.config.include_sky_ray_loss and self.training and batch is not None:
            # all rays that go from cameras into the sky don't intersect the scene
            # so we know that in the opposite direction the DDF should predict the
            # distance from the DDF sphere to the camera origin, this is a ground truth
            # distance that we can use to train the DDF

            # first get the sky rays
            sky_ray_bundle = batch["sky_ray_bundle"]

            camera_origins = sky_ray_bundle.origins.reshape(-1, 3)
            camera_directions = sky_ray_bundle.directions.reshape(-1, 3)

            # we need the intersection points of the sky rays with the DDF sphere
            points_on_sphere = ray_sphere_intersection(positions=camera_origins, directions=camera_directions, radius=self.ddf_radius)

            # we need the ground truth distance from the camera origin to the intersection point
            # this is the distance that the DDF should predict
            distance_to_camera_origins = torch.norm(camera_origins - points_on_sphere, dim=-1)

            # reverse directions (Origins to Sky -> DDF to Origin) and transform such that [0, 1, 0] is facing the origin
            rotation_matrices = self.get_localised_transforms(points_on_sphere)
            transformed_directions = torch.einsum('ijl,ij->il', rotation_matrices, -camera_directions)

            ray_samples = RaySamples(
                frustums=Frustums(
                    origins=points_on_sphere,
                    directions=transformed_directions,
                    starts=torch.zeros_like(points_on_sphere),
                    ends=torch.zeros_like(points_on_sphere),
                    pixel_area=torch.ones_like(points_on_sphere[:, 0]),
                ),
            )
        
            field_outputs = self.field.forward(ray_samples)

            outputs["sky_ray_termination_dist"] = distance_to_camera_origins
            outputs["sky_ray_expected_termination_dist"] = field_outputs[RENINeuSFieldHeadNames.TERMINATION_DISTANCE]


        if H is not None and W is not None:
            for key, value in outputs.items():
                outputs[key] = value.reshape(H, W, 1, -1)

        return outputs
    
    def forward(self, ray_bundle: RayBundle, batch, reni_neus) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        # if self.collider is not None:
        #     ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, batch, reni_neus)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        
        # the sdf value at the predicted termination distance
        # should be zero
        loss_dict = {}

        if 'sdf_at_termination' in outputs:
            sdf_loss = self.sdf_loss(
                outputs["sdf_at_termination"] * batch["mask"],
                torch.zeros_like(outputs["sdf_at_termination"]) * batch["mask"],
            )
            loss_dict["sdf_loss"] = sdf_loss * self.config.sdf_loss_mult

        # match the depth of the sdf model
        if 'expected_termination_dist' in outputs:
            if self.config.include_depth_loss_scene_center_weight:
                mask = batch["mask"] * outputs['distance_weight']
            else:
                mask = batch["mask"]
            
            depth_loss = self.depth_loss(
                outputs["expected_termination_dist"].unsqueeze(1) * mask,
                batch["termination_dist"] * mask,
            )

            loss_dict["depth_loss"] = depth_loss * self.config.depth_loss_mult

        if 'expected_probability_of_hit' in outputs:
            # this should be matching the mask and use binary cross entropy
            probability_loss = self.probability_loss(
                outputs["expected_probability_of_hit"],
                batch["mask"].squeeze(-1),
            )
            loss_dict["probability_loss"] = probability_loss * self.config.prob_hit_loss_mult
                
        if 'predicted_normals' in outputs:
            normal_loss = self.normal_loss(
                outputs["predicted_normals"] * batch["mask"].unsqueeze(-1),
                batch["normals"] * batch["mask"].unsqueeze(-1),
            ).sum()
            loss_dict["normal_loss"] = normal_loss * self.config.normal_loss_mult

        if 'multi_view_termintation_dist' in outputs:
            # multi_view_expected_termination_dist must be less than multi_view_termintation_dist
            # so penalise anything over
            multi_view_loss = torch.zeros_like(outputs["multi_view_expected_termination_dist"])
            multi_view_loss = torch.max(multi_view_loss, outputs["multi_view_expected_termination_dist"] - outputs["multi_view_termintation_dist"])
            loss_dict["multi_view_loss"] = torch.mean(multi_view_loss) * self.config.multi_view_loss_mult

        if 'sky_ray_termination_dist' in outputs:
            sky_ray_loss = self.depth_loss(
                outputs["sky_ray_expected_termination_dist"],
                outputs["sky_ray_termination_dist"],
            ) * self.config.sky_ray_loss_mult

            loss_dict["sky_ray_loss"] = sky_ray_loss

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, reni_neus=None, show_progress=False
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)

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
                    if self.config.compute_normals:
                        with torch.enable_grad():
                            outputs = self.forward(ray_bundle=ray_bundle, batch=None, reni_neus=reni_neus)
                    else:
                        outputs = self.forward(ray_bundle=ray_bundle, batch=None, reni_neus=reni_neus)
                    # move to cpu
                    outputs = {k: v.cpu() for k, v in outputs.items()}
                    for output_name, output in outputs.items():  # type: ignore
                        if not torch.is_tensor(output):
                            # TODO: handle lists of tensors as well
                            continue
                        outputs_lists[output_name].append(output)
                    progress.update(task, completed=i)
        else:
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle, batch=None, reni_neus=reni_neus)
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1).to(dtype=torch.float32)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict = {}
        images_dict = {}

        gt_accumulations = batch["accumulations"]
        gt_termination_dist = batch["termination_dist"]
        expected_termination_dist = outputs["expected_termination_dist"]

        # ensure gt is on the same device as the model
        gt_accumulations = gt_accumulations.to(expected_termination_dist.device)
        gt_termination_dist = gt_termination_dist.to(expected_termination_dist.device)

        gt_depth = colormaps.apply_depth_colormap(
            gt_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
            colormap_options=ColormapOptions(normalize=False,
                                             colormap_min=0.0,
                                             colormap_max=2.0)
        )

        depth = colormaps.apply_depth_colormap(
            expected_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
            colormap_options=ColormapOptions(normalize=False,
                                             colormap_min=0.0,
                                             colormap_max=2.0)
        )

        combined_depth = torch.cat([gt_depth, depth], dim=1)
        images_dict["depth"] = combined_depth

        depth_error = torch.abs(gt_termination_dist * batch['mask'].type_as(gt_termination_dist) - expected_termination_dist * batch['mask'].type_as(expected_termination_dist))
        depth_error_normalized = (depth_error - torch.min(depth_error)) / (torch.max(depth_error) - torch.min(depth_error))
        images_dict["depth_error"] = depth_error_normalized

        if "expected_probability_of_hit" in outputs:
            expected_probability_of_hit = outputs["expected_probability_of_hit"]

            combined_probability_of_hit = torch.cat([gt_accumulations, expected_probability_of_hit], dim=1)

            images_dict["probability_of_hit"] = combined_probability_of_hit


        if 'predicted_normals' in outputs:
            normals = outputs["predicted_normals"]
            normals = (normals + 1.0) / 2.0

            gt_normal = batch["normals"].to(normals.device)
            gt_normal = (gt_normal + 1.0) / 2.0

            combined_normal = torch.cat([gt_normal, normals], dim=1)

            images_dict["normals"] = combined_normal

        return metrics_dict, images_dict
