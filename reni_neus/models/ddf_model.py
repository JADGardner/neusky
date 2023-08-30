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
Model for Spherical Directional Distance Fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
from collections import defaultdict

import torch
from torch.nn import Parameter
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig
from reni_neus.utils.utils import random_points_on_unit_sphere, ray_sphere_intersection
from reni_neus.field_components.reni_neus_fieldheadnames import RENINeuSFieldHeadNames

CONSOLE = Console(width=120)


@dataclass
class DDFModelConfig(ModelConfig):
    """DDF Model Config"""

    _target: Type = field(default_factory=lambda: DDFModel)
    ddf_field: DirectionalDistanceFieldConfig = DirectionalDistanceFieldConfig()
    """DDF field configuration"""
    compute_normals: bool = False
    """Whether to compute normals"""
    include_depth_loss_scene_center_weight: bool = False
    """Whether to include depth loss scene center weight"""
    scene_center_weight_exp: float = 1.0
    """Exponent for the scene center weight"""
    scene_center_weight_include_z: bool = False
    """Whether to use xyz or xy for the scene center weight"""
    mask_to_circumference: bool = True
    """Whether to set depths outside of accumulation mask to the radius of the DDF sphere"""
    loss_inclusions: Dict[str, bool] = to_immutable_dict(
        {
            "depth_l1_loss": True,
            "depth_l2_loss": False,
            "sdf_l1_loss": True,
            "sdf_l2_loss": False,
            "prob_hit_loss": False,
            "normal_loss": False,
            "multi_view_loss": False,
            "sky_ray_loss": False,
        }
    )
    """Dictionary of loss inclusions"""


class DDFModel(Model):
    """Directional Distance Field model

    Args:
        config: DDFModelConfig configuration to instantiate model
    """

    config: DDFModelConfig

    def __init__(self, config: DDFModelConfig, ddf_radius, **kwargs) -> None:
        self.ddf_radius = ddf_radius
        self.viewer_control = ViewerControl()  # no arguments

        def on_sphere_look_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(0, 1, 0), look_at=(0, 0, 0), instant=False)

        self.viewer_button = ViewerButton(name="Camera on DDF", cb_hook=on_sphere_look_at_origin)

        super().__init__(config=config, **kwargs)

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
        if self.config.include_depth_loss_scene_center_weight:
            reduction = "none"
        else:
            reduction = "mean"
        if self.config.loss_inclusions["depth_l1_loss"]:
            self.depth_l1_loss = nn.L1Loss(reduction=reduction)
        if self.config.loss_inclusions["depth_l2_loss"]:
            self.depth_l2_loss = nn.MSELoss(reduction=reduction)
        if self.config.loss_inclusions["sdf_l1_loss"]:
            self.sdf_l1_loss = nn.L1Loss()
        if self.config.loss_inclusions["sdf_l2_loss"]:
            self.sdf_l2_loss = nn.MSELoss()
        if self.config.loss_inclusions["prob_hit_loss"]:
            self.prob_hit_loss = torch.nn.BCELoss()
        if self.config.loss_inclusions["normal_loss"]:
            self.normal_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.config.loss_inclusions["multi_view_loss"]:
            self.multi_view_loss = nn.MSELoss()
        if self.config.loss_inclusions["sky_ray_loss"]:
            self.sky_ray_loss = nn.L1Loss()

        # metrics
        self.psnr = PeakSignalNoiseRatio((0.0, self.ddf_radius))
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
        up_vector = torch.tensor([0, 0, 1]).type_as(
            positions
        )  # Assuming world up-vector is along z-axis as is the case in nerfstudio
        up_vector = up_vector.expand_as(positions)  # Expand to match the shape of positions

        positions = -positions  # negate to ensure [0, 1, 0] direction is facing origin

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

    def get_outputs(self, ray_bundle: RayBundle, batch, reni_neus, stop_gradients: bool = True):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # get H, W from ray_bundle if it's shape is (H, W, 3) and not (N, 3)
        # this occurs when using viewer
        H, W = None, None
        if len(ray_bundle.origins.shape) in [3, 4]:
            H, W = ray_bundle.origins.shape[:2]

        positions = ray_bundle.origins.reshape(-1, 3)  # (N, 3)
        directions = ray_bundle.directions.reshape(-1, 3)  # (N, 3)

        rotation_matrices = self.get_localised_transforms(positions)  # (N, 3, 3)

        # we transform directions so the model is only conditioned on positions,
        # directions are now independent of the position of the point
        transformed_directions = torch.einsum("ijl,ij->il", rotation_matrices, directions)  # (N, 3)

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

        if self.config.include_depth_loss_scene_center_weight and self.training and batch is not None:
            gt_termination_points = positions + directions * batch["termination_dist"].repeat(1, 3)

            if self.config.scene_center_weight_include_z:
                # use XYZ
                distance_from_center = torch.norm(positions, dim=-1)
            else:
                # use only the XY plane, ignoring the Z coordinate
                distance_from_center = torch.norm(positions[..., :2], dim=-1)

            # normalize to [0, 1]
            distance_from_center = distance_from_center / self.ddf_radius
            # invert so that points closer to the center have higher weight
            distance_weight = 1.0 - distance_from_center**self.config.scene_center_weight_exp
            outputs["distance_weight"] = distance_weight

        # get sdf at expected termination distance for loss
        if (self.config.loss_inclusions["sdf_l1_loss"] or self.config.loss_inclusions["sdf_l2_loss"]) and self.training:
            if reni_neus is not None:
                termination_points = positions + directions * expected_termination_dist.unsqueeze(-1)
                if stop_gradients:
                    with torch.no_grad():
                        sdf_at_termination = reni_neus.field.get_sdf_at_pos(termination_points)
                        sdf_at_termination = sdf_at_termination.detach()
                        sdf_at_termination.requires_grad = False
                else:
                    sdf_at_termination = reni_neus.field.get_sdf_at_pos(termination_points)
                outputs["sdf_at_termination"] = sdf_at_termination
            elif batch is not None and "sdf_at_termination" in batch:
                sdf_at_termination = batch["sdf_at_termination"]
                outputs["sdf_at_termination"] = sdf_at_termination

        # # Compute the gradient of the depths with respect to the ray origins
        if self.config.compute_normals:
            d_output = torch.ones_like(
                expected_termination_dist, requires_grad=False, device=expected_termination_dist.device
            )
            gradients = torch.autograd.grad(
                outputs=expected_termination_dist,
                inputs=ray_samples.frustums.origins,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Normalize the gradient to obtain the predicted normal
            n_hat = gradients / torch.norm(gradients, dim=-1, keepdim=True)

            # Choose the sign of n_hat such that n_hat * direction < 0
            varsigma = torch.sign(-torch.sum(n_hat * ray_samples.frustums.directions, dim=-1, keepdim=True))
            n_hat = varsigma * n_hat

            outputs["predicted_normals"] = n_hat

        if self.config.loss_inclusions["multi_view_loss"] and self.training and batch is not None:
            # for every gt termination point we choose a random other position on the sphere
            # we then sample the ddf at that point and in the direction of the random sample
            # to the gt termination point to predict the termination distance. This distance
            # should be no greater than the distance from the random point to the gt termination point.

            # get gt_termination_points using gt_termination_dist
            gt_termination_points = positions + directions * batch["termination_dist"].repeat(1, 3)

            # for every termination point we choose a random other position on the sphere
            points_on_sphere = random_points_on_unit_sphere(num_points=gt_termination_points.shape[0]).type_as(
                gt_termination_points
            )

            # ensure they are positive z by flipping if not
            points_on_sphere[:, 2] = torch.abs(points_on_sphere[:, 2])

            # get directions from points_on_sphere to termination_points
            direction_to_term_points = gt_termination_points - points_on_sphere

            # distance is the norm of the direction vector (this will be used in loss)
            distance_to_term_points = torch.norm(direction_to_term_points, dim=-1)

            # normalize the direction vector
            direction_to_term_points = direction_to_term_points / distance_to_term_points.unsqueeze(-1)

            # normalise directions such that [0, 1, 0] is facing the origin
            rotation_matrices = self.get_localised_transforms(points_on_sphere)
            transformed_directions = torch.einsum("ijl,ij->il", rotation_matrices, direction_to_term_points)

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

        if self.config.loss_inclusions["sky_ray_loss"] and self.training and batch is not None:
            # all rays that go from cameras into the sky don't intersect the scene
            # so we know that in the opposite direction the DDF should predict the
            # distance from the DDF sphere to the camera origin, this is a ground truth
            # distance that we can use to train the DDF

            # first get the sky rays
            sky_ray_bundle = batch["sky_ray_bundle"]

            camera_origins = sky_ray_bundle.origins.reshape(-1, 3)
            camera_directions = sky_ray_bundle.directions.reshape(-1, 3)

            # we need the intersection points of the sky rays with the DDF sphere
            points_on_sphere = ray_sphere_intersection(
                positions=camera_origins, directions=camera_directions, radius=self.ddf_radius
            )

            # we need the ground truth distance from the camera origin to the intersection point
            # this is the distance that the DDF should predict
            distance_to_camera_origins = torch.norm(camera_origins - points_on_sphere, dim=-1)

            # reverse directions (Origins to Sky -> DDF to Origin) and transform such that [0, 1, 0] is facing the origin
            rotation_matrices = self.get_localised_transforms(points_on_sphere)
            # -camera_directions as we are going from the sphere back towards the camera origin
            transformed_directions = torch.einsum("ijl,ij->il", rotation_matrices, -camera_directions)

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

    def forward(self, ray_bundle: RayBundle, batch, reni_neus, stop_gradients: bool = True) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        return self.get_outputs(ray_bundle, batch, reni_neus, stop_gradients=stop_gradients)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}

        mask = batch["mask"]
        gt_termination_dist = batch["termination_dist"]
        expected_termination_dist = outputs["expected_termination_dist"].unsqueeze(1)

        # ensure gt is on the same device as the model
        mask = mask.to(expected_termination_dist.device)
        gt_termination_dist = gt_termination_dist.to(expected_termination_dist.device)

        masked_depth = expected_termination_dist * mask
        masked_gt_depth = gt_termination_dist * mask

        depth_psnr = self.psnr(preds=masked_depth, target=masked_gt_depth)

        metrics_dict["depth_psnr"] = depth_psnr

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # the sdf value at the predicted termination distance
        # should be zero
        loss_dict = {}

        if self.config.mask_to_circumference:
            expected_termination_dist = outputs["expected_termination_dist"].unsqueeze(1)
            gt_termination_dist = batch["termination_dist"]
            gt_termination_dist[batch["mask"] == 0] = self.ddf_radius * 2
        else:
            expected_termination_dist = outputs["expected_termination_dist"].unsqueeze(1) * batch["mask"]
            gt_termination_dist = batch["termination_dist"] * batch["mask"]

        if self.config.loss_inclusions["depth_l1_loss"]:
            if self.config.include_depth_loss_scene_center_weight:
                loss = self.depth_l1_loss(expected_termination_dist, gt_termination_dist)
                # now we weight by distance_weight and reduce using mean
                loss_dict["depth_l1_loss"] = torch.mean(loss * outputs["distance_weight"].unsqueeze(-1))
            else:
                loss_dict["depth_l1_loss"] = self.depth_l1_loss(expected_termination_dist, gt_termination_dist)

        if self.config.loss_inclusions["depth_l2_loss"]:
            if self.config.include_depth_loss_scene_center_weight:
                loss = self.depth_l2_loss(expected_termination_dist, gt_termination_dist)
                # now we weight by distance_weight and reduce using mean
                loss_dict["depth_l2_loss"] = torch.mean(loss * outputs["distance_weight"].unsqueeze(-1))
            else:
                loss_dict["depth_l2_loss"] = self.depth_l2_loss(expected_termination_dist, gt_termination_dist)

        if self.config.loss_inclusions["sdf_l1_loss"]:
            loss_dict["sdf_l1_loss"] = self.sdf_l1_loss(
                outputs["sdf_at_termination"] * batch["mask"],
                torch.zeros_like(outputs["sdf_at_termination"]) * batch["mask"],
            )

        if self.config.loss_inclusions["sdf_l2_loss"]:
            loss_dict["sdf_l2_loss"] = self.sdf_l2_loss(
                outputs["sdf_at_termination"] * batch["mask"],
                torch.zeros_like(outputs["sdf_at_termination"]) * batch["mask"],
            )

        if self.config.loss_inclusions["prob_hit_loss"]:
            loss_dict["prob_hit_loss"] = self.prob_hit_loss(
                outputs["expected_probability_of_hit"],
                batch["mask"].squeeze(-1),
            )

        if self.config.loss_inclusions["normal_loss"]:
            loss_dict["normal_loss"] = self.normal_loss(
                outputs["predicted_normals"] * batch["mask"].unsqueeze(-1),
                batch["normals"] * batch["mask"].unsqueeze(-1),
            ).sum()

        if self.config.loss_inclusions["multi_view_loss"]:
            # multi_view_expected_termination_dist must be less than multi_view_termintation_dist
            # so penalise anything over
            loss_dict["multi_view_loss"] = torch.mean(
                torch.nn.functional.relu(
                    outputs["multi_view_expected_termination_dist"] - outputs["multi_view_termintation_dist"]
                )
                ** 2
            )

        if self.config.loss_inclusions["sky_ray_loss"]:
            # use l1 loss between sky_ray_expected_termination_dist and sky_ray_termination_dist
            loss_dict["sky_ray_loss"] = self.sky_ray_loss(
                outputs["sky_ray_expected_termination_dist"],
                outputs["sky_ray_termination_dist"],
            )

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
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

        masked_depth = expected_termination_dist * gt_accumulations + (1 - gt_accumulations)
        masked_gt_depth = gt_termination_dist * gt_accumulations + (1 - gt_accumulations)

        # need to reshape for metrics from H, W, C to B, C, H, W
        masked_depth = masked_depth.unsqueeze(0).permute(0, 3, 1, 2)
        masked_gt_depth = masked_gt_depth.unsqueeze(0).permute(0, 3, 1, 2)

        depth_psnr = self.psnr(preds=masked_depth, target=masked_gt_depth)
        depth_ssim = self.ssim(preds=masked_depth, target=masked_gt_depth)

        # # lpips expects images in range 0 - 1 so we need to normalize
        # masked_depth = (masked_depth - torch.min(masked_depth)) / (torch.max(masked_depth) - torch.min(masked_depth))
        # masked_gt_depth = (masked_gt_depth - torch.min(masked_gt_depth)) / (torch.max(masked_gt_depth) - torch.min(masked_gt_depth))
        # depth_lpips = self.lpips(masked_depth, masked_gt_depth)

        metrics_dict["depth_psnr"] = depth_psnr
        metrics_dict["depth_ssim"] = depth_ssim
        # metrics_dict["depth_lpips"] = depth_lpips

        gt_depth = colormaps.apply_depth_colormap(
            gt_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
            colormap_options=ColormapOptions(normalize=False, colormap_min=0.0, colormap_max=2.0),
        )

        depth = colormaps.apply_depth_colormap(
            expected_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
            colormap_options=ColormapOptions(normalize=False, colormap_min=0.0, colormap_max=2.0),
        )

        combined_depth = torch.cat([gt_depth, depth], dim=1)
        images_dict["depth"] = combined_depth

        depth_error = torch.abs(
            gt_termination_dist * batch["mask"].type_as(gt_termination_dist)
            - expected_termination_dist * batch["mask"].type_as(expected_termination_dist)
        )
        depth_error_normalized = (depth_error - torch.min(depth_error)) / (
            torch.max(depth_error) - torch.min(depth_error)
        )
        images_dict["depth_error"] = depth_error_normalized

        if "expected_probability_of_hit" in outputs:
            expected_probability_of_hit = outputs["expected_probability_of_hit"]
            combined_probability_of_hit = torch.cat([gt_accumulations, expected_probability_of_hit], dim=1)
            images_dict["probability_of_hit"] = combined_probability_of_hit

        if "predicted_normals" in outputs:
            normals = outputs["predicted_normals"]
            normals = (normals + 1.0) / 2.0
            gt_normal = batch["normals"].to(normals.device)
            gt_normal = (gt_normal + 1.0) / 2.0
            combined_normal = torch.cat([gt_normal, normals], dim=1)
            images_dict["normals"] = combined_normal

        return metrics_dict, images_dict

    def get_image_dict(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        images_dict = {}

        expected_termination_dist = outputs["expected_termination_dist"]

        expected_probability_of_hit = None
        if RENINeuSFieldHeadNames.PROBABILITY_OF_HIT in outputs:
            expected_probability_of_hit = outputs[RENINeuSFieldHeadNames.PROBABILITY_OF_HIT]
            expected_probability_of_hit = expected_probability_of_hit.unsqueeze(-1)
            accumulation = colormaps.apply_colormap(outputs["accumulation"])
            images_dict["ddf_accumulation"] = accumulation

        depth = colormaps.apply_depth_colormap(
            expected_termination_dist,
            accumulation=expected_probability_of_hit,
            near_plane=0.0,
            far_plane=self.ddf_radius * 2,
            colormap_options=ColormapOptions(normalize=False, colormap_min=0.0, colormap_max=2.0),
        )

        images_dict["ddf_depth"] = depth

        return images_dict
