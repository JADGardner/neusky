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
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

from reni_neus.fields.directional_distance_field import DirectionalDistanceField, DirectionalDistanceFieldConfig
from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions
from reni_neus.model_components.ddf_sdf_sampler import DDFSDFSampler
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames

CONSOLE = Console(width=120)


@dataclass
class DDFModelConfig(ModelConfig):
    """DDF Model Config"""

    _target: Type = field(default_factory=lambda: DDFModel)
    ddf_field: DirectionalDistanceFieldConfig = DirectionalDistanceFieldConfig()
    """DDF field configuration"""
    sdf_loss_mult: float = 1.0
    """Multiplier for the sdf loss"""
    depth_loss_mult: float = 1.0
    """Multiplier for the depth loss"""
    prob_hit_loss_mult: float = 1.0
    """Multiplier for the probability of hit loss"""
    normal_loss_mult: float = 1.0
    """Multiplier for the normal loss"""

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
        self.sdf_loss = MSELoss()
        self.depth_loss = MSELoss()
        self.normal_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, reni_neus):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # get H, W from ray_bundle if it's shape is (H, W, 3) and not (N, 3)
        H, W = None, None
        if len(ray_bundle.origins.shape) in [3, 4]:
            H, W = ray_bundle.origins.shape[:2]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=ray_bundle.origins.reshape(-1, 3),
                directions=ray_bundle.directions.reshape(-1, 3),
                starts=torch.zeros_like(ray_bundle.origins),
                ends=torch.zeros_like(ray_bundle.origins),
                pixel_area=ray_bundle.pixel_area.reshape(-1, 1),
            ),
        )

        field_outputs = self.field.forward(ray_samples)
        expected_termination_dist = field_outputs[RENINeuSFieldHeadNames.TERMINATION_DISTANCE]

        # get sdf at expected termination distance for loss
        termination_points = (
            ray_samples.frustums.origins + ray_samples.frustums.directions * expected_termination_dist.unsqueeze(-1)
        )
        sdf_at_termination = reni_neus.field.get_sdf_at_pos(termination_points)

        outputs = {
            "sdf_at_termination": sdf_at_termination,
            "expected_termination_dist": expected_termination_dist,
        }

        if RENINeuSFieldHeadNames.PROBABILITY_OF_HIT in field_outputs:
            outputs["expected_probability_of_hit"] = field_outputs[RENINeuSFieldHeadNames.PROBABILITY_OF_HIT]

        # # Compute the gradient of depth with respect to position
        # grad_expected_term = torch.autograd.grad(expected_termination_dist.sum(), ray_samples.frustums.origins, create_graph=True)[0]
        
        # # Normalize the gradient to obtain the predicted normal
        # n_hat_unnormalized = grad_expected_term.t()
        # n_hat = n_hat_unnormalized / torch.norm(n_hat_unnormalized, dim=-1, keepdim=True)
        
        # # Choose the sign of n_hat such that n_hat^T * direction < 0
        # varsigma = torch.sign(-torch.sum(n_hat * ray_samples.frustums.directions, dim=-1, keepdim=True))
        # n_hat = varsigma * n_hat

        # outputs["predicted_normal"] = n_hat

        for key, value in outputs.items():
            if H is not None and W is not None:
                outputs[key] = value.reshape(H, W, 1, -1)

        return outputs
    
    def forward(self, ray_bundle: RayBundle, reni_neus) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, reni_neus)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        
        # the sdf value at the predicted termination distance
        # should be zero
        loss_dict = {}

        sdf_loss = self.sdf_loss(
            outputs["sdf_at_termination"] * batch["mask"],
            torch.zeros_like(outputs["sdf_at_termination"]) * batch["mask"],
        )

        # match the depth of the sdf model
        depth_loss = self.depth_loss(
            outputs["expected_termination_dist"] * batch["mask"],
            batch["termination_dist"] * batch["mask"],
        )

        loss_dict["sdf_loss"] = sdf_loss * self.config.sdf_loss_mult
        loss_dict["depth_loss"] = depth_loss * self.config.depth_loss_mult
        
        if 'predicted_normal' in outputs:
            normal_loss = self.normal_loss(
                outputs["predicted_normal"] * batch["mask"],
                batch["normal"] * batch["mask"],
            )
            loss_dict["normal_loss"] = normal_loss * self.config.normal_loss_mult
        
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, reni_neus=None, show_progress=True
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
                    outputs = self.forward(ray_bundle=ray_bundle, reni_neus=reni_neus)
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
                outputs = self.forward(ray_bundle=ray_bundle, reni_neus=reni_neus)
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_accumulations = batch["accumulations"]
        gt_termination_dist = batch["termination_dist"]
        gt_normals = batch["normals"]

        expected_termination_dist = outputs["expected_termination_dist"]

        if RENINeuSFieldHeadNames.PROBABILITY_OF_HIT in outputs:
            expected_probability_of_hit = outputs["expected_probability_of_hit"]

        gt_depth = colormaps.apply_depth_colormap(
            gt_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
        )

        depth = colormaps.apply_depth_colormap(
            expected_termination_dist,
            accumulation=gt_accumulations,
            near_plane=self.collider.near_plane,
            far_plane=self.collider.radius * 2,
        )

        combined_depth = torch.cat([gt_depth, depth], dim=1)

        metrics_dict = {}

        images_dict = {"depth": combined_depth}
        return metrics_dict, images_dict
