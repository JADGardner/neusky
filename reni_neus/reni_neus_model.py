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
from typing import Any, Dict, List, Optional, Tuple, Type
from collections import defaultdict
import random
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

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

from reni_neus.illumination_fields.base_illumination_field import IlluminationFieldConfig
from reni_neus.model_components.renderers import RGBLambertianRendererWithVisibility
from reni_neus.model_components.illumination_samplers import IlluminationSamplerConfig
from reni_neus.utils.utils import RENITestLossMask, get_directions
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
    render_only_albedo: bool = False
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
        super().__init__(config, scene_box, num_train_data, **kwargs)

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
        return param_groups

    def get_illumination_field(self):
        if self.training and not self.fitting_eval_latents:
            illumination_field = self.illumination_field_train
        else:
            illumination_field = (
                self.illumination_field_test if self.test_mode == "test" else self.illumination_field_val
            )

        return illumination_field

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

            # Calculate gap
            gap = torch.tensor([(max_coord[i] - min_coord[i]) / self.config.hashgrid_density_loss_sample_resolution for i in range(3)])

            # Generate random perturbations
            perturbations = torch.rand_like(positions) * gap - gap / 2

            # Apply perturbations
            positions += perturbations
            
            # generate random normalised directions of shape positions
            # these are needed for 
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

        albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)

        if not self.config.render_only_albedo:
            rgb = self.lambertian_renderer(
                albedos=field_outputs[RENINeuSFieldHeadNames.ALBEDO],
                normals=field_outputs[FieldHeadNames.NORMALS],
                light_directions=illumination_directions,
                light_colors=hdr_illumination_colours,
                visibility=None,
                background_illumination=background_colours,
                weights=weights,
            )
            p2p_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
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
            "directions_norm": ray_bundle.metadata["directions_norm"]
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

        if 'grid_density' in samples_and_field_outputs:
            outputs['grid_density'] = samples_and_field_outputs['grid_density']

        return outputs

    def get_loss_dict(
        self, outputs: Dict[str, Any], batch: Dict[str, Any], metrics_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the loss dictionary, including interlevel loss for proposal networks."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            if "fg_mask" in batch:
                fg_label = batch["fg_mask"].float().to(self.device)
                sky_label = 1 - fg_label
                loss_dict["illumination_loss"] = (
                    self.direct_illumination_loss(
                        inputs=outputs["background_colours"],
                        targets=batch["image"].to(self.device),
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
            envmap, _ = illumination_field(idx, None, D, "envmap")
            envmap = envmap.reshape(1, H, W, 3).squeeze(0)
            images_dict["RENI"] = envmap

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, show_progress=False
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
                    outputs = self.forward(ray_bundle=ray_bundle)
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
                outputs = self.forward(ray_bundle=ray_bundle)
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
                        # rgb = batch["envmap"].to(self.device)
                        # directions = get_directions(rgb.shape[1])
                        # sineweight = get_sineweight(rgb.shape[1])
                        # rgb = rgb.unsqueeze(0)  # [B, H, W, 3]
                        # rgb = rgb.reshape(rgb.shape[0], -1, 3)  # [B, H*W, 3]
                        # D = directions.type_as(rgb).repeat(rgb.shape[0], 1, 1)  # [B, H*W, 3]
                        # S = sineweight.type_as(rgb).repeat(rgb.shape[0], 1, 1)  # [B, H*W, 3]
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
