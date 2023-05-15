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
from typing import Dict, List, Tuple, Type
from pathlib import Path
import yaml
from torch.utils.data import Dataset

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
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

from reni_neus.fields.directional_distance_field import DirectionalDistanceField, DirectionalDistanceFieldConfig
from reni_neus.utils.utils import random_points_on_unit_sphere, random_inward_facing_directions


@dataclass
class DDFModelConfig(ModelConfig):
    """DDF Model Config"""

    _target: Type = field(default_factory=lambda: DDFModel)
    ddf_field: DirectionalDistanceFieldConfig = DirectionalDistanceFieldConfig()
    """DDF field configuration"""
    reni_neus_ckpt_path: str = ""
    """Path to reni_neus checkpoint"""
    reni_neus_ckpt_step: int = 10000
    """Step of reni_neus checkpoint"""


class DDFModel(Model):
    """mip-NeRF model

    Args:
        config: DDFModelConfig configuration to instantiate model
    """

    config: DDFModelConfig

    def populate_modules(self):
        """Set the fields and modules"""

        # setting up fields
        self.field = self.config.ddf_field.setup()

        # setting up reni_neus for pseudo ground truth
        ckpt = torch.load(
            self.config.reni_neus_ckpt_path
            + "/nerfstudio_models"
            + f"/step-{self.config.reni_neus_ckpt_step:09d}.ckpt",
            map_location=self.device,
        )
        model_dict = {}
        for key in ckpt["pipeline"].keys():
            if key.startswith("_model."):
                model_dict[key[7:]] = ckpt["pipeline"][key]

        num_train_data = model_dict["illumination_field_train.reni.mu"].shape[0]
        num_eval_data = model_dict["illumination_field_eval.reni.mu"].shape[0]
        aabb = model_dict["field.aabb"]

        # load yaml checkpoint config
        reni_neus_config = Path(self.config.reni_neus_ckpt_path) / "config.yml"
        reni_neus_config = yaml.load(reni_neus_config.open(), Loader=yaml.Loader)

        self.reni_neus = reni_neus_config.pipeline.model.setup(
            scene_box=aabb, num_train_data=num_train_data, num_eval_data=num_eval_data
        )
        self.reni_neus.to(self.device)
        self.reni_neus.load_state_dict(model_dict)
        self.reni_neus.eval()

        # # samplers
        # self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

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

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict


class DDFSDFDataset(Dataset):
    def __init__(self, num_directions, ddf_sphere_radius, sdf_function, repeat_position, device):
        self.sdf_function = sdf_function
        self.ddf_sphere_radius = ddf_sphere_radius
        self.num_directions = num_directions
        self.repeat_position = repeat_position
        self.device = device
        # generate a H and W from number of directions
        self.H, self.W = self._find_factors(self.num_directions)[-1]

        origins = torch.zeros(self.H, self.W, 1, 3, device=device)
        directions = torch.ones(self.H, self.W, 1, 3, device=device)
        pixel_area = torch.ones(self.H, self.W, 1, 1, device=device)
        directions_norm = torch.ones(self.H, self.W, 1, 1, device=device)
        # camera indices should be int64
        camera_indices = torch.zeros(self.H, self.W, 1, 1, device=device, dtype=torch.int64)
        nears = torch.ones(self.H, self.W, 1, 1, device=device) * sdf_function.scene_box.near
        fars = torch.ones(self.H, self.W, 1, 1, device=device) * sdf_function.scene_box.far

        metadata = {"directions_norm": directions_norm}

        self.ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            metadata=metadata,
            camera_indices=camera_indices,
            nears=nears,
            fars=fars,
        )

    def __len__(self):
        return self.num_positions

    def _find_factors(self, value):
        factors = []
        for i in range(2, int(value**0.5) + 1):
            if value % i == 0:
                factors.append((i, value // i))
        return factors

    def _generate_sample(self):
        if self.repeat_position:
            positions = torch.tensor([[0.6421, 0.7197, -0.2640]])
            positions = positions / torch.norm(positions)
        else:
            positions = random_points_on_unit_sphere(1, cartesian=True)  # (1, 3)

        directions = random_inward_facing_directions(self.num_directions, normals=-positions)  # (1, num_directions, 3)

        # scale positions from unit sphere to ddf_sphere_radius
        positions = positions * self.ddf_sphere_radius

        pos_ray = positions.unsqueeze(1).unsqueeze(1).expand(self.H, self.W, 1, -1).to(self.device)
        dir_ray = directions.reshape(self.H, self.W, 3).unsqueeze(2).to(self.device)

        self.ray_bundle.origins = pos_ray
        self.ray_bundle.directions = dir_ray
        field_outputs = self.sdf_function.get_outputs_for_camera_ray_bundle(self.ray_bundle)
        accumultation = field_outputs["accumulation"].reshape(-1, 1).squeeze()
        termination_dist = field_outputs["p2p_dist"].reshape(-1, 1).squeeze()
        normals = field_outputs["normal"].reshape(-1, 3).squeeze()

        directions = directions.squeeze(0)
        positions = positions.repeat(directions.shape[0], 1)

        return positions, directions, accumultation, termination_dist, normals

    def __getitem__(self, idx):
        return self._generate_sample()
