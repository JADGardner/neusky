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

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils import colormaps

from reni_neus.illumination_fields.reni_field import RENIFieldConfig, RENIField
from reni_neus.illumination_fields.base_illumination_field import IlluminationFieldConfig, IlluminationField
from reni_neus.model_components.renderers import RGBLambertianRendererWithVisibility
from reni_neus.model_components.illumination_samplers import IlluminationSamplerConfig, IlluminationSampler
from reni_neus.utils.utils import RENITestLossMask, get_directions
from reni_neus.reni_neus_fieldheadnames import RENINeuSFieldHeadNames


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
    fg_mask_loss_multi: float = 0.01
    """Weight for the fg mask loss"""


class RENINeuSFactoModel(NeuSFactoModel):
    """NeuSFactoModel extends NeuSModel for a more efficient sampling strategy.

    The model improves the rendering speed and quality by incorporating a learning-based
    proposal distribution to guide the sampling process.(similar to mipnerf-360)

    Args:
        config: NeuS configuration to instantiate model
    """

    config: RENINeuSFactoModelConfig

    def __init__(
        self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, num_eval_data: int, **kwargs
    ) -> None:
        self.num_eval_data = num_eval_data
        self.fitting_eval_latents = False
        super().__init__(config, scene_box, num_train_data, **kwargs)

    def populate_modules(self):
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        self.illumination_field_train = self.config.illumination_field.setup(num_latent_codes=self.num_train_data)
        self.illumination_field_eval = self.config.illumination_field.setup(num_latent_codes=self.num_eval_data)
        self.illumination_sampler = self.config.illumination_sampler.setup()

        self.field_background = None

        self.albedo_renderer = RGBRenderer(background_color=torch.tensor([1.0, 1.0, 1.0]))
        self.lambertian_renderer = RGBLambertianRendererWithVisibility()

        self.direct_illumination_loss = RENITestLossMask(
            alpha=self.config.illumination_field_prior_loss_weight,
            beta=self.config.illumination_field_cosine_loss_weight,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return a dictionary with the parameters of the proposal networks."""
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["illumination_field"] = list(self.illumination_field_train.parameters())
        return param_groups

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

        if self.training:
            illumination_field = (
                self.illumination_field_train if not self.fitting_eval_latents else self.illumination_field_eval
            )
        else:
            illumination_field = self.illumination_field_eval

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

        # TODO Add visibility here?

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
        albedo = self.albedo_renderer(rgb=field_outputs[RENINeuSFieldHeadNames.ALBEDO], weights=weights)

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

    def get_loss_dict(
        self, outputs: Dict[str, Any], batch: Dict[str, Any], metrics_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the loss dictionary, including interlevel loss for proposal networks."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            if "fg_mask" in batch:
                fg_label = batch["fg_mask"].float().to(self.device)
                # TODO somehow make this not an if statement here
                # Maybe get_loss_dict for the illumination_field should be a function?
                loss_dict["illumination_loss"] = (
                    self.direct_illumination_loss(
                        inputs=outputs["background_colours"],
                        targets=batch["image"].to(self.device),
                        mask=fg_label,
                        Z=self.illumination_field_train.get_latents(),
                    )
                    * self.config.illumination_field_loss_weight
                )

                # foreground mask loss
                if self.config.background_model != "none":
                    if self.config.fg_mask_loss_mult > 0.0:
                        w = outputs["weights_bg"]
                        weights_sum = w.sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                        loss_dict["bg_mask_loss"] = (
                            F.binary_cross_entropy_with_logits(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                        )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images, including the proposal depth for each iteration."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
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
            envmap, _ = self.illumination_field_eval(idx, None, D, "envmap")
            envmap = envmap.reshape(1, H, W, 3).squeeze(0)
            images_dict["RENI"] = envmap

        return metrics_dict, images_dict

    def fit_latent_codes_for_eval(self, datamanager, gt_source, epochs, learning_rate):
        return
