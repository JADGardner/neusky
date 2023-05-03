"""
RENI-NeuS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig

from reni_neus.data.nerfosr_cityscapes_dataparser import NeRFOSRCityScapesDataParserConfig
from reni_neus.reni_neus_model import RENINeuSFactoModelConfig
from reni_neus.illumination_fields.reni_field import RENIFieldConfig
from reni_neus.model_components.illumination_samplers import IcosahedronSamplerConfig
from reni_neus.reni_neus_pipeline import RENINeuSPipelineConfig
from reni_neus.data.reni_neus_datamanager import RENINeuSDataManagerConfig
from reni_neus.fields.sdf_albedo_field import SDFAlbedoFieldConfig


RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        pipeline=RENINeuSPipelineConfig(
            eval_latent_optimisation_source="image_half",
            eval_latent_optimisation_epochs=50,
            eval_latent_optimisation_lr=1e-2,
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                ),
                train_num_rays_per_batch=256,
                eval_num_rays_per_batch=256,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=RENINeuSFactoModelConfig(
                # proposal network allows for signifanctly smaller sdf/color network
                sdf_field=SDFAlbedoFieldConfig(
                    use_grid_feature=True,
                    num_layers=5,
                    hidden_dim=256,
                    num_layers_color=5,
                    hidden_dim_color=256,
                    bias=0.5,
                    beta_init=0.3,
                    use_appearance_embedding=False,
                    inside_outside=False,
                ),
                illumination_field=RENIFieldConfig(
                    checkpoint_path="/workspace/reni_neus/checkpoints/reni_weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1589.ckpt",
                    fixed_decoder=True,
                    optimise_exposure_scale=True,
                ),
                illumination_sampler=IcosahedronSamplerConfig(
                    icosphere_order=11,
                    apply_random_rotation=True,
                    remove_lower_hemisphere=False,
                ),
                eval_num_rays_per_chunk=256,
                illumination_field_prior_loss_weight=1e-7,
                illumination_field_cosine_loss_weight=1e-1,
                illumination_field_loss_weight=1.0,
                fg_mask_loss_multi=1.0,
                background_model="none",
                use_average_appearance_embedding=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=1.0, max_steps=100001),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=1.0, max_steps=100001),
            },
            "illumination_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for RENI-NeuS.",
)


NeuSFactoNeRFOSR = MethodSpecification(
    config=TrainerConfig(
        method_name="neus-facto-nerfosr",
        steps_per_eval_image=10000,
        steps_per_eval_batch=100000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                ),
                train_num_rays_per_batch=256,
                eval_num_rays_per_batch=256,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NeuSFactoModelConfig(
                # proposal network allows for signifanctly smaller sdf/color network
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=True,
                    inside_outside=False,
                ),
                background_model="mlp",
                eval_num_rays_per_chunk=2048,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=20001, milestones=(10000, 1500, 18000)),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for NeuS Facto NeRF-OSR.",
)


NeRFFactoNeRFOSR = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-nerfosr",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for NeuS Facto NeRF-OSR.",
)
