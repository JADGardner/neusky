"""
RENI-NeuS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.sdf_datamanager import SDFDataManagerConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from reni_neus.data.nerfosr_cityscapes_dataparser import NeRFOSRCityScapesDataParserConfig
from reni_neus.reni_neus_model import RENINeuSFactoModel, RENINeuSFactoModelConfig
from reni_neus.illumination_fields.reni_field import RENIFieldConfig, RENIField
from reni_neus.model_components.illumination_samplers import IcosahedronSamplerConfig, IcosahedronSampler
from reni_neus.fields.sdf_albedo_field import SDFAlbedoFieldConfig, SDFAlbedoField
from reni_neus.reni_neus_pipeline import RENINeuSPipelineConfig
from reni_neus.data.reni_neus_datamanager import RENINeuSDataManagerConfig


RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=RENINeuSPipelineConfig(
            eval_latent_optimisation_source="image_half",
            eval_latent_optimisation_epochs=100,
            eval_latent_optimisation_lr=0.1,
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                ),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=RENINeuSFactoModelConfig(
                # proposal network allows for signifanctly smaller sdf/color network
                sdf_field=SDFAlbedoFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                    inside_outside=False,
                ),
                illumination_field=RENIFieldConfig(
                    checkpoint_path="/workspace/reni_neus/checkpoints/reni_weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1589.ckpt",
                    fixed_decoder=True,
                    exposure_scale=True,
                ),
                illumination_sampler=IcosahedronSamplerConfig(
                    icosphere_order=11,
                    apply_random_rotation=True,
                    remove_lower_hemisphere=False,
                ),
                illumination_field_prior_loss_weight=1e-7,
                illumination_field_cosine_loss_weight=1e-1,
                illumination_field_loss_weight=1.0,
                visibility_loss_mse_mult=0.01,
                background_model="none",
                eval_num_rays_per_chunk=2048,
                use_average_appearance_embedding=False,
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
            "illumination_field": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for RENI-NeuS.",
)
