"""
RENI-NeuS configuration file.
"""
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSRDataParserConfig

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
from reni_neus.ddf_model import DDFModelConfig
from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig


RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        load_dir=Path("/workspace/outputs/unnamed/reni-neus/2023-05-05_105353/nerfstudio_models/"),
        load_step=30000,
        pipeline=RENINeuSPipelineConfig(
            eval_latent_optimisation_source="image_half",
            eval_latent_optimisation_epochs=50,
            eval_latent_optimisation_lr=1e-2,
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=False,
                ),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=512,
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
                eval_num_rays_per_chunk=512,
                illumination_field_prior_loss_weight=1e-7,
                illumination_field_cosine_loss_weight=1e-1,
                illumination_field_loss_weight=1.0,
                fg_mask_loss_mult=1.0,
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


DirectionalDistanceField = MethodSpecification(
    config=TrainerConfig(
        method_name="ddf",
        steps_per_eval_image=100,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=10001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NeRFOSRDataParserConfig(
                    scene="lk2",
                ),
                train_num_rays_per_batch=256,
                eval_num_rays_per_batch=256,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=DDFModelConfig(
                ddf_field=DirectionalDistanceFieldConfig(
                    position_encoding_type="none",
                    direction_encoding_type="none",
                    network_type="siren",
                    hidden_layers=5,
                    hidden_features=256,
                    predict_probability_of_hit=False,
                ),
                eval_num_rays_per_chunk=256,
                reni_neus_ckpt_path="/workspace/outputs/unnamed/reni-neus/2023-05-08_123628",
                reni_neus_ckpt_step=95000,
                num_sample_directions=256,
                ddf_radius=1.0,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=10001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for NeuS Facto NeRF-OSR.",
)
