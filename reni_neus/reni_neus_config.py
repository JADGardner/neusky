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
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
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
from reni_neus.ddf_pipeline import DDFPipelineConfig
from reni_neus.data.ddf_datamanager import DDFDataManagerConfig
from reni_neus.reni_neus_with_vis_model import RENINeuSFactoWithVisibilityModelConfig



RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        steps_per_eval_image=10000,
        steps_per_eval_batch=100000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        load_dir=Path("/workspace/outputs/unnamed/reni-neus/2023-05-23_104452/nerfstudio_models/"),
        load_step=50000,
        pipeline=RENINeuSPipelineConfig(
            eval_latent_optimisation_source="image_full",
            eval_latent_optimisation_epochs=50,
            eval_latent_optimisation_lr=1e-2,
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=True,
                    crop_to_equal_size=True,
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
                    num_layers=2,
                    hidden_dim=256,
                    num_layers_color=2,
                    hidden_dim_color=256,
                    bias=0.1,
                    beta_init=0.1,
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
                render_only_albedo=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100001),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100001),
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


RENINeuSWithVis = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus-with-vis",
        steps_per_eval_image=10000,
        steps_per_eval_batch=100000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        # load_dir=Path("/workspace/outputs/unnamed/reni-neus/2023-05-23_104452/nerfstudio_models/"),
        # load_step=50000,
        pipeline=RENINeuSPipelineConfig(
            eval_latent_optimisation_source="image_full",
            eval_latent_optimisation_epochs=50,
            eval_latent_optimisation_lr=1e-2,
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=True,
                    crop_to_equal_size=True,
                ),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=512,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=RENINeuSFactoWithVisibilityModelConfig(
                # proposal network allows for signifanctly smaller sdf/color network
                sdf_field=SDFAlbedoFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    hidden_dim=256,
                    num_layers_color=2,
                    hidden_dim_color=256,
                    bias=0.1,
                    beta_init=0.1,
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
                visibility_field=DDFModelConfig(
                    ddf_field=DirectionalDistanceFieldConfig(
                      position_encoding_type="none",
                      direction_encoding_type="none",
                      network_type="siren",
                      termination_output_activation="sigmoid",
                      probability_of_hit_output_activation="sigmoid",
                      hidden_layers=5,
                      hidden_features=256,
                      predict_probability_of_hit=False,
                      ),
                    sdf_loss_mult=0.5,
                    depth_loss_mult=3.0,
                    prob_hit_loss_mult=0.0,
                    normal_loss_mult=1.0,
                ),
                ddf_radius="AABB",
                eval_num_rays_per_chunk=512,
                illumination_field_prior_loss_weight=1e-7,
                illumination_field_cosine_loss_weight=1e-1,
                illumination_field_loss_weight=1.0,
                fg_mask_loss_mult=1.0,
                background_model="none",
                use_average_appearance_embedding=False,
                render_only_albedo=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100001),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100001),
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
        pipeline=DDFPipelineConfig(
            datamanager=DDFDataManagerConfig(
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=512,
                num_test_images_to_generate=1,
                test_image_cache_dir=Path("/workspace/outputs/ddf/cache/"),
                accumulation_mask_threshold=0.7,
                train_data="rand_pnts_on_sphere", # "rand_pnts_on_sphere, single_camera"
            ),
            model=DDFModelConfig(
                ddf_field=DirectionalDistanceFieldConfig(
                    ddf_type="pddf", # pddf
                    position_encoding_type="none",
                    direction_encoding_type="none",
                    network_type="siren",
                    termination_output_activation="sigmoid",
                    probability_of_hit_output_activation="sigmoid",
                    hidden_layers=5,
                    hidden_features=256,
                    predict_probability_of_hit=False,
                ),
                sdf_loss_mult=1.0,
                depth_loss_mult=1.0,
                prob_hit_loss_mult=0.0,
                normal_loss_mult=1.0,
                eval_num_rays_per_chunk=128,
            ),
            reni_neus_ckpt_path=Path("/workspace/outputs/unnamed/reni-neus/2023-05-23_191641"),
            reni_neus_ckpt_step=70000,
            ddf_radius="AABB",
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=10001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Directional Distance Field.",
)


NeRFactoNeRFOSR = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-nerfosr",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=False,
                ),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3)
                ),
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                hidden_dim_color=128,
                hidden_dim_transient=128,
                max_res=3000,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Nerfacto NeRF-OSR.",
)


NeusFactoNeRFOSR = MethodSpecification(
    config=TrainerConfig(
        method_name="neus-facto-nerfosr",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=False,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
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
                    use_appearance_embedding=False,
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
    description="Base config for Neusfacto nerfosr.",
)
