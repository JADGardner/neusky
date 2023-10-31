"""
RENI-NeuS configuration file.
"""
from pathlib import Path

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)

from reni_neus.data.dataparsers.nerfosr_cityscapes_dataparser import NeRFOSRCityScapesDataParserConfig
from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig
from reni_neus.pipelines.reni_neus_pipeline import RENINeuSPipelineConfig
from reni_neus.data.datamanagers.reni_neus_datamanager import RENINeuSDataManagerConfig
from reni_neus.fields.sdf_albedo_field import SDFAlbedoFieldConfig
from reni_neus.models.ddf_model import DDFModelConfig
from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig
from reni_neus.model_components.ddf_sampler import VMFDDFSamplerConfig
from reni_neus.data.reni_neus_pixel_sampler import RENINeuSPixelSamplerConfig


RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        experiment_name="reni-neus",
        steps_per_eval_image=10,
        steps_per_eval_batch=100000,
        steps_per_save=5000,
        steps_per_eval_all_images=100000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        # load_dir=Path("/workspace/outputs/unnamed/reni-neus/2023-08-09_075320/nerfstudio_models"),
        # load_step=50000,
        pipeline=RENINeuSPipelineConfig(
          test_mode='test',
            datamanager=RENINeuSDataManagerConfig(
                dataparser=NeRFOSRCityScapesDataParserConfig(
                    scene="lk2",
                    auto_scale_poses=True,
                    crop_to_equal_size=True,
                    pad_to_equal_size=False,
                    scene_scale=1.0,  # AABB
                    mask_vegetation=False,
                ),
                train_num_images_to_sample_from=-1,
                train_num_times_to_repeat_images=-1,  # # Iterations before resample a new subset
                pixel_sampler=RENINeuSPixelSamplerConfig(),
                images_on_gpu=False,
                masks_on_gpu=False,
                train_num_rays_per_batch=64,
                eval_num_rays_per_batch=256,
            ),
            model=RENINeuSFactoModelConfig(
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
                    predict_shininess=False,
                ),
                illumination_field=RENIFieldConfig(
                    conditioning="Attention",
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z",  # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",
                    latent_dim=100,
                    hidden_features=128,
                    hidden_layers=9,
                    mapping_layers=5,
                    mapping_features=128,
                    num_attention_heads=8,
                    num_attention_layers=6,
                    output_activation="None",
                    last_layer_linear=True,
                    fixed_decoder=True,
                    trainable_scale=True,
                ),
                illumination_sampler=IcosahedronSamplerConfig(
                    num_directions=64,
                    apply_random_rotation=True,
                    remove_lower_hemisphere=False,
                ),
                loss_inclusions={
                    "rgb_l1_loss": True,
                    "rgb_l2_loss": False,
                    "cosine_colour_loss": False,
                    "eikonal loss": True,
                    "fg_mask_loss": True,
                    "normal_loss": False,
                    "depth_loss": False,
                    "interlevel_loss": True,
                    "sky_pixel_loss": {"enabled": True, "cosine_weight": 0.1},
                    "hashgrid_density_loss": {
                        "enabled": True,
                        "grid_resolution": 10,
                    },
                    "ground_plane_loss": True,
                    "visibility_sigmoid_loss": {
                        "visibility_threshold_method": "learnable",  # "learnable", "fixed", "exponential_decay"
                        "optimise_sigmoid_bias": True,
                        "optimise_sigmoid_scale": False,
                        "target_min_bias": 0.1,
                        "target_max_scale": 25,
                        "steps_until_min_bias": 50000,  # if sigmoid_bias_method is exponential_decay
                    },
                },
                loss_coefficients={
                    "rgb_l1_loss": 1.0,
                    "rgb_l2_loss": 0.0,
                    "cosine_colour_loss": 1.0,
                    "eikonal loss": 0.1,
                    "fg_mask_loss": 1.0,
                    "normal_loss": 1.0,
                    "depth_loss": 1.0,
                    "interlevel_loss": 1.0,
                    "sky_pixel_loss": 1.0,
                    "hashgrid_density_loss": 1e-4,
                    "ground_plane_loss": 0.1,
                    "visibility_sigmoid_loss": 0.01,  # if learnable
                },
                eval_latent_optimizer={
                    "eval_latents": {
                        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                        # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=250),
                        "scheduler": CosineDecaySchedulerConfig(
                            warm_up_end=50, learning_rate_alpha=0.05, max_steps=250
                        ),
                    },
                },
                eval_latent_optimise_method="nerf_osr_holdout",  # per_image, nerf_osr_holdout, nerf_osr_envmap
                eval_latent_sample_region="full_image",
                illumination_field_ckpt_path=Path("outputs/reni/reni_plus_plus_models/latent_dim_100/"),
                illumination_field_ckpt_step=50000,
                fix_test_illumination_directions=True,
                eval_num_rays_per_chunk=256,
                use_visibility=True,
                fit_visibility_field=True,  # if true, train visibility field, else visibility is static
                sdf_to_visibility_stop_gradients="depth",  # "depth", "sdf", "both", "none" # if both then visibility losses can't update sdf
                only_upperhemisphere_visibility=True,
                scene_contraction_order="L2",  # L2, Linf
                collider_shape="sphere",
                use_average_appearance_embedding=False,
                background_model="none",
            ),
            visibility_field=DDFModelConfig(  # DDFModelConfig or None
                ddf_field=DirectionalDistanceFieldConfig(
                    ddf_type="ddf",  # pddf
                    position_encoding_type="hash",  # none, hash, nerf, sh
                    direction_encoding_type="nerf",
                    conditioning="FiLM",  # FiLM, Concat, Attention
                    termination_output_activation="sigmoid",
                    probability_of_hit_output_activation="sigmoid",
                    hidden_layers=5,
                    hidden_features=256,
                    mapping_layers=5,
                    mapping_features=256,
                    num_attention_heads=8,
                    num_attention_layers=6,
                    predict_probability_of_hit=False,
                ),
                loss_inclusions={
                    "depth_l1_loss": True,
                    "depth_l2_loss": False,
                    "sdf_l1_loss": False,
                    "sdf_l2_loss": True,
                    "prob_hit_loss": False,
                    "normal_loss": False,
                    "multi_view_loss": True,
                    "sky_ray_loss": True,
                },
                loss_coefficients={
                    "depth_l1_loss": 1.0,
                    "depth_l2_loss": 0.0,
                    "sdf_l1_loss": 1.0,
                    "sdf_l2_loss": 0.01,
                    "prob_hit_loss": 0.01,
                    "normal_loss": 1.0,
                    "multi_view_loss": 0.01,
                    "sky_ray_loss": 1.0,
                },
                include_depth_loss_scene_center_weight=True,
                compute_normals=False,  # This currently does not work, the input to the network need changing to work with autograd
                eval_num_rays_per_chunk=1024,
                scene_center_weight_exp=3.0,
                scene_center_weight_include_z=False,  # only xy
                mask_to_circumference=False,
                inverse_depth_weight=False,
                log_depth=False,
            ),
            visibility_train_sampler=VMFDDFSamplerConfig(
                num_samples_on_sphere=8,
                num_rays_per_sample=128,  # 8 * 128 = 1024 rays per batch
                only_sample_upper_hemisphere=True,
                concentration=20.0,
            ),
            # visibility_ckpt_path=Path('/workspace/outputs/unnamed/ddf/2023-06-20_085448/'),
            # visibility_ckpt_step=20000,
            # reni_neus_ckpt_path=Path('/workspace/outputs/unnamed/reni-neus/2023-08-02_102036/'),
            # reni_neus_ckpt_step=55000,
            visibility_field_radius="AABB",  # From dataparser
            visibility_accumulation_mask_threshold=0.0,  # 0.0 means no mask as mask = accum > threshold
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=100001),
            },
            "visibility_sigmoid": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=4000, lr_final=1e-4, max_steps=100001),
            },
            "ddf_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for RENI-NeuS.",
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
                    scene="lk2", auto_scale_poses=True, crop_to_equal_size=True, mask_source="original"
                ),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
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
