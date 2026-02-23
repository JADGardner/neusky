"""
NeuSky configuration for synthetic multi-illumination datasets.

Uses the custom dataparser that reads transforms.json + PNG images
from train/validation/test split directories.
"""
from pathlib import Path

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)

from neusky.data.dataparsers.custom_neusky_dataparser import CustomNeuskyDataparserConfig
from neusky.models.neusky_model import NeuSkyFactoModelConfig
from neusky.pipelines.neusky_pipeline import NeuSkyPipelineConfig
from neusky.data.datamanagers.neusky_datamanager import NeuSkyDataManagerConfig
from neusky.fields.sdf_albedo_field import SDFAlbedoFieldConfig
from neusky.models.ddf_model import DDFModelConfig
from neusky.fields.directional_distance_field import DirectionalDistanceFieldConfig
from neusky.model_components.ddf_sampler import VMFDDFSamplerConfig
from neusky.data.neusky_pixel_sampler import NeuSkyPixelSamplerConfig


NeuSkySynthetic = MethodSpecification(
    config=TrainerConfig(
        method_name="neusky-synthetic",
        experiment_name="synthetic",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100002,
        steps_per_save=5000,
        steps_per_eval_all_images=100000,
        max_num_iterations=100001,
        mixed_precision=False,
        pipeline=NeuSkyPipelineConfig(
            test_mode=None,
            stop_sdf_gradients=False,
            datamanager=NeuSkyDataManagerConfig(
                dataparser=CustomNeuskyDataparserConfig(
                    center_method_sfm=True,
                    center_method="none",
                    auto_scale_poses=False,
                    scene_scale=1.0,
                    mask_vegetation=False,
                    include_sidewalk_in_ground_mask=True,
                ),
                train_num_images_to_sample_from=-1,
                train_num_times_to_repeat_images=-1,
                pixel_sampler=NeuSkyPixelSamplerConfig(),
                images_on_gpu=True,
                masks_on_gpu=True,
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=256,
                camera_res_scale_factor=0.25,
            ),
            model=NeuSkyFactoModelConfig(
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
                    axis_of_invariance="z",
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
                    num_directions=256,
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
                    "sdf_level_set_visibility_loss": True,
                    "interlevel_loss": True,
                    "sky_pixel_loss": {"enabled": True, "cosine_weight": 0.1},
                    "hashgrid_density_loss": {
                        "enabled": True,
                        "grid_resolution": 10,
                    },
                    "ground_plane_loss": True,
                    "visibility_sigmoid_loss": {
                        "visibility_threshold_method": "learnable",
                        "optimise_sigmoid_bias": True,
                        "optimise_sigmoid_scale": False,
                        "target_min_bias": 0.1,
                        "target_max_scale": 25,
                        "steps_until_min_bias": 50000,
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
                    "sdf_level_set_visibility_loss": 1.0,
                    "interlevel_loss": 1.0,
                    "sky_pixel_loss": 1.0,
                    "hashgrid_density_loss": 1e-4,
                    "ground_plane_loss": 0.1,
                    "visibility_sigmoid_loss": 0.01,
                },
                eval_latent_optimizer={
                    "eval_latents": {
                        "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
                        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=250),
                    },
                },
                eval_latent_optimise_method="per_image",
                eval_latent_sample_region="full_image",
                illumination_field_ckpt_path=Path("/workspace/reni_models/reni_plus_plus_models/latent_dim_100/"),
                illumination_field_ckpt_step=50000,
                fix_test_illumination_directions=True,
                eval_num_rays_per_chunk=256,
                use_visibility=True,
                fit_visibility_field=True,
                sdf_to_visibility_stop_gradients="depth",
                only_upperhemisphere_visibility=True,
                scene_contraction_order="L2",
                collider_shape="sphere",
                background_model="none",
            ),
            visibility_field=DDFModelConfig(
                ddf_field=DirectionalDistanceFieldConfig(
                    ddf_type="ddf",
                    position_encoding_type="hash",
                    direction_encoding_type="nerf",
                    conditioning="FiLM",
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
                compute_normals=False,
                eval_num_rays_per_chunk=1024,
                scene_center_weight_exp=3.0,
                scene_center_weight_include_z=False,
                mask_to_circumference=False,
                inverse_depth_weight=False,
                log_depth=False,
            ),
            visibility_train_sampler=VMFDDFSamplerConfig(
                num_samples_on_sphere=4,
                num_rays_per_sample=64,
                only_sample_upper_hemisphere=True,
                concentration=20.0,
            ),
            visibility_field_radius="AABB",
            visibility_accumulation_mask_threshold=0.0,
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
    description="NeuSky config for synthetic multi-illumination datasets.",
)
