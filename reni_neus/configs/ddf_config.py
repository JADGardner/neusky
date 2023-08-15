"""
Directional Distance Field configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
)

from reni_neus.models.ddf_model import DDFModelConfig
from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig
from reni_neus.pipelines.ddf_pipeline import DDFPipelineConfig
from reni_neus.data.ddf_datamanager import DDFDataManagerConfig
from reni_neus.model_components.ddf_sampler import VMFDDFSamplerConfig

DirectionalDistanceField = MethodSpecification(
    config=TrainerConfig(
        method_name="ddf",
        experiment_name="ddf",
        steps_per_eval_image=500,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=DDFPipelineConfig(
            datamanager=DDFDataManagerConfig(
                num_test_images_to_generate=8,
                test_image_cache_dir=Path("outputs/ddf/cache/"),
                accumulation_mask_threshold=0.7,
                training_data_type="rand_pnts_on_sphere", # "rand_pnts_on_sphere", "single_camera", "all_cameras"
                train_data_idx=5, # idx if using single_camera for training data
                ddf_sampler=VMFDDFSamplerConfig(
                    num_samples_on_sphere=8,
                    num_rays_per_sample=128, # 8 * 128 = 1024 rays per batch
                    only_sample_upper_hemisphere=True,
                    concentration=20.0,
                ),
                num_of_sky_ray_samples=256,
            ),
            model=DDFModelConfig(
                ddf_field=DirectionalDistanceFieldConfig(
                    ddf_type="ddf", # pddf
                    position_encoding_type="hash", # none, hash, nerf, sh
                    direction_encoding_type="nerf",
                    conditioning="Attention", # FiLM, Concat, Attention
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
                depth_loss="L1", # L2, L1, Log_Loss
                include_sdf_loss=True,
                include_multi_view_loss=True,
                include_sky_ray_loss=True,
                multi_view_loss_stop_gradient=False,
                include_depth_loss_scene_center_weight=True,
                compute_normals=False, # This currently does not work, the input to the network need changing to work with autograd
                sdf_loss_mult=100.0,
                multi_view_loss_mult=0.1,
                sky_ray_loss_mult=1.0,
                depth_loss_mult=20.0,
                prob_hit_loss_mult=1.0,
                normal_loss_mult=1.0,
                eval_num_rays_per_chunk=1024,
                scene_center_weight_exp=3.0,
                scene_center_use_xyz=False, # only xy
                mask_depth_to_circumference=False, # force depth under mask to circumference of ddf (not implemented)
            ),
            reni_neus_ckpt_path=Path("outputs/unnamed/reni-neus/2023-08-09_150349"),
            reni_neus_ckpt_step=85000,
            ddf_radius="AABB",
        ),
        optimizers={
            "ddf_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Directional Distance Field.",
)

