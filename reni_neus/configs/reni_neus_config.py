"""
RENI-NeuS configuration file.
"""
from pathlib import Path

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)

from reni_neus.data.nerfosr_cityscapes_dataparser import NeRFOSRCityScapesDataParserConfig
from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig
from reni_neus.pipelines.reni_neus_pipeline import RENINeuSPipelineConfig
from reni_neus.data.reni_neus_datamanager import RENINeuSDataManagerConfig
from reni_neus.fields.sdf_albedo_field import SDFAlbedoFieldConfig
from reni_neus.models.ddf_model import DDFModelConfig
from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig
from reni_neus.model_components.ddf_sampler import VMFDDFSamplerConfig


RENINeuS = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-neus",
        experiment_name="reni-neus",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
        # load_dir=Path("/workspace/outputs/unnamed/reni-neus/2023-08-09_075320/nerfstudio_models"),
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
                    conditioning='Attention',
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z", # Nerfstudio world space is z-up
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
                    num_directions=300,
                    apply_random_rotation=True,
                    remove_lower_hemisphere=False,
                ),
                illumination_field_ckpt_path=Path("outputs/unnamed/reni/2023-08-07_154753/"),
                illumination_field_ckpt_step=50000,
                fix_test_illumination_directions=True,
                eval_num_rays_per_chunk=256,
                illumination_field_prior_loss_weight=1e-7,
                illumination_field_cosine_loss_weight=1e-1,
                illumination_field_loss_weight=1.0,
                fg_mask_loss_mult=1.0,
                background_model="none",
                use_average_appearance_embedding=False,
                render_only_albedo=False,
                include_hashgrid_density_loss=True,
                hashgrid_density_loss_weight=1e-4,
                hashgrid_density_loss_sample_resolution=10,
                include_ground_plane_normal_alignment=True,
                ground_plane_normal_alignment_multi=0.1,
                use_visibility=False,
                visibility_threshold=(1.0, 0.1), # "learnable", float, tuple(start, end) ... tuple will exponentially decay from start to end
                steps_till_min_visibility_threshold=10000,
                only_upperhemisphere_visibility=True,
                scene_contraction_order="L2", # L2, Linf
                collider_shape="sphere",
            ),
            visibility_field=DDFModelConfig( # DDFModelConfig or None
                
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
              compute_normals=False, # This currently does not work, the input to the network needs changing to work with autograd
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
          visibility_train_sampler=VMFDDFSamplerConfig(
              concentration=20.0,
          ),
          # visibility_ckpt_path=Path('/workspace/outputs/unnamed/ddf/2023-06-20_085448/'),
          # visibility_ckpt_step=20000,
          # reni_neus_ckpt_path=Path('/workspace/outputs/unnamed/reni-neus/2023-08-02_102036/'),
          # reni_neus_ckpt_step=55000,
          fit_visibility_field=False, # if true, train visibility field, else visibility is static
          visibility_field_radius="AABB",
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
            "visibility_threshold": {
                "optimizer": AdamOptimizerConfig(lr=1e-6, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100001),
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


# DirectionalDistanceField = MethodSpecification(
#     config=TrainerConfig(
#         method_name="ddf",
#         experiment_name="ddf",
#         steps_per_eval_image=500,
#         steps_per_eval_batch=100000,
#         steps_per_save=1000,
#         steps_per_eval_all_images=1000,  # set to a very large model so we don't eval with all images
#         max_num_iterations=20001,
#         mixed_precision=False,
#         pipeline=DDFPipelineConfig(
#             datamanager=DDFDataManagerConfig(
#                 num_test_images_to_generate=8,
#                 test_image_cache_dir=Path("outputs/ddf/cache/"),
#                 accumulation_mask_threshold=0.7,
#                 training_data_type="rand_pnts_on_sphere", # "rand_pnts_on_sphere", "single_camera", "all_cameras"
#                 train_data_idx=5, # idx if using single_camera for training data
#                 ddf_sampler=VMFDDFSamplerConfig(
#                     num_samples_on_sphere=8,
#                     num_rays_per_sample=128, # 8 * 128 = 1024 rays per batch
#                     only_sample_upper_hemisphere=True,
#                     concentration=20.0,
#                 ),
#                 num_of_sky_ray_samples=256,
#             ),
#             model=DDFModelConfig(
#                 ddf_field=DirectionalDistanceFieldConfig(
#                     ddf_type="ddf", # pddf
#                     position_encoding_type="hash", # none, hash, nerf, sh
#                     direction_encoding_type="nerf",
#                     conditioning="Attention", # FiLM, Concat, Attention
#                     termination_output_activation="sigmoid",
#                     probability_of_hit_output_activation="sigmoid",
#                     hidden_layers=5,
#                     hidden_features=256,
#                     mapping_layers=5,
#                     mapping_features=256,
#                     num_attention_heads=8,
#                     num_attention_layers=6,
#                     predict_probability_of_hit=False,
#                 ),
#                 depth_loss="L1", # L2, L1, Log_Loss
#                 include_sdf_loss=True,
#                 include_multi_view_loss=True,
#                 include_sky_ray_loss=True,
#                 multi_view_loss_stop_gradient=False,
#                 include_depth_loss_scene_center_weight=True,
#                 compute_normals=False, # This currently does not work, the input to the network need changing to work with autograd
#                 sdf_loss_mult=100.0,
#                 multi_view_loss_mult=0.1,
#                 sky_ray_loss_mult=1.0,
#                 depth_loss_mult=20.0,
#                 prob_hit_loss_mult=1.0,
#                 normal_loss_mult=1.0,
#                 eval_num_rays_per_chunk=1024,
#                 scene_center_weight_exp=3.0,
#                 scene_center_use_xyz=False, # only xy
#                 mask_depth_to_circumference=False, # force depth under mask to circumference of ddf (not implemented)
#             ),
#             reni_neus_ckpt_path=Path("outputs/unnamed/reni-neus/2023-08-09_150349"),
#             reni_neus_ckpt_step=85000,
#             ddf_radius="AABB",
#         ),
#         optimizers={
#             "ddf_field": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Directional Distance Field.",
# )


# NeRFactoNeRFOSR = MethodSpecification(
#     config=TrainerConfig(
#         method_name="nerfacto-nerfosr",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=100000,
#         mixed_precision=True,
#         pipeline=VanillaPipelineConfig(
#             datamanager=RENINeuSDataManagerConfig(
#                 dataparser=NeRFOSRCityScapesDataParserConfig(
#                     scene="lk2",
#                     auto_scale_poses=False,
#                 ),
#                 train_num_rays_per_batch=2048,
#                 eval_num_rays_per_batch=2048,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3", optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3)
#                 ),
#             ),
#             model=NerfactoModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#                 num_nerf_samples_per_ray=128,
#                 num_proposal_samples_per_ray=(512, 256),
#                 hidden_dim=128,
#                 hidden_dim_color=128,
#                 hidden_dim_transient=128,
#                 max_res=3000,
#                 proposal_weights_anneal_max_num_iters=5000,
#                 log2_hashmap_size=21,
#             ),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#             "fields": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Nerfacto NeRF-OSR.",
# )


# NeusFactoNeRFOSR = MethodSpecification(
#     config=TrainerConfig(
#         method_name="neus-facto-nerfosr",
#         steps_per_eval_image=5000,
#         steps_per_eval_batch=5000,
#         steps_per_save=2000,
#         steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
#         max_num_iterations=20001,
#         mixed_precision=False,
#         pipeline=VanillaPipelineConfig(
#             datamanager=RENINeuSDataManagerConfig(
#                 dataparser=NeRFOSRCityScapesDataParserConfig(
#                     scene="lk2",
#                     auto_scale_poses=False,
#                 ),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
#                 ),
#             ),
#             model=NeuSFactoModelConfig(
#                 # proposal network allows for signifanctly smaller sdf/color network
#                 sdf_field=SDFFieldConfig(
#                     use_grid_feature=True,
#                     num_layers=2,
#                     num_layers_color=2,
#                     hidden_dim=256,
#                     bias=0.5,
#                     beta_init=0.8,
#                     use_appearance_embedding=False,
#                 ),
#                 background_model="mlp",
#                 eval_num_rays_per_chunk=2048,
#             ),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": MultiStepSchedulerConfig(max_steps=20001, milestones=(10000, 1500, 18000)),
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
#             },
#             "field_background": {
#                 "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Neusfacto nerfosr.",
# )
