"""
RENI-NeuS configuration file.
"""
from pathlib import Path

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import get_external_methods

from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.generfacto import GenerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

from reni_neus.data.dataparsers.nerfosr_cityscapes_dataparser import NeRFOSRCityScapesDataParserConfig
from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig
from reni_neus.pipelines.reni_neus_pipeline import RENINeuSPipelineConfig
from reni_neus.data.datamanagers.reni_neus_datamanager import RENINeuSDataManagerConfig
from reni_neus.fields.sdf_albedo_field import SDFAlbedoFieldConfig
from reni_neus.models.ddf_model import DDFModelConfig
from reni_neus.fields.directional_distance_field import DirectionalDistanceFieldConfig
from reni_neus.model_components.ddf_sampler import VMFDDFSamplerConfig
from reni_neus.data.reni_neus_pixel_sampler import RENINeuSPixelSamplerConfig
from reni_neus.data.dataparsers.pnerf_dataparser import PNeRFDataParserConfig
from reni_neus.data.datasets.pnerf_dataset import PNeRFDataset
from reni_neus.models.pnerf_model import PNeRFModelConfig

# PNeRF = MethodSpecification(
#     config=TrainerConfig(
#         method_name="pnerf",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=VanillaPipelineConfig(
#             datamanager=VanillaDataManagerConfig(
#                 dataparser=PNeRFDataParserConfig(
#                     data=Path("/workspace/data/pnerf_data/shakespeare"),
#                     images_path=Path("/workspace/data/pnerf_data/shakespeare/images/"),
#                     # masks_path=Path("/workspace/data/pnerf_data/shakespeare/masks/"),
#                     center_method="focus",
#                     downscale_factor=2,
#                 ),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#                 # camera_optimizer=CameraOptimizerConfig(
#                 #     mode="SO3xR3",
#                 #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
#                 #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
#                 # ),
#             ),
#             model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, disable_scene_contraction=True),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Nerfacto NeRF-OSR.",
# )

# PNeRF = MethodSpecification(
#     config=TrainerConfig(
#         method_name="pnerf",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=VanillaPipelineConfig(
#             datamanager=VanillaDataManagerConfig(
#                 _target=VanillaDataManager[SDFDataset],
#                 # dataparser=ColmapDataParserConfig(
#                 #     data=Path("/workspace/data/pnerf_data/shakespeare"),
#                 #     images_path=Path("/workspace/data/pnerf_data/shakespeare/images/"),
#                 #     # masks_path=Path("/workspace/data/pnerf_data/shakespeare/masks/"),
#                 #     center_method="focus",
#                 #     downscale_factor=1,
#                 # ),
#                 dataparser=SDFStudioDataParserConfig(
#                     data=Path("/workspace/data/sdfstudio-demo-data/dtu-scan65"),
#                     skip_every_for_val_split=10,
#                 ),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3",
#                     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
#                     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
#                 ),
#             ),
#             model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, background_color="random"),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Nerfacto NeRF-OSR.",
# )


PNeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="pnerf",
        steps_per_eval_image=3000,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[PNeRFDataset],
                dataparser=PNeRFDataParserConfig(
                    data=Path("data/pnerf_data/shakespeare"),
                    images_path=Path("data/pnerf_data/shakespeare/images/"),
                    masks_path=Path("data/pnerf_data/shakespeare/masks/"),
                    center_method="focus",
                    downscale_factor=1,
                ),
                # dataparser=SDFStudioDataParserConfig(
                #     data=Path("/workspace/data/sdfstudio-demo-data/dtu-scan65"),
                #     skip_every_for_val_split=10,
                # ),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                # ),
            ),
            model=PNeRFModelConfig(
                # proposal network allows for significantly smaller sdf/color network
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.3,
                    beta_init=0.5,
                    use_appearance_embedding=False,
                    inside_outside=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=2048,
                fg_mask_loss_mult=10.0,
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
    description="Base config for Nerfacto NeRF-OSR.",
)
