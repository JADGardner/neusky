"""
PNerF configuration file.
"""
from pathlib import Path

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, MultiStepSchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from reni_neus.data.dataparsers.pnerf_dataparser import PNeRFDataParserConfig
from reni_neus.data.datasets.pnerf_dataset import PNeRFDataset
from reni_neus.models.pnerf_model import PNeRFModelConfig

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
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
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
