import torch
import yaml
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import plotly.graph_objects as go
from torch.utils.data import Dataset

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSR, NeRFOSRDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaDataManager
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.colormaps import apply_depth_colormap
from nerfstudio.field_components.encodings import SHEncoding, NeRFEncoding
import tinycudann as tcnn

from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig, RENINeuSFactoModel
from reni_neus.utils.utils import get_directions, get_sineweight, look_at_target, random_points_on_unit_sphere
from reni_neus.illumination_fields.reni_field import RENIField
from reni_neus.data.reni_neus_datamanager import RENINeuSDataManagerConfig, RENINeuSDataManager
from reni_neus.configs.reni_neus_config import RENINeuS as RENINeuSMethodSpecification, DirectionalDistanceField
from reni_neus.configs.reni_neus_config import RENINeuS
from reni_neus.illumination_fields.environment_map import EnvironmentMapConfig

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Return 3D rotation matrix for rotating around the given axis by the given angle.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    # convert to pytorch
    rotation = torch.from_numpy(rotation).float()
    return rotation

# setup config
test_mode = 'val'
world_size = 1
local_rank = 0
device = 'cuda:0'

reni_neus_ckpt_path = '/workspace/outputs/unnamed/reni-neus/2023-06-07_141907/' # model without vis
step = 85000

ckpt = torch.load(reni_neus_ckpt_path + '/nerfstudio_models' + f'/step-{step:09d}.ckpt', map_location=device)
reni_neus_model_dict = {}
for key in ckpt['pipeline'].keys():
    if key.startswith('_model.'):
        reni_neus_model_dict[key[7:]] = ckpt['pipeline'][key]

# vision model checkpoint
vision_ckpt_path = '/workspace/outputs/unnamed/ddf/2023-06-20_085448/' # model without vis
step = 20000

ckpt = torch.load(vision_ckpt_path + '/nerfstudio_models' + f'/step-{step:09d}.ckpt', map_location=device)
vision_model_dict = {}
for key in ckpt['pipeline'].keys():
    if key.startswith('_model.'):
        vision_model_dict[key[7:]] = ckpt['pipeline'][key]

# update reni_neus_model_dict with vision_model_dict
for key in vision_model_dict.keys():
    reni_neus_model_dict['visibility_field.' + key] = vision_model_dict[key]

datamanager: RENINeuSDataManager = RENINeuS.config.pipeline.datamanager.setup(
    device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank,
)
datamanager.to(device)

# instantiate model with config with vis
model = RENINeuS.config.pipeline.model.setup(
    scene_box=datamanager.train_dataset.scene_box,
    num_train_data=len(datamanager.train_dataset),
    num_val_data=datamanager.num_val,
    num_test_data=datamanager.num_test,
    test_mode=test_mode,
)

model.to(device)
model.load_state_dict(reni_neus_model_dict)
model.eval()

print('Model loaded')

ray_bundle, batch = datamanager.fixed_indices_eval_dataloader.get_data_from_image_idx(3)
model.config.use_visibility = True
model.render_shadow_map = False
model.visibility_threshold = 0.1
model.shadow_map_threshold.value = 0.1
model.config.fix_test_illumination_directions = True
model.accumulation_mask_threshold.value = 0.7 
model.render_illumination_animation(ray_bundle=ray_bundle,
                                    batch=None,
                                    num_frames=100,
                                    fps=20,
                                    visibility_threshold=0.1,
                                    output_path='/workspace/outputs/renders/')