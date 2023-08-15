# %%|
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

def get_reni_image(model, outputs, batch, R):
    idx = torch.tensor(batch["image_idx"], device=model.device)
    W = 512
    H = W // 2
    D = get_directions(W).to(model.device)  # [B, H*W, 3]
    envmap, _ = model.get_illumination_field()(idx, None, D, R, "envmap")
    envmap = envmap.reshape(1, H, W, 3).squeeze(0)
    return envmap

# setup config
test_mode = 'val'
world_size = 1
local_rank = 0
device = 'cuda:0'

reni_neus_ckpt_path = '/workspace/outputs/unnamed/reni-neus/2023-06-07_141907/' # model without vis
step_reni = 85000

ckpt_reni = torch.load(reni_neus_ckpt_path + '/nerfstudio_models' + f'/step-{step_reni:09d}.ckpt', map_location=device)

# vision model checkpoint
vision_ckpt_path = '/workspace/outputs/unnamed/ddf/2023-06-20_085448/' # model without vis
step = 20000

ckpt = torch.load(vision_ckpt_path + '/nerfstudio_models' + f'/step-{step:09d}.ckpt', map_location=device)
vision_model_dict = {}
for key in ckpt['pipeline'].keys():
    if key.startswith('_model.'):
        vision_model_dict[key[7:]] = ckpt['pipeline'][key]

# %%
# update ckpt to include vision model
for key in vision_model_dict.keys():
    ckpt_reni['pipeline']['_model.visibility_field.' + key] = vision_model_dict[key]

# # save reni_neus_model_dict
torch.save(ckpt_reni, '/workspace/outputs/nerfstudio_models' + f'/step-{step_reni:09d}.ckpt')
# %%
