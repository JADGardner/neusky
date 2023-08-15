import argparse
import os
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
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.utils.colormaps import apply_depth_colormap
from nerfstudio.field_components.encodings import SHEncoding, NeRFEncoding
import tinycudann as tcnn

from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig, RENINeuSFactoModel
from reni_neus.utils.utils import get_directions, get_sineweight, look_at_target, random_points_on_unit_sphere
from reni_neus.data.reni_neus_datamanager import RENINeuSDataManagerConfig, RENINeuSDataManager
from reni_neus.configs.reni_neus_config import RENINeuS as RENINeuSMethodSpecification, DirectionalDistanceField
from reni_neus.configs.reni_neus_config import RENINeuS
from reni_neus.utils.utils import find_nerfstudio_project_root, rot_z

def main(args):
    project_root = find_nerfstudio_project_root(Path(os.getcwd()))
    os.chdir(project_root)

    test_mode = 'val'
    world_size = 1
    local_rank = 0

    reni_neus_config = RENINeuS
    reni_neus_config.config.pipeline.visibility_ckpt_path = Path(args.visibility_ckpt_path)
    reni_neus_config.config.pipeline.visibility_ckpt_step = args.visibility_ckpt_step
    reni_neus_config.config.pipeline.reni_neus_ckpt_path = Path(args.reni_neus_ckpt_path)
    reni_neus_config.config.pipeline.reni_neus_ckpt_step = args.reni_neus_ckpt_step
    reni_neus_config.config.pipeline.model.use_visibility = True
    reni_neus_config.config.pipeline.model.visibility_threshold = args.visibility_threshold

    pipeline = reni_neus_config.config.pipeline.setup(device=args.device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)
    datamanager = pipeline.datamanager
    model = pipeline.model
    model = model.eval()

    ray_bundle, batch = datamanager.fixed_indices_eval_dataloader.get_data_from_image_idx(3)
    model.render_illumination_animation(ray_bundle=ray_bundle, 
                                        batch=batch,
                                        num_frames=args.num_frames,
                                        fps=args.fps,
                                        visibility_threshold=args.visibility_threshold,
                                        output_path='outputs/renders',
                                        render_final_animation=args.render_final_animation,
                                        start_frame=args.start_frame,
                                        end_frame=args.end_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process command line inputs for rendering.")
    parser.add_argument('--device', default='cuda:1', help='Specify device for computation.')
    parser.add_argument('--visibility_ckpt_path', required=True, help='Path to visibility checkpoint.')
    parser.add_argument('--visibility_ckpt_step', type=int, required=True, help='Visibility checkpoint step.')
    parser.add_argument('--reni_neus_ckpt_path', required=True, help='Path to reni-neus checkpoint.')
    parser.add_argument('--reni_neus_ckpt_step', type=int, required=True, help='Reni-neus checkpoint step.')
    parser.add_argument('--visibility_threshold', type=float, default=0.01, help='Visibility threshold.')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for the animation.')
    parser.add_argument('--fps', type=float, default=20.0, help='Frames per second for the animation.')
    parser.add_argument('--render_final_animation', action='store_false', help='Render final animation if specified.')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame for animation.')
    parser.add_argument('--end_frame', type=int, default=100, help='End frame for animation.')
    args = parser.parse_args()

    main(args)


# python3 reni_neus/reni_neus/scripts/animation_generation.py --device cuda:0 --visibility_ckpt_path outputs/unnamed/ddf/2023-08-11_065642/ --visibility_ckpt_step 20000 --reni_neus_ckpt_path outputs/unnamed/reni-neus/2023-08-09_150349/ --reni_neus_ckpt_step 85000 --start_frame 0 --end_frame 25
