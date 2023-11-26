# %%
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
import pyexr
import math
from datetime import datetime

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSR, NeRFOSRDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaDataManager
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.utils.colormaps import apply_depth_colormap
from nerfstudio.field_components.encodings import SHEncoding, NeRFEncoding
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.utils import colormaps
from nerfstudio.utils.io import load_from_json
from nerfstudio.cameras.cameras import Cameras, CameraType
import tinycudann as tcnn

from reni_neus.models.reni_neus_model import RENINeuSFactoModelConfig, RENINeuSFactoModel
from reni_neus.utils.utils import look_at_target, random_points_on_unit_sphere
from reni_neus.data.datamanagers.reni_neus_datamanager import RENINeuSDataManagerConfig, RENINeuSDataManager
from reni_neus.configs.ddf_config import DirectionalDistanceField
from reni_neus.configs.reni_neus_config import RENINeuS
from reni_neus.utils.utils import find_nerfstudio_project_root, rot_z

# teaser figure lighting 
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB

def get_envmap(illumination_idx, rotation=None):
    with torch.no_grad():
        illumination_latents, scales = model.get_illumination_field()
        ray_samples = model.equirectangular_sampler.generate_direction_samples()
        ray_samples = ray_samples.to(model.device)
        if rotation is not None:
            if len(rotation.shape) == 2:
                rotation = rotation
            elif len(rotation.shape) == 3:
                rotation = rotation[ray_samples.camera_indices[:, 0]]
        ray_samples.camera_indices = torch.ones_like(ray_samples.camera_indices) * illumination_idx
        illumination_field_outputs = model.illumination_field(ray_samples=ray_samples,
                                                              latent_codes=illumination_latents[ray_samples.camera_indices[:, 0]],
                                                              scale=scales[ray_samples.camera_indices[:, 0]],
                                                              rotation=rotation
        )

        hdr_envmap = illumination_field_outputs[RENIFieldHeadNames.RGB]
        hdr_envmap = model.illumination_field.unnormalise(hdr_envmap)  # N, 3
        ldr_envmap = linear_to_sRGB(hdr_envmap)  # N, 3
        # reshape to H, W, 3
        height = model.equirectangular_sampler.height
        width = model.equirectangular_sampler.width
        ldr_envmap = ldr_envmap.reshape(height, width, 3)
        return ldr_envmap

project_root = find_nerfstudio_project_root(Path(os.getcwd()))
# set current working directory to nerfstudio project root
os.chdir(project_root)

# setup config
test_mode = 'test'
world_size = 1
local_rank = 0
device = 'cuda:0'

scene = 'site1'

reni_neus_config = RENINeuS
reni_neus_config.config.load_dir = Path('/workspace/reni_neus/models/ablations/stronger_sdf_ddf_weighting/nerfstudio_models')
reni_neus_config.config.load_step = 100000
reni_neus_config.config.pipeline.datamanager.dataparser.scene = scene


if scene == 'site1':
    reni_neus_config.config.pipeline.datamanager.dataparser.session_holdout_indices=[0, 0, 0, 0, 0]
elif scene == 'site2':
    reni_neus_config.config.pipeline.datamanager.dataparser.session_holdout_indices=[1, 2, 2, 7, 9]
elif scene == 'site3':
    reni_neus_config.config.pipeline.datamanager.dataparser.session_holdout_indices=[0, 6, 6, 2, 11]

trainer = reni_neus_config.config.setup(local_rank=local_rank, world_size=world_size)
trainer.setup(test_mode=test_mode)
pipeline = trainer.pipeline
datamanager = pipeline.datamanager
model = pipeline.model
model = model.eval()

# load camera poses
# %%
scene = 'site1'
keyframe_for_illumination_rotation = 4
illumination_idx = 143
seconds_for_rotation = 8
camera_poses_path = f'/users/jadg502/scratch/code/nerfstudio/reni_neus/publication/{scene}_demo_path_2.json'
meta = load_from_json(Path(camera_poses_path))
fps = meta['fps']

assert keyframe_for_illumination_rotation < len(meta['keyframes'])

# create folder in /workspace/reni_neus/publication/animations/{scene}_datetime
# save all rendered images in this folder
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f'/users/jadg502/scratch/code/nerfstudio/reni_neus/publication/animations/{scene}_{datetime_str}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

render_height = meta['render_height']
render_width = meta['render_width']
render_height = 1080 # 1080, 144
render_width = 1920 # 1920, 256
cx = render_width / 2.0
cy = render_height / 2.0
fov = meta['keyframes'][0]['fov']
aspect = render_width / render_height
fx = render_width / (2 * math.tan(math.radians(fov) / 2))
fy = fx
c2w = torch.eye(4)[:3, :4]

camera = Cameras(camera_to_worlds=c2w,
                 fy=fy,
                 fx=fx,
                 cx=cx,
                 cy=cy,
                 camera_type=CameraType.PERSPECTIVE)

base_ray_bundle = datamanager.train_dataset.cameras[0].generate_rays(0)
base_ray_bundle = base_ray_bundle.to(device)

model.viewing_training_image = True

def save_model_output(model_output, frame_num, c2w, path, rotation = None):
    rendered_image = model_output['rgb'].detach().cpu().numpy()
    normal = model_output['normal'].cpu()
    normal = normal @ c2w[:3, :3]
    normal_scaled = (normal + 1) / 2
    normal_scaled[normal.norm(dim=-1) < 0.5] = 1
    normal_scaled = normal_scaled.detach().cpu().numpy()
    normal = model_output['normal'].detach().cpu().numpy()
    normal_world = (normal + 1.0) / 2.0 # world space normal mapped to [0, 1]
    # clamp to [0, 1]
    normal_world = np.clip(normal_world, 0, 1)
    depth = colormaps.apply_depth_colormap(
          model_output['depth'],
          accumulation=model_output['accumulation'],
    )
    depth = depth.detach().cpu().numpy()
    albedo = model_output['albedo'].detach().cpu().numpy()
    ldr_envmap = get_envmap(illumination_idx, rotation).detach().cpu().numpy()
    # now save everything
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_render.png', rendered_image)
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_normal.png', normal_world)
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_normal_scaled.png', normal_scaled)
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_depth.png', depth)
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_albedo.png', albedo)
    plt.imsave(f'{path}/frame_{str(frame_num).zfill(3)}_ldr_envmap.png', ldr_envmap)

# Add a parameter for an optional output folder
def process_scene(scene, camera_poses_path, illumination_idx, keyframe_for_illumination_rotation, seconds_for_rotation, optional_output_folder=None):
    meta = load_from_json(Path(camera_poses_path))
    fps = meta['fps']

    assert keyframe_for_illumination_rotation < len(meta['keyframes'])

    # Use the provided output folder or create a new one based on 'scene' and 'datetime'
    if optional_output_folder:
        folder_path = optional_output_folder
    else:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = f'/users/jadg502/scratch/code/nerfstudio/reni_neus/publication/animations/{scene}_{datetime_str}'
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Your existing setup code here...

    illumination_rotated = False
    keyframe_matrix = np.fromstring(meta['keyframes'][keyframe_for_illumination_rotation]['matrix'].strip('[]'), sep=',').reshape((4, 4)).transpose()
    
    for frame_idx, frame in enumerate(meta['camera_path']):       
        camera_to_world = torch.from_numpy(np.array(frame['camera_to_world']).reshape((4, 4))).to(torch.float32)
        camera.camera_to_worlds = camera_to_world[:3, :4]
        ray_bundle = camera.generate_rays(0)
        ray_bundle = ray_bundle.to(device)
        ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * illumination_idx
        if np.allclose(camera_to_world, keyframe_matrix, rtol=1e-03, atol=1e-02) and illumination_rotated == False:
        # perform illumination rotation
            for i in range(fps*seconds_for_rotation):
                output_file_path = f'{folder_path}/frame_{str(frame_idx + i).zfill(3)}_render.png'
                if os.path.exists(output_file_path):
                    print(f'Skipping idx {frame_idx + i}')
                    continue
                # we do a full 360 degree rotation, need in radians
                angle = 2 * math.pi * i / (fps*seconds_for_rotation)
                angle_degrees = angle * 180 / math.pi
                rotation = rot_z(torch.tensor(angle)).to(device)
                model_output = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle=ray_bundle, rotation=rotation)
                print(f'Rendering frame_idx: {frame_idx + i}, with rotation angle: {angle_degrees}')
                save_model_output(model_output, frame_idx + i, camera_to_world[:3, :4], folder_path, rotation)
            illumination_rotated = True
        else:
            if illumination_rotated == True:
                frame_idx = frame_idx + fps*seconds_for_rotation - 1
            # Check if the file for the current idx already exists
            output_file_path = f'{folder_path}/frame_{str(frame_idx).zfill(3)}_render.png'
            if os.path.exists(output_file_path):
                print(f'Skipping idx {frame_idx}')
                continue
            # just render the frame normally
            print(f'Rendering frame_idx: {frame_idx}')
            model_output = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle=ray_bundle)
            save_model_output(model_output, frame_idx, camera_to_world[:3, :4], folder_path)

# Call with an optional output folder
process_scene(scene, camera_poses_path, illumination_idx, keyframe_for_illumination_rotation, seconds_for_rotation)
model.viewing_training_image = False
# %%
import os
import cv2
import imageio

def create_animation(folder, image_type, fps, format='gif'):
    """
    Creates an animation from images of a specific type in the given folder.

    :param folder: Path to the folder containing the images.
    :param image_type: Type of the image (e.g., 'render', 'albedo', 'normal').
    :param fps: Frames per second for the output animation.
    :param format: Output format of the animation ('gif' or 'mp4').
    """
    # List all files in the folder
    files = sorted([f for f in os.listdir(folder) if f.endswith(f"{image_type}.png")])

    if not files:
        print("No images found for the specified type.")
        return

    # Read images and store them in a list
    images = []
    for file in files:
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Failed to read image: {img_path}")

    # Create animation
    output_path = os.path.join(folder, f"animation_{image_type}.{format}")
    if format == 'gif':
        imageio.mimsave(output_path, images, fps=fps)
    elif format == 'mp4':
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
        for img in images:
            writer.append_data(img)
        writer.close()
    else:
        print("Unsupported format. Please choose 'gif' or 'mp4'.")

    print(f"Animation saved at {output_path}")

folder_path = '/workspace/reni_neus/publication/animations/site2_20231124_091227'
create_animation(folder_path, 'render', 24, 'mp4')
create_animation(folder_path, 'normal', 24, 'mp4')
create_animation(folder_path, 'normal_scaled', 24, 'mp4')
create_animation(folder_path, 'albedo', 24, 'mp4')
create_animation(folder_path, 'depth', 24, 'mp4')
create_animation(folder_path, 'ldr_envmap', 24, 'mp4')