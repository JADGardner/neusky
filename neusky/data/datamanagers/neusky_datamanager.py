# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from itertools import cycle

import numpy as np
import torch
from PIL import Image
from rich.progress import Console
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    variable_res_collate,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator

from neusky.data.neusky_pixel_sampler import NeuSkyPixelSampler
from neusky.data.datasets.neusky_dataset import NeuSkyDataset
from neusky.data.utils.dataloaders import SelectedIndicesCacheDataloader

CONSOLE = Console(width=120)


@dataclass
class NeuSkyDataManagerConfig(VanillaDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: NeuSkyDataManager)
    """Target class to instantiate."""


class NeuSkyDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NeuSkyDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[NeuSkyPixelSampler] = None
    eval_pixel_sampler: Optional[NeuSkyPixelSampler] = None

    def __init__(
        self,
        config: NeuSkyDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.eval_latent_optimise_method = kwargs.get("eval_latent_optimise_method", None)
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        # TODO This is a mess, can only have test or val at one time anyway so just use one variable
        # This will need updating in pipeline and model too
        if self.eval_latent_optimise_method == "per_image":
            self.num_test = len(self.test_outputs.image_filenames)
            if self.test_mode == "test":
                self.num_val = len(self.test_outputs.image_filenames)
            else:
                self.num_val = len(self.val_outputs.image_filenames)
        else:
            self.num_test = len(self.eval_dataset.metadata["session_to_indices"].keys())
            self.num_val = len(self.eval_dataset.metadata["session_to_indices"].keys())
        
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and 'mask' in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and 'image' in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break

        super(VanillaDataManager, self).__init__()  # Call grandparent class constructor ignoring parent class

    def create_train_dataset(self) -> NeuSkyDataset:
        return NeuSkyDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split="train",
        )

    def create_eval_dataset(self) -> NeuSkyDataset:
        self.test_outputs = self.dataparser.get_dataparser_outputs("test")
        self.val_outputs = self.dataparser.get_dataparser_outputs("val")
        # self.num_test = len(test_outputs.image_filenames)
        # self.num_val = len(val_outputs.image_filenames)
        return NeuSkyDataset(
            dataparser_outputs=self.test_outputs if self.test_mode == "test" else self.val_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split=self.test_split,
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        if self.eval_latent_optimise_method == "per_image":
            self.eval_image_dataloader = CacheDataloader(
                self.eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
            )
            self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
            self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
            self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))

            self.eval_dataloader = RandIndicesEvalDataloader(
                    input_dataset=self.eval_dataset,
                    device=self.device,
                    num_workers=self.world_size * 4,
                )
        else:
            ### This is for NeRF-OSR relighting benchmark ###
            session_image_idxs = self.eval_dataset.metadata["session_holdout_indices"] # idx of holdout relative to session
            session_to_indices = self.eval_dataset.metadata["session_to_indices"] # maps session idx to image idxs
            self.indices_to_session = self.eval_dataset.metadata["indices_to_session"] # maps image idxs to session idxs
            # currently session_image_idxs is the image idxs relative to session
            # but we want it to be relative to the whole dataset
            image_idxs_holdout = [
                session_to_indices[key][index] for key, index in zip(session_to_indices.keys(), session_image_idxs)
            ]
            self.eval_session_holdout_dataloader = SelectedIndicesCacheDataloader(
                self.eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                # exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
                selected_indices=image_idxs_holdout,
            )
            self.iter_eval_session_holdout_dataloader = iter(self.eval_session_holdout_dataloader)
            self.holdout_eval_dataloader = FixedIndicesEvalDataloader(
                input_dataset=self.eval_dataset,
                image_indices=tuple(image_idxs_holdout),
                device=self.device,
                num_workers=self.world_size * 4,
            )
            self.iter_holdout_eval_dataloader = iter(self.holdout_eval_dataloader)
            # image_idxs_eval = [x for x in range(len(self.eval_dataset))]
            # image_idxs_eval = [idx for idx in image_idxs_eval if idx not in image_idxs_holdout]
            image_idxs_eval = list(self.eval_dataset.test_eval_mask_dict.keys())
            self.eval_session_compare_dataloader = SelectedIndicesCacheDataloader(
                self.eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                # exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
                selected_indices=image_idxs_eval,
            )
            self.iter_eval_session_compare_dataloader = iter(self.eval_session_compare_dataloader)

            self.eval_dataloader = FixedIndicesEvalDataloader(
                input_dataset=self.eval_dataset,
                image_indices=tuple(image_idxs_eval),
                device=self.device,
                num_workers=self.world_size * 4,
            )
            self.iter_eval_dataloader = iter(self.eval_dataloader)


    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        if not self.eval_latent_optimise_method == "per_image":
            # nerf osr holdout
            if self.eval_dataloader.count >= len(self.eval_dataloader.image_indices):
                self.eval_dataloader.count = 0
            camera, batch = next(self.iter_eval_dataloader)
            camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            # the dataloader yields a [idx:idx+1] camera slice that carries no
            # index (its generated rays always report camera 0), so take the
            # true image index from the batch
            camera_idx = int(batch["image_idx"])
            # we need to use the indices_to_session mapping to get the session idx
            # as all images from a sessioin have the same latent code
            image_idx = self.indices_to_session[camera_idx]
            batch["image_idx"] = image_idx
            # we also need to update camera_ray_bundle.camera_indices which is shape [H, W, 1]
            # to also just be same shape but all image_idx
            camera_ray_bundle.camera_indices = torch.ones_like(camera_ray_bundle.camera_indices) * image_idx
            return image_idx, camera_ray_bundle, batch
        else:
            for camera, batch in self.eval_dataloader:
                camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                image_idx = batch["image_idx"]
                camera_ray_bundle.camera_indices = torch.ones_like(camera_ray_bundle.camera_indices) * image_idx
                return image_idx, camera_ray_bundle, batch
            raise ValueError("No more eval images")
    
    def next_holdout_image(self, step: int):
        if self.holdout_eval_dataloader.count >= len(self.holdout_eval_dataloader.image_indices):
            self.holdout_eval_dataloader.count = 0
        camera, batch = next(self.iter_holdout_eval_dataloader)
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
        # as in next_eval_image: the camera slice carries no index, use the batch
        camera_idx = int(batch["image_idx"])
        # we need to use the indices_to_session mapping to get the session idx
        # as all images from a sessioin have the same latent code
        image_idx = self.indices_to_session[camera_idx]
        batch["image_idx"] = image_idx
        # we also need to update camera_ray_bundle.camera_indices which is shape [H, W, 1]
        # to also just be same shape but all image_idx
        camera_ray_bundle.camera_indices = torch.ones_like(camera_ray_bundle.camera_indices) * image_idx
        return image_idx, camera_ray_bundle, batch

    def get_sky_ray_bundle(self, number_of_rays: int) -> Tuple[RayBundle, Dict]:
        """Returns a sky ray bundle for the given step."""
        # choose random
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        if isinstance(image_batch["image"], list):
            batch = self.train_pixel_sampler.collate_sky_ray_batch_list(image_batch, num_rays_per_batch=number_of_rays)
        else:
            batch = self.train_pixel_sampler.collate_sky_ray_batch(image_batch, num_rays_per_batch=number_of_rays)
        ray_indices = batch["indices"].cpu()
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle

    def get_eval_image_half_bundle(
        self, sample_region: Literal["left_image_half", "right_image_half", "full_image"]
    ) -> Tuple[RayBundle, Dict]:
        """Returns a ray bundle of rays within the left image half and within standard mask."""
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        if isinstance(image_batch["image"], list):
            batch = self.eval_pixel_sampler.collate_image_half_list(
                batch=image_batch, num_rays_per_batch=self.config.eval_num_rays_per_batch, sample_region=sample_region
            )
        else:
            batch = self.eval_pixel_sampler.collate_image_half(
                batch=image_batch, num_rays_per_batch=self.config.eval_num_rays_per_batch, sample_region=sample_region
            )
        ray_indices = batch["indices"].cpu()
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def get_nerfosr_lighting_eval_bundle(self, stage: Literal["optimise", "compare"]) -> Tuple[RayBundle, Dict]:
        """Returns a ray bundle of rays from only the selected IDX from each test session.
        The test datasets contain multiple capture sessions at different dates. We get a
        single image from each session as specified by the session_idxs list."""
        assert stage in ["optimise", "compare"]
        if stage == "optimise":
            image_batch = next(self.iter_eval_session_holdout_dataloader)
        else:
            image_batch = next(self.iter_eval_session_compare_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        # we need to update the image indices to be the session indices so as to use the same RENI illumination for all images from a session
        batch["indices"][:, 0] = torch.tensor(
            [self.indices_to_session[i.item()] for i in batch["indices"][:, 0]]
        ).type_as(batch["indices"][:, 0])
        # we also need to update the camera_ray_bundle.camera_indices which is shape [N, 1]
        ray_bundle.camera_indices = batch["indices"][:, 0][:, None]
        return ray_bundle, batch

    def get_nerfosr_envmap_lighting_optimisation_bundle(
        self,
        saturation_scale: float = 10.0,
        saturation_threshold: float = 0.95,
        envmap_height: int = 64,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """GT session envmaps for the nerf_osr_envmap relighting protocol.

        For each test session this loads the session's ENV_MAP_CC panorama,
        applies the NeRF-OSR pseudo-HDR scaling (pixels above
        saturation_threshold multiplied by saturation_scale) and maps the
        panorama pixel directions into the model world frame using the KNOWN
        per-session rotation from <scene_dir>/envmap_rotations.json composed
        with the dataparser's auto-orientation transform (the same way the
        dataparser orients the SfM points).

        envmap_rotations.json is produced by scripts/register_envmaps.py and
        maps ERP panorama directions (its convention: lon 0 at the centre
        column increasing rightwards, lat -pi/2 at the top row / zenith,
        d_pano = [cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)], so
        d_world = rotation @ d_pano) into the dataset-normalised world frame
        of the raw pose/*.txt files, i.e. BEFORE auto-orientation. The
        optional "envmap_image" key pins the panorama filename within the
        session folder (default: first, alphabetically).

        Returns {session_idx: {"directions": [N, 3], "rgb": [N, 3],
        "sineweight": [N, 1]}} (CPU); sineweight corrects the equirectangular
        oversampling towards the poles, as in RENI++ inversion.
        """
        metadata = self.eval_dataset.metadata
        scene_dir = metadata.get("scene_dir")
        session_names = metadata.get("session_names")
        transform = metadata.get("orientation_transform")
        envmap_filenames = metadata.get("envmap_filenames") or []
        if scene_dir is None or session_names is None or not envmap_filenames:
            raise ValueError(
                "nerf_osr_envmap protocol needs session envmaps; the eval dataparser "
                "did not provide scene_dir/session_names/ENV_MAP_CC metadata "
                "(NeRF-OSR scenes with ENV_MAP_CC only)."
            )

        rotations_path = Path(scene_dir) / "envmap_rotations.json"
        if not rotations_path.exists():
            raise FileNotFoundError(
                f"nerf_osr_envmap protocol requires known per-session envmap rotations but none were "
                f"found at the expected path: {rotations_path}. Generate them with "
                "scripts/register_envmaps.py (format {\"<session>\": {\"rotation\": [[3x3 row-major]]}}, "
                "mapping panorama directions into the dataset-normalised world frame of "
                "final/<split>/pose/*.txt, before auto-orientation)."
            )
        session_rotations = json.loads(rotations_path.read_text())

        # Rays in the model frame need the same orientation transform the
        # dataparser applied to the poses/SfM points (rotation part only).
        orientation_rotation = torch.eye(3) if transform is None else torch.as_tensor(transform)[:3, :3].float()

        # ERP pixel grid directions in the register_envmaps.py convention.
        height, width = envmap_height, envmap_height * 2
        lon = 2 * torch.pi * (torch.arange(width, dtype=torch.float32) + 0.5) / width - torch.pi
        lat = torch.pi * (torch.arange(height, dtype=torch.float32) + 0.5) / height - torch.pi / 2
        lat, lon = torch.meshgrid(lat, lon, indexing="ij")  # [H, W]
        d_pano = torch.stack(
            (torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)), dim=-1
        ).reshape(-1, 3)  # [N, 3]
        # Solid-angle weighting of the ERP grid (sin of polar = cos of lat).
        sineweight = torch.cos(lat).reshape(-1, 1)  # [N, 1]

        bundle = {}
        for session_idx, session_name in session_names.items():
            if session_name not in session_rotations:
                raise KeyError(
                    f"Session '{session_name}' missing from {rotations_path}; "
                    f"available sessions: {sorted(session_rotations.keys())}"
                )
            entry = session_rotations[session_name]
            panorama_to_pose_frame = torch.as_tensor(entry["rotation"], dtype=torch.float32)
            assert panorama_to_pose_frame.shape == (3, 3), f"rotation for '{session_name}' must be 3x3"

            filenames = sorted(
                f for f in envmap_filenames if f"{os.sep}{session_name}{os.sep}" in f
            )
            if not filenames:
                raise FileNotFoundError(f"No ENV_MAP_CC panorama found for session '{session_name}'")
            if entry.get("envmap_image") is not None:
                pinned = os.path.basename(entry["envmap_image"])
                filenames = [f for f in filenames if os.path.basename(f) == pinned]
                if not filenames:
                    raise FileNotFoundError(
                        f"Panorama '{pinned}' pinned in {rotations_path} not found in "
                        f"session '{session_name}'"
                    )

            # Panorama pixels -> linear-ish [0, 1] -> NeRF-OSR pseudo-HDR.
            envmap = torch.from_numpy(
                np.array(Image.open(filenames[0]).convert("RGB"), dtype="float32") / 255.0
            ).permute(2, 0, 1)[None]  # [1, 3, H, W]
            envmap = torch.nn.functional.interpolate(envmap, size=(height, width), mode="area")
            envmap[envmap > saturation_threshold] *= saturation_scale
            rgb = envmap.squeeze(0).permute(1, 2, 0).reshape(-1, 3)  # [N, 3]

            # Panorama directions -> pose frame (known rotation) -> model frame.
            rotation = orientation_rotation @ panorama_to_pose_frame
            directions = d_pano @ rotation.T  # [N, 3]

            bundle[session_idx] = {"directions": directions, "rgb": rgb, "sineweight": sineweight}
        return bundle
