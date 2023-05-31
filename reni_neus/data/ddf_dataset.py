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
SDFStudio dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path


"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union, Literal

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

import yaml


class DDFDataset(Dataset):
    """Dataset that returns images.

    Args:
        reni_neus_checkpoint_path: The path to the Reni Neus checkpoint.
        test_mode: The test mode to use. One of "train", "test", or "val".
        num_generated_imgs: The number of images to generate.
        cache_dir: The directory to cache/load the generated images.
    """

    cameras: Cameras

    def __init__(
        self,
        reni_neus_ckpt_path,
        reni_neus_ckpt_step: int,
        test_mode: Literal["train", "test", "val"] = "train",
        num_generated_imgs: int = 10,
        cache_dir: Union[Path, None] = Path.home() / ".nerfstudio" / "cache",
    ):
        super().__init__()
        self.reni_neus_ckpt_path = reni_neus_ckpt_path
        self.reni_neus_ckpt_step = reni_neus_ckpt_step
        self.test_mode = test_mode
        self.num_generated_imgs = num_generated_imgs
        self.cache_dir = cache_dir

        self._setup_reni()

    def __len__(self):
        return self.num_generated_imgs

    def _setup_reni(self):
        # setting up reni_neus for pseudo ground truth
        ckpt = torch.load(
            self.reni_neus_ckpt_path + "/nerfstudio_models" + f"/step-{self.reni_neus_ckpt_step:09d}.ckpt",
        )
        model_dict = {}
        for key in ckpt["pipeline"].keys():
            if key.startswith("_model."):
                model_dict[key[7:]] = ckpt["pipeline"][key]

        num_train_data = model_dict["illumination_field_train.reni.mu"].shape[0]
        num_val_data = model_dict["illumination_field_val.reni.mu"].shape[0]
        num_test_data = model_dict["illumination_field_test.reni.mu"].shape[0]

        # load yaml checkpoint config
        reni_neus_config = Path(self.reni_neus_ckpt_path) / "config.yml"
        reni_neus_config = yaml.load(reni_neus_config.open(), Loader=yaml.Loader)

        self.reni_neus = reni_neus_config.pipeline.model.setup(
            scene_box=self.scene_box, num_train_data=num_train_data, num_eval_data=num_eval_data
        )

        self.reni_neus.load_state_dict(model_dict)
        self.reni_neus.eval()

    def _generate_images(self):
        pass

    def _ddf_rays(self):
        pass

    def _get_generated_image(self, image_idx: int):
        pass

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if self.test_mode == "train":
            data = self._ddf_rays()
        else:
            data = self._get_generated_image(image_idx)
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data


class DDFSDFSampler(Sampler):
    def __init__(self, num_samples, ddf_sphere_radius, sdf_function):
        super().__init__(num_samples=num_samples)
        self.sdf_function = sdf_function
        self.ddf_sphere_radius = ddf_sphere_radius

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        return_gt: bool = True,
    ):
        device = self.sdf_function.device
        if ray_bundle is None:
            num_samples = num_samples or self.num_samples

            positions = random_points_on_unit_sphere(1, cartesian=True)  # (1, 3)
            directions = random_inward_facing_directions(num_samples, normals=-positions)  # (1, num_directions, 3)

            positions = positions * self.ddf_sphere_radius

            pos_ray = positions.repeat(num_samples, 1).to(device)
            dir_ray = directions.reshape(-1, 3).to(device)
            pixel_area = torch.ones(num_samples, 1, device=device)
            camera_indices = torch.zeros(num_samples, 1, device=device, dtype=torch.int64)
            metadata = {"directions_norm": torch.ones(num_samples, 1, device=device)}

            ray_bundle = RayBundle(
                origins=pos_ray,
                directions=dir_ray,
                pixel_area=pixel_area,
                camera_indices=camera_indices,
                metadata=metadata,
            )

        accumulations = None
        termination_dist = None
        normals = None
        if return_gt:
            field_outputs = self.sdf_function(ray_bundle)
            accumulations = field_outputs["accumulation"].reshape(-1, 1).squeeze()
            termination_dist = field_outputs["p2p_dist"].reshape(-1, 1).squeeze()
            normals = field_outputs["normal"].reshape(-1, 3).squeeze()

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=ray_bundle.origins.reshape(-1, 3),
                directions=ray_bundle.directions.reshape(-1, 3),
                starts=torch.zeros_like(ray_bundle.origins),
                ends=torch.zeros_like(ray_bundle.origins),
                pixel_area=torch.ones_like(ray_bundle.origins),
            ),
        )

        return ray_samples, accumulations, termination_dist, normals
