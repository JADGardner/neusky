# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from typing import Any, Callable, List, Optional, Sized, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset

from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.utils.dataloaders import CacheDataloader


class SelectedIndicesCacheDataloader(CacheDataloader):
    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        selected_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        self.selected_indices = selected_indices
        super().__init__(
            dataset,
            num_images_to_sample_from,
            num_times_to_repeat_images,
            device,
            collate_fn,
            exclude_batch_keys_from_device,
            **kwargs,
        )

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        assert isinstance(self.dataset, Sized)
        if self.selected_indices is not None:
            indices = self.selected_indices
        else:
            indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)

        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list
