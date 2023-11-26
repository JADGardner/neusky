# The Sky’s the Limit

### Official implementation of NeuSky.

Paper: The Sky's the Limit: Re-lightable Outdoor Scenes via a Sky-pixel Constrained Illumination Prior and Outside-In Visibility

<img src="imgs/teaser.png" width="85%"/>

<!-- <img src="imgs/animation.gif" width="50%"/> -->

## Installation

We build on top of Nerfstudio. However, since Nerfstudio is still in very activate develeopment with farily large codebase changes still occuring compatibility might be an issue. Pull requests and issues are very welcome.

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create Environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

a. Install nerfstudio

```bash
pwd # -> Should be in /Some/Path/Ending/In/SupplementaryMaterial/Code

conda create --name nerfstudio -y python=3.8

conda activate nerfstudio

pip install --upgrade pip

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install --upgrade pip setuptools

pip install -e .
```

b. Install RENI++

```bash
sudo apt install libopenexr-dev

conda install -c conda-forge openexr

cd reni_neus/ns_reni

pip install -e .
```

c. Install NeuSky

```bash
cd ..

pip install -e .
```

d. Setup Nerfstudio CLI

```bash
ns-install-cli
```

e. Close and reopen your terminal and source virtual environment again:

```bash
conda activate nerfstudio
```

## Download Data

```bash
ns-download-data nerfosr --save-dir data --capture-name lk2
```

```bash
pwd # -> Should be in /Some/Path/Ending/In/SupplementaryMaterial/Code

python copy_segmentation_masks.py
```

## Start Training

You can now launch training for scene 'lk2'

```bash
ns-train reni-neus --vis wandb
```

If you find you run out of GPU memory you can try updating some or all of these settings in

```bash
reni_neus/reni_neus/configs/reni_neus_config.py
```

```bash
train_num_images_to_sample_from=-1, # Set to integer value if out of GPU memory
train_num_times_to_repeat_images=-1, # Iterations before resampling a new subset, set to integer value if out of GPU memory
images_on_gpu=True, # set False if out of GPU memory
masks_on_gpu=True, # set False if out of GPU memory
train_num_rays_per_batch=1024, # Lower to 512, 256, or 128 if out of GPU memory
eval_num_rays_per_batch=1024, # Lower to 512, 256, or 128 if out of GPU memory
```
