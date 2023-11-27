#!/bin/bash

source ~/.bashrc

cd /users/$USER/scratch/nerfstudio

python3.10 -m venv /mnt/nerfstudio

source /mnt/nerfstudio/bin/activate

pip install pip==23.0.1 --upgrade

pip install --upgrade setuptools pathtools promise pybind11

# Install pytorch and submodules
CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && pip install \
    torch==2.0.1+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

pip install omegaconf

pip install -e .

cd neusky/ns_reni

pip install -e .

cd ..

pip install -e .