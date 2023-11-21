# The Sky’s the Limit

### Official implementation of NeuSky.

Paper: The Sky's the Limit: Re-lightable Outdoor Scenes via a Sky-pixel Constrained Illumination Prior and Outside-In Visibility

<img src="imgs/teaser.png" width="85%"/>

<!-- <img src="imgs/animation.gif" width="50%"/> -->

### Setup

We build ontop of Nerfstudio and in future this dependancy will be a simple 'pip install'.However, since Nerfstudio is still in very activate develeopment with farily large codebase changes still occuring a copy of a codebase has been included to ensure compatability.

1. It is reccomended to use a pip venv to install our model

a.

```bash
cd nerfstudio

python3.10 -m venv .venv/nerfstudio

source .venv/nerfstudio/bin/activate

```

b. We need a specific version of pip for tinycuda

```bash
pip install pip==23.0.1 --upgrade

pip install --upgrade setuptools pathtools promise pybind11

```

c. Install pytorch and submodules

```bash
CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && pip install \
    torch==2.0.1+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

pip install omegaconf

```

d. Install nerfstudio

```bash
cd nerfstudio

pip install -e .

```

e. Install RENI++

```bash
cd reni_neus/ns_reni

pip install -e .

```

f. Install RENINeuS

```bash
cd ..

pip install -e .

```
