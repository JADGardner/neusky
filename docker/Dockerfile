# Dockerfile — NeuSky research container
# Multi-stage build: devel stage compiles CUDA extensions, runtime stage is smaller.
#
# NeuSky is a nerfstudio extension for outdoor neural scene reconstruction
# with sky-pixel constrained illumination priors.
#
# Includes: CUDA 12.8, PyTorch 2.x, tiny-cuda-nn, nvdiffrast, nerfstudio,
#           COLMAP, and all pure-Python dependencies.
#
# Usage:
#   docker compose build research
#   docker compose run research bash
#
# Requires the NVIDIA Container Toolkit on the host:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

# =============================================================================
# Stage 1: BUILDER — devel image with compilers for CUDA extension builds
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# -- System dependencies (build tools + runtime libs) -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libopenexr-dev \
    openexr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -- Miniconda ----------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# -- Conda environment --------------------------------------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n research python=3.12 -y && \
    conda run -n research conda install -c conda-forge colmap sqlite ffmpeg ninja -y && \
    conda clean -afy

# Set CUDA_HOME to system CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# -- PyTorch + CUDA 12.8 ------------------------------------------------------
RUN conda run -n research pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128

# -- Set up LD_LIBRARY_PATH for CUDA libs from torch --------------------------
ENV LD_LIBRARY_PATH=/opt/conda/envs/research/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/research/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/conda/envs/research/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# -- CUDA extensions (require devel headers + compilers) ----------------------
# tiny-cuda-nn: required by nerfstudio hash-grid encodings
ARG CUDA_ARCHITECTURES="90;89;86;80"
RUN conda run -n research bash -c " \
    export TCNN_CUDA_ARCHITECTURES='${CUDA_ARCHITECTURES}' && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch \
    && pip cache purge && rm -rf /tmp/*"

# -- nvdiffrast (differentiable rasterization) --------------------------------
RUN conda run -n research bash -c " \
    export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0+PTX' && \
    ln -sf \$CONDA_PREFIX/lib/python3.12/site-packages/triton/backends/nvidia/include/crt \$CONDA_PREFIX/include/ && \
    pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git \
    && pip cache purge && rm -rf /tmp/*"

# -- nerfstudio (mainline) ----------------------------------------------------
# Shallow clone to save space. Installed as editable so nerfstudio CLI works.
RUN conda run -n research bash -c " \
    git clone --depth 1 https://github.com/nerfstudio-project/nerfstudio.git /opt/nerfstudio && \
    cd /opt/nerfstudio && \
    pip install --no-cache-dir -e . \
    && pip cache purge && rm -rf /tmp/*"

# -- gsplat (pre-compile CUDA kernels) ----------------------------------------
# nerfstudio pulls in gsplat which JIT-compiles CUDA kernels on first use.
# Pre-compile here so the runtime stage (no CUDA headers) works correctly.
RUN conda run -n research bash -c " \
    export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0+PTX' && \
    python -c 'import gsplat' \
    " || true

# -- All pure-Python dependencies in ONE layer ---------------------------------
# Scientific Python + 3D/NeRF + NeuSky deps + ns_reni deps
RUN conda run -n research pip install --no-cache-dir \
    numpy scipy pandas matplotlib seaborn \
    scikit-learn scikit-image \
    opencv-python-headless Pillow \
    tqdm wandb tensorboard \
    jupyterlab ipywidgets \
    open3d trimesh plyfile pycolmap \
    tyro rich jaxtyping viser nerfview splines \
    einops pyyaml packaging requests \
    pyexr OpenEXR roma \
    icosphere wget yacs ftfy regex kaleido omegaconf \
    "imageio[all]" \
    && pip cache purge && rm -rf /tmp/*

# nerfstudio CLI completions
RUN conda run -n research ns-install-cli || true

# Final cleanup in builder
RUN conda clean -afy && \
    find /opt/conda -name '*.pyc' -delete && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Stage 2: RUNTIME — smaller base without CUDA compilers/headers
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# -- Runtime system dependencies only (no build-essential, cmake, ninja) -------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    # OpenEXR runtime libs (not -dev headers)
    libopenexr-3-1-30 \
    openexr \
    # Image / video libraries used by CV packages
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -- Copy conda from builder --------------------------------------------------
ENV CONDA_DIR=/opt/conda
COPY --from=builder /opt/conda /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# -- Copy nerfstudio source (installed as editable) ---------------------------
COPY --from=builder /opt/nerfstudio /opt/nerfstudio

# -- Environment variables ----------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/research/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/research/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/conda/envs/research/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

# Pre-create HOME dir and pip cache so they're writable even when Docker volume mounts
# create parent dirs as root
RUN mkdir -p /tmp/home/.cache/pip && chmod -R 777 /tmp/home

# Make conda env writable by any UID so `pip install -e` works at runtime
# when the container runs as a non-root user (matching host UID).
RUN chmod -R a+rwX /opt/conda/envs/research/lib/python3.12/site-packages && \
    chmod -R a+rwX /opt/conda/envs/research/bin

# Activate the conda environment by default
RUN echo "conda activate research" >> /etc/profile.d/conda.sh
SHELL ["conda", "run", "-n", "research", "/bin/bash", "-c"]

# Entrypoint: install mounted code editably, then run the user command
COPY docker/entrypoint.sh /entrypoint.sh
# Reset SHELL — chmod doesn't need conda, and conda run triggers a broken
# nerfstudio bash completion script that exits with code 127.
SHELL ["/bin/bash", "-c"]
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
