#!/bin/bash

apptainer shell --nv --no-home \
  -B /data/home/$USER:/data/home/$USER \
  -B /data/home/$USER/.bashrc:/data/home/$USER/.bashrc \
  -B /data/home/$USER/.config:/data/home/$USER/.config \
  -B /data/home/$USER/.local:/data/home/$USER/.local \
  -B /data/home/$USER/.apptainer/pip_venvs:/mnt \
  -B /data/home/$USER/.vscode-server:/data/home/$USER/.vscode-server \
  -B /data/home/$USER/.jupyter:/data/home/$USER/.jupyter \
  --env MPLCONFIGDIR=/data/home/$USER/.config/matplotlib \
  /data/home/$USER/.apptainer/containers/nerfstudio.sif