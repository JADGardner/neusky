#!/bin/bash

apptainer shell --nv --no-home \
  -B /mnt/scratch/users/$USER:/users/$USER/scratch \
  -B /users/$USER/.bashrc:/users/$USER/.bashrc \
  -B /users/$USER/scratch/.config:/users/$USER/.config \
  -B /users/$USER/scratch/.local:/users/$USER/.local \
  -B /users/$USER/scratch/.apptainer/pip_venvs:/mnt \
  -B /users/$USER/.vscode-server:/users/$USER/.vscode-server \
  -B /users/$USER/.jupyter:/users/$USER/.jupyter \
  --env MPLCONFIGDIR=/users/$USER/scratch/.config/matplotlib \
  /users/$USER/scratch/.apptainer/containers/nerfstudio.sif