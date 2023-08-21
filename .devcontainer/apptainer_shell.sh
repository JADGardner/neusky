#!/bin/bash

apptainer exec --nv --no-home \
  -B /mnt/scratch/users/$USER:/users/$USER/scratch \
  -B /users/$USER/.bashrc:/users/$USER/.bashrc \
  -B /users/$USER/scratch/.config:/users/$USER/.config \
  -B /users/$USER/scratch/.local:/users/$USER/.local \
  -B /users/$USER/scratch/.singularity/pip_venvs:/mnt \
  -B /users/$USER/.vscode-server:/users/$USER/.vscode-server \
  -B /users/$USER/.jupyter:/users/$USER/.jupyter \
  --env MPLCONFIGDIR=/users/$USER/scratch/.config/matplotlib \
  /users/$USER/scratch/.singularity/containers/nerfstudio_old.sif /bin/bash -c "source /mnt/nerfstudio/bin/activate && /bin/bash --norc"