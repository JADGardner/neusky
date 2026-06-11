#!/bin/bash
# Train NeuSky on one scene via the umbrella Apptainer wrapper.
#
# Usage (from the neusky repo root on the cluster):
#   SCENE=apartment_building_prepared sbatch .apptainer/train_neusky.sbatch
#   # NeRF-OSR sites: SCENE=lk2 DATA_ROOT=/path/to/NeRF-OSR/Data METHOD=neusky
#
#SBATCH --job-name=neusky
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCENE="${SCENE:?set SCENE=<scene>_prepared (synthetic) or <site> (NeRF-OSR)}"
METHOD="${METHOD:-neusky-synthetic}"
DATA_ROOT="${DATA_ROOT:-/mnt/scratch/users/${USER}/data/neusky_synthetic_data/renders}"
EXPERIMENT="${EXPERIMENT:-synthetic/${SCENE}}"

echo "scene=${SCENE} method=${METHOD} data=${DATA_ROOT}/${SCENE} node=$(hostname)"

module load Apptainer/latest 2>/dev/null || true

.apptainer/apptainer.sh exec -- ns-train "${METHOD}" \
    --data "${DATA_ROOT}/${SCENE}" \
    --experiment-name "${EXPERIMENT}" \
    --output-dir outputs \
    --vis tensorboard \
    --viewer.quit-on-train-completion True
