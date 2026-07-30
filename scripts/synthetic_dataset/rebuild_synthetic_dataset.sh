#!/bin/bash
# Full re-render and re-prepare of the NeuSky synthetic dataset.
#
# Per scene renders three passes (see scene_render_configs.json):
#   150 frames  train profile          (varied views, focal/exposure variation)
#   100 frames  train_curated profile  (whole-building framing, train appearance)
#    50 frames  eval profile           (whole-building framing, fixed focal/exposure)
# then rebuilds <data>/renders/<scene>_prepared: train = varied + curated (250),
# validation/test = 25 + 25 from the eval render. The previous prepared dataset
# is kept at <scene>_prepared_pre_rebuild.
#
# Usage:
#   ./scripts/rebuild_synthetic_dataset.sh [scene ...]   # default: all five
#
# ~2h per scene on an RTX 4090 (~10h for all five).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${NEUSKY_SYN_DATA:-$HOME/data/neusky_synthetic_data}"

DEFAULT_SCENES=(apartment_building glass_building abandoned_buildings
                interstellar_house arlanda_uppsala_cathedral)
SCENES=("${@:-${DEFAULT_SCENES[@]}}")

for s in "${SCENES[@]}"; do
    echo "############################################################"
    echo "##### ${s}: $(date)"
    echo "############################################################"

    python3 "${SCRIPT_DIR}/render_eval_views.py" "$s" --profile train
    python3 "${SCRIPT_DIR}/render_eval_views.py" "$s" --profile train_curated
    python3 "${SCRIPT_DIR}/render_eval_views.py" "$s" --profile eval

    # Keep the previous prepared dataset (first rebuild only)
    if [[ -d "${DATA}/renders/${s}_prepared" && ! -d "${DATA}/renders/${s}_prepared_pre_rebuild" ]]; then
        mv "${DATA}/renders/${s}_prepared" "${DATA}/renders/${s}_prepared_pre_rebuild"
    fi

    python3 "${SCRIPT_DIR}/prepare_synthetic_data.py" \
        --input "${DATA}/renders/${s}_train_varied" "${DATA}/renders/${s}_train_curated" \
        --output "${DATA}/renders/${s}_prepared" --n-val 0 --n-test 0

    python3 "${SCRIPT_DIR}/prepare_synthetic_data.py" \
        --input "${DATA}/renders/${s}_eval" \
        --output "${DATA}/renders/${s}_prepared" \
        --update-eval --n-val 25 --n-test 25

    # Point cloud for Gaussian-splat initialisation (from the varied render)
    python3 "${SCRIPT_DIR}/extract_pointcloud.py" \
        --input "${DATA}/renders/${s}_train_varied" --max_depth 200 --subsample 4
    cp "${DATA}/renders/${s}_train_varied/points3d.ply" "${DATA}/renders/${s}_prepared/"

    echo "##### ${s} complete: $(date)"
done

echo "ALL SCENES REBUILT: $(date)"
