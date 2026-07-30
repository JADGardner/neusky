#!/bin/bash
# Regenerate vegetation in all 5 NeuSky scenes.
#
# Removes existing trees/boulders/shrubs and scatters new ones with
# collision avoidance against buildings and cliffs.
# Cliffs (CoastalCliff_*) are preserved.
#
# Usage:
#   NEUSKY_SYN_DATA=/path/to/neusky_synthetic_data \
#     ./scripts/synthetic_dataset/scene_setup/regenerate_all_vegetation.sh
#
# Requires Blender 4.x+ on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="${NEUSKY_SYN_DATA:-$HOME/data/neusky_synthetic_data}"

SCENES_DIR="$DATA/scenes"
BG_ASSETS="$DATA/background_assets"
REGEN_SCRIPT="$SCRIPT_DIR/regenerate_vegetation.py"

SCENES=(
    interstellar_house
    apartment_building
    abandoned_buildings
    glass_building
    arlanda_uppsala_cathedral
)

# Each scene gets a unique deterministic seed
SEEDS=(7001 7002 7003 7004 7005)

for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    seed="${SEEDS[$i]}"
    blend="$SCENES_DIR/${scene}.blend"

    if [ ! -f "$blend" ]; then
        echo "SKIP: $blend not found"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Regenerating vegetation: $scene  (seed=$seed)"
    echo "============================================================"

    blender --background "$blend" --python "$REGEN_SCRIPT" -- \
        --bg_assets_dir "$BG_ASSETS" \
        --output "$blend" \
        --seed "$seed" \
        --margin 5.0 \
        --min_spacing 8.0

    echo "Done: $scene"
done

echo ""
echo "All scenes updated."
