#!/usr/bin/env python3
"""Launch a NeuSky synthetic-scene training run.

Imports the method config directly (like train_nerfosr.py) so it works with
PYTHONPATH-only setups where nerfstudio's entry-point method registry cannot
see the neusky package.

Usage (inside the research container, cwd = code/neusky):
    PYTHONPATH=. python scripts/train_synthetic.py \
        --data /workspace/data/neusky_synthetic_data/renders/glass_building_prepared
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Scene dir, e.g. .../renders/<scene>_prepared")
    parser.add_argument("--experiment-name", default=None,
                        help="Default: synthetic/<scene dir name without _prepared>")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--vis", default="wandb")
    parser.add_argument("--max-num-iterations", type=int, default=None)
    parser.add_argument("--reni-ckpt", type=Path, default=None,
                        help="Override illumination_field_ckpt_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from nerfstudio.scripts.train import main as ns_train_main
    from neusky.configs.neusky_synthetic_config import NeuSkySynthetic

    config = copy.deepcopy(NeuSkySynthetic.config)
    data = args.data.expanduser()
    config.data = data
    config.pipeline.datamanager.dataparser.data = data

    scene = data.name.removesuffix("_prepared")
    config.experiment_name = args.experiment_name or f"synthetic/{scene}"
    config.project_name = "neusky"
    config.output_dir = args.output_dir
    config.vis = args.vis
    if args.max_num_iterations is not None:
        config.max_num_iterations = args.max_num_iterations
    if args.reni_ckpt is not None:
        config.pipeline.model.illumination_field_ckpt_path = args.reni_ckpt.expanduser()

    ns_train_main(config)


if __name__ == "__main__":
    main()
