#!/usr/bin/env python3
"""Launch a NeuSky NeRF-OSR training run with a scene override.

The neusky method embeds NeRFOSRCityScapesDataParserConfig, which is not in
nerfstudio's dataparser-union subcommands, so `ns-train` cannot override
`dataparser.scene` from the CLI. This wrapper sets it programmatically and
hands the config to nerfstudio's standard train entry point.

Usage (inside the research container, cwd = code/neusky):
    PYTHONPATH=. python scripts/train_nerfosr.py --scene site2 \
        --data ~/data/NeRF-OSR/Data --experiment-name st_refit_sfm
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True,
                        choices=["site1", "site2", "site3", "lk2", "st", "lwp", "trevi"])
    parser.add_argument("--data", type=Path, default=Path("data/NeRF-OSR/Data"))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--vis", default="wandb")
    parser.add_argument("--max-num-iterations", type=int, default=None)
    parser.add_argument("--reni-ckpt", type=Path, default=None,
                        help="Override illumination_field_ckpt_path")
    parser.add_argument("--no-sfm", action="store_true",
                        help="Disable SfM point-cloud rescaling (camera auto-scale instead)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from nerfstudio.scripts.train import main as ns_train_main
    from neusky.configs.neusky_config import NeuSky

    config = copy.deepcopy(NeuSky.config)
    dp = config.pipeline.datamanager.dataparser
    dp.scene = args.scene
    dp.data = args.data.expanduser()
    if args.no_sfm:
        dp.center_method_sfm = False

    scene_label = {"site1": "lk2", "site2": "st", "site3": "lwp"}.get(args.scene, args.scene)
    config.experiment_name = args.experiment_name or f"{scene_label}_refit_sfm"
    config.project_name = "neusky"
    config.output_dir = args.output_dir
    config.vis = args.vis
    config.data = dp.data
    if args.max_num_iterations is not None:
        config.max_num_iterations = args.max_num_iterations
    if args.reni_ckpt is not None:
        config.pipeline.model.illumination_field_ckpt_path = args.reni_ckpt.expanduser()

    ns_train_main(config)


if __name__ == "__main__":
    main()
