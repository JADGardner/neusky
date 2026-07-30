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
    parser.add_argument("--train-gt-supervision", action="store_true",
                        help="Enable train-time synthetic GT albedo/normal/depth losses.")
    parser.add_argument("--gt-albedo-loss-weight", type=float, default=1.0)
    parser.add_argument("--gt-normal-loss-weight", type=float, default=0.1)
    parser.add_argument("--gt-depth-loss-weight", type=float, default=1.0)
    parser.add_argument("--fit-gt-envmap-latents", action="store_true",
                        help="Fit train/eval RENI++ latents directly to correctly rotated synthetic GT HDRIs.")
    parser.add_argument("--gt-envmap-loss-weight", type=float, default=0.1)
    parser.add_argument("--gt-envmap-loss-num-directions", type=int, default=128)
    parser.add_argument("--gt-envmap-fit-num-directions", type=int, default=2048)
    parser.add_argument("--gt-envmap-fit-num-images-per-step", type=int, default=8)
    parser.add_argument("--gt-envmap-fit-width", type=int, default=128)
    parser.add_argument("--train-num-images-to-sample-from", type=int, default=None,
                        help="Override datamanager train_num_images_to_sample_from for resource-limited smoke tests.")
    parser.add_argument("--train-num-times-to-repeat-images", type=int, default=None,
                        help="Override datamanager train_num_times_to_repeat_images for resource-limited smoke tests.")
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
    if args.train_num_images_to_sample_from is not None:
        config.pipeline.datamanager.train_num_images_to_sample_from = args.train_num_images_to_sample_from
    if args.train_num_times_to_repeat_images is not None:
        config.pipeline.datamanager.train_num_times_to_repeat_images = args.train_num_times_to_repeat_images
    if args.reni_ckpt is not None:
        config.pipeline.model.illumination_field_ckpt_path = args.reni_ckpt.expanduser()

    config.pipeline.model.gt_envmap_loss_num_directions = args.gt_envmap_loss_num_directions
    config.pipeline.model.gt_envmap_fit_num_directions = args.gt_envmap_fit_num_directions
    config.pipeline.model.gt_envmap_fit_num_images_per_step = args.gt_envmap_fit_num_images_per_step
    config.pipeline.model.gt_envmap_fit_width = args.gt_envmap_fit_width

    if args.fit_gt_envmap_latents:
        config.pipeline.model.eval_latent_optimise_method = "synthetic_gt_envmap"
        loss_inclusions = dict(config.pipeline.model.loss_inclusions)
        loss_inclusions["gt_envmap_loss"] = True
        config.pipeline.model.loss_inclusions = loss_inclusions
        loss_coefficients = dict(config.pipeline.model.loss_coefficients)
        loss_coefficients["gt_envmap_loss"] = args.gt_envmap_loss_weight
        config.pipeline.model.loss_coefficients = loss_coefficients

    if args.train_gt_supervision:
        if args.train_num_images_to_sample_from is None:
            config.pipeline.datamanager.train_num_images_to_sample_from = 64
        if args.train_num_times_to_repeat_images is None:
            config.pipeline.datamanager.train_num_times_to_repeat_images = 64

        loss_inclusions = dict(config.pipeline.model.loss_inclusions)
        loss_inclusions.update({
            "gt_albedo_loss": True,
            "gt_normal_loss": True,
            "gt_depth_loss": True,
        })
        if args.fit_gt_envmap_latents:
            loss_inclusions["gt_envmap_loss"] = True
        config.pipeline.model.loss_inclusions = loss_inclusions

        loss_coefficients = dict(config.pipeline.model.loss_coefficients)
        loss_coefficients.update({
            "gt_albedo_loss": args.gt_albedo_loss_weight,
            "gt_normal_loss": args.gt_normal_loss_weight,
            "gt_depth_loss": args.gt_depth_loss_weight,
        })
        if args.fit_gt_envmap_latents:
            loss_coefficients["gt_envmap_loss"] = args.gt_envmap_loss_weight
        config.pipeline.model.loss_coefficients = loss_coefficients

    ns_train_main(config)


if __name__ == "__main__":
    main()
