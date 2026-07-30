#!/usr/bin/env python3
"""Render curated eval (val/test) views for the NeuSky synthetic dataset.

The production 300-frame renders sample train AND eval views from one random
camera distribution, so eval frames are often wall close-ups or mostly sky.
This script re-renders ONLY the eval views with a curated camera profile
(defined per scene in scene_render_configs.json) and can merge them into the
existing prepared dataset, leaving the 250 train frames untouched.

Runs locally (Blender on PATH, e.g. the 4090 box). Typical use:

    # 1. Quick visual check of the eval framing (8 low-res PNGs)
    python scripts/render_eval_views.py apartment_building --preview

    # 2. Full eval render (50 EXR frames) + split passes + merge into prepared
    python scripts/render_eval_views.py apartment_building --update-prepared

    # All scenes:
    for s in arlanda_uppsala_cathedral interstellar_house apartment_building \
             abandoned_buildings glass_building; do
        python scripts/render_eval_views.py $s --update-prepared
    done

Outputs:
    <data_root>/renders/<scene>_eval/            raw eval render (EXR mode)
    <data_root>/renders/<scene>_eval_preview/    preview render (PNG mode)
    <data_root>/renders/<scene>_prepared/        updated in place (--update-prepared)
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path.home() / "data" / "neusky_synthetic_data"


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Render curated eval views for a synthetic scene")
    parser.add_argument("scene", help="Scene name (key in scene_render_configs.json)")
    parser.add_argument("--profile", default="eval",
                        choices=["eval", "train_curated", "train"],
                        help="Camera/appearance profile: 'eval' (fixed focal/exposure, "
                             "whole-building framing), 'train_curated' (whole-building "
                             "framing with train appearance variation), 'train' "
                             "(production varied profile). Default: eval")
    parser.add_argument("--preview", action="store_true",
                        help="Fast PNG preview (8 frames, 960x540, 32 samples) "
                             "to visually check framing; no split/prepare")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of frames (default: eval 50, train_curated 100, "
                             "train 150; preview 8)")
    parser.add_argument("--n_val", type=int, default=25,
                        help="Validation frames when merging (default: 25)")
    parser.add_argument("--n_test", type=int, default=25,
                        help="Test frames when merging (default: 25)")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"Dataset root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--blender", default="blender",
                        help="Blender binary (default: blender on PATH)")
    parser.add_argument("--config", type=Path,
                        default=SCRIPT_DIR / "scene_render_configs.json",
                        help="Per-scene camera config file")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip rendering; only split + merge an existing eval render")
    parser.add_argument("--update-prepared", action="store_true", dest="update_prepared",
                        help="After rendering, merge the eval frames into "
                             "<data_root>/renders/<scene>_prepared (replaces its "
                             "validation/ and test/ splits)")
    parser.add_argument("--extra", default="",
                        help="Extra args appended to the Blender render command, "
                             "e.g. --extra '--sphere_radius_min 25'")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if args.scene not in config["scenes"]:
        sys.exit(f"Unknown scene '{args.scene}'. Known: {', '.join(config['scenes'])}")
    scene_cfg = config["scenes"][args.scene]

    blend = args.data_root / "scenes" / f"{args.scene}.blend"
    hdri_dir = args.data_root / "hdris"
    if not blend.exists():
        sys.exit(f"Scene file not found: {blend}")
    if not hdri_dir.is_dir():
        sys.exit(f"HDRI directory not found: {hdri_dir}")

    # Compose the profile: common appearance args + scene geometry + scene
    # camera args. 'train_curated' uses the scene's *eval* camera framing with
    # the *train* appearance variation (focal/exposure ranges, its own seed).
    profile_spec = {
        "eval": ("eval", "eval", 50, "_eval"),
        "train_curated": ("train_curated", "eval", 100, "_train_curated"),
        "train": ("train", "train", 150, "_train_varied"),
    }
    common_key, scene_key, default_frames, suffix = profile_spec[args.profile]

    if args.preview:
        suffix += "_preview"
    out_dir = args.data_root / "renders" / f"{args.scene}{suffix}"
    num_frames = args.num_frames or (8 if args.preview else default_frames)

    render_args = (
        shlex.split(config["common"][common_key])
        + shlex.split(scene_cfg["geometry"])
        + shlex.split(scene_cfg[scene_key])
        + ["--num_frames", str(num_frames)]
        + shlex.split(args.extra)
    )
    if args.preview:
        # Override quality settings for speed; 4K HDRIs load faster than 16K
        render_args = [a for a in render_args if a != "--hdri_16k"]
        render_args += ["--format", "png", "--samples", "32",
                        "--resolution", "960", "540"]
    else:
        render_args += ["--format", "exr"]

    if not args.skip_render:
        run([args.blender, "--background", blend,
             "--python", SCRIPT_DIR / "blender_render_scene.py", "--",
             "--output", out_dir, "--hdri_dir", hdri_dir] + render_args)

    if args.preview:
        print(f"\nPreview PNGs: {out_dir}/train/rgb/")
        return

    # Split multipart EXRs into per-pass files
    run([sys.executable, SCRIPT_DIR / "split_multipass_exr.py", "--input", out_dir])

    if args.update_prepared:
        if args.profile != "eval":
            sys.exit("--update-prepared only applies to the eval profile; "
                     "train renders are merged with prepare_synthetic_data.py "
                     "(multiple --input dirs)")
        prepared = args.data_root / "renders" / f"{args.scene}_prepared"
        run([sys.executable, SCRIPT_DIR / "prepare_synthetic_data.py",
             "--input", out_dir, "--output", prepared,
             "--update-eval", "--n-val", str(args.n_val),
             "--n-test", str(args.n_test)])
    else:
        print(f"\n{args.profile} render complete: {out_dir}")


if __name__ == "__main__":
    main()
