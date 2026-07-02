#!/usr/bin/env python
"""Render an upstream NeRF-OSR checkpoint into the synthetic benchmark contract.

Expected workflow:

1. Run ``prepare_nerfosr_synthetic.py`` to create a NeRF-OSR datadir/config.
2. Train upstream NeRF-OSR with that config.
3. Run this script from inside the NeuSky container with the same config.

The script writes ``*_rgb.png`` plus raw decomposition outputs where available
(``*_albedo.png``, ``*_normal.exr``, ``*_depth.exr``) so
``scripts/synthetic_benchmark/evaluate.py`` can score it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
NERF_OSR_ROOT = REPO_ROOT / "thirdparty" / "nerf-osr"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate import write_exr  # noqa: E402


def to8b(x: np.ndarray) -> np.ndarray:
    return (255 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def _load_manifest(datadir: Path, scene: str) -> dict:
    path = datadir / scene / "neusky_synthetic_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest {path}. Run prepare_nerfosr_synthetic.py first."
        )
    return json.loads(path.read_text())


def _config_args(config: Path, overrides: list[str], config_parser) -> argparse.Namespace:
    parser = config_parser()
    return parser.parse_args(["--config", str(config), *overrides])


def _ensure_single_gpu_args(args: argparse.Namespace, world_size: int, resolution_level: float) -> None:
    args.world_size = world_size
    args.resolution_level = resolution_level
    if args.world_size != 1:
        raise ValueError(
            "render_nerfosr_predictions.py currently writes files from one process. "
            "Use --world-size 1; upstream ddp_test_nerf.py remains available for multi-GPU RGB previews."
        )


def _read_mask(source_scene_dir: Path, split: str, stem: str) -> np.ndarray | None:
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    path = source_scene_dir / split / "cityscapes_mask" / f"{stem}.png"
    if not path.exists():
        return None
    arr = iio.imread(path)
    if arr.ndim == 2:
        return arr > 0
    sky = np.array([70, 130, 180], dtype=np.uint8)
    return ~np.all(arr[..., :3].astype(np.uint8) == sky, axis=-1)


def _write_png(path: Path, arr: np.ndarray) -> None:
    data = to8b(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, data)
    except ImportError:
        from PIL import Image

        Image.fromarray(data).save(path)


def render_predictions(args: argparse.Namespace) -> Path:
    import torch

    if str(NERF_OSR_ROOT) not in sys.path:
        sys.path.insert(0, str(NERF_OSR_ROOT))
    from data_loader_split import load_data_split
    from ddp_train_nerf import cleanup, config_parser, create_nerf, render_single_image, setup, setup_logger

    nerfosr_args = _config_args(args.config, args.config_overrides, config_parser)
    _ensure_single_gpu_args(nerfosr_args, args.world_size, args.resolution_level)
    if args.checkpoint is not None:
        nerfosr_args.ckpt_path = str(args.checkpoint)
    if args.test_env is not None:
        nerfosr_args.test_env = str(args.test_env)
    else:
        nerfosr_args.test_env = str(Path(nerfosr_args.datadir) / nerfosr_args.scene / "env_sh" / args.split)

    manifest = _load_manifest(Path(nerfosr_args.datadir), nerfosr_args.scene)
    metric_depth_scale = float(manifest["normalisation"]["metric_depth_scale"])
    source_scene_dir = Path(manifest["source_scene_dir"])

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    setup_logger()
    setup(0, 1)
    try:
        start, models = create_nerf(0, nerfosr_args)
        ray_samplers = load_data_split(
            nerfosr_args.datadir,
            nerfosr_args.scene,
            args.split,
            try_load_min_depth=nerfosr_args.load_min_depth,
            resolution_level=nerfosr_args.resolution_level,
            use_ray_jitter=False,
        )
        for idx, ray_sampler in enumerate(ray_samplers):
            stem = Path(ray_sampler.img_path).stem if ray_sampler.img_path else f"{idx:04d}"
            ret = render_single_image(
                0,
                1,
                models,
                ray_sampler,
                nerfosr_args.chunk_size,
                start,
                img_name=f"{stem}.png",
            )[-1]

            rgb = np.asarray(ret["rgb"].numpy(), dtype=np.float32)
            albedo = np.asarray(ret["fg_albedo"].numpy(), dtype=np.float32)
            # NeRF-OSR's fg_normal is anti-parallel to the dataset's outward
            # world-frame (Z-up) normals on this benchmark: scored as-written
            # it gives ~152° mean angular error vs ~28° after negation
            # (verified against interstellar_house GT, 2026-07-02). Negate at
            # export so *_normal.exr matches the prediction contract.
            normal = -np.asarray(ret["fg_normal"].numpy(), dtype=np.float32)
            depth = np.asarray(ret["fg_depth"].numpy(), dtype=np.float32) * metric_depth_scale
            bg_lambda = np.asarray(ret["bg_lambda"].numpy(), dtype=np.float32)

            # NeRF-OSR has no sky-depth convention. Mark likely no-hit pixels as
            # sky depth; the benchmark masks GT sky anyway, but this keeps the
            # raw output sane when inspected.
            no_hit = bg_lambda > args.no_hit_transmittance
            gt_mask = _read_mask(source_scene_dir, args.split, stem)
            if gt_mask is not None and gt_mask.shape == depth.shape:
                no_hit |= ~gt_mask
            depth = depth.copy()
            depth[no_hit] = 1.0e10

            _write_png(args.pred_dir / f"{stem}_rgb.png", rgb)
            # NeRF-OSR's learned albedo lives in the same sRGB-ish gauge as its
            # training PNGs. Writing PNG lets the benchmark linearise it.
            _write_png(args.pred_dir / f"{stem}_albedo.png", albedo)
            write_exr(args.pred_dir / f"{stem}_normal.exr", {
                "X": normal[..., 0],
                "Y": normal[..., 1],
                "Z": normal[..., 2],
            })
            write_exr(args.pred_dir / f"{stem}_depth.exr", {"Y": depth})
            torch.cuda.empty_cache()
    finally:
        cleanup()

    manifest_out = {
        "method": "nerf_osr",
        "source": "code/neusky/thirdparty/nerf-osr",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "latest",
        "split": args.split,
        "test_env": nerfosr_args.test_env,
        "source_manifest": manifest,
    }
    (args.pred_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    return args.pred_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Generated NeRF-OSR config, e.g. thirdparty/nerf-osr/configs/neusky_synthetic/scene.txt")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Specific model_*.pth checkpoint. Defaults to latest under basedir/expname.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test", "train"])
    parser.add_argument("--test-env", type=Path, default=None,
                        help="Directory of per-frame SH txt files. Defaults to <datadir>/<scene>/env_sh/<split>.")
    parser.add_argument("--resolution-level", type=float, default=1.0,
                        help="Render resolution. Use 1 for benchmark scoring.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--no-hit-transmittance", type=float, default=0.5)
    parser.add_argument("config_overrides", nargs=argparse.REMAINDER,
                        help="Optional raw NeRF-OSR configargparse overrides after --, e.g. -- --basedir ...")
    args = parser.parse_args()
    if args.config_overrides and args.config_overrides[0] == "--":
        args.config_overrides = args.config_overrides[1:]
    return args


def main() -> None:
    render_predictions(parse_args())


if __name__ == "__main__":
    main()
