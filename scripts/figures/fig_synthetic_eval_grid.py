"""Synthetic benchmark eval grid from saved prediction directories.

Rows are synthetic test frames; columns are GT RGB, predicted RGB, GT albedo,
predicted albedo, GT normal, predicted normal. Inputs are the method-agnostic
benchmark contract produced by scripts/synthetic_benchmark/render_neusky_predictions.py
and scored by scripts/synthetic_benchmark/evaluate.py, so the figure is tied to
the same artifacts as tab:synthetic.

Example:
    PYTHONPATH=. python scripts/figures/fig_synthetic_eval_grid.py \
        --scenes abandoned_buildings --frame-indices 0 \
        --output outputs/figures/synthetic_eval_grid_abandoned_preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from _common import (OUTPUTS_ROOT, SYNTHETIC_ROOT, SYNTHETIC_SCENES,
                     add_common_args, canonical_scene, save_figure, seed_all)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "scripts" / "synthetic_benchmark"

# Frame index per scene, chosen for good viewpoints.
DEFAULT_FRAMES = {
    "abandoned_buildings_prepared": 10,
    "apartment_building_prepared": 10,
    "arlanda_uppsala_cathedral_prepared": 10,
    "glass_building_prepared": 0,
    "interstellar_house_prepared": 12,
}

COLUMNS = ("GT RGB", "Pred RGB", "GT Albedo", "Pred Albedo", "GT Normal", "Pred Normal")

# Multi-method mode: benchmark pred roots per method (the same contract dirs
# the evaluate.py scores read; NeuSky's wave-3 dirs land via the Isambard
# eval jobs, baselines via their render_*_predictions.py scripts).
METHOD_LABELS = {"neusky": "NeuSky (ours)", "nerf_osr": "NeRF-OSR", "gs_ir": "GS-IR"}
MODALITIES = ("rgb", "albedo", "normal")


def _import_evaluator():
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    import evaluate

    return evaluate


def read_png(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_exr_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    evaluate = _import_evaluator()
    return evaluate._first_channels(evaluate.read_exr(path), 3).astype(np.float32)[..., :3]


def albedo_vis(albedo: np.ndarray) -> np.ndarray:
    evaluate = _import_evaluator()
    return evaluate.linear_to_srgb(np.clip(albedo[..., :3], 0.0, 1.0))


def normal_vis(normal: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normal, axis=-1)
    vis = (normal[..., :3] + 1.0) * 0.5
    vis[norm < 0.5] = 1.0
    return np.clip(vis, 0.0, 1.0)


def resize_for_display(img: np.ndarray, max_width: int) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    height = max(1, round(img.shape[0] * max_width / img.shape[1]))
    pil = Image.fromarray((img * 255.0).round().astype(np.uint8))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil.resize((max_width, height), resampling), dtype=np.float32) / 255.0


def scene_prediction_dir(pred_root: Path, scene: str, track: str) -> Path:
    stem = canonical_scene(scene).removesuffix("_prepared")
    return pred_root / f"{stem}_{track}"


def read_modality(base: Path, stem: str, modality: str, is_gt: bool) -> np.ndarray:
    if modality == "rgb":
        return read_png(base / "rgb" / f"{stem}.png" if is_gt
                        else base / f"{stem}_rgb.png")
    if modality == "albedo":
        return albedo_vis(read_exr_rgb(
            base / "albedo" / f"{stem}.exr" if is_gt else base / f"{stem}_albedo.exr"))
    return normal_vis(read_exr_rgb(
        base / "normal" / f"{stem}.exr" if is_gt else base / f"{stem}_normal.exr"))


def render_method_row(scene: str, frame_idx: int, data_root: Path,
                      method_roots: dict, track: str, modality: str,
                      max_width: int):
    """GT then one cell per method for a single modality. Missing method
    predictions become grey placeholders so partial matrices still render."""
    scene = canonical_scene(scene)
    stem = f"{frame_idx:04d}"
    cells = [read_modality(data_root / scene / "test", stem, modality, True)]
    for method, root in method_roots.items():
        pred_dir = scene_prediction_dir(Path(root), scene, track)
        try:
            cells.append(read_modality(pred_dir, stem, modality, False))
        except FileNotFoundError:
            print(f"[warn] {method}/{scene}/{modality}: missing - placeholder")
            cells.append(np.full_like(cells[0], 0.85))
    return [resize_for_display(c, max_width) for c in cells]


def render_scene_cells(scene: str, frame_idx: int, data_root: Path, pred_root: Path,
                       track: str, max_width: int):
    scene = canonical_scene(scene)
    stem = f"{frame_idx:04d}"
    scene_dir = data_root / scene / "test"
    pred_dir = scene_prediction_dir(pred_root, scene, track)

    cells = [
        read_png(scene_dir / "rgb" / f"{stem}.png"),
        read_png(pred_dir / f"{stem}_rgb.png"),
        albedo_vis(read_exr_rgb(scene_dir / "albedo" / f"{stem}.exr")),
        albedo_vis(read_exr_rgb(pred_dir / f"{stem}_albedo.exr")),
        normal_vis(read_exr_rgb(scene_dir / "normal" / f"{stem}.exr")),
        normal_vis(read_exr_rgb(pred_dir / f"{stem}_normal.exr")),
    ]
    return [resize_for_display(cell, max_width) for cell in cells]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "synthetic_eval_grid", needs_device=False)
    parser.add_argument("--scenes", nargs="+", default=list(SYNTHETIC_SCENES),
                        help="Synthetic scene names (bare stems accepted)")
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None,
                        help="Test frame per scene (default: curated views)")
    parser.add_argument("--track", choices=["gt", "estimated"], default="gt",
                        help="Prediction track directory suffix to visualise")
    parser.add_argument("--data-root", type=Path, default=SYNTHETIC_ROOT,
                        help="Synthetic renders root containing *_prepared dirs")
    parser.add_argument("--pred-root", type=Path,
                        default=OUTPUTS_ROOT / "synthetic_benchmark" / "neusky",
                        help="Root containing <scene>_<track> prediction dirs")
    parser.add_argument("--max-width", type=int, default=480,
                        help="Display width per cell; predictions are scored at native resolution")
    parser.add_argument("--labels", action="store_true",
                        help="Draw row/column labels (omit when LaTeX adds them)")
    parser.add_argument("--methods", nargs="+", default=None,
                        metavar="NAME[=PRED_ROOT]",
                        help="Multi-method mode: GT | method columns for one "
                             "--modality. Default roots: "
                             "outputs/synthetic_benchmark/<name>")
    parser.add_argument("--modality", choices=MODALITIES, default="rgb",
                        help="Which modality to show in --methods mode")
    args = parser.parse_args()
    seed_all(args.seed)

    scenes = [canonical_scene(s) for s in args.scenes]
    frames = args.frame_indices or [DEFAULT_FRAMES.get(s, 0) for s in scenes]
    if len(frames) != len(scenes):
        raise SystemExit("--frame-indices must give one index per scene")

    if args.methods:
        method_roots = {}
        for spec in args.methods:
            name, _, root = spec.partition("=")
            method_roots[name] = Path(root) if root else (
                OUTPUTS_ROOT / "synthetic_benchmark" / name)
        columns = ["GT"] + [METHOD_LABELS.get(m, m) for m in method_roots]
        rows = [render_method_row(scene, frame, args.data_root, method_roots,
                                  args.track, args.modality, args.max_width)
                for scene, frame in zip(scenes, frames)]
        n_rows, n_cols = len(rows), len(columns)
        h, w = rows[0][0].shape[:2]
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows * h / w),
            squeeze=False)
        for r, row in enumerate(rows):
            for c, img in enumerate(row):
                axs[r, c].imshow(img)
                axs[r, c].axis("off")
                if args.labels and r == 0:
                    axs[r, c].set_title(columns[c], fontsize=9)
        plt.tight_layout(pad=0.2)
        save_figure(fig, args.output, svg=args.svg)
        return

    rows = [render_scene_cells(scene, frame, args.data_root, args.pred_root,
                               args.track, args.max_width)
            for scene, frame in zip(scenes, frames)]

    n_rows, n_cols = len(rows), len(COLUMNS)
    h, w = rows[0][0].shape[:2]
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows * h / w), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            if args.labels and r == 0:
                axs[r, c].set_title(COLUMNS[c], fontsize=8)
        if args.labels:
            axs[r, 0].set_ylabel(SYNTHETIC_SCENES.get(scenes[r], scenes[r]), fontsize=8)
    plt.tight_layout(pad=0.2)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
