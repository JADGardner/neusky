#!/usr/bin/env python
"""Render a trained GS-IR checkpoint into the NeuSky synthetic benchmark contract.

Workflow:
    1. Run ``prepare_gsir_synthetic.py`` for one or more NeuSky synthetic scenes.
    2. Train upstream GS-IR from ``thirdparty/gs-ir/train.py``.
    3. Run this adapter to export RGB/decomposition predictions that
       ``scripts/synthetic_benchmark/evaluate.py`` can score directly.

The adapter intentionally reuses GS-IR's unmodified ``Scene``, ``GaussianModel``
and ``gaussian_renderer.render`` APIs. It does not implement relighting; it
exports the trained test/train render plus albedo, normal, depth, roughness and
metallic maps where GS-IR provides them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
GSIR_ROOT = REPO_ROOT / "thirdparty" / "gs-ir"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def add_gsir_to_path() -> None:
    for p in (GSIR_ROOT, GSIR_ROOT / "gs-ir"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def latest_checkpoint(model_path: Path) -> Path:
    candidates = []
    for p in model_path.glob("chkpnt*.pth"):
        m = re.fullmatch(r"chkpnt(\d+)", p.stem)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No GS-IR chkpnt*.pth found in {model_path}")
    return max(candidates, key=lambda item: item[0])[1]


def tensor_chw_to_hwc(tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def tensor_hw(tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    return np.ascontiguousarray(np.squeeze(arr), dtype=np.float32)


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


def write_rgb_exr(path: Path, image: np.ndarray) -> None:
    from evaluate import write_exr

    write_exr(path, {"RGB": np.ascontiguousarray(image[..., :3], dtype=np.float32)})


def write_scalar_exr(path: Path, image: np.ndarray) -> None:
    from evaluate import write_exr

    write_exr(path, {"Y": np.ascontiguousarray(image, dtype=np.float32)})


def write_normal_exr(path: Path, normal: np.ndarray) -> None:
    from evaluate import write_exr

    write_exr(
        path,
        {
            "X": np.ascontiguousarray(normal[..., 0], dtype=np.float32),
            "Y": np.ascontiguousarray(normal[..., 1], dtype=np.float32),
            "Z": np.ascontiguousarray(normal[..., 2], dtype=np.float32),
        },
    )


def make_dataset_args(args: argparse.Namespace):
    from arguments import GroupParams

    dataset = GroupParams()
    dataset.sh_degree = args.sh_degree
    dataset.source_path = str(args.data.expanduser().resolve())
    dataset.model_path = str(args.model_path.expanduser().resolve())
    dataset.images = "images"
    dataset.resolution = args.resolution
    dataset.white_background = args.white_background
    dataset.data_device = "cuda"
    dataset.eval = True
    return dataset


def make_pipeline_args(args: argparse.Namespace):
    from arguments import GroupParams

    pipe = GroupParams()
    pipe.convert_SHs_python = args.convert_SHs_python
    pipe.compute_cov3D_python = args.compute_cov3D_python
    pipe.debug = args.debug
    return pipe


def load_scene(args: argparse.Namespace):
    import torch
    from gaussian_renderer import GaussianModel
    from scene import Scene

    checkpoint = args.checkpoint or latest_checkpoint(args.model_path.expanduser().resolve())
    dataset = make_dataset_args(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    state = torch.load(str(checkpoint), map_location="cuda", weights_only=False)
    if "gaussians" not in state:
        raise KeyError(f"{checkpoint} does not look like a GS-IR checkpoint: missing 'gaussians'")
    gaussians.restore(state["gaussians"])
    return scene, checkpoint


def render_predictions(args: argparse.Namespace) -> None:
    add_gsir_to_path()

    import torch
    from gaussian_renderer import render
    from tqdm import tqdm
    from utils.general_utils import safe_state

    if not torch.cuda.is_available():
        raise RuntimeError("GS-IR rendering requires CUDA")

    safe_state(args.quiet)
    scene, checkpoint = load_scene(args)
    pipe = make_pipeline_args(args)
    views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    if not views:
        raise ValueError(f"No {args.split} cameras found. For GS-IR prepared data, train with --eval.")

    pred_dir = args.pred_dir.expanduser().resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    background = torch.tensor(args.background, dtype=torch.float32, device="cuda")
    frames = []

    with torch.no_grad():
        for view in tqdm(views, desc=f"Rendering GS-IR {args.split}"):
            result = render(
                viewpoint_camera=view,
                pc=scene.gaussians,
                pipe=pipe,
                bg_color=background,
                inference=True,
                pad_normal=True,
                derive_normal=True,
            )
            stem = view.image_name
            rgb = tensor_chw_to_hwc(result["render"])
            albedo = tensor_chw_to_hwc(result["albedo_map"])
            normal = tensor_chw_to_hwc(result["normal_map"])
            depth = tensor_hw(result["depth_map"])
            roughness = tensor_hw(result["roughness_map"])
            metallic = tensor_hw(result["metallic_map"])

            save_png(pred_dir / f"{stem}_rgb.png", rgb)
            write_rgb_exr(pred_dir / f"{stem}_albedo.exr", albedo)
            write_normal_exr(pred_dir / f"{stem}_normal.exr", normal)
            write_scalar_exr(pred_dir / f"{stem}_depth.exr", depth)
            write_scalar_exr(pred_dir / f"{stem}_roughness.exr", roughness)
            write_scalar_exr(pred_dir / f"{stem}_metallic.exr", metallic)
            frames.append(stem)

    manifest = {
        "method": "gs_ir",
        "data": str(args.data.expanduser().resolve()),
        "model_path": str(args.model_path.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "resolution": args.resolution,
        "frames": frames,
        "outputs": ["rgb", "albedo", "normal", "depth", "roughness", "metallic"],
        "notes": [
            "RGB is GS-IR's direct Gaussian render, not external HDRI relighting.",
            "Depth is GS-IR rasterizer depth in the prepared scene coordinate system; evaluator also reports scale-aligned depth.",
        ],
    }
    (pred_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[rendered] {len(frames)} frames -> {pred_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared GS-IR scene directory from prepare_gsir_synthetic.py")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="GS-IR training output directory containing cfg_args/chkpnt*.pth")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Specific GS-IR chkpnt*.pth. Defaults to latest under --model-path.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--resolution", type=int, default=1,
                        help="GS-IR image downscale. Use 1 for benchmark scoring.")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_predictions(args)


if __name__ == "__main__":
    main()
