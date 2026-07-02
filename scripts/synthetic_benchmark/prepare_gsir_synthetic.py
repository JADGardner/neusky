#!/usr/bin/env python
"""Prepare NeuSky synthetic scenes for the upstream GS-IR Blender reader.

GS-IR already supports Blender-style ``transforms_train.json`` and
``transforms_test.json`` files. The NeuSky synthetic data stores all splits in
one ``transforms.json`` and keeps images under ``train/``, ``validation/`` and
``test/``. This script creates a small adapter directory with the transform
files and an initial point cloud, plus materialises RGB files by hardlink/copy.

This gets GS-IR to the point where its standard commands can run, but it does
not make GS-IR a relighting benchmark renderer by itself. A renderer-to-contract
adapter is still needed once GS-IR checkpoints are trained.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _scene_name(scene_dir: Path) -> str:
    return scene_dir.name.removesuffix("_prepared")


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "symlink":
        os.symlink(os.path.relpath(src, dst.parent), dst)
        return "symlink"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(src, dst)
        return "copy"


def write_ascii_xyzrgb_ply(dst: Path, xyz: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z in xyz:
            f.write(f"{x:.8g} {y:.8g} {z:.8g} 0 0 0 128 128 128\n")


def write_ply_from_source(src: Path, dst: Path, max_points: int | None) -> None:
    if not src.exists():
        return
    try:
        from plyfile import PlyData, PlyElement

        ply = PlyData.read(str(src))
        vertex = ply["vertex"]
        n = len(vertex)
        if max_points is not None and n > max_points:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n, size=max_points, replace=False))
            source = vertex.data[idx]
        else:
            source = vertex.data

        names = source.dtype.names or ()
        out_dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
        data = np.empty(len(source), dtype=out_dtype)
        for field in ("x", "y", "z"):
            data[field] = source[field].astype(np.float32)
        for field in ("nx", "ny", "nz"):
            data[field] = source[field].astype(np.float32) if field in names else 0.0
        for field in ("red", "green", "blue"):
            data[field] = source[field].astype(np.uint8) if field in names else 128

        dst.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(dst))
    except ImportError:
        from neusky.data.dataparsers.sfm_utils import load_ply_points

        xyz = load_ply_points(src)
        if xyz is None:
            return
        if max_points is not None and len(xyz) > max_points:
            rng = np.random.default_rng(42)
            xyz = xyz[np.sort(rng.choice(len(xyz), size=max_points, replace=False))]
        write_ascii_xyzrgb_ply(dst, xyz)


def prepare_scene(args: argparse.Namespace, scene_dir: Path) -> Path:
    scene_dir = scene_dir.expanduser().resolve()
    scene_name = args.scene_name or _scene_name(scene_dir)
    out_dir = args.output_root.expanduser().resolve() / scene_name
    root = json.loads((scene_dir / "transforms.json").read_text())

    split_map = {
        "train": "transforms_train.json",
        "validation": "transforms_val.json",
        "test": "transforms_test.json",
    }
    counts = {}
    link_counts: dict[str, int] = {}
    for split, filename in split_map.items():
        frames = []
        for frame in root["frames"]:
            path = Path(frame["file_path"])
            if path.parts[0] != split:
                continue
            stem = path.stem
            rel_no_ext = f"{split}/rgb/{stem}"
            copied = link_or_copy(scene_dir / frame["file_path"], out_dir / f"{rel_no_ext}.png", args.link_mode)
            link_counts[copied] = link_counts.get(copied, 0) + 1
            new_frame = dict(frame)
            new_frame["file_path"] = rel_no_ext
            frames.append(new_frame)
        if split == "validation" and not args.write_validation:
            continue
        payload = {
            "camera_angle_x": root["camera_angle_x"],
            "camera_angle_y": root.get("camera_angle_y"),
            "fl_x": root.get("fl_x"),
            "fl_y": root.get("fl_y"),
            "cx": root.get("cx"),
            "cy": root.get("cy"),
            "w": root.get("w"),
            "h": root.get("h"),
            "frames": frames,
        }
        (out_dir / filename).write_text(json.dumps(payload, indent=2))
        counts[split] = len(frames)

    write_ply_from_source(scene_dir / "points3d.ply", out_dir / "points3d.ply", args.max_points)
    manifest = {
        "method": "gs_ir",
        "source_scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "splits": counts,
        "link_counts": link_counts,
        "note": "GS-IR's Blender reader uses one camera_angle_x for all frames; per-frame focal metadata is preserved only in this manifest/transform payload.",
    }
    (out_dir / "neusky_synthetic_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prepared] {scene_name}: {counts} -> {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-name", default=None,
                        help="Override scene name. Only valid with one --data scene.")
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "copy", "symlink"],
                        default="auto")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Subsample points3d.ply for GS-IR initialisation; 0 copies all points.")
    parser.add_argument("--write-validation", action="store_true",
                        help="Also write transforms_val.json for inspection. GS-IR uses train/test.")
    args = parser.parse_args()
    if args.scene_name and len(args.data) != 1:
        parser.error("--scene-name can only be used with exactly one --data scene")
    if args.max_points == 0:
        args.max_points = None
    return args


def main() -> None:
    args = parse_args()
    for scene_dir in args.data:
        prepare_scene(args, scene_dir)


if __name__ == "__main__":
    main()
