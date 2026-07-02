#!/usr/bin/env python
"""Prepare NeuSky synthetic scenes for the upstream NeRF-OSR code.

The upstream NeRF-OSR loader expects a NeRF++-style directory:

    <datadir>/<scene>/{train,validation,test}/
        rgb/*.png
        mask/*.png
        intrinsics/*.txt
        pose/*.txt
        max_depth.txt

This script derives that layout from a NeuSky synthetic ``*_prepared`` scene,
normalises poses into the unit sphere expected by NeRF-OSR, and optionally
projects the known HDRI + rotation for each frame to the 9x3 spherical harmonic
text files consumed by NeRF-OSR's ``--test_env`` option.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SKY_COLOUR = np.array([70, 130, 180], dtype=np.uint8)

SH_BANDS = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
DIFFUSE_KERNEL = np.array(
    [math.pi, 2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0,
     math.pi / 4.0, math.pi / 4.0, math.pi / 4.0, math.pi / 4.0, math.pi / 4.0],
    dtype=np.float64,
)


@dataclass
class Normalisation:
    center: np.ndarray
    scale: float
    source: str
    sfm_scale_uncapped: float | None = None
    camera_radius_uncapped: float | None = None


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _scene_name(scene_dir: Path) -> str:
    return scene_dir.name.removesuffix("_prepared")


def _split_from_file_path(file_path: str) -> str:
    return Path(file_path).parts[0]


def _stem_from_file_path(file_path: str) -> str:
    return Path(file_path).stem


def _camera_positions(frames: Iterable[dict]) -> np.ndarray:
    return np.asarray([np.asarray(f["transform_matrix"], dtype=np.float64)[:3, 3] for f in frames])


def compute_normalisation(
    scene_dir: Path,
    frames: list[dict],
    max_camera_radius: float,
    sfm_outlier_percentile: float,
    sfm_scale_percentile: float,
    sfm_target_radius: float,
) -> Normalisation:
    """Match NeuSky's SfM percentile scaling, without changing world up axes."""
    camera_positions = _camera_positions(frames)
    points_path = scene_dir / "points3d.ply"
    if points_path.exists():
        from neusky.data.dataparsers.sfm_utils import compute_sfm_centering, load_ply_points

        points = load_ply_points(points_path)
        if points is not None and len(points) > 0:
            center, scale = compute_sfm_centering(
                points,
                outlier_percentile=sfm_outlier_percentile,
                scale_percentile=sfm_scale_percentile,
                target_radius=sfm_target_radius,
            )
            shifted_cameras = camera_positions - center[None, :]
            max_dist = float(np.linalg.norm(shifted_cameras, axis=-1).max())
            cap = max_camera_radius / max(max_dist, 1.0e-6)
            uncapped_scale = float(scale)
            uncapped_radius = float(uncapped_scale * max_dist)
            if scale > cap:
                print(
                    f"[scale] {scene_dir.name}: SfM scale {scale:.6f} would put "
                    f"cameras at radius {uncapped_radius:.3f}; capping to {cap:.6f}"
                )
                scale = cap
            return Normalisation(
                center=np.asarray(center, dtype=np.float64),
                scale=float(scale),
                source="sfm_percentile",
                sfm_scale_uncapped=uncapped_scale,
                camera_radius_uncapped=uncapped_radius,
            )

    center = camera_positions.mean(axis=0)
    max_dist = float(np.linalg.norm(camera_positions - center[None, :], axis=-1).max())
    scale = max_camera_radius / max(max_dist, 1.0e-6)
    print(f"[scale] {scene_dir.name}: no SfM cloud; using camera mean/cap scale {scale:.6f}")
    return Normalisation(center=center, scale=scale, source="camera_cap")


def opengl_to_nerfosr_c2w(c2w: np.ndarray, norm: Normalisation) -> np.ndarray:
    """Convert Blender/OpenGL c2w to NeRF-OSR/OpenCV c2w and normalise origin."""
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, 3] = (out[:3, 3] - norm.center) * norm.scale
    out[:3, 1:3] *= -1.0
    return out


def intrinsics_matrix(frame: dict, root: dict) -> np.ndarray:
    w = float(frame.get("w", root.get("w")))
    h = float(frame.get("h", root.get("h")))
    fx = float(frame.get("fl_x", root.get("fl_x")))
    fy = float(frame.get("fl_y", root.get("fl_y", fx)))
    cx = float(frame.get("cx", root.get("cx", w * 0.5)))
    cy = float(frame.get("cy", root.get("cy", h * 0.5)))
    k = np.eye(4, dtype=np.float64)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k


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


def write_mask(src: Path, dst: Path) -> None:
    try:
        import imageio.v3 as iio

        arr = iio.imread(src)
        writer = lambda path, data: iio.imwrite(path, data)
    except ImportError:
        from PIL import Image

        arr = np.asarray(Image.open(src))
        writer = lambda path, data: Image.fromarray(data).save(path)

    if arr.ndim == 2:
        mask = arr > 0
    else:
        rgb = arr[..., :3].astype(np.uint8)
        mask = ~np.all(rgb == SKY_COLOUR, axis=-1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer(dst, mask.astype(np.uint8) * 255)


def sh_basis(dirs: np.ndarray) -> np.ndarray:
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    return np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=1,
    )


def read_hdr(path: Path) -> np.ndarray:
    try:
        import pyexr

        arr = pyexr.open(str(path)).get()
    except ImportError:
        import imageio.v3 as iio

        arr = iio.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return np.nan_to_num(arr[..., :3], nan=0.0, posinf=0.0, neginf=0.0)


def project_hdri_to_diffuse_sh(hdri_path: Path, yaw: float, target_width: int, exposure: float) -> np.ndarray:
    """Project a yaw-rotated equirectangular HDRI to NeRF-OSR's 9x3 SH env."""
    hdr = read_hdr(hdri_path)
    h, w = hdr.shape[:2]
    stride = max(1, int(math.ceil(w / target_width)))
    hdr = hdr[::stride, ::stride, :].astype(np.float64) * float(exposure)
    hs, ws = hdr.shape[:2]

    u = (np.arange(ws, dtype=np.float64) + 0.5) / ws
    v = (np.arange(hs, dtype=np.float64) + 0.5) / hs
    lon = (u[None, :] * 2.0 * math.pi) + yaw
    lat = (0.5 - v[:, None]) * math.pi
    cos_lat = np.cos(lat)

    dirs = np.stack(
        [
            -cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.broadcast_to(np.sin(lat), (hs, ws)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    basis = sh_basis(dirs)
    weights = (
        np.broadcast_to(np.cos(lat), (hs, ws)) * (2.0 * math.pi / ws) * (math.pi / hs)
    ).reshape(-1)
    radiance_coeffs = np.einsum("nc,nk,n->kc", hdr.reshape(-1, 3), basis, weights)
    return (radiance_coeffs * DIFFUSE_KERNEL[:, None]).astype(np.float32)


def resolve_hdri_path(hdri_root: Path, env_name: str) -> Path:
    candidates = [
        hdri_root / f"{env_name}.exr",
        hdri_root.parent / "hdris_16k" / f"{env_name}.exr",
        hdri_root.parent / "hdris" / f"{env_name}.exr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"HDRI not found for {env_name}; tried " + ", ".join(str(c) for c in candidates)
    )


def write_env_sh(
    frame: dict,
    output_path: Path,
    hdri_root: Path,
    target_width: int,
    exposure: float,
    cache: dict[tuple[str, float], np.ndarray],
) -> None:
    env_name = frame.get("envmap_name")
    if not env_name:
        return
    yaw = float((frame.get("envmap_rotation") or [0.0, 0.0, 0.0])[2])
    hdri_path = resolve_hdri_path(hdri_root, env_name)
    key = (str(hdri_path), yaw)
    if key not in cache:
        cache[key] = project_hdri_to_diffuse_sh(hdri_path, yaw, target_width, exposure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, cache[key], fmt="%.9g")


def write_config(
    config_path: Path,
    *,
    datadir: str,
    scene_name: str,
    expname: str,
    basedir: str,
    n_iters: int,
    resolution_level: float,
    world_size: int,
    i_weights: int,
    i_img: int,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""### INPUT
datadir = {datadir}
scene = {scene_name}
expname = {expname}
basedir = {basedir}
config = None
ckpt_path = None
no_reload = False
testskip = 1

### TRAINING
N_iters = {n_iters}
N_rand = 1024
lrate = 0.0005
lrate_decay_factor = 0.1
lrate_decay_steps = 50000000

### CASCADE
cascade_level = 2
cascade_samples = 64,128

### TESTING
chunk_size = 8192
render_splits = test
resolution_level = {resolution_level}
world_size = {world_size}

### RENDERING
det = False
max_freq_log2 = 12
max_freq_log2_viewdirs = 4
N_anneal = 30000
N_anneal_min_freq = 8
N_anneal_min_freq_viewdirs = 4
netdepth = 8
netwidth = 256
use_viewdirs = False
activation = relu

### CONSOLE AND TENSORBOARD
i_img = {i_img}
i_print = 100
i_weights = {i_weights}
"""
    config_path.write_text(text)


def prepare_scene(args: argparse.Namespace, scene_dir: Path) -> Path:
    scene_dir = scene_dir.expanduser().resolve()
    root = _read_json(scene_dir / "transforms.json")
    frames = root["frames"]
    scene_name = args.scene_name or _scene_name(scene_dir)
    output_scene = args.output_root.expanduser().resolve() / scene_name
    hdri_root = args.hdri_root.expanduser().resolve() if args.hdri_root else scene_dir.parent.parent / "hdris"

    norm = compute_normalisation(
        scene_dir,
        frames,
        max_camera_radius=args.max_camera_radius,
        sfm_outlier_percentile=args.sfm_outlier_percentile,
        sfm_scale_percentile=args.sfm_scale_percentile,
        sfm_target_radius=args.sfm_target_radius,
    )

    splits = set(args.splits)
    counts = {split: 0 for split in splits}
    link_counts: dict[str, int] = {}
    env_cache: dict[tuple[str, float], np.ndarray] = {}

    for frame in frames:
        split = _split_from_file_path(frame["file_path"])
        if split not in splits:
            continue
        stem = _stem_from_file_path(frame["file_path"])
        src_rgb = scene_dir / frame["file_path"]
        src_mask = scene_dir / split / "cityscapes_mask" / f"{stem}.png"
        split_dir = output_scene / split

        link_kind = link_or_copy(src_rgb, split_dir / "rgb" / f"{stem}.png", args.link_mode)
        link_counts[link_kind] = link_counts.get(link_kind, 0) + 1
        write_mask(src_mask, split_dir / "mask" / f"{stem}.png")
        (split_dir / "intrinsics").mkdir(parents=True, exist_ok=True)
        (split_dir / "pose").mkdir(parents=True, exist_ok=True)
        np.savetxt(split_dir / "intrinsics" / f"{stem}.txt", intrinsics_matrix(frame, root), fmt="%.10g")
        c2w = opengl_to_nerfosr_c2w(np.asarray(frame["transform_matrix"]), norm)
        np.savetxt(split_dir / "pose" / f"{stem}.txt", c2w, fmt="%.10g")
        (split_dir / "max_depth.txt").write_text(f"{args.max_depth:.8g}\n")

        if args.write_env_sh and split in args.env_sh_splits:
            write_env_sh(
                frame,
                output_scene / "env_sh" / split / f"{stem}.txt",
                hdri_root=hdri_root,
                target_width=args.env_sh_width,
                exposure=args.env_sh_exposure,
                cache=env_cache,
            )
        counts[split] += 1

    manifest = {
        "source_scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "normalisation": {
            "center": norm.center.tolist(),
            "scale": norm.scale,
            "metric_depth_scale": 1.0 / norm.scale,
            "source": norm.source,
            "sfm_scale_uncapped": norm.sfm_scale_uncapped,
            "camera_radius_uncapped": norm.camera_radius_uncapped,
            "max_camera_radius": args.max_camera_radius,
        },
        "camera_axes": "source OpenGL c2w converted to NeRF-OSR/OpenCV by negating c2w Y/Z columns",
        "splits": counts,
        "link_counts": link_counts,
        "env_sh": {
            "written": args.write_env_sh,
            "root": str(output_scene / "env_sh"),
            "hdri_root": str(hdri_root),
            "target_width": args.env_sh_width,
            "splits": args.env_sh_splits if args.write_env_sh else [],
            "diffuse_convolved": True,
        },
    }
    output_scene.mkdir(parents=True, exist_ok=True)
    (output_scene / "neusky_synthetic_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.config_dir is not None:
        config_datadir = args.config_datadir or str(args.output_root.expanduser().resolve())
        write_config(
            args.config_dir.expanduser().resolve() / f"{scene_name}.txt",
            datadir=config_datadir,
            scene_name=scene_name,
            expname=args.expname or f"synthetic_{scene_name}_nerfosr",
            basedir=args.basedir,
            n_iters=args.n_iters,
            resolution_level=args.resolution_level,
            world_size=args.world_size,
            i_weights=args.i_weights,
            i_img=args.i_img,
        )

    print(f"[prepared] {scene_name}: {counts} -> {output_scene}")
    return output_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="One or more NeuSky synthetic *_prepared scene dirs.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="NeRF-OSR datadir root to create.")
    parser.add_argument("--scene-name", default=None,
                        help="Override scene name. Only valid with one --data scene.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "copy", "symlink"],
                        default="auto",
                        help="How RGB files are materialised. auto tries hardlink then copy.")
    parser.add_argument("--max-camera-radius", type=float, default=0.95)
    parser.add_argument("--sfm-outlier-percentile", type=float, default=95.0)
    parser.add_argument("--sfm-scale-percentile", type=float, default=50.0)
    parser.add_argument("--sfm-target-radius", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--hdri-root", type=Path, default=None)
    parser.add_argument("--no-env-sh", dest="write_env_sh", action="store_false",
                        help="Skip HDRI projection to NeRF-OSR SH .txt files.")
    parser.set_defaults(write_env_sh=True)
    parser.add_argument("--env-sh-width", type=int, default=256,
                        help="Downsampled equirect width used for SH projection.")
    parser.add_argument("--env-sh-exposure", type=float, default=1.0)
    parser.add_argument("--env-sh-splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"],
                        help="Splits for which to write known-illumination SH files.")
    parser.add_argument("--config-dir", type=Path,
                        default=Path("thirdparty/nerf-osr/configs/neusky_synthetic"),
                        help="Directory for generated NeRF-OSR config files; pass empty string to skip.")
    parser.add_argument("--config-datadir", default=None,
                        help="datadir string written into generated configs, e.g. /workspace/data/...")
    parser.add_argument("--basedir", default="/workspace/outputs/synthetic_benchmark/nerf_osr_logs")
    parser.add_argument("--expname", default=None)
    parser.add_argument("--n-iters", type=int, default=500001)
    parser.add_argument("--resolution-level", type=float, default=4)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--i-weights", type=int, default=5000)
    parser.add_argument("--i-img", type=int, default=5000)
    args = parser.parse_args()
    if args.scene_name and len(args.data) != 1:
        parser.error("--scene-name can only be used with exactly one --data scene")
    if args.config_dir is not None and str(args.config_dir) == "":
        args.config_dir = None
    if args.link_mode == "auto":
        args.link_mode = "auto"
    return args


def main() -> None:
    args = parse_args()
    for scene_dir in args.data:
        prepare_scene(args, scene_dir)


if __name__ == "__main__":
    main()
