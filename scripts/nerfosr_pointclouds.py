#!/usr/bin/env python3
"""Manage NeRF-OSR SfM point clouds for the NeuSky dataparser rescaling.

The NeRF-OSR scenes (lk2, st, lwp) have COLMAP point clouds triangulated
against the dataset's known cam_dict_norm.json poses (so they live in the
same normalized space as the final/*/pose/*.txt files the dataparser reads).
The dataparser's `center_method_sfm` option uses Data/<scene>/final/points3d.ply
to compute a robust scene center and percentile-based scale.

Subcommands:
  convert  Read Data/<scene>/colmap/sparse_triangulated/points3D.bin and write
           Data/<scene>/final/points3d.ply for each scene.
  package  Stage only the converted PLYs (plus a README) into a standalone
           zip.
  install  Extract a downloaded zip (or copy from an extracted folder) into a
           NeRF-OSR Data tree, placing each points3d.ply under <scene>/final/.

Typical usage:
  python scripts/nerfosr_pointclouds.py convert --data ~/data/NeRF-OSR/Data
  python scripts/nerfosr_pointclouds.py package --data ~/data/NeRF-OSR/Data \
      --output ~/Downloads/nerfosr_neusky_point_clouds.zip
  python scripts/nerfosr_pointclouds.py install --zip nerfosr_neusky_point_clouds.zip \
      --data /path/to/NeRF-OSR/Data
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neusky.data.dataparsers import sfm_utils  # noqa: E402

DEFAULT_SCENES = ["lk2", "st", "lwp"]

README_TEXT = """NeuSky NeRF-OSR SfM point clouds
================================

One points3d.ply per scene, destined for NeRF-OSR/Data/<scene>/final/points3d.ply.

Provenance: COLMAP feature extraction + matching over all splits, then
`colmap point_triangulator` with poses FIXED to the dataset's known
cam_dict_norm.json cameras (identical to kai_cameras_normalized.json and to
inv(W2C) of the per-image pose/*.txt files). The clouds therefore sit in the
dataset-normalized coordinate space and need no registration.

Used by the NeuSky `NeRFOSRCityScapes` dataparser when `center_method_sfm=True`
to compute a robust scene center and percentile-based scale (instead of
camera-position-based auto-scaling).

Install:
  python scripts/nerfosr_pointclouds.py install --zip <this zip> --data <NeRF-OSR/Data>

PLY format: binary_little_endian, x/y/z float32 + red/green/blue uchar.
"""


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write xyz float32 + rgb uint8 points to a binary PLY file."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
    print(f"Wrote {n:,} points to {path}")


def cmd_convert(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    failures = 0
    for scene in args.scenes:
        bin_path = data / scene / "colmap" / "sparse_triangulated" / "points3D.bin"
        out_path = data / scene / "final" / "points3d.ply"
        loaded = sfm_utils.load_colmap_points3d_bin(bin_path)
        if loaded is None:
            print(f"[{scene}] SKIP: {bin_path} not found or unreadable")
            failures += 1
            continue
        xyz, rgb = loaded
        if out_path.exists() and not args.force:
            print(f"[{scene}] SKIP: {out_path} exists (use --force to overwrite)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_ply(out_path, xyz, rgb)
        # Round-trip sanity check via the same loader the dataparser uses
        reread = sfm_utils.load_ply_points(out_path)
        assert reread is not None and len(reread) == len(xyz), f"round-trip failed for {scene}"
        assert np.allclose(reread, xyz.astype(np.float32), atol=1e-6), f"round-trip mismatch for {scene}"
    return failures


def cmd_package(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nerfosr_point_clouds/README.txt", README_TEXT)
        for scene in args.scenes:
            ply = data / scene / "final" / "points3d.ply"
            if not ply.exists():
                print(f"[{scene}] MISSING {ply} — run `convert` first")
                return 1
            zf.write(ply, f"nerfosr_point_clouds/{scene}/final/points3d.ply")
            print(f"[{scene}] added {ply}")
    print(f"Packaged -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    zip_path = Path(args.zip).expanduser()
    if not zip_path.exists():
        print(f"Zip not found: {zip_path}")
        return 1
    installed = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("points3d.ply"):
                continue
            # nerfosr_point_clouds/<scene>/final/points3d.ply -> <scene>/final/points3d.ply
            parts = Path(name).parts
            scene = parts[-3]
            dest = data / scene / "final" / "points3d.ply"
            if dest.exists() and not args.force:
                print(f"[{scene}] SKIP: {dest} exists (use --force to overwrite)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out:
                out.write(src.read())
            print(f"[{scene}] installed -> {dest}")
            installed += 1
    print(f"Installed {installed} point cloud(s) into {data}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="COLMAP points3D.bin -> final/points3d.ply per scene")
    p_convert.add_argument("--data", default="~/data/NeRF-OSR/Data", help="NeRF-OSR Data root")
    p_convert.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p_convert.add_argument("--force", action="store_true", help="Overwrite existing PLYs")
    p_convert.set_defaults(func=cmd_convert)

    p_package = sub.add_parser(
        "package", help="Stage only the converted PLYs into a zip"
    )
    p_package.add_argument("--data", default="~/data/NeRF-OSR/Data", help="NeRF-OSR Data root")
    p_package.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p_package.add_argument("--output", default="~/Downloads/nerfosr_neusky_point_clouds.zip")
    p_package.set_defaults(func=cmd_package)

    p_install = sub.add_parser("install", help="Extract a downloaded zip into a NeRF-OSR Data tree")
    p_install.add_argument("--zip", required=True, help="Path to nerfosr_neusky_point_clouds.zip")
    p_install.add_argument("--data", default="~/data/NeRF-OSR/Data", help="NeRF-OSR Data root")
    p_install.add_argument("--force", action="store_true", help="Overwrite existing PLYs")
    p_install.set_defaults(func=cmd_install)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
