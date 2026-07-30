#!/usr/bin/env python3
"""Extract a coloured point cloud from rendered EXR passes + transforms.

Reads the depth maps, RGB images, and camera transforms from a rendered
dataset, unprojects visible pixels into 3D, and saves a PLY point cloud
suitable for initialising Gaussian Splatting.

Input structure (after split_multipass_exr.py):
    <input>/
        transforms_train.json
        train/
            rgb/Image0001.exr, ...
            depth/Image0001.exr, ...
            mask/Image0001.exr, ...

Output:
    <input>/points3d.ply

Usage:
    python extract_pointcloud.py --input /path/to/rendered/dataset \
        --max_depth 200 --subsample 4
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    raise ImportError("OpenEXR required: pip install OpenEXR")


def read_exr_channel(path, channel_name=None):
    """Read a single channel from an EXR file as a numpy array."""
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    if channel_name is None:
        channel_name = channels[0]

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(channel_name, pt)
    return np.frombuffer(raw, dtype=np.float32).reshape(h, w)


def read_exr_rgb(path):
    """Read RGB channels from an EXR file."""
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
    g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
    b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)
    return np.stack([r, g, b], axis=-1)


def write_ply(path, points, colors):
    """Write a point cloud to PLY format.

    Args:
        path: Output file path.
        points: (N, 3) float32 array of xyz positions.
        colors: (N, 3) uint8 array of rgb colours.
    """
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
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<fff", *points[i]))
            f.write(struct.pack("<BBB", *colors[i]))
    print(f"Wrote {n:,} points to {path}")


def unproject_depth(depth, mask, rgb, intrinsics, c2w, max_depth, subsample):
    """Unproject depth map pixels to 3D world coordinates.

    Args:
        depth: (H, W) depth map (z-depth from camera).
        mask: (H, W) alpha mask (0-1).
        rgb: (H, W, 3) colour image (linear float).
        intrinsics: dict with fl_x, fl_y, cx, cy.
        c2w: (4, 4) camera-to-world transform matrix.
        max_depth: Maximum depth to include.
        subsample: Take every Nth pixel in each dimension.

    Returns:
        points: (N, 3) world coordinates.
        colors: (N, 3) uint8 RGB.
    """
    h, w = depth.shape
    fl_x = intrinsics["fl_x"]
    fl_y = intrinsics["fl_y"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Create pixel grid (subsampled)
    ys = np.arange(0, h, subsample)
    xs = np.arange(0, w, subsample)
    xx, yy = np.meshgrid(xs, ys)

    # Sample values at subsampled positions
    d = depth[yy, xx]
    m = mask[yy, xx]
    r = rgb[yy, xx]

    # Filter: valid depth and object pixels
    valid = (d > 0) & (d < max_depth) & (m > 0.5)
    d = d[valid]
    r = r[valid]
    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Unproject to camera space (Blender convention: -Z forward)
    x_cam = (xx - cx) / fl_x * d
    y_cam = (yy - cy) / fl_y * d
    z_cam = -d  # Blender depth is along -Z

    pts_cam = np.stack([x_cam, -y_cam, z_cam, np.ones_like(d)], axis=-1)

    # Transform to world space
    c2w_mat = np.array(c2w, dtype=np.float64)
    pts_world = (c2w_mat @ pts_cam.T).T[:, :3]

    # Convert linear RGB to sRGB uint8
    r_clamped = np.clip(r, 0, 1)
    # Simple linear-to-sRGB gamma
    srgb = np.where(r_clamped <= 0.0031308,
                    12.92 * r_clamped,
                    1.055 * np.power(r_clamped, 1.0 / 2.4) - 0.055)
    colors = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

    return pts_world.astype(np.float32), colors


def main():
    parser = argparse.ArgumentParser(
        description="Extract point cloud from rendered EXR dataset"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to rendered dataset directory")
    parser.add_argument("--max_depth", type=float, default=200.0,
                        help="Maximum depth to include (default: 200)")
    parser.add_argument("--subsample", type=int, default=4,
                        help="Subsample factor (default: 4, every 4th pixel)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY path (default: <input>/points3d.ply)")
    parser.add_argument("--target_points", type=int, default=None,
                        help="Randomly downsample to this many points (default: keep all)")
    parser.add_argument("--voxel_size", type=float, default=None,
                        help="Voxel size for spatial downsampling (e.g. 0.05 = 5cm grid). "
                             "Keeps one point per voxel for uniform density.")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Std dev of Gaussian noise added to positions (simulates COLMAP)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for downsampling and noise")
    args = parser.parse_args()

    input_dir = Path(args.input)
    transforms_path = input_dir / "transforms_train.json"

    with open(transforms_path) as f:
        transforms = json.load(f)

    intrinsics = {
        "fl_x": transforms["fl_x"],
        "fl_y": transforms["fl_y"],
        "cx": transforms["cx"],
        "cy": transforms["cy"],
    }

    # Build per-frame intrinsics lookup (overrides global when present)
    frame_intrinsics = {}
    for frame in transforms["frames"]:
        if "fl_x" in frame:
            fp = frame["file_path"]
            # Extract stem: handle both "train/Image0001.exr" and "train/rgb/Image0001.exr"
            frame_key = Path(fp).stem
            frame_intrinsics[frame_key] = {
                "fl_x": frame["fl_x"],
                "fl_y": frame["fl_y"],
                "cx": frame["cx"],
                "cy": frame["cy"],
            }
    if frame_intrinsics:
        print(f"Found per-frame intrinsics for {len(frame_intrinsics)} frames")

    rgb_dir = input_dir / "train" / "rgb"
    depth_dir = input_dir / "train" / "depth"
    mask_dir = input_dir / "train" / "mask"

    for d in [rgb_dir, depth_dir, mask_dir]:
        if not d.exists():
            print(f"Error: {d} not found. Run split_multipass_exr.py first.")
            return

    all_points = []
    all_colors = []

    frames = transforms["frames"]
    print(f"Processing {len(frames)} frames...")

    for i, frame in enumerate(frames):
        stem = Path(frame["file_path"]).stem
        c2w = frame["transform_matrix"]

        # Use per-frame intrinsics if available, else global fallback
        frame_intr = frame_intrinsics.get(stem, intrinsics)

        depth = read_exr_channel(depth_dir / f"{stem}.exr")
        mask = read_exr_channel(mask_dir / f"{stem}.exr")
        rgb = read_exr_rgb(rgb_dir / f"{stem}.exr")

        pts, cols = unproject_depth(
            depth, mask, rgb, frame_intr, c2w,
            args.max_depth, args.subsample,
        )

        all_points.append(pts)
        all_colors.append(cols)
        valid_depth = depth[(depth > 0) & (depth < args.max_depth)]
        if len(valid_depth) > 0:
            print(f"  [{i+1}/{len(frames)}] {stem}: {len(pts):,} points "
                  f"(depth {valid_depth.min():.1f}-{valid_depth.max():.1f})")
        else:
            print(f"  [{i+1}/{len(frames)}] {stem}: {len(pts):,} points (no valid depth)")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"\nRaw total: {len(points):,} points from {len(frames)} frames")

    rng = np.random.default_rng(args.seed)

    # Voxel downsampling: one point per grid cell for uniform spatial density
    if args.voxel_size is not None and args.voxel_size > 0:
        voxel_coords = np.floor(points / args.voxel_size).astype(np.int64)
        # Unique voxels — keep first point encountered per cell
        _, unique_idx = np.unique(
            voxel_coords, axis=0, return_index=True
        )
        unique_idx.sort()
        points = points[unique_idx]
        colors = colors[unique_idx]
        print(f"Voxel downsampled (size={args.voxel_size}): {len(points):,} points")

    # Downsample to target count
    if args.target_points and len(points) > args.target_points:
        idx = rng.choice(len(points), args.target_points, replace=False)
        idx.sort()
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {len(points):,} points")

    # Add positional noise to simulate COLMAP triangulation error
    if args.noise_std > 0:
        noise = rng.normal(0, args.noise_std, size=points.shape).astype(np.float32)
        points += noise
        print(f"Added Gaussian noise (std={args.noise_std})")

    out_path = args.output or str(input_dir / "points3d.ply")
    write_ply(out_path, points, colors)
    print(f"Final: {len(points):,} points")


if __name__ == "__main__":
    main()
