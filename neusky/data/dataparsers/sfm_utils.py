"""Shared SfM point-cloud helpers for scene centering and scaling.

Used by both the synthetic CustomNeuskyDataparser and the NeRF-OSR
NeRFOSRCityScapes dataparser to derive a robust scene center and scale from a
sparse point cloud instead of from camera positions.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rich.console import Console

CONSOLE = Console(width=120)


def load_ply_points(ply_path: Path) -> Optional[np.ndarray]:
    """Load xyz positions from a PLY file. Returns (N, 3) float32 array or None."""
    ply_path = Path(ply_path)
    if not ply_path.exists():
        CONSOLE.log(f"[yellow]SfM point cloud not found: {ply_path}")
        return None

    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
        CONSOLE.log(f"Loaded {len(points)} SfM points from {ply_path}")
        return points
    except ImportError:
        CONSOLE.log("[yellow]plyfile not installed, falling back to numpy PLY loading")
        return _load_ply_numpy(ply_path)
    except Exception as e:  # noqa: BLE001
        CONSOLE.log(f"[yellow]Failed to load SfM points: {e}")
        return None


def _load_ply_numpy(ply_path: Path) -> Optional[np.ndarray]:
    """Fallback PLY loader using numpy for binary_little_endian x,y,z[,rgb] format."""
    try:
        with open(ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii").strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            n_vertices = 0
            is_binary = False
            for line in header_lines:
                if line.startswith("element vertex"):
                    n_vertices = int(line.split()[-1])
                if "binary_little_endian" in line:
                    is_binary = True

            if n_vertices == 0:
                return None

            if is_binary:
                dtype = np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                    ]
                )
                data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)
                return np.stack([data["x"], data["y"], data["z"]], axis=-1)
            data = np.loadtxt(f, max_rows=n_vertices)
            return data[:, :3].astype(np.float32)
    except Exception as e:  # noqa: BLE001
        CONSOLE.log(f"[yellow]Numpy PLY loading failed: {e}")
        return None


def load_colmap_points3d_bin(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a COLMAP points3D.bin file without pycolmap.

    The binary layout is stable across COLMAP versions (including 3.13):
    uint64 num_points, then per point: uint64 id, 3x double xyz, 3x uint8 rgb,
    double error, uint64 track_length, track_length x (uint32, uint32).

    Returns (xyz float32 [N, 3], rgb uint8 [N, 3]) or None.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            buf = f.read()
        (n_points,) = struct.unpack_from("<Q", buf, 0)
        offset = 8
        xyz = np.empty((n_points, 3), dtype=np.float64)
        rgb = np.empty((n_points, 3), dtype=np.uint8)
        for i in range(n_points):
            _, x, y, z, r, g, b, _err = struct.unpack_from("<QdddBBBd", buf, offset)
            offset += 8 + 24 + 3 + 8
            (track_len,) = struct.unpack_from("<Q", buf, offset)
            offset += 8 + track_len * 8
            xyz[i] = (x, y, z)
            rgb[i] = (r, g, b)
        CONSOLE.log(f"Loaded {n_points} COLMAP points from {path}")
        return xyz.astype(np.float32), rgb
    except Exception as e:  # noqa: BLE001
        CONSOLE.log(f"[yellow]Failed to read COLMAP points3D.bin: {e}")
        return None


def compute_sfm_centering(
    points: np.ndarray,
    outlier_percentile: float = 95.0,
    scale_percentile: float = 50.0,
    target_radius: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Compute scene center and scale from SfM points.

    Filters outliers (keeps closest ``outlier_percentile``% of points around the
    median), then computes the inlier mean center and a percentile-based scale.
    Using a percentile (default: median) instead of the mean makes the scale
    robust to ground-plane and distant-feature points that dominate the tail.

    Returns (center [3], scale) such that ``(p - center) * scale`` maps the
    ``scale_percentile`` inlier distance to ``target_radius``.
    """
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)

    threshold = np.percentile(dists, outlier_percentile)
    inliers = points[dists <= threshold]
    CONSOLE.log(
        f"SfM centering: {len(inliers)}/{len(points)} inlier points "
        f"(percentile={outlier_percentile}%)"
    )

    center = np.mean(inliers, axis=0)
    dists_from_center = np.linalg.norm(inliers - center, axis=1)

    target_dist = np.percentile(dists_from_center, scale_percentile)
    scale = target_radius / max(target_dist, 1e-6)

    CONSOLE.log(
        f"SfM center: {center}, scale: {scale:.4f} "
        f"(p{scale_percentile:.0f}={target_dist:.2f} -> r={target_radius})"
    )
    return center, scale
