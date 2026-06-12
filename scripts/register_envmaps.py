#!/usr/bin/env python3
"""Recover NeRF-OSR session environment-map rotations via COLMAP registration.

Each NeRF-OSR scene (lk2, st, lwp) ships per-session 360deg environment
photos under Data/<scene>/final/ENV_MAP_CC/<session>/<photo.jpg>. These are
2:1 equirectangular (ERP) panoramas (verified: 5660x2830 for all sessions)
shot near the scene, but their orientation relative to the dataset's
normalized camera frame is unknown. This script recovers that rotation by:

  1. Rendering N perspective crops from the ERP with known crop->panorama
     rotations (yaw x pitch grid, ~60deg FoV).
  2. Registering the crops into the scene's existing COLMAP model
     (Data/<scene>/colmap/sparse_triangulated, COLMAP 3.13 binary format
     with rigs.bin/frames.bin; poses fixed to cam_dict_norm.json, so the
     model lives in the same dataset-normalized frame as final/*/pose/*.txt).
     Feature extraction + matching run against a COPY of database.db; the
     originals under Data/ are never mutated.
  3. Solving R_pano_world per registered crop and robustly averaging.

ERP direction convention (used for crop rendering AND for interpreting the
output rotation). For ERP pixel indices (u, v), u in [0, W), v in [0, H):

    lon = 2*pi * (u + 0.5) / W - pi     # 0 at the centre column, increasing
                                        # towards the right of the image
    lat = pi * (v + 0.5) / H - pi/2     # -pi/2 at the TOP row (zenith),
                                        # +pi/2 at the bottom row (nadir)
    d_pano = [ cos(lat)*sin(lon),       # +X: centre-right quarter (lon=+90deg)
               sin(lat),                # +Y: nadir (image bottom) -- y-down,
                                        #     matching the OpenCV-style world
               cos(lat)*cos(lon) ]      # +Z: centre pixel of the panorama

i.e. the panorama frame is OpenCV-like (x right-of-centre, y down, z out of
the image centre); the panorama up direction is (0, -1, 0). With this
convention an identity crop rotation reproduces the centre of the ERP with
the same left/right and up/down content order as the stored image (no
mirroring). Crop rotations are R_pano_from_cam = R_y(yaw) @ R_x(pitch), with
yaw > 0 panning towards +X (right in the ERP) and pitch > 0 tilting up
(towards the zenith / top of the ERP).

Pose algebra. COLMAP yields w2c per crop: d_cam = R_w2c @ d_world. Crops were
rendered by sampling the ERP at d_pano = R_pano_from_cam @ d_cam. Therefore

    d_world = R_w2c^T @ d_cam = R_w2c^T @ R_pano_from_cam^T @ d_pano
    => R_pano_world = R_c2w @ R_pano_from_cam^T

R_pano_world maps panorama directions (convention above) into the
dataset-normalized world frame (same frame as final/*/pose/*.txt). Per-crop
estimates are averaged with a chordal-L2 (SVD) mean plus iterative outlier
rejection on geodesic residuals.

If a session directory contains multiple exposures, the middle file in
sorted order is used (lk2/st/lwp currently have exactly one photo each).

Output: Data/<scene>/final/envmap_rotations.json
    {"<session>": {"rotation": [[3x3]], "envmap_image": "<path relative to
     Data/<scene>>", "num_registered": N, "mean_residual_deg": x,
     "centre": [3], ...diagnostics}}

Subcommands:
  register  Run the full pipeline for one or more scenes/sessions. Requires
            the `colmap` binary and `pycolmap` (available in the phd
            research docker container).
  compare   Render the panorama from a recovered rotation towards a chosen
            model image's viewpoint and save a side-by-side PNG (visual
            sanity check).
  package   Stage the per-scene envmap_rotations.json files into a zip for
            the NeuSky NeRF-OSR extras Dropbox folder.
  install   Extract a downloaded zip into a NeRF-OSR Data tree.

Typical usage (register/compare inside the research container):
  python scripts/register_envmaps.py register --data ~/data/NeRF-OSR/Data --scenes lk2
  python scripts/register_envmaps.py compare --data ~/data/NeRF-OSR/Data \
      --scene lk2 --session 01-08_07_30 --image C6_DSC_9_5 --output /tmp/cmp.png
  python scripts/register_envmaps.py package --data ~/data/NeRF-OSR/Data \
      --output ~/Downloads/nerfosr_neusky_eval_extras.zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_SCENES = ["lk2", "st", "lwp"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

README_TEXT = """NeuSky NeRF-OSR environment-map rotations
=========================================

One envmap_rotations.json per scene, destined for
NeRF-OSR/Data/<scene>/final/envmap_rotations.json.

Each entry maps a session name (e.g. "01-08_07_30") to the rotation of that
session's ERP environment photo (final/ENV_MAP_CC/<session>/<photo.jpg>)
relative to the dataset-normalized world frame (the frame of the
final/*/pose/*.txt cameras and of cam_dict_norm.json).

ERP direction convention: for pixel (u, v) of a WxH equirectangular image,
  lon = 2*pi*(u+0.5)/W - pi          (0 at centre column, + to the right)
  lat = pi*(v+0.5)/H - pi/2          (-pi/2 at top row / zenith)
  d_pano = [cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)]
(x right-of-centre, y down/nadir, z out of the image centre; panorama up is
(0,-1,0)). Then  d_world = rotation @ d_pano.

Provenance: perspective crops rendered from each panorama on a yaw x pitch
grid were registered into the scene's COLMAP model (triangulated against the
known dataset poses) with `colmap image_registrator`; the per-crop rotation
estimates were averaged (chordal-L2/SVD with outlier rejection). "centre" is
the mean registered crop camera centre (where the panorama was captured),
"mean_residual_deg" the mean angular deviation of inlier crops from the
average, "up_deviation_deg" the angle between the recovered panorama up and
the scene's mean camera up (gravity-consistency check).

Install:
  python scripts/register_envmaps.py install --zip <this zip> --data <NeRF-OSR/Data>
"""


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def rot_y(deg: float) -> np.ndarray:
    """Rotation about the pano +Y axis; +yaw pans the camera towards +X."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    """Rotation about the pano +X axis; +pitch tilts the camera up (-Y)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def crop_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R_pano_from_cam for a crop: d_pano = R @ d_cam (OpenCV cam frame)."""
    return rot_y(yaw_deg) @ rot_x(pitch_deg)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angular distance between two rotations in degrees."""
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def chordal_mean(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal-L2 mean rotation via SVD projection of the summed matrices."""
    M = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R


def robust_mean_rotation(
    rotations: list[np.ndarray], reject_deg: float = 5.0, iters: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively re-weighted chordal mean with outlier rejection.

    Returns (mean_rotation, residuals_deg for ALL inputs, inlier_mask).
    Inliers are rotations within max(reject_deg, 2.5 * median residual).
    """
    mask = np.ones(len(rotations), dtype=bool)
    mean = chordal_mean(rotations)
    for _ in range(iters):
        res = np.array([geodesic_deg(R, mean) for R in rotations])
        thresh = max(reject_deg, 2.5 * float(np.median(res[mask])))
        new_mask = res < thresh
        if not new_mask.any():
            break
        mask = new_mask
        mean = chordal_mean([R for R, m in zip(rotations, mask) if m])
    res = np.array([geodesic_deg(R, mean) for R in rotations])
    return mean, res, mask


# ---------------------------------------------------------------------------
# ERP sampling / crop rendering
# ---------------------------------------------------------------------------


def pano_dirs_to_uv(d: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Map unit directions (..., 3) in the pano frame to ERP pixel indices."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u = (lon + np.pi) / (2.0 * np.pi) * width - 0.5
    v = (lat + np.pi / 2.0) / np.pi * height - 0.5
    return u, v


def render_view(
    erp: np.ndarray,
    R_pano_from_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Render a perspective view from an ERP image (OpenCV pinhole, no dist).

    (cx, cy) follow the COLMAP convention where the centre of pixel index i
    is at continuous coordinate i + 0.5.
    """
    import cv2

    hgt, wid = erp.shape[:2]
    i = np.arange(out_w, dtype=np.float64)
    j = np.arange(out_h, dtype=np.float64)
    ii, jj = np.meshgrid(i, j)
    d_cam = np.stack(
        [(ii + 0.5 - cx) / fx, (jj + 0.5 - cy) / fy, np.ones_like(ii)], axis=-1
    )
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_pano = d_cam @ R_pano_from_cam.T
    u, v = pano_dirs_to_uv(d_pano, wid, hgt)
    return cv2.remap(
        erp,
        u.astype(np.float32),
        np.clip(v, 0, hgt - 1).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def crop_grid(num_yaw: int, pitches_deg: list[float]) -> list[tuple[str, float, float]]:
    """(name, yaw, pitch) for the crop grid."""
    crops = []
    for pitch in pitches_deg:
        for k in range(num_yaw):
            yaw = 360.0 * k / num_yaw
            name = f"crop_y{int(round(yaw)):03d}_p{int(round(pitch)):+03d}.jpg"
            crops.append((name, yaw, pitch))
    return crops


# ---------------------------------------------------------------------------
# Scene / session discovery
# ---------------------------------------------------------------------------


def session_envmap(scene_dir: Path, session: str) -> Path:
    """Pick the session's ERP photo (middle exposure if several)."""
    sess_dir = scene_dir / "final" / "ENV_MAP_CC" / session
    files = sorted(p for p in sess_dir.iterdir() if p.suffix in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"no environment photos in {sess_dir}")
    if len(files) > 1:
        print(f"  [{session}] {len(files)} exposures; using middle: {files[len(files) // 2].name}")
    return files[len(files) // 2]


def list_sessions(scene_dir: Path) -> list[str]:
    env_dir = scene_dir / "final" / "ENV_MAP_CC"
    return sorted(p.name for p in env_dir.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# COLMAP helpers
# ---------------------------------------------------------------------------


def run_colmap(args: list[str], log_path: Path) -> None:
    """Run a colmap subcommand, teeing output to a log file."""
    with open(log_path, "a") as log:
        log.write(f"\n$ colmap {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run(
            ["colmap", *args], stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        tail = "".join(open(log_path).readlines()[-30:])
        raise RuntimeError(
            f"colmap {args[0]} failed (exit {proc.returncode}); log tail:\n{tail}"
        )


def load_reconstruction(path: Path):
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pycolmap is required for the register/compare subcommands; "
            "run inside the phd research container"
        ) from exc
    return pycolmap.Reconstruction(str(path))


def model_world_up(rec) -> np.ndarray:
    """Mean camera up direction of the registered model images (world frame).

    The dataset-normalized world frame is OpenCV-like (y down), so this is
    close to (0, -1, 0); the actual mean absorbs the photographers' average
    camera tilt and serves as the gravity reference.
    """
    ups = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        R_w2c = np.asarray(img.cam_from_world().rotation.matrix())
        ups.append(-R_w2c[1])  # c2w @ (0,-1,0): cam -y (up) in world coords
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def register_session(
    scene: str,
    scene_dir: Path,
    session: str,
    args: argparse.Namespace,
    world_up: np.ndarray,
) -> dict | None:
    """Run the crop->COLMAP pipeline for one session; return the JSON entry."""
    import cv2

    env_path = session_envmap(scene_dir, session)
    erp = cv2.imread(str(env_path), cv2.IMREAD_COLOR)
    if erp is None:
        print(f"  [{session}] ERROR: cannot read {env_path}")
        return None
    hgt, wid = erp.shape[:2]
    if abs(wid / hgt - 2.0) > 0.01:
        print(
            f"  [{session}] WARNING: {env_path.name} is {wid}x{hgt} "
            f"(aspect {wid / hgt:.3f}, expected 2:1 equirectangular)"
        )

    workdir = Path(args.temp_root) / f"envreg_{scene}_{session}"
    if workdir.exists():
        shutil.rmtree(workdir)
    crops_dir = workdir / "crops"
    out_model_dir = workdir / "registered"
    crops_dir.mkdir(parents=True)
    out_model_dir.mkdir()
    log_path = workdir / "colmap.log"

    # --- 1. Render perspective crops with known crop->pano rotations -------
    size = args.crop_size
    f = size / (2.0 * np.tan(np.deg2rad(args.fov_deg) / 2.0))
    cx = cy = size / 2.0  # COLMAP convention (pixel i centre at i + 0.5)
    crops = crop_grid(args.num_yaw, args.pitches)
    crop_R = {}
    for name, yaw, pitch in crops:
        R_pc = crop_rotation(yaw, pitch)
        view = render_view(erp, R_pc, f, f, cx, cy, size, size)
        cv2.imwrite(str(crops_dir / name), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_R[name] = R_pc
    print(f"  [{session}] rendered {len(crops)} crops "
          f"({args.num_yaw} yaws x pitches {args.pitches}, fov {args.fov_deg} deg, {size}px)")

    # --- 2. COLMAP: features + matches into a COPY of the database ---------
    db_src = scene_dir / "colmap" / "database.db"
    db = workdir / "database.db"
    shutil.copy2(db_src, db)

    run_colmap(
        [
            "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(crops_dir),
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params", f"{f},{f},{cx},{cy}",
            "--ImageReader.single_camera_per_folder", "1",
            "--FeatureExtraction.use_gpu", "0" if not args.use_gpu else "1",
            "--SiftExtraction.max_num_features", "8192",
        ],
        log_path,
    )

    # Match each crop against every pre-existing image only (bounded work,
    # and the existing-existing pairs are already matched in the database).
    conn = sqlite3.connect(str(db))
    names = [row[0] for row in conn.execute("SELECT name FROM images")]
    conn.close()
    crop_names = set(crop_R)
    existing_all = sorted(n for n in names if n not in crop_names)
    # Always include same-session model images (the test split contains
    # photos taken under the same lighting as the panorama -- by far the
    # best match candidates), plus every match_stride-th other image.
    existing = sorted(
        set(existing_all[:: args.match_stride])
        | {n for n in existing_all if session in n}
    )
    pairs_path = workdir / "pairs.txt"
    with open(pairs_path, "w") as fh:
        for cn in sorted(crop_names):
            for en in existing:
                fh.write(f"{cn} {en}\n")
    run_colmap(
        [
            "matches_importer",
            "--database_path", str(db),
            "--match_list_path", str(pairs_path),
            "--match_type", "pairs",
            "--FeatureMatching.use_gpu", "0" if not args.use_gpu else "1",
        ],
        log_path,
    )

    # --- 3. Register crops against the fixed triangulated model ------------
    run_colmap(
        [
            "image_registrator",
            "--database_path", str(db),
            "--input_path", str(scene_dir / "colmap" / "sparse_triangulated"),
            "--output_path", str(out_model_dir),
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.abs_pose_min_num_inliers", str(args.min_inliers),
        ],
        log_path,
    )

    rec = load_reconstruction(out_model_dir)
    R_panos, centres, used = [], [], []
    for img in rec.images.values():
        if img.name not in crop_R or not img.has_pose:
            continue
        cfw = img.cam_from_world()
        R_w2c = np.asarray(cfw.rotation.matrix())
        R_c2w = R_w2c.T
        R_pw = R_c2w @ crop_R[img.name].T
        # Consistency: R_pw must carry this crop's pano-frame forward to its
        # COLMAP world-frame forward.
        fwd_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        fwd_pano = crop_R[img.name] @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(R_pw @ fwd_pano, fwd_world, atol=1e-9)
        assert abs(np.linalg.det(R_pw) - 1.0) < 1e-6
        R_panos.append(R_pw)
        centres.append(np.asarray(img.projection_center()))
        used.append(img.name)

    if len(R_panos) < args.min_registered:
        print(
            f"  [{session}] FAILED: only {len(R_panos)}/{len(crops)} crops "
            f"registered (min {args.min_registered}); see {log_path}"
        )
        return None

    mean_R, residuals, inliers = robust_mean_rotation(R_panos, args.reject_deg)
    centres = np.array(centres)
    centre = centres[inliers].mean(axis=0)
    centre_spread = float(np.linalg.norm(centres[inliers] - centre, axis=1).max())

    # Gravity check: panorama up (0,-1,0) vs scene mean camera up.
    pano_up_world = mean_R @ np.array([0.0, -1.0, 0.0])
    up_dev = float(
        np.rad2deg(np.arccos(np.clip(pano_up_world @ world_up, -1.0, 1.0)))
    )

    n_in = int(inliers.sum())
    print(
        f"  [{session}] registered {len(R_panos)}/{len(crops)} crops, "
        f"{n_in} inliers; residual mean {residuals[inliers].mean():.2f} / "
        f"max {residuals[inliers].max():.2f} deg "
        f"(rejected: {[f'{r:.1f}' for r in residuals[~inliers]]}); "
        f"centre {np.round(centre, 3).tolist()} (spread {centre_spread:.3f}); "
        f"up deviation {up_dev:.2f} deg"
    )
    for name, res, ok in sorted(zip(used, residuals, inliers)):
        print(f"      {name}: {res:6.2f} deg {'   ' if ok else '(outlier)'}")

    if not args.keep_temp:
        shutil.rmtree(workdir)

    return {
        "rotation": mean_R.tolist(),
        "envmap_image": str(env_path.relative_to(scene_dir)),
        "num_registered": len(R_panos),
        "mean_residual_deg": round(float(residuals[inliers].mean()), 4),
        "centre": [round(float(c), 6) for c in centre],
        # diagnostics
        "num_crops": len(crops),
        "num_inliers": n_in,
        "max_residual_deg": round(float(residuals[inliers].max()), 4),
        "centre_spread": round(centre_spread, 6),
        "up_deviation_deg": round(up_dev, 4),
    }


def cmd_register(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    failures = 0
    for scene in args.scenes:
        scene_dir = data / scene
        model_dir = scene_dir / "colmap" / "sparse_triangulated"
        if not model_dir.exists():
            print(f"[{scene}] SKIP: no COLMAP model at {model_dir}")
            failures += 1
            continue
        rec = load_reconstruction(model_dir)
        world_up = model_world_up(rec)
        print(
            f"[{scene}] model: {rec.num_images()} images, {rec.num_points3D()} points; "
            f"world up {np.round(world_up, 4).tolist()}"
        )
        sessions = args.sessions or list_sessions(scene_dir)
        out_path = scene_dir / "final" / "envmap_rotations.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        for session in sessions:
            try:
                entry = register_session(scene, scene_dir, session, args, world_up)
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"  [{session}] FAILED: {exc}")
                entry = None
            if entry is None:
                failures += 1
                continue
            results[session] = entry
            out_path.write_text(json.dumps(results, indent=2) + "\n")
            print(f"  [{session}] -> {out_path}")
    return failures


# ---------------------------------------------------------------------------
# compare (visual sanity check)
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    """Side-by-side: a model image vs the panorama rendered with the same
    camera orientation (translation differs by pano-vs-camera baseline)."""
    import cv2

    data = Path(args.data).expanduser()
    scene_dir = data / args.scene
    rotations = json.loads((scene_dir / "final" / "envmap_rotations.json").read_text())
    if args.session not in rotations:
        print(f"session '{args.session}' not in envmap_rotations.json "
              f"(have: {sorted(rotations)})")
        return 1
    entry = rotations[args.session]
    R_pano_world = np.array(entry["rotation"])
    erp = cv2.imread(str(scene_dir / entry["envmap_image"]), cv2.IMREAD_COLOR)

    rec = load_reconstruction(scene_dir / "colmap" / "sparse_triangulated")
    img = next(
        (im for im in rec.images.values() if args.image in im.name and im.has_pose),
        None,
    )
    if img is None:
        print(f"no registered model image matching '{args.image}'")
        return 1
    cam = rec.cameras[img.camera_id]
    fx, fy, cx, cy = cam.params  # PINHOLE
    R_c2w = np.asarray(img.cam_from_world().rotation.matrix()).T
    # Pano view sharing the image's orientation: d_pano = R_pw^T @ R_c2w @ d_cam
    R_pano_from_cam = R_pano_world.T @ R_c2w
    pano_view = render_view(erp, R_pano_from_cam, fx, fy, cx, cy, cam.width, cam.height)

    # Model image names are paths relative to colmap/all_images (and the
    # all_images symlinks may point at stale container mounts), so resolve
    # the relative name lexically and fall back to searching the rgb splits.
    candidates = [
        Path(os.path.normpath(scene_dir / "colmap" / "all_images" / img.name))
    ]
    candidates += sorted((scene_dir / "final").glob(f"*/rgb/{Path(img.name).name}"))
    img_path = next((p for p in candidates if p.is_file()), None)
    real = cv2.imread(str(img_path)) if img_path else None
    if real is None:
        print(f"cannot read model image pixels for {img.name} (tried {candidates})")
        return 1
    if real.shape != pano_view.shape:
        real = cv2.resize(real, (pano_view.shape[1], pano_view.shape[0]))
    sep = np.full((pano_view.shape[0], 8, 3), 255, np.uint8)
    side = np.concatenate([real, sep, pano_view], axis=1)
    out = Path(args.output).expanduser()
    cv2.imwrite(str(out), side)
    print(f"wrote {out} (left: {img.name}; right: {args.session} panorama "
          f"rendered with the recovered rotation at the same orientation)")
    return 0


# ---------------------------------------------------------------------------
# package / install
# ---------------------------------------------------------------------------


def cmd_package(args: argparse.Namespace) -> int:
    data = Path(args.data).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nerfosr_eval_extras/README.txt", README_TEXT)
        for scene in args.scenes:
            src = data / scene / "final" / "envmap_rotations.json"
            if not src.exists():
                print(f"[{scene}] MISSING {src} — run `register` first")
                if args.require_all:
                    return 1
                continue
            zf.write(src, f"nerfosr_eval_extras/{scene}/final/envmap_rotations.json")
            print(f"[{scene}] added {src}")
            added += 1
    if added == 0:
        print("nothing packaged")
        return 1
    print(f"Packaged -> {output} ({output.stat().st_size / 1e3:.1f} kB)")
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
            if not name.endswith("envmap_rotations.json"):
                continue
            scene = Path(name).parts[-3]
            dest = data / scene / "final" / "envmap_rotations.json"
            if dest.exists() and not args.force:
                print(f"[{scene}] SKIP: {dest} exists (use --force to overwrite)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            print(f"[{scene}] installed -> {dest}")
            installed += 1
    print(f"Installed {installed} rotation file(s) into {data}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="Register session env maps into the COLMAP models")
    p_reg.add_argument("--data", default="~/data/NeRF-OSR/Data", help="NeRF-OSR Data root")
    p_reg.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p_reg.add_argument("--sessions", nargs="+", default=None,
                       help="Restrict to these sessions (default: all found)")
    p_reg.add_argument("--num-yaw", type=int, default=12, help="Yaw steps over 360 deg")
    p_reg.add_argument("--pitches", type=float, nargs="+", default=[0.0, 20.0],
                       help="Pitch rows in degrees (+ is up)")
    p_reg.add_argument("--fov-deg", type=float, default=60.0, help="Crop horizontal FoV")
    p_reg.add_argument("--crop-size", type=int, default=800, help="Crop size in pixels")
    p_reg.add_argument("--min-registered", type=int, default=3,
                       help="Minimum registered crops to accept a session")
    p_reg.add_argument("--reject-deg", type=float, default=5.0,
                       help="Base outlier-rejection threshold for rotation averaging")
    p_reg.add_argument("--match-stride", type=int, default=1,
                       help="Match crops against every Nth existing image "
                            "(speeds up CPU matching; default 1 = all)")
    p_reg.add_argument("--min-inliers", type=int, default=30,
                       help="Mapper.abs_pose_min_num_inliers for image_registrator "
                            "(COLMAP default 30; lower for distant/small landmarks)")
    p_reg.add_argument("--temp-root", default="/tmp", help="Where to put COLMAP workspaces")
    p_reg.add_argument("--keep-temp", action="store_true", help="Keep temp workspaces")
    p_reg.add_argument("--use-gpu", action="store_true",
                       help="Use GPU SIFT extraction/matching (default: CPU)")
    p_reg.set_defaults(func=cmd_register)

    p_cmp = sub.add_parser("compare", help="Side-by-side panorama render vs a model image")
    p_cmp.add_argument("--data", default="~/data/NeRF-OSR/Data")
    p_cmp.add_argument("--scene", required=True)
    p_cmp.add_argument("--session", required=True)
    p_cmp.add_argument("--image", required=True, help="Substring of a model image name")
    p_cmp.add_argument("--output", default="/tmp/envmap_compare.png")
    p_cmp.set_defaults(func=cmd_compare)

    p_pkg = sub.add_parser("package", help="Zip envmap_rotations.json files for Dropbox")
    p_pkg.add_argument("--data", default="~/data/NeRF-OSR/Data")
    p_pkg.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p_pkg.add_argument("--output", default="~/Downloads/nerfosr_neusky_eval_extras.zip")
    p_pkg.add_argument("--require-all", action="store_true",
                       help="Fail if any scene is missing its rotations file")
    p_pkg.set_defaults(func=cmd_package)

    p_ins = sub.add_parser("install", help="Extract a downloaded zip into a NeRF-OSR Data tree")
    p_ins.add_argument("--zip", required=True)
    p_ins.add_argument("--data", default="~/data/NeRF-OSR/Data")
    p_ins.add_argument("--force", action="store_true")
    p_ins.set_defaults(func=cmd_install)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
