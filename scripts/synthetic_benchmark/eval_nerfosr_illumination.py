"""Score NeRF-OSR's fitted per-train-frame SH illumination against GT.

NeRF-OSR optimises a 9x3 diffuse-SH environment per training image from the
photographs alone; the synthetic dataset records each frame's GT HDRI, yaw
and exposure. The GT is projected to the same diffuse SH via the prepare
script's own projection (identical basis, frame and exposure conventions),
so predicted and GT illumination live in the same representation by
construction. Reported per scene: mean/median sun angular error (argmax of
the SH-rendered ERP) and mean log-domain PSNR between the SH-rendered ERPs
(both p99-normalised, since NeRF-OSR's SH scale is only defined up to the
benchmark exposure gauge).

    PYTHONPATH=. python scripts/synthetic_benchmark/eval_nerfosr_illumination.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_nerfosr_synthetic import (  # noqa: E402
    project_hdri_to_diffuse_sh, resolve_hdri_path, sh_basis)

REPO = HERE.parents[1]
ENV_DIR = REPO / "outputs" / "synthetic_benchmark" / "nerfosr_env_params"
DATA_ROOT = Path("/home/james/data/neusky_synthetic_data")
SCENES = ("abandoned_buildings", "apartment_building",
          "arlanda_uppsala_cathedral", "glass_building", "interstellar_house")
SH_TARGET_WIDTH = 128
GRID_H, GRID_W = 64, 128


def erp_dirs(h, w):
    """Direction grid in the prepare script's frame convention."""
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    lon = u[None, :] * 2.0 * math.pi
    lat = (0.5 - v[:, None]) * math.pi
    cos_lat = np.cos(lat)
    return np.stack([
        -cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.broadcast_to(np.sin(lat), (h, w)),
    ], axis=-1).reshape(-1, 3)


def render_sh(coeffs, basis):
    """[9,3] SH -> [N,3] ERP radiance (their illuminate_vec contraction)."""
    return np.clip(basis @ coeffs, 0.0, None)


def yaw_from_rotation(rot):
    if isinstance(rot[0], list):
        return math.atan2(rot[1][0], rot[0][0])
    return float(rot[-1])


def main():
    dirs = erp_dirs(GRID_H, GRID_W)
    basis = sh_basis(dirs)
    lat_w = np.cos((0.5 - (np.arange(GRID_H) + 0.5) / GRID_H) * math.pi)
    area_w = np.repeat(lat_w, GRID_W)

    summary = {}
    for scene in SCENES:
        npz = np.load(ENV_DIR / f"synthetic_{scene}_nerfosr.npz")
        tj = json.loads((DATA_ROOT / "renders" / f"{scene}_prepared" /
                         "transforms.json").read_text())
        frames = [fr for fr in tj["frames"] if "train" in fr.get("file_path", "")]
        gt_sh_cache = {}
        angles, psnrs, matched = [], [], 0
        for fr in frames:
            stem = fr["file_path"].split("/")[-1].split(".")[0]
            key = f"train/rgb/{stem}-png"   # NeRF-OSR img_name convention
            if key not in npz:
                continue
            pred_sh = npz[key].astype(np.float64)
            name = fr["envmap_name"]
            yaw = yaw_from_rotation(fr["envmap_rotation"])
            exposure = float(fr.get("exposure", 1.0))
            key = (name, round(yaw, 5), round(exposure, 5))
            if key not in gt_sh_cache:
                hdri = resolve_hdri_path(DATA_ROOT / "hdris", name)
                gt_sh_cache[key] = project_hdri_to_diffuse_sh(
                    hdri, yaw, SH_TARGET_WIDTH, exposure).astype(np.float64)
            gt_sh = gt_sh_cache[key]

            pred_erp = render_sh(pred_sh, basis)
            gt_erp = render_sh(gt_sh, basis)

            def sun(e):
                i = int(np.argmax(e.mean(-1)))
                return dirs[i]
            cosang = float(np.clip(np.dot(sun(pred_erp), sun(gt_erp)), -1, 1))
            angles.append(math.degrees(math.acos(cosang)))

            p = pred_erp / max(np.percentile(pred_erp, 99), 1e-9)
            g = gt_erp / max(np.percentile(gt_erp, 99), 1e-9)
            a, b = np.log1p(p), np.log1p(g)
            mse = float((area_w[:, None] * (a - b) ** 2).sum()
                        / (area_w.sum() * 3))
            psnrs.append(-10 * math.log10(max(mse, 1e-12)))
            matched += 1

        summary[scene] = {
            "n_frames": matched,
            "sun_angle_mean_deg": float(np.mean(angles)),
            "sun_angle_median_deg": float(np.median(angles)),
            "envmap_psnr_log_mean": float(np.mean(psnrs)),
        }
        print(f"{scene}: n={matched} sun mean {np.mean(angles):.1f} deg "
              f"median {np.median(angles):.1f} deg  "
              f"envmap log-PSNR {np.mean(psnrs):.2f}")

    out = ENV_DIR / "illumination_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    means = {k: float(np.mean([s[k] for s in summary.values()]))
             for k in ("sun_angle_mean_deg", "sun_angle_median_deg",
                       "envmap_psnr_log_mean")}
    print("MEAN over scenes:", json.dumps(means))


if __name__ == "__main__":
    main()
