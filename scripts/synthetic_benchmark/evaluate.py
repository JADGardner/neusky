#!/usr/bin/env python
"""Method-agnostic scorer for the NeuSky synthetic outdoor benchmark.

Compares a prediction directory (see README.md, "Prediction contract")
against the ground-truth layers of one scene/split of the NeuSky synthetic
dataset and writes per-frame CSV + aggregate JSON metrics.

Usage:
    python evaluate.py --pred-dir PRED --data SCENE_DIR \
        [--split test] [--tracks nvs decomposition] \
        [--output metrics.json] [--csv per_frame.csv] [--no-lpips]

Deterministic and CPU-only. Dependencies: numpy, imageio, scikit-image,
OpenEXR>=3.2 (or pyexr as fallback). torch is imported ONLY if LPIPS is
requested and the `lpips` package is installed; otherwise LPIPS is skipped
with a note in the output.

All metric definitions and data conventions are specified in README.md.
The docstrings below are the implementation-level reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dataset conventions (verified empirically; see README.md "Data conventions")
# ---------------------------------------------------------------------------

SKY_COLOUR = (70, 130, 180)        # cityscapes "sky" colour in cityscapes_mask
DEPTH_SKY_VALUE = 1.0e9            # GT depth >= this is sky / no-hit (stored 1e10)
NORMAL_VALID_MIN_NORM = 0.5        # GT normals: ||n||~1 on surfaces, 0 in sky

GT_LAYER_DIRS = {
    "rgb": "rgb",
    "albedo": "albedo",
    "normal": "normal",
    "depth": "depth",
    "roughness": "roughness",
    "metallic": "metallic",
    "mask": "cityscapes_mask",
}

# Prediction-contract suffixes per layer, in priority order.
PRED_SUFFIXES = {
    "rgb": ["_rgb.png", "_rgb.exr"],
    "albedo": ["_albedo.exr", "_albedo.npy", "_albedo.png"],
    "normal": ["_normal.exr", "_normal.npy"],
    "depth": ["_depth.exr", "_depth.npy"],
    "roughness": ["_roughness.exr", "_roughness.npy", "_roughness.png"],
    "metallic": ["_metallic.exr", "_metallic.npy", "_metallic.png"],
}

LPIPS_NET = "alex"  # pinned; see README dependency pins


# ---------------------------------------------------------------------------
# EXR I/O (OpenEXR>=3.2 preferred, pyexr fallback)
# ---------------------------------------------------------------------------

def read_exr(path) -> dict:
    """Read an EXR file, returning {channel_name: float32 array}.

    With the OpenEXR>=3.2 bindings, like-named channels (R/G/B/A) are
    returned stacked under a single key (e.g. "RGBA" -> (H, W, 4)); separate
    channels such as the dataset's normal X/Y/Z or depth Y stay scalar
    (H, W). The pyexr fallback reproduces that behaviour for the channel
    layouts used by this benchmark.
    """
    path = str(path)
    try:
        import OpenEXR  # >=3.2 "File" API

        with OpenEXR.File(path) as f:
            return {k: np.asarray(v.pixels, dtype=np.float32) for k, v in f.channels().items()}
    except ImportError:
        pass
    try:
        import pyexr
    except ImportError as e:
        raise ImportError(
            "Reading EXR requires either OpenEXR>=3.2 (`pip install OpenEXR`) "
            "or pyexr (`pip install pyexr`)."
        ) from e
    with pyexr.open(path) as f:
        chans = list(f.channel_map.keys())
        if set(chans) >= {"R", "G", "B"}:
            keys = ["R", "G", "B"] + (["A"] if "A" in chans else [])
            arr = np.stack([f.get(c)[..., 0] for c in keys], axis=-1).astype(np.float32)
            return {"".join(keys): arr}
        return {c: f.get(c)[..., 0].astype(np.float32) for c in chans}


def write_exr(path, channels: dict) -> None:
    """Write {channel_name: float32 array} to an EXR (used by tests/renderer)."""
    try:
        import OpenEXR

        OpenEXR.File({}, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in channels.items()}).write(str(path))
        return
    except ImportError:
        pass
    import pyexr

    names = list(channels.keys())
    if len(names) == 1 and channels[names[0]].ndim == 3:
        # stacked RGB(A) under one key
        arr = channels[names[0]]
        pyexr.write(str(path), arr.astype(np.float32),
                    channel_names=list("RGBA"[: arr.shape[-1]]))
    else:
        arr = np.stack([channels[n] for n in names], axis=-1).astype(np.float32)
        pyexr.write(str(path), arr, channel_names=names)


def _first_channels(exr_dict: dict, want: int) -> np.ndarray:
    """Flatten an EXR channel dict to an (H, W, want) or (H, W) array."""
    if len(exr_dict) == 1:
        arr = next(iter(exr_dict.values()))
        if arr.ndim == 2:
            return arr if want == 1 else np.repeat(arr[..., None], want, axis=-1)
        return arr[..., :want]
    for keyset in (("R", "G", "B"), ("X", "Y", "Z")):
        if all(k in exr_dict for k in keyset[:want]):
            return np.stack([exr_dict[k] for k in keyset[:want]], axis=-1)
    if want == 1 and "Y" in exr_dict:
        return exr_dict["Y"]
    raise ValueError(f"Cannot interpret EXR channels {sorted(exr_dict)} as {want}-channel image")


# ---------------------------------------------------------------------------
# Generic image loading
# ---------------------------------------------------------------------------

def _imread_png(path) -> np.ndarray:
    import imageio.v3 as iio

    arr = iio.imread(str(path))
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32)


def load_prediction(path: Path, channels: int) -> np.ndarray:
    """Load a prediction file (.png/.exr/.npy) as float32, (H, W[, C])."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    elif suffix == ".exr":
        arr = _first_channels(read_exr(path), channels)
    else:
        arr = _imread_png(path)
    arr = np.squeeze(arr)
    if channels == 1:
        if arr.ndim == 3:
            arr = arr[..., 0]
    else:
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], channels, axis=-1)
        arr = arr[..., :channels]
    return np.ascontiguousarray(arr, dtype=np.float32)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF (IEC 61966-2-1)."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, None)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def gt_frame_stems(scene_dir: Path, split: str) -> list:
    split_dir = scene_dir / split / GT_LAYER_DIRS["rgb"]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No GT split directory: {split_dir}")
    return sorted(p.stem for p in split_dir.glob("*.png"))


def load_gt(scene_dir: Path, split: str, stem: str) -> dict:
    """Load all GT layers for one frame.

    Returns dict with keys rgb (H,W,3 sRGB-encoded [0,1]), mask (H,W bool,
    True = not sky), albedo (H,W,3 linear), normal (H,W,3 world Z-up),
    normal_valid (H,W bool), depth (H,W metric z-depth), depth_valid
    (H,W bool), roughness (H,W), metallic (H,W). Layers missing on disk are
    simply absent from the dict.
    """
    base = scene_dir / split
    out = {}
    out["rgb"] = _imread_png(base / "rgb" / f"{stem}.png")[..., :3]

    mask_path = base / GT_LAYER_DIRS["mask"] / f"{stem}.png"
    if mask_path.exists():
        m = (_imread_png(mask_path) * 255.0).round().astype(np.uint8)[..., :3]
        out["mask"] = ~np.all(m == np.array(SKY_COLOUR, dtype=np.uint8), axis=-1)
    else:
        out["mask"] = np.ones(out["rgb"].shape[:2], dtype=bool)

    def _exr(layer, want):
        p = base / layer / f"{stem}.exr"
        return _first_channels(read_exr(p), want) if p.exists() else None

    albedo = _exr("albedo", 3)
    if albedo is not None:
        out["albedo"] = albedo

    normal = _exr("normal", 3)
    if normal is not None:
        norm = np.linalg.norm(normal, axis=-1)
        out["normal"] = normal
        out["normal_valid"] = norm > NORMAL_VALID_MIN_NORM

    depth = _exr("depth", 1)
    if depth is not None:
        out["depth"] = depth
        out["depth_valid"] = depth < DEPTH_SKY_VALUE

    for layer in ("roughness", "metallic"):
        arr = _exr(layer, 1)
        if arr is not None:
            out[layer] = arr
    return out


def find_pred(pred_dir: Path, stem: str, layer: str):
    for suffix in PRED_SUFFIXES[layer]:
        p = pred_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """PSNR with data range 1. PSNR = -10 log10(MSE); MSE over all channels
    of the pixels selected by `mask` (all pixels if mask is None).
    Identical images -> inf."""
    err = (pred.astype(np.float64) - gt.astype(np.float64)) ** 2
    if mask is not None:
        if err.ndim == 3:
            err = err[mask]
        else:
            err = err[mask]
    mse = float(np.mean(err)) if err.size else float("nan")
    if mse == 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """SSIM (scikit-image, data_range=1, default 7x7 window; channel_axis=-1
    for colour). Masked variant: mean of the full SSIM map over mask==True."""
    from skimage.metrics import structural_similarity

    kwargs = dict(data_range=1.0, full=True)
    if pred.ndim == 3:
        kwargs["channel_axis"] = -1
    _, smap = structural_similarity(gt, pred, **kwargs)
    if mask is None:
        return float(np.mean(smap))
    if smap.ndim == 3:
        return float(np.mean(smap[mask]))
    return float(np.mean(smap[mask]))


class LPIPSScorer:
    """Optional LPIPS (pinned net: alex). Constructed once; CPU-only."""

    def __init__(self):
        import lpips  # noqa: F401  (raises ImportError if unavailable)
        import torch

        self._torch = torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fn = lpips.LPIPS(net=LPIPS_NET, verbose=False)
        self._fn.eval()
        self.version = getattr(sys.modules["lpips"], "__version__", "unknown")

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        torch = self._torch
        with torch.no_grad():
            a = torch.from_numpy(pred).permute(2, 0, 1)[None] * 2 - 1
            b = torch.from_numpy(gt).permute(2, 0, 1)[None] * 2 - 1
            return float(self._fn(a.float(), b.float()).item())


def scale_align_per_channel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel least-squares scale alignment (no intercept).

    For each channel c: alpha_c = sum(mask * pred_c * gt_c) / sum(mask * pred_c^2)
    (alpha_c = 1 if the denominator is 0). Returns clip(alpha * pred, 0, 1).
    """
    aligned = np.empty_like(pred)
    for c in range(pred.shape[-1]):
        p, g = pred[..., c][mask].astype(np.float64), gt[..., c][mask].astype(np.float64)
        denom = float(np.sum(p * p))
        alpha = float(np.sum(p * g)) / denom if denom > 0 else 1.0
        aligned[..., c] = pred[..., c] * alpha
    return np.clip(aligned, 0.0, 1.0)


def exposure_align(pred_srgb: np.ndarray, gt_srgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Single-scalar exposure alignment in linear space.

    Both images are sRGB-decoded; one scalar s* = sum(p*g)/sum(p*p) is fitted
    over all channels of mask==True pixels; the scaled prediction is
    re-encoded to sRGB and clipped to [0, 1]. Compensates the dataset's
    per-image 98th-percentile exposure normalisation (see README).
    """
    p = srgb_to_linear(np.clip(pred_srgb, 0.0, 1.0))
    g = srgb_to_linear(np.clip(gt_srgb, 0.0, 1.0))
    pm, gm = p[mask].astype(np.float64), g[mask].astype(np.float64)
    denom = float(np.sum(pm * pm))
    s = float(np.sum(pm * gm)) / denom if denom > 0 else 1.0
    return np.clip(linear_to_srgb(p * s), 0.0, 1.0)


def angular_error_deg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Mean/median angular error in degrees between unit-normalised vectors.

    Both inputs are renormalised; pixels where either has near-zero norm are
    excluded (in addition to `valid`).
    """
    pn = np.linalg.norm(pred, axis=-1)
    gn = np.linalg.norm(gt, axis=-1)
    ok = valid & (pn > 1e-6) & (gn > 1e-6)
    if not ok.any():
        return float("nan"), float("nan"), 0
    p = pred[ok] / pn[ok][:, None]
    g = gt[ok] / gn[ok][:, None]
    dot = np.clip(np.sum(p * g, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return float(np.mean(ang)), float(np.median(ang)), int(ok.sum())


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------

def _check_shape(pred, gt, stem, layer):
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"{stem} {layer}: prediction resolution {pred.shape[:2]} != GT {gt.shape[:2]}. "
            "Predictions must be at the GT resolution; no resizing is performed."
        )


def evaluate_rgb(pred_dir: Path, stem: str, gt: dict, lpips_fn, prefix: str) -> dict:
    p = find_pred(pred_dir, stem, "rgb")
    if p is None:
        return {}
    pred = np.clip(load_prediction(p, 3), 0.0, 1.0)
    _check_shape(pred, gt["rgb"], stem, "rgb")
    mask = gt["mask"]
    m = {
        f"{prefix}/psnr": psnr(pred, gt["rgb"]),
        f"{prefix}/psnr_masked": psnr(pred, gt["rgb"], mask),
        f"{prefix}/ssim": ssim(pred, gt["rgb"]),
        f"{prefix}/ssim_masked": ssim(pred, gt["rgb"], mask),
    }
    pred_ea = exposure_align(pred, gt["rgb"], mask)
    m[f"{prefix}/psnr_masked_ea"] = psnr(pred_ea, gt["rgb"], mask)
    if lpips_fn is not None:
        m[f"{prefix}/lpips"] = lpips_fn(pred, gt["rgb"])
    return m


def evaluate_decomposition(pred_dir: Path, stem: str, gt: dict, notes: list) -> dict:
    m = {}
    mask = gt["mask"]

    p = find_pred(pred_dir, stem, "albedo")
    if p is not None and "albedo" in gt:
        pred = load_prediction(p, 3)
        if p.suffix.lower() == ".png":
            pred = srgb_to_linear(np.clip(pred, 0.0, 1.0)).astype(np.float32)
        _check_shape(pred, gt["albedo"], stem, "albedo")
        gt_albedo = np.clip(gt["albedo"], 0.0, 1.0)
        aligned = scale_align_per_channel(pred, gt_albedo, mask)
        m["decomposition/albedo_psnr_masked"] = psnr(aligned, gt_albedo, mask)
        m["decomposition/albedo_ssim_masked"] = ssim(aligned, gt_albedo, mask)

    p = find_pred(pred_dir, stem, "normal")
    if p is not None and "normal" in gt:
        pred = load_prediction(p, 3)
        _check_shape(pred, gt["normal"], stem, "normal")
        mean_deg, median_deg, n = angular_error_deg(pred, gt["normal"], mask & gt["normal_valid"])
        m["decomposition/normal_mae_deg"] = mean_deg
        m["decomposition/normal_median_ae_deg"] = median_deg

    p = find_pred(pred_dir, stem, "depth")
    if p is not None and "depth" in gt:
        pred = load_prediction(p, 1)
        _check_shape(pred, gt["depth"], stem, "depth")
        valid = mask & gt["depth_valid"] & np.isfinite(pred) & (pred < DEPTH_SKY_VALUE)
        pd, gd = pred[valid].astype(np.float64), gt["depth"][valid].astype(np.float64)
        if pd.size:
            m["decomposition/depth_rmse"] = float(np.sqrt(np.mean((pd - gd) ** 2)))
            m["decomposition/depth_mae"] = float(np.mean(np.abs(pd - gd)))
            denom = float(np.sum(pd * pd))
            s = float(np.sum(pd * gd)) / denom if denom > 0 else 1.0
            m["decomposition/depth_rmse_scale_aligned"] = float(np.sqrt(np.mean((s * pd - gd) ** 2)))
            m["decomposition/depth_scale_factor"] = s
        else:
            notes.append(f"{stem}: no valid depth pixels (all sky/non-finite)")

    for layer in ("roughness", "metallic"):
        p = find_pred(pred_dir, stem, layer)
        if p is not None and layer in gt:
            pred = load_prediction(p, 1)
            _check_shape(pred, gt[layer], stem, layer)
            m[f"decomposition/{layer}_mse_masked"] = float(
                np.mean((pred[mask].astype(np.float64) - gt[layer][mask].astype(np.float64)) ** 2)
            )
    return m


# ---------------------------------------------------------------------------
# Aggregation / driver
# ---------------------------------------------------------------------------

def _aggregate(per_frame: dict) -> dict:
    keys = sorted({k for v in per_frame.values() for k in v})
    agg = {}
    for k in keys:
        vals = [v[k] for v in per_frame.values() if k in v]
        finite = [x for x in vals if math.isfinite(x)]
        if len(finite) < len(vals):
            # e.g. PSNR=inf on identical images: the mean is inf by definition
            agg[k] = float("inf") if all(x == float("inf") or math.isfinite(x) for x in vals) else float("nan")
        else:
            agg[k] = float(np.mean(finite)) if finite else float("nan")
        agg[f"{k}__n"] = len(vals)
    return agg


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def evaluate(pred_dir, scene_dir, split="test", tracks=("nvs", "decomposition"),
             use_lpips=True, frames=None) -> dict:
    """Evaluate one prediction directory. Returns the result dict
    (see README "Output schema")."""
    pred_dir, scene_dir = Path(pred_dir), Path(scene_dir)
    notes = []
    tracks = list(tracks)
    rgb_track = "estimation" if "estimation" in tracks else "nvs"

    lpips_fn = None
    lpips_info = {"requested": bool(use_lpips), "available": False,
                  "net": LPIPS_NET, "version": None}
    if use_lpips and ({"nvs", "estimation"} & set(tracks)):
        try:
            lpips_fn = LPIPSScorer()
            lpips_info.update(available=True, version=lpips_fn.version)
        except ImportError:
            notes.append(
                "lpips not installed; LPIPS skipped "
                "(pip install lpips torch -- see README dependency pins)"
            )
            warnings.warn(notes[-1])

    stems = gt_frame_stems(scene_dir, split)
    if frames:
        wanted = {f"{int(f):04d}" if str(f).isdigit() else str(f) for f in frames}
        stems = [s for s in stems if s in wanted]
        if not stems:
            raise ValueError(f"No GT frames match --frames {sorted(wanted)}")

    layers_missing = {layer: 0 for layer in PRED_SUFFIXES}
    per_frame = {}
    for stem in stems:
        gt = load_gt(scene_dir, split, stem)
        m = {}
        if {"nvs", "estimation"} & set(tracks):
            m.update(evaluate_rgb(pred_dir, stem, gt, lpips_fn, rgb_track))
        if "decomposition" in tracks:
            m.update(evaluate_decomposition(pred_dir, stem, gt, notes))
        for layer in layers_missing:
            if find_pred(pred_dir, stem, layer) is None:
                layers_missing[layer] += 1
        per_frame[stem] = m

    for layer, n_missing in layers_missing.items():
        if n_missing == len(stems):
            notes.append(f"layer '{layer}' not predicted for any frame; its metrics were skipped")
        elif n_missing:
            notes.append(f"layer '{layer}' missing for {n_missing}/{len(stems)} frames")

    return {
        "scene_dir": str(scene_dir),
        "scene": scene_dir.name,
        "split": split,
        "pred_dir": str(pred_dir),
        "tracks": tracks,
        "n_frames": len(stems),
        "lpips": lpips_info,
        "notes": notes,
        "aggregate": _aggregate(per_frame),
        "per_frame": per_frame,
    }


def write_csv(result: dict, csv_path):
    keys = sorted({k for v in result["per_frame"].values() for k in v})
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + keys)
        for stem in sorted(result["per_frame"]):
            row = result["per_frame"][stem]
            w.writerow([stem] + [row.get(k, "") for k in keys])
        w.writerow(["aggregate_mean"] + [result["aggregate"].get(k, "") for k in keys])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred-dir", required=True, help="prediction directory (see README contract)")
    ap.add_argument("--data", required=True, help="scene directory, e.g. .../abandoned_buildings_prepared")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--tracks", nargs="+", default=["nvs", "decomposition"],
                    choices=["nvs", "decomposition", "estimation"])
    ap.add_argument("--output", default=None, help="aggregate JSON path (default: <pred-dir>/metrics.json)")
    ap.add_argument("--csv", default=None, help="per-frame CSV path (default: <pred-dir>/metrics_per_frame.csv)")
    ap.add_argument("--frames", nargs="*", default=None, help="evaluate only these frame stems/indices")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS even if installed")
    args = ap.parse_args(argv)

    result = evaluate(args.pred_dir, args.data, split=args.split, tracks=args.tracks,
                      use_lpips=not args.no_lpips, frames=args.frames)

    out_json = Path(args.output) if args.output else Path(args.pred_dir) / "metrics.json"
    out_csv = Path(args.csv) if args.csv else Path(args.pred_dir) / "metrics_per_frame.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(_jsonable(result), f, indent=2)
    write_csv(result, out_csv)

    print(f"[evaluate] {result['scene']} {result['split']}: {result['n_frames']} frames")
    for note in result["notes"]:
        print(f"[evaluate] NOTE: {note}")
    for k, v in result["aggregate"].items():
        if not k.endswith("__n"):
            print(f"  {k:45s} {v:.4f}" if math.isfinite(v) else f"  {k:45s} {v}")
    print(f"[evaluate] wrote {out_json} and {out_csv}")
    return result


if __name__ == "__main__":
    main()
