"""Tests for evaluate.py on a tiny synthetic fixture (no dataset needed).

Run from this directory:  pytest test_evaluate.py -q

Builds a miniature scene (2 test frames, 24x32) with the exact GT layout and
conventions of the real dataset (sRGB PNG rgb, RGBA-EXR linear albedo,
X/Y/Z-EXR world normals, Y-EXR metric z-depth with 1e10 sky, cityscapes-style
sky mask), then checks:
  - GT-as-prediction scores at the identity (PSNR=inf, SSIM=1, 0 errors);
  - perturbations degrade exactly the right metrics by the expected amount;
  - sky-only corruption leaves masked metrics at the identity;
  - missing layers are skipped with a note instead of erroring.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402

pytest.importorskip("imageio")
pytest.importorskip("skimage")

H, W = 24, 32
STEMS = ["0000", "0001"]


def _write_png(path, arr01):
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), (np.clip(arr01, 0, 1) * 255).round().astype(np.uint8))


def _gt_frame(seed):
    """Deterministic GT layers for one frame (sky in the top-left corner)."""
    rng = np.random.RandomState(seed)
    sky = np.zeros((H, W), dtype=bool)
    sky[:6, :10] = True

    rgb = rng.rand(H, W, 3).astype(np.float32)
    albedo = (rng.rand(H, W, 3) * 0.9).astype(np.float32)
    normal = rng.randn(H, W, 3).astype(np.float32)
    normal[..., 2] = np.abs(normal[..., 2]) + 0.5  # mostly-up, like ground
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal[sky] = 0.0
    depth = (rng.rand(H, W) * 50 + 5).astype(np.float32)
    depth[sky] = 1.0e10
    roughness = rng.rand(H, W).astype(np.float32)
    metallic = rng.rand(H, W).astype(np.float32)
    return dict(sky=sky, rgb=rgb, albedo=albedo, normal=normal, depth=depth,
                roughness=roughness, metallic=metallic)


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    pytest.importorskip("OpenEXR")  # or pyexr; fixture writing uses evaluate.write_exr
    root = tmp_path_factory.mktemp("toy_scene")
    for i, stem in enumerate(STEMS):
        g = _gt_frame(seed=i)
        base = root / "test"
        _write_png(base / "rgb" / f"{stem}.png", g["rgb"])
        mask = np.full((H, W, 3), 70, dtype=np.uint8)
        mask[g["sky"]] = evaluate.SKY_COLOUR
        _write_png(base / "cityscapes_mask" / f"{stem}.png", mask / 255.0)
        for d in ("albedo", "normal", "depth", "roughness", "metallic"):
            (base / d).mkdir(parents=True, exist_ok=True)
        evaluate.write_exr(base / "albedo" / f"{stem}.exr",
                           {"RGBA": np.dstack([g["albedo"], np.ones((H, W, 1), np.float32)])})
        evaluate.write_exr(base / "normal" / f"{stem}.exr",
                           {"X": g["normal"][..., 0], "Y": g["normal"][..., 1],
                            "Z": g["normal"][..., 2]})
        evaluate.write_exr(base / "depth" / f"{stem}.exr", {"Y": g["depth"]})
        evaluate.write_exr(base / "roughness" / f"{stem}.exr", {"Y": g["roughness"]})
        evaluate.write_exr(base / "metallic" / f"{stem}.exr", {"Y": g["metallic"]})
    return root


def _make_pred_from_gt(scene_dir, pred_dir, mutate=None):
    """Copy GT layers into prediction-contract files, optionally mutating."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(STEMS):
        g = _gt_frame(seed=i)
        # round-trip rgb through the 8-bit PNG exactly as stored
        rgb = (np.clip(g["rgb"], 0, 1) * 255).round() / 255.0
        layers = dict(rgb=rgb.astype(np.float32), albedo=g["albedo"],
                      normal=g["normal"], depth=g["depth"],
                      roughness=g["roughness"], metallic=g["metallic"])
        if mutate:
            mutate(stem, layers, g)
        _write_png(pred_dir / f"{stem}_rgb.png", layers["rgb"])
        evaluate.write_exr(pred_dir / f"{stem}_albedo.exr", {"RGB": layers["albedo"]})
        evaluate.write_exr(pred_dir / f"{stem}_normal.exr",
                           {"X": layers["normal"][..., 0], "Y": layers["normal"][..., 1],
                            "Z": layers["normal"][..., 2]})
        evaluate.write_exr(pred_dir / f"{stem}_depth.exr", {"Y": layers["depth"]})
        evaluate.write_exr(pred_dir / f"{stem}_roughness.exr", {"Y": layers["roughness"]})
        evaluate.write_exr(pred_dir / f"{stem}_metallic.exr", {"Y": layers["metallic"]})


def test_gt_as_prediction_is_identity(scene_dir, tmp_path):
    pred = tmp_path / "pred_gt"
    _make_pred_from_gt(scene_dir, pred)
    res = evaluate.evaluate(pred, scene_dir, use_lpips=False)
    agg = res["aggregate"]
    assert math.isinf(agg["nvs/psnr"]) and math.isinf(agg["nvs/psnr_masked"])
    assert agg["nvs/ssim"] == pytest.approx(1.0, abs=1e-6)
    # exposure alignment linearises and re-encodes (float round trip), so the
    # identity scores ~150 dB rather than exactly inf
    assert agg["nvs/psnr_masked_ea"] > 100
    assert math.isinf(agg["decomposition/albedo_psnr_masked"])
    assert agg["decomposition/albedo_ssim_masked"] == pytest.approx(1.0, abs=1e-6)
    assert agg["decomposition/normal_mae_deg"] == pytest.approx(0.0, abs=1e-2)
    assert agg["decomposition/depth_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert agg["decomposition/depth_mae"] == pytest.approx(0.0, abs=1e-6)
    assert agg["decomposition/depth_rmse_scale_aligned"] == pytest.approx(0.0, abs=1e-6)
    assert agg["decomposition/roughness_mse_masked"] == pytest.approx(0.0, abs=1e-12)
    assert agg["decomposition/metallic_mse_masked"] == pytest.approx(0.0, abs=1e-12)


def test_noisy_albedo_degrades_albedo_only(scene_dir, tmp_path):
    rng = np.random.RandomState(7)

    def mutate(stem, layers, g):
        layers["albedo"] = np.clip(
            layers["albedo"] + rng.normal(0, 0.1, layers["albedo"].shape).astype(np.float32), 0, 1
        )

    pred = tmp_path / "pred_noisy_albedo"
    _make_pred_from_gt(scene_dir, pred, mutate)
    agg = evaluate.evaluate(pred, scene_dir, use_lpips=False)["aggregate"]
    assert math.isinf(agg["nvs/psnr"])  # rgb untouched
    assert math.isfinite(agg["decomposition/albedo_psnr_masked"])
    assert 10 < agg["decomposition/albedo_psnr_masked"] < 30  # sigma=0.1 -> ~20 dB
    assert agg["decomposition/albedo_ssim_masked"] < 0.999
    assert agg["decomposition/depth_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_albedo_scale_invariance(scene_dir, tmp_path):
    """A per-channel rescale of albedo must be absorbed by the alignment."""

    def mutate(stem, layers, g):
        layers["albedo"] = layers["albedo"] * np.array([0.5, 0.8, 0.6], np.float32)

    pred = tmp_path / "pred_scaled_albedo"
    _make_pred_from_gt(scene_dir, pred, mutate)
    agg = evaluate.evaluate(pred, scene_dir, use_lpips=False)["aggregate"]
    assert agg["decomposition/albedo_psnr_masked"] > 50  # ~identity after alignment


def test_rotated_normals_give_expected_angle(scene_dir, tmp_path):
    angle = np.radians(5.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]], dtype=np.float32)

    def mutate(stem, layers, g):
        layers["normal"] = layers["normal"] @ R.T

    pred = tmp_path / "pred_rot_normal"
    _make_pred_from_gt(scene_dir, pred, mutate)
    agg = evaluate.evaluate(pred, scene_dir, use_lpips=False)["aggregate"]
    # rotation about Z by 5 deg: error <= 5 deg, > 0 (normals are mostly-up)
    assert 0.5 < agg["decomposition/normal_mae_deg"] <= 5.001


def test_depth_gauge_drift_caught_by_raw_but_not_aligned(scene_dir, tmp_path):
    def mutate(stem, layers, g):
        layers["depth"] = layers["depth"] * 1.1

    pred = tmp_path / "pred_scaled_depth"
    _make_pred_from_gt(scene_dir, pred, mutate)
    agg = evaluate.evaluate(pred, scene_dir, use_lpips=False)["aggregate"]
    assert agg["decomposition/depth_rmse"] > 1.0  # ~0.1 * mean depth
    assert agg["decomposition/depth_rmse_scale_aligned"] == pytest.approx(0.0, abs=1e-3)
    assert agg["decomposition/depth_scale_factor"] == pytest.approx(1 / 1.1, rel=1e-3)


def test_sky_corruption_ignored_by_masked_metrics(scene_dir, tmp_path):
    def mutate(stem, layers, g):
        rgb = layers["rgb"].copy()
        rgb[g["sky"]] = 1.0 - rgb[g["sky"]]
        layers["rgb"] = rgb

    pred = tmp_path / "pred_sky_corrupt"
    _make_pred_from_gt(scene_dir, pred, mutate)
    agg = evaluate.evaluate(pred, scene_dir, use_lpips=False)["aggregate"]
    assert math.isfinite(agg["nvs/psnr"])          # full-image metric sees it
    assert math.isinf(agg["nvs/psnr_masked"])      # masked metric does not


def test_missing_layers_are_skipped_with_note(scene_dir, tmp_path):
    pred = tmp_path / "pred_rgb_only"
    _make_pred_from_gt(scene_dir, pred)
    for f in list(pred.iterdir()):
        if "_rgb" not in f.name:
            f.unlink()
    res = evaluate.evaluate(pred, scene_dir, use_lpips=False)
    agg = res["aggregate"]
    assert math.isinf(agg["nvs/psnr"])
    assert not any(k.startswith("decomposition/") for k in agg)
    assert any("albedo" in n for n in res["notes"])
    assert any("depth" in n for n in res["notes"])


def test_estimation_track_label(scene_dir, tmp_path):
    pred = tmp_path / "pred_est"
    _make_pred_from_gt(scene_dir, pred)
    agg = evaluate.evaluate(pred, scene_dir, tracks=["estimation"], use_lpips=False)["aggregate"]
    assert math.isinf(agg["estimation/psnr"])
    assert not any(k.startswith("nvs/") for k in agg)


def test_cli_round_trip(scene_dir, tmp_path):
    pred = tmp_path / "pred_cli"
    _make_pred_from_gt(scene_dir, pred)
    out_json = tmp_path / "metrics.json"
    out_csv = tmp_path / "per_frame.csv"
    evaluate.main([
        "--pred-dir", str(pred), "--data", str(scene_dir),
        "--split", "test", "--tracks", "nvs", "decomposition",
        "--output", str(out_json), "--csv", str(out_csv), "--no-lpips",
    ])
    data = json.loads(out_json.read_text())
    assert data["aggregate"]["nvs/psnr"] == "inf"  # non-finite serialised as string
    assert data["n_frames"] == len(STEMS)
    assert out_csv.exists()
    header = out_csv.read_text().splitlines()[0]
    assert header.startswith("frame,")
