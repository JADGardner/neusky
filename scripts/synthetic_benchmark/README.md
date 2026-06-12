# NeuSky synthetic outdoor benchmark — evaluation suite

A method-agnostic evaluation protocol for inverse rendering of outdoor
scenes under varying natural illumination, on the NeuSky synthetic dataset
(5 Blender scenes, Poly Haven HDRIs). Any method that can write the
prediction contract below can be scored with `evaluate.py`; no part of the
scorer depends on NeuSky.

```
evaluate.py                   method-agnostic scorer (CPU-only, deterministic)
render_neusky_predictions.py  NeuSky-specific renderer into the contract (GPU)
test_evaluate.py              pytest suite on a tiny synthetic fixture
```

## Dataset

Scenes (`<root>` = `~/data/neusky_synthetic_data`, override with the path
you store it at):

```
<root>/renders/{abandoned_buildings,apartment_building,
                arlanda_uppsala_cathedral,glass_building,
                interstellar_house}_prepared/
    transforms.json      all splits in one file; frames keyed by file_path prefix
    points3d.ply         metric world-frame point cloud (~14.5M pts, xyz+rgb)
    train/               250 frames: rgb/, cityscapes_mask/ only
    validation/          25 frames: rgb/ + all GT layers below
    test/                25 frames: rgb/ + all GT layers below
<root>/hdris/<name>.exr      4K HDRIs   (linear RGB(A), equirectangular)
<root>/hdris_16k/<name>.exr  16K HDRIs
```

Per-frame `transforms.json` metadata: `transform_matrix` (camera-to-world),
`fl_x fl_y cx cy` (pixels; `w=1920 h=1080`), `focal_mm`, `exposure_ev`,
`envmap_name`, `envmap_url`, `envmap_rotation`.

### Split design (verified on all 5 scenes)

- 250 train / 25 validation / 25 test frames per scene.
- Every test (and validation) frame uses a **different** HDRI; all of these
  HDRIs **also appear in train**, but no test `(envmap, rotation)` pair
  occurs in train (0/25 per scene). The current tracks therefore measure
  relighting at **novel rotations of train-seen illuminations**, not unseen
  illumination generalisation (see "Planned track" below).
- `exposure_ev = 0` for all test and validation frames; train frames have
  random `exposure_ev` in roughly [-1.5, 1.5].

## Confirmed data conventions

Everything below was verified empirically on the actual files (checks noted
in parentheses).

| Item | Convention |
|---|---|
| Camera poses | `transform_matrix` is camera-to-world, **OpenGL/Blender** axes (x right, y up, z backward looking down -z), world **Z-up**, positions in **metres** (Blender scene units). Verified by projecting `points3d.ply` into a test view: 392K points land in-frustum with OpenGL axes, 0 with OpenCV. |
| Pixel convention | `u = fl_x * x_cam / (-z_cam) + cx` with pixel centres at `+0.5`; `cx=960, cy=540`. |
| `rgb/*.png` | 8-bit sRGB-encoded. The PNG is `sRGB(linear / q98)` where `q98` is the **per-image 98th percentile** of the linear render (then clipped, quantised). Verified against the raw linear EXR renders: max abs difference 1/255. **Consequence: absolute brightness is a per-image gauge a method cannot know**, hence the exposure-aligned PSNR below. |
| `albedo/*.exr` | float32, channels `R,G,B,A` (alpha = 1 everywhere; ignore). Linear base-colour in [0, 1]. |
| `normal/*.exr` | float32, channels `X,Y,Z`. **World frame, Z-up**, unit length on surfaces, `(0,0,0)` in sky, non-unit at anti-aliased edges. Verified: ground pixels have mean normal ≈ (0, 0, 0.98) across cameras with different yaws. |
| `depth/*.exr` | float32, single channel `Y`. **z-depth** (distance along the camera optical axis, NOT ray length), **metric** (same units as `transform_matrix` translations), sky = `1e10`. Verified vs `points3d.ply`: 37.5% of projected points match z-depth within 1% vs 4.1% for the ray-distance hypothesis (the remainder are occluded points). |
| `roughness/metallic/ior/transmission/*.exr` | float32, single channel `Y`. Sky = 0. |
| `cityscapes_mask/*.png` | Cityscapes-palette RGB PNG with exactly two colours: building `(70,70,70)` and **sky `(70,130,180)`**. Produced by thresholding the Blender film-transparent alpha (alpha > 0.5 → object). Sky pixels agree with `depth >= 1e9` for 99.999% of pixels (disagreements are AA edge pixels). **Valid-pixel mask = not sky colour.** The dataset has no transients; sky is the only masked region. |
| `envmap_rotation` | `[0, 0, yaw]`, yaw in **radians**. The HDRI is rotated about world Z: a world direction `d=(x,y,z)` sees HDRI pixel `lon = atan2(y, -x) - yaw`, `lat = asin(z)`, sampled at `u = ((lon/2π) mod 1)·W`, `v = (0.5 - lat/π)·H` (v=0 at zenith). Equivalently: roll the HDRI left by `yaw/2π·W` pixels to align column 0 with `lon = 0`. Verified on sky pixels of a raw linear render vs the HDRI: median relative error 0.66%; every other axis/sign convention errs > 20%. |
| HDRIs | `hdris/<envmap_name>.exr` (4K) and `hdris_16k/` (16K), linear RGB(A) float. |

### Open questions

- **`ior` / `transmission` semantics at layered materials.** Populated for
  glass-heavy scenes (glass_building, apartment_building); zero everywhere in
  abandoned_buildings test views. What `depth`/`normal` mean exactly at
  transmissive surfaces (front surface vs through-glass) has not been
  verified geometrically; these layers are not scored.
- **Out-of-range AA values.** `metallic` reaches 3.1 and `roughness` 1.06 at
  a small number of pixels (anti-aliased material boundaries in Blender's
  Cycles aux passes). GT is used as stored; the masked MSE is dominated by
  in-range pixels.
- **RGB tonemap provenance.** The q98-exposure + standard sRGB encode was
  verified for `abandoned_buildings` test frame 0000 against its raw EXR and
  is implemented in `scripts/prepare_synthetic_data.py` (phd repo); not
  re-verified per-frame for all scenes.

## Tracks

A *prediction directory* corresponds to one (method, scene, split, track)
combination. The tracks differ only in what information the method was
allowed to use; the scorer is told the track label so results are
self-describing.

### Track 1 — Known-illumination NVS (`--tracks nvs`) — primary

The method is given the test frame's GT HDRI **and** its known rotation
(`envmap_name`, `envmap_rotation`) and renders the held-out view. Scores
RGB only. Rationale: this isolates geometry + material + light-transport
quality from illumination estimation, and is the fairest comparison across
methods with different illumination models.

### Track 2 — Illumination estimation (`--tracks estimation`) — secondary

The method may look at the **left half** of each test image to estimate the
illumination, then renders the full frame; the same RGB metrics are
reported (the right half is the genuinely unseen part, but full-frame
metrics keep numbers comparable with Track 1). Fitting on the **full** test
image is explicitly rejected as test-set fitting; the NeuSky renderer
exposes it only as `--illumination full-image-diagnostic`, and such
predictions must not be reported as a benchmark track.

### Track 3 — Decomposition (`--tracks decomposition`)

Albedo, normal, depth (+ roughness/metallic if predicted) against GT
layers, with the exact alignment rules below. Evaluated on whichever
prediction directory the method designates (conventionally the Track 1
directory; decomposition outputs should not depend on test illumination
handling).

### Planned track — unseen-HDRI relighting (placeholder)

All current test HDRIs are train-seen at novel rotations. A future track
will hold out a disjoint set of HDRIs (and scenes re-rendered under them)
to measure generalisation to genuinely unseen illumination. Not yet
specified; do not report numbers under this name.

## Prediction contract

One flat directory per (method, scene, split, track). Files are named by
the GT frame stem (`0000` … `0024` for the test split):

| File | Required | Format |
|---|---|---|
| `<stem>_rgb.png` (or `_rgb.exr`) | for nvs/estimation | 8-bit (or 16-bit) sRGB-encoded PNG at GT resolution (1920x1080). An EXR/float input is interpreted as sRGB-encoded values in [0,1], NOT linear. |
| `<stem>_albedo.exr` / `.npy` (or `.png`) | optional | linear RGB in [0,1]. A PNG is assumed sRGB-encoded and is linearised before scoring. |
| `<stem>_normal.exr` / `.npy` | optional | world-frame (Z-up) unit vectors; EXR channels `X,Y,Z` or a (H,W,3) npy. |
| `<stem>_depth.exr` / `.npy` | optional | **metric z-depth** in scene units (metres); EXR channel `Y` or (H,W) npy. Sky/no-hit may hold any value >= 1e9 or non-finite (such pixels are excluded anyway). |
| `<stem>_roughness.exr` / `.npy` / `.png` | optional | scalar in [0,1] |
| `<stem>_metallic.exr` / `.npy` / `.png` | optional | scalar in [0,1] |
| `manifest.json` | recommended | free-form provenance (method, checkpoint, track) |

Rules:

- Predictions must be at GT resolution; the scorer never resizes (mismatch
  is an error, not a silent rescale).
- A missing layer skips that layer's metrics for the method with a note in
  the output — it is not an error.
- Multi-suffix priority when several files exist: the order listed in
  `evaluate.PRED_SUFFIXES` (EXR before npy before PNG, except RGB where PNG
  is canonical).

## Metric definitions

All metrics are computed in float64 over the GT-resolution image. The
*valid mask* `M` is `cityscapes_mask != sky colour`; layer-specific
validity is intersected on top (`depth < 1e9` and finite for depth,
`||n_gt|| > 0.5` for normals).

**RGB (nvs / estimation)** — computed on sRGB-encoded values in [0,1]:

- `psnr` / `psnr_masked`: `-10 log10( mean (p-g)^2 )` over all / masked
  pixels x channels, data range 1. Identical inputs give `inf` (serialised
  as the string `"inf"` in JSON, `inf` in CSV).
- `ssim` / `ssim_masked`: scikit-image `structural_similarity`,
  `data_range=1`, default 7x7 uniform window, `channel_axis=-1`; the masked
  variant averages the full SSIM map over `M`.
- `psnr_masked_ea` (exposure-aligned, secondary): both images are
  sRGB-decoded; a single scalar `s* = Σ p·g / Σ p·p` is fitted over masked
  pixels; `clip(sRGB(s*·p),0,1)` is scored with masked PSNR. This
  compensates the dataset's per-image q98 exposure gauge (see conventions).
  Note the linearise/re-encode float round-trip means identical inputs
  score ≈ 152 dB, not `inf`.
- `lpips`: LPIPS with the **pinned net `alex`** (package `lpips==0.1.4`,
  full image, inputs scaled to [-1,1]). Optional dependency: if `lpips` /
  `torch` are not installed it is skipped with a note in the output.

**Decomposition** (all masked by `M`):

- `albedo_psnr_masked`, `albedo_ssim_masked`: prediction is first aligned
  to GT by a **per-channel least-squares scale** (no intercept):
  `alpha_c = Σ_M (p_c · g_c) / Σ_M (p_c^2)` (1 if the denominator is 0),
  then `clip(alpha_c · p_c, 0, 1)` is scored against `clip(g, 0, 1)` in
  linear space. This removes the standard albedo/illumination scale
  ambiguity per colour channel while preserving structure errors.
- `normal_mae_deg`, `normal_median_ae_deg`: both vectors renormalised;
  mean/median of `arccos(clip(p̂·ĝ, -1, 1))` in degrees over
  `M ∧ ||n_gt||>0.5 ∧ ||n_pred||>1e-6`.
- `depth_rmse`, `depth_mae`: in scene units (metres), **no scale
  alignment** — poses are metric, so the gauge is part of the task.
- `depth_rmse_scale_aligned` (secondary, for methods with gauge drift) with
  `depth_scale_factor`: `s* = Σ p·g / Σ p^2`, RMSE of `s*·p` vs `g`.
- `roughness_mse_masked`, `metallic_mse_masked`: plain masked MSE (no
  alignment).

**Aggregation**: per-frame values are averaged arithmetically over frames
(PSNR included, the field convention); if any per-frame value is `inf` the
aggregate is `inf`. `<metric>__n` records how many frames contributed.
Cross-scene numbers are the unweighted mean of the 5 per-scene aggregates.

## Output schema

`metrics.json`:

```json
{
  "scene": "...", "split": "test", "pred_dir": "...", "tracks": ["nvs", "decomposition"],
  "n_frames": 25,
  "lpips": {"requested": true, "available": false, "net": "alex", "version": null},
  "notes": ["layer 'roughness' not predicted for any frame; ..."],
  "aggregate": {"nvs/psnr": 26.22, "nvs/psnr__n": 25, "...": "..."},
  "per_frame": {"0000": {"nvs/psnr": 26.43, "...": "..."}}
}
```

Non-finite values are serialised as strings `"inf"`/`"-inf"`/`"nan"`.
`metrics_per_frame.csv` has one row per frame plus an `aggregate_mean` row;
columns are the flattened `track/metric` names.

## Dependencies

```
python >= 3.9
numpy            (validated with 2.2)
imageio >= 2.27  (PNG I/O)
scikit-image >= 0.22  (SSIM; validated with 0.25.2)
OpenEXR >= 3.2   (EXR I/O; `pip install OpenEXR`)  — or pyexr as fallback
# optional, only for LPIPS:
torch (CPU is fine), lpips == 0.1.4   (net pinned to "alex" in evaluate.py)
```

`cv2.imread` CANNOT be used for the GT EXRs: the normal maps store
channels named `X/Y/Z`, which OpenCV silently mangles into a single
channel.

## Worked example

```bash
# score a prediction directory (any method) on one scene
python evaluate.py \
    --pred-dir /path/to/predictions/my_method/abandoned_buildings_gt_illum \
    --data ~/data/neusky_synthetic_data/renders/abandoned_buildings_prepared \
    --split test --tracks nvs decomposition \
    --output metrics.json --csv metrics_per_frame.csv

# Track 2 directory: label it as estimation
python evaluate.py --pred-dir .../my_method/abandoned_buildings_estimated \
    --data .../abandoned_buildings_prepared --tracks estimation

# render NeuSky into the contract (GPU; repo root, trained checkpoint under outputs/)
PYTHONPATH=. python scripts/synthetic_benchmark/render_neusky_predictions.py \
    --scene abandoned_buildings --illumination gt \
    --pred-dir outputs/synthetic_benchmark/abandoned_buildings_gt
PYTHONPATH=. python scripts/synthetic_benchmark/render_neusky_predictions.py \
    --scene abandoned_buildings --illumination estimated \
    --pred-dir outputs/synthetic_benchmark/abandoned_buildings_estimated
```

## Validation

`test_evaluate.py` (pytest, runs anywhere — builds its own tiny fixture)
covers identity, perturbation, masking, alignment-invariance and
missing-layer behaviour. Additionally the scorer was validated against the
real dataset (abandoned_buildings, frames 0000-0002):

GT-as-prediction (GT layers symlinked into a prediction dir):

```
nvs/psnr inf   nvs/psnr_masked inf   nvs/ssim 1.0000   nvs/psnr_masked_ea 151.89
decomposition/albedo_psnr_masked inf   albedo_ssim_masked 1.0000
decomposition/normal_mae_deg 0.0024 (float32 arccos noise)
decomposition/depth_rmse 0.0   depth_mae 0.0   depth_scale_factor 1.0
roughness/metallic mse 0.0
```

Perturbed predictions (rgb +N(0,0.05), albedo +N(0,0.1), normals rotated
10° about Z, depth x1.05):

```
nvs/psnr 26.22          (theory for sigma=0.05: 26.02 dB)
decomposition/albedo_psnr_masked 21.57   (sigma=0.1 -> ~20 dB + clipping)
decomposition/normal_mae_deg 8.25, median 9.95   (<= 10°, less near zenith)
decomposition/depth_rmse 1.82 (raw)   depth_rmse_scale_aligned 0.0000
decomposition/depth_scale_factor 0.9524 (= 1/1.05)
```
