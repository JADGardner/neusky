---
pretty_name: NeuSky Synthetic
license: cc-by-4.0
size_categories:
  - 1K<n<10K
tags:
  - image
  - 3d
  - computer-vision
  - inverse-rendering
  - neural-rendering
  - relighting
  - intrinsic-decomposition
  - synthetic
  - hdr
  - outdoor-scenes
---

# NeuSky Synthetic

![NeuSky Synthetic scenes and ground-truth layers](assets/synthetic_dataset_overview.png)

NeuSky Synthetic is a five-scene, multi-illumination outdoor dataset for
evaluating neural inverse rendering, intrinsic decomposition, novel-view
synthesis and illumination estimation. It contains 1,500 rendered views at
1920 x 1080 resolution, with 250 training, 25 validation and 25 test views per
scene.

This benchmark was developed as a thesis extension to
[The Sky's the Limit](https://arxiv.org/abs/2311.16937). It was not part of
the original ECCV 2024 evaluation.

| Property | Value |
|---|---:|
| Scenes | 5 |
| Total views | 1,500 |
| Training views | 1,250 |
| Validation views | 125 |
| Test views | 125 |
| Resolution | 1920 x 1080 |
| Illumination environments | 167 |
| Release | @@RELEASE_VERSION@@ |
| Generator commit | [`@@NEUSKY_COMMIT@@`](https://github.com/JADGardner/neusky/tree/@@NEUSKY_COMMIT@@) |

## Scenes

- `abandoned_buildings`
- `apartment_building`
- `arlanda_uppsala_cathedral`
- `glass_building`
- `interstellar_house`

Each scene is rendered under many Poly Haven HDR environments with known
horizontal rotation. Training views additionally vary camera focal length and
exposure. Camera intrinsics, camera-to-world transforms, exposure, environment
identity and environment rotation are recorded per frame.

## Splits And Layers

The training split contains only the inputs used for reconstruction:

- tone-mapped RGB;
- foreground/sky semantic masks.

Validation and test additionally provide the ground-truth decomposition:

- diffuse albedo;
- world-space surface normals;
- metric depth;
- roughness;
- metallic;
- transmission;
- index of refraction.

Every scene also contains `points3d.ply`, used for scene centring and
Gaussian-splatting initialisation, and `transforms.json`, containing all camera
and illumination metadata.

Each independently downloadable archive extracts to:

```text
scenes/<scene>/
  transforms.json
  points3d.ply
  train/
    rgb/
    cityscapes_mask/
  validation/
    rgb/
    cityscapes_mask/
    albedo/
    normal/
    depth/
    roughness/
    metallic/
    transmission/
    ior/
  test/
    <same layers as validation>
```

The mask images use the Cityscapes colours expected by NeuSky: sky is
`(70, 130, 180)` and the reconstructed scene is `(70, 70, 70)`.

## Download

Download the complete release with the Hugging Face CLI, verify it, then
extract the five scene archives:

```bash
hf download @@REPO_ID@@ \
  --repo-type dataset \
  --revision v@@RELEASE_VERSION@@ \
  --local-dir neusky-synthetic

(cd neusky-synthetic && sha256sum -c SHA256SUMS)
for archive in neusky-synthetic/archives/*.tar.zst; do
  tar --zstd -xf "$archive" -C neusky-synthetic
done
```

Download one scene:

```bash
hf download @@REPO_ID@@ \
  --repo-type dataset \
  --revision v@@RELEASE_VERSION@@ \
  archives/interstellar_house.tar.zst \
  --local-dir neusky-synthetic

tar --zstd -xf \
  neusky-synthetic/archives/interstellar_house.tar.zst \
  -C neusky-synthetic
```

An individual scene can also be downloaded without the CLI:

```bash
curl -L \
  "https://huggingface.co/datasets/@@REPO_ID@@/resolve/v@@RELEASE_VERSION@@/archives/interstellar_house.tar.zst?download=true" \
  -o interstellar_house.tar.zst
```

`zstd` is required for extraction. `MANIFEST.json` records the size and SHA256
of every release file, and `SHA256SUMS` can be checked before extraction.

## NeuSky

The loader, training code and deterministic generation pipeline are in the
[NeuSky repository](https://github.com/JADGardner/neusky).

```bash
git clone --recurse-submodules https://github.com/JADGardner/neusky.git
cd neusky

PYTHONPATH=. python scripts/train_synthetic.py \
  --data /path/to/neusky-synthetic/scenes/abandoned_buildings
```

See `provenance/scene_sources.json` for the source models and scene hashes,
and `provenance/scene_render_configs.json` for the accepted camera and
rendering profiles.

## Baseline Results

These are the accepted means over all five scenes. The full per-scene results
and additional raw/exposure-aligned columns are in
`benchmark/synthetic_results.csv`.

| Method | Known PSNR EA ↑ | SSIM ↑ | LPIPS ↓ | Left-Half PSNR ↑ | SSIM ↑ | LPIPS ↓ | Albedo PSNR ↑ | Normal MAE ↓ | Depth RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NeuSky | **16.54** | 0.506 | 0.602 | **15.94** | **0.575** | 0.570 | 17.84 | 46.49 | **24.91** |
| NeRF-OSR | 15.48 | 0.335 | 0.591 | 12.79 | 0.468 | **0.523** | **18.43** | **38.44** | 27.13 |
| GS-IR | 16.11 | **0.597** | **0.415** | - | - | - | 17.30 | 55.46 | 25.86 |

Known-illumination novel-view synthesis uses the ground-truth HDR environment
and rotation. Its PSNR is exposure-aligned. In the left-half fitting track,
illumination is fitted using only the left half of each held-out image and the
full image is scored. GS-IR has no entry because its scene-wide lighting has
no per-image estimation mechanism. Decomposition metrics use non-sky,
valid-object masks; albedo PSNR is scale-aligned and normal MAE is measured in
degrees.

## Provenance

The 167 HDR environments are CC0 assets from
[Poly Haven](https://polyhaven.com/). Their identifiers, current download
metadata and accepted-generation checksums are included under `provenance/`.
Poly Haven has made minor upstream revisions to 36 EXRs since the accepted
render. Current upstream files are supported for source re-rendering; this
prepared release is canonical for reproducing the reported benchmark.

The editable Blender scenes are not included. They contain extractable
third-party models that cannot be redistributed in source form under their
licences. `provenance/scene_sources.json` records their authors, source pages,
licences and accepted scene hashes.

## Licence

The prepared renders, masks, camera metadata, point clouds and ground-truth
layers in this repository are released under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
See `LICENSE.md`.

This licence does not apply to the external HDRIs or editable third-party
models referenced by the provenance files. The HDRIs are distributed
separately by Poly Haven under CC0; the editable models are not included.

## Limitations

- The dataset contains five synthetic outdoor architectural scenes and does
  not represent the full diversity of real outdoor captures.
- Camera paths are sampled around the principal scene content and do not
  reproduce a specific handheld capture trajectory.
- The training data contains deliberately broad illumination and exposure
  variation; methods assuming a single fixed scene illumination are outside
  their intended setting.
- Ground-truth decomposition layers are reserved for validation and test and
  must not be used as training supervision when reproducing the reported
  protocol.

## Citation

The synthetic benchmark should be cited as part of the accompanying thesis.
For NeuSky, cite:

```bibtex
@inproceedings{gardner2024sky,
  author    = {James A. D. Gardner and Evgenii Kashin and
               Bernhard Egger and William A. P. Smith},
  title     = {The Sky's the Limit: Relightable Outdoor Scenes via a
               Sky-Pixel Constrained Illumination Prior and Outside-In
               Visibility},
  booktitle = {European Conference on Computer Vision},
  pages     = {126--143},
  year      = {2024}
}
```

Release prepared on @@RELEASE_DATE@@.
