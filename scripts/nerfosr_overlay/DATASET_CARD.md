---
pretty_name: NeuSky additions for NeRF-OSR
license: cc-by-4.0
tags:
  - image
  - 3d
  - computer-vision
  - inverse-rendering
  - neural-rendering
  - relighting
  - semantic-segmentation
  - outdoor-scenes
---

# NeuSky Additions for NeRF-OSR

This repository contains the additional files used to train and evaluate
[NeuSky](https://github.com/JADGardner/neusky) on three scenes from the
[NeRF-OSR dataset](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/):

- Cityscapes semantic segmentation masks;
- scene point clouds used for SfM-based centring and scaling;
- rotations that align the captured environment maps with the dataset camera
  coordinate frame.

It is an overlay for an official NeRF-OSR download. It does **not** contain
NeRF-OSR RGB images, camera poses, environment maps, COLMAP models, or any
other original dataset files.

| Property | Value |
|---|---:|
| Scenes | 3 |
| Segmentation masks | 1,021 |
| Point clouds | 3 |
| Environment-map rotation records | 16 |
| Release | @@RELEASE_VERSION@@ |
| Release date | @@RELEASE_DATE@@ |
| NeuSky commit | [`@@NEUSKY_COMMIT@@`](https://github.com/JADGardner/neusky/tree/@@NEUSKY_COMMIT@@) |

## Contents

Each independently downloadable archive preserves the target paths within the
official dataset:

```text
Data/<scene>/final/
  points3d.ply
  envmap_rotations.json
  train/cityscapes_mask/*.png
  validation/cityscapes_mask/*.png
  test/cityscapes_mask/*.png
```

The included scenes and mask counts are:

| Scene | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `lk2` | 160 | 5 | 95 | 260 |
| `lwp` | 258 | 5 | 96 | 359 |
| `st` | 301 | 5 | 96 | 402 |

Every mask has the same filename stem as its corresponding official RGB
image. The masks use the Cityscapes palette consumed by the NeuSky
dataparser.

The binary PLY files contain XYZ and RGB values. They were converted from
COLMAP point clouds triangulated against the known, dataset-normalised
NeRF-OSR camera poses. NeuSky uses them to choose a robust scene centre and
scale while keeping scene geometry and cameras inside the unit sky sphere.

Each `envmap_rotations.json` record maps directions in the associated
equirectangular environment image into the dataset-normalised world frame.
Fifteen rotations were recovered by registering perspective crops from the
environment panoramas into the corresponding scene model. The remaining
`st/01-09_14_00` rotation was obtained by panorama-to-panorama composition and
is marked with its method and diagnostics in the JSON.

## Download and Install

First obtain the NeRF-OSR dataset from its
[official project page](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/). Then download
and verify this overlay:

```bash
hf download @@REPO_ID@@ \
  --repo-type dataset \
  --revision v@@RELEASE_VERSION@@ \
  --local-dir neusky-nerfosr-overlay

(cd neusky-nerfosr-overlay && sha256sum -c SHA256SUMS)
```

Extract the archives over the directory that contains the official `Data/`
folder:

```bash
for archive in neusky-nerfosr-overlay/archives/*.tar.zst; do
  tar --zstd -xf "$archive" -C /path/to/NeRF-OSR
done
```

Download a single scene:

```bash
hf download @@REPO_ID@@ \
  --repo-type dataset \
  --revision v@@RELEASE_VERSION@@ \
  archives/lk2.tar.zst \
  --local-dir neusky-nerfosr-overlay

tar --zstd -xf \
  neusky-nerfosr-overlay/archives/lk2.tar.zst \
  -C /path/to/NeRF-OSR
```

An individual archive can also be downloaded without the CLI:

```bash
curl -L \
  "https://huggingface.co/datasets/@@REPO_ID@@/resolve/v@@RELEASE_VERSION@@/archives/lk2.tar.zst?download=true" \
  -o lk2.tar.zst
```

`zstd` is required for extraction. `OVERLAY_CONTENTS.json` records the
destination, size and SHA256 of every extracted file. `MANIFEST.json` and
`SHA256SUMS` cover the downloadable release files.

## Reproducibility

The relevant NeuSky scripts are:

- `scripts/nerfosr_pointclouds.py`, which converts the registered COLMAP
  points to the shipped PLY format;
- `scripts/register_envmaps.py`, which recovers, inspects and installs the
  environment-map rotations;
- `scripts/nerfosr_overlay/build_hf_release.py`, which validates the exact
  additions and builds this allowlisted release.

The release builder checks that every mask corresponds to an official RGB
filename, validates the PLY headers and rotation matrices, and stages only
the three additions listed above.

## Licence

The NeuSky-created masks, point clouds and rotation metadata in this overlay
are licensed under CC BY 4.0. This licence does not apply to the original
NeRF-OSR dataset, which is not included and remains subject to its own terms.
See `LICENSE.md`.

## Citation

Please cite both NeuSky and NeRF-OSR when using these additions:

```bibtex
@inproceedings{gardner2024neusky,
  title     = {The Sky's the Limit: Relightable Outdoor Scenes via a
               Sky-pixel Constrained Illumination Prior and Outside-In
               Visibility},
  author    = {Gardner, James A. D. and Egger, Bernhard and Smith, William A. P.},
  booktitle = {European Conference on Computer Vision},
  year      = {2024}
}

@inproceedings{rudnev2022nerfosr,
  title     = {{NeRF} for Outdoor Scene Relighting},
  author    = {Rudnev, Viktor and Elgharib, Mohamed and Smith, William and
               Liu, Lingjie and Golyanik, Vladislav and Theobalt, Christian},
  booktitle = {European Conference on Computer Vision},
  year      = {2022}
}
```
