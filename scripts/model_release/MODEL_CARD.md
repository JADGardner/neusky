---
library_name: nerfstudio
tags:
- inverse-rendering
- neural-rendering
- relighting
- outdoor-scenes
- nerfstudio
datasets:
- jadgardner/neusky-synthetic
- jadgardner/neusky-nerfosr-overlay
---

# NeuSky Checkpoints

This repository contains the scene-specific checkpoints used for the NeuSky
experiments in *Learning Representations for Incomplete Spherical Scene
Signals*.

Two collections are provided:

- `nerf-osr`: the final `lk2`, `lwp` and `st` models used for the NeRF-OSR
  evaluation and thesis figures.
- `synthetic`: the final five NeuSky models used in the accepted synthetic
  benchmark matrix.

Every model directory contains its nerfstudio `config.yml`,
`dataparser_transforms.json`, and final checkpoint. `MODEL_MANIFEST.json`
records the original source run, byte size and SHA256 of every file. Failed,
superseded and diagnostic runs are not included.

All models use the exact channelwise, two-bracket RENI prior released as
`neusky-prior` in
[jadgardner/reni-models](https://huggingface.co/jadgardner/reni-models).
They do not use the later joint-frame RENI thesis model.

## Download

From a clone of NeuSky, download all three NeRF-OSR models:

```bash
python scripts/download_models.py model-storage/neusky
```

Download one scene or the synthetic collection:

```bash
python scripts/download_models.py model-storage/neusky --scene lk2
python scripts/download_models.py model-storage/neusky --collection synthetic
```

The required RENI prior is downloaded from the `ns_reni` submodule:

```bash
python ns_reni/scripts/download_models.py \
  model-storage/reni \
  --group neusky-prior
```

Files can also be retrieved directly:

```bash
hf download jadgardner/neusky-models \
  --revision v1.0 \
  --include "nerf-osr/lk2/*" "nerf-osr/lk2/**/*" \
  --local-dir model-storage/neusky
```

## Data

- [NeuSky Synthetic v1.0](https://huggingface.co/datasets/jadgardner/neusky-synthetic)
  contains the five released synthetic scene datasets.
- [NeuSky NeRF-OSR Overlay v1.0](https://huggingface.co/datasets/jadgardner/neusky-nerfosr-overlay)
  contains only the NeuSky masks, registered point clouds and environment-map
  rotations to layer over the official NeRF-OSR data.

The original NeRF-OSR images, poses and environment maps are not redistributed
here.

## Licence

No licence has yet been assigned to the NeuSky checkpoint release. The source
datasets and third-party assets retain their own licences.
