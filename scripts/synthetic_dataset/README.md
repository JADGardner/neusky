# NeuSky Synthetic Dataset Generator

This directory contains the rendering and preparation pipeline for NeuSky
Synthetic v1. The generator lives in the NeuSky repository so that its
implementation is versioned with the training and evaluation code.

The prepared RGB and ground-truth benchmark data can be redistributed. The
current Blender source scenes incorporate third-party Royalty Free models whose
licences do not permit distributing an editable source asset. Those `.blend`
files must remain in the private reproducibility archive unless the original
authors grant redistribution permission. See the
[BlenderKit licence](https://www.blenderkit.com/docs/licenses/) and
[licensing FAQ](https://www.blenderkit.com/docs/licenses/licensing-faq/).
`scene_sources.json` is the machine-readable provenance record for the exact
accepted scenes. `synthetic_scenes.md` gives the corresponding construction
notes and human-readable links.

## Reproducibility Levels

There are two supported routes:

1. **Reproduce the reported experiments.** Download the prepared five-scene
   dataset and use the committed training and evaluation code. This is the
   canonical route and does not require Blender or any editable third-party
   models.
2. **Re-render the dataset from source.** Acquire the base models listed in
   `scene_sources.json`, restore or reconstruct the accepted scenes, download
   the HDRIs, and run the pipeline below.

The second route is not yet a fully public, bitwise clean-room rebuild. The
accepted `.blend` files contain extractable Royalty Free models and therefore
cannot be redistributed under the current BlenderKit and ArtStation terms.
Four source listings remain available, but the Interstellar Cooper House has
been removed from BlenderKit. Exact source re-rendering therefore requires
either permission from the relevant authors or a user's own previously
licensed copy. The accepted scene filenames, sizes and SHA256 hashes are
recorded so that private or author-approved copies can be verified.

## Dataset Layout

Set `NEUSKY_SYN_DATA` to a writable dataset directory:

```text
neusky_synthetic_data/
  scenes/                    # acquired or privately restored source scenes
    abandoned_buildings.blend
    apartment_building.blend
    arlanda_uppsala_cathedral.blend
    glass_building.blend
    interstellar_house.blend
  background_assets/         # shared Poly Haven CC0 assets
  textures/                  # shared Poly Haven CC0 textures
  hdris_16k/
  renders/
```

The `.blend` files use relative paths into `background_assets/` and `textures/`.
Keep this layout when restoring a scene from the private archive. Poly Haven
HDRIs are not included in the prepared-data bundle. They are CC0 assets and
can be downloaded from the pinned 167-asset list.

## Download The HDRIs

The downloader resolves the pinned assets through the Poly Haven API, verifies
their published MD5 checksums, and resumes interrupted files:

```bash
python scripts/synthetic_dataset/download_hdris.py \
  --output "$NEUSKY_SYN_DATA/hdris_16k"
```

`hdris_16k_manifest.json` pins the currently resolved file URLs, sizes and
checksums. `hdris_16k_generation_md5.txt` records the files used for the
accepted render. Poly Haven has since made minor upstream revisions to 36 of
the 167 EXRs under the same asset identifiers. A pixel-level spot check found
only sparse image changes, consistent with retouching or reprocessing. Current
Poly Haven downloads are accepted for source re-rendering; they are not
expected to reproduce the historical render pixels bit for bit.

The distributed prepared dataset is the canonical input for reproducing the
reported results. Regenerate the current download manifest only when
intentionally updating the pinned upstream files:

```bash
python scripts/synthetic_dataset/download_hdris.py \
  --refresh-manifest --manifest-only
```

`POLYHAVEN_SOURCE_AUDIT.md` records the checksum and pixel-level investigation
that distinguishes these upstream revisions from a downloader error.

The sorted asset order is part of the dataset definition because the render
profiles select environments using fixed offsets. The distributed prepared
dataset remains the canonical input for reproducing the reported results.

## Rebuild The Prepared Dataset

Blender 5.0 or later must be available as `blender`. Rebuild all five scenes:

```bash
NEUSKY_SYN_DATA=/path/to/neusky_synthetic_data \
  scripts/synthetic_dataset/rebuild_synthetic_dataset.sh
```

To rebuild selected scenes:

```bash
NEUSKY_SYN_DATA=/path/to/neusky_synthetic_data \
  scripts/synthetic_dataset/rebuild_synthetic_dataset.sh \
  apartment_building glass_building
```

Each scene is rendered with three deterministic profiles:

| Profile | Frames | Purpose |
|---|---:|---|
| `train` | 150 | Varied cameras, focal lengths and exposures |
| `train_curated` | 100 | Whole-building views with training variation |
| `eval` | 50 | Fixed-focal validation and test views |

The prepared output combines the first two profiles into 250 training frames
and splits the evaluation profile into 25 validation and 25 test frames. The
training split contains RGB and foreground masks. Validation and test also
contain the ground-truth decomposition layers used by the benchmark.

The rebuild produces large raw multipart EXR directories before preparation.
Only the resulting five `*_prepared` directories are part of the public
prepared-data release.

## Components

- `blender_render_scene.py`: deterministic Cycles rendering, camera sampling,
  environment selection and multipart render passes.
- `scene_render_configs.json`: the accepted five-scene camera profiles and
  seeds.
- `render_eval_views.py`: profile launcher for one scene.
- `split_multipass_exr.py`: extracts the multipart render layers.
- `prepare_synthetic_data.py`: writes the RGB, masks, decomposition layers and
  split metadata consumed by NeuSky.
- `extract_pointcloud.py`: produces the metric point cloud used for scene
  centering and Gaussian-splatting initialisation.
- `rebuild_synthetic_dataset.sh`: end-to-end five-scene rebuild.
- `scene_sources.json`: exact model provenance, source availability, accepted
  scene hashes and Poly Haven dependencies.
- `POLYHAVEN_SOURCE_AUDIT.md`: investigation of the 36 revised upstream HDRIs.
- `synthetic_scenes.md`: human-readable scene provenance and construction
  notes.
- `scene_setup/`: the original scene assembly, cliff placement, vegetation
  regeneration and inspection utilities. These require separately acquired
  source models where redistribution is restricted.

Method-specific NeRF-OSR and GS-IR adapters are under
`scripts/synthetic_benchmark/`; they are derived from the prepared release and
are not distributed as separate datasets.
