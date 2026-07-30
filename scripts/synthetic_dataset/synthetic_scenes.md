# Synthetic Outdoor Multi-Illumination Scenes

Synthetic datasets for evaluating models' ability to accurately predict materials
and illumination. Each scene provides ground truth geometry (depth, normals),
albedo, object masks, and per-frame HDRI environment maps with known rotations.

> **Redistribution note:** The prepared 2D renders and benchmark layers are
> released publicly. Several editable source scenes incorporate BlenderKit
> Royalty Free models, which cannot be redistributed as extractable `.blend`
> assets under the current BlenderKit terms. Keep the source scenes in the
> private archive unless their authors grant permission. Users rebuilding from
> source must acquire the listed models under their own BlenderKit accounts.
> See the [BlenderKit licence](https://www.blenderkit.com/docs/licenses/) and
> [licensing FAQ](https://www.blenderkit.com/docs/licenses/licensing-faq/).
> `scene_sources.json` records the exact listing identifiers, source filenames,
> cached asset identifiers, accepted scene hashes and current availability.

## Dataset Purpose

Test a model's ability to decompose a scene into:
- **Geometry** — depth and surface normals
- **Materials** — diffuse albedo, roughness
- **Illumination** — HDRI environment map identity and orientation

Each frame is rendered under a different HDRI with a random Z-axis rotation,
so the model must disentangle material appearance from lighting.

---

## Scenes

### Scene 1: Arlanda Uppsala Cathedral (+ Medieval Fort)

- **Blend file**: `data/neusky_synthetic_data/scenes/arlanda_uppsala_cathedral.blend`
- **Cathedral source**: [Arlanda Uppsala Cathedral on ArtStation](https://www.artstation.com/marketplace/p/MpNz/arlanda-uppsala-cathedral)
- **Author**: Tiv Sol
- **Viewer**: [interactive Sketchfab page](https://sketchfab.com/3d-models/arlanda-uppsala-cathedral-9c91370bec27494ca3c64a3a3ea9964d)
- **License**: ArtStation Standard Use License
- **Acquired file**: `Arlanda_Uppsala_Cathedral.fbx`
- **Additional assets**:
  - [`modular_fort_01`](https://polyhaven.com/a/modular_fort_01) (Poly Haven, CC0) — medieval fort architecture with 8K PBR
- **External shared assets** (in `background_assets/` and `textures/`):
  - [`coastal_cliff_04`](https://polyhaven.com/a/coastal_cliff_04) (Poly Haven, CC0) — cliff background geometry
  - [`patterned_cobblestone`](https://polyhaven.com/a/patterned_cobblestone) (Poly Haven, CC0) — ground material
- **Note**: This scene has its own ground plane and cliff backdrop (not created by `setup_scene.py`). It requires restricted azimuth (`--azimuth_min -45 --azimuth_max 90`) because cliffs surround the scene on the north and south sides.
- **Camera params**: `--sphere_radius 80 --sphere_center -30 -28 20 --azimuth_min -45 --azimuth_max 90`
- **Status**: Integrated into standardised pipeline. Textures remapped and verified. Ready for production render.

### Scene 2: Interstellar Farmhouse

- **Source**: [BlenderKit — Interstellar House](https://www.blenderkit.com/asset-gallery-detail/02092ebd-0c3a-4474-9a9d-d54474b01445/)
- **Author**: Abin Suresh
- **License**: Royalty Free (BlenderKit)
- **Availability**: removed from BlenderKit as of 2026-07-30; the URL is
  retained as the original provenance record
- **Faces**: 582,607
- **Style**: Rural farmhouse, highly detailed, cinematic (Cooper house from Interstellar)
- **Blend file**: `data/neusky_synthetic_data/scenes/interstellar_house.blend`
- **Ground texture**: grass_path_2 (Poly Haven CC0)
- **Status**: Scene set up with PBR ground + background vegetation. Test renders verified. Ready for production render.

### Scene 3: Modern Apartment Building

- **Source**: [BlenderKit — Modern Apartment Building](https://www.blenderkit.com/asset-gallery-detail/b3faf096-32df-466d-b04d-7c8b3b3983df/)
- **Author**: Pawel Walasiewicz
- **License**: Royalty Free (BlenderKit)
- **Faces**: 66,048
- **Style**: Realistic modern residential building based on real project
- **Blend file**: `data/neusky_synthetic_data/scenes/apartment_building.blend`
- **Ground texture**: concrete_floor_02 (Poly Haven CC0)
- **Status**: Scene set up. Building shifted down to sit flush on ground. Test renders verified. Ready for production render.

### Scene 4: Abandoned Buildings

- **Source**: [BlenderKit — Building Pack Lowpoly](https://www.blenderkit.com/asset-gallery-detail/57b02da8-288e-4bf3-b326-b1fdddac89de/)
- **Author**: Fridqeir Haynura
- **License**: Royalty Free (BlenderKit)
- **Faces**: 8,280
- **Style**: Historic brick buildings with AI-generated textures, urban/abandoned
- **Blend file**: `data/neusky_synthetic_data/scenes/abandoned_buildings.blend`
- **Ground texture**: cobblestone_floor_04 (Poly Haven CC0)
- **Status**: Scene set up. Test renders verified. Ready for production render.

### Scene 5: Modern Glass Building

- **Source**: [BlenderKit — Modern Glass Building](https://www.blenderkit.com/asset-gallery-detail/dd5d37c8-79e6-4171-81ba-4909aabde32d/)
- **Author**: Abbos Mirzaev
- **License**: Royalty Free (BlenderKit)
- **Faces**: 5,417
- **Style**: Contemporary glass office building with reflective surfaces
- **Blend file**: `data/neusky_synthetic_data/scenes/glass_building.blend`
- **Ground texture**: concrete_floor_02 (Poly Haven CC0)
- **Status**: Scene set up. Test renders verified. Ready for production render.

## Shared Resources

### HDRIs (Multi-Illumination)

- **Location**: `data/neusky_synthetic_data/hdris_16k/`
- **Count**: 167 selected EXR files
- **Source**: Poly Haven (CC0)
- Each frame uses a different HDRI with a random Z-axis rotation
- Download with `scripts/synthetic_dataset/download_hdris.py`
- `hdris_16k_legacy.txt` lists the 36 accepted-generation files that Poly
  Haven has subsequently replaced

### Ground Textures (Poly Haven CC0, 2K)

Located in `data/neusky_synthetic_data/textures/`:

| Texture | Files | Used by |
|---------|-------|---------|
| `concrete_floor_02` | diff, nor_gl, rough | Apartment, Glass building |
| `grass_path_2` | diff, nor_gl, rough | Interstellar house |
| `cobblestone_floor_04` | diff, nor_gl, rough | Building pack |

### Background Assets (Poly Haven CC0, 1K .blend)

Located in `data/neusky_synthetic_data/background_assets/`:

| Asset | Placement | Count | Scale |
|-------|-----------|-------|-------|
| `tree_small_02_1k.blend` | r=80 ring | 12 | 2.0–3.5× |
| `boulder_01_1k.blend` | r=56 ring | 8 | 3.0–6.0× |
| `shrub_04_1k.blend` | r=68 ring | 16 | 3.0–5.0× |
| `island_tree_01_1k.blend` | r=104 ring | 8 | 1.5–2.5× |
| `boulder_01_1k.blend` | r=80 ring (outer) | 12 | 2.0–4.0× |
| `coastal_cliff_04_8k.blend` | — | — | — |
| `rock_face_01_8k.blend` | — | — | — |

The cliff and rock assets (8K, Poly Haven CC0) are available for use as background geometry in any scene. The ground plane (200×200) has terrain deformation: flat within r=80, then smoothstep rise to 15 units at the edges (100×100 vertex grid with smooth shading).

---

## Rendering Pipeline

The accepted camera parameters, frame counts, seeds and post-processing steps
are defined by `scene_render_configs.json` and
`rebuild_synthetic_dataset.sh`. Do not reconstruct production commands from
the historical values in individual scene notes.

```bash
NEUSKY_SYN_DATA=/path/to/neusky_synthetic_data \
  scripts/synthetic_dataset/rebuild_synthetic_dataset.sh
```

See `README.md` for the full directory contract, HDRI restoration procedure
and selective-scene rebuild syntax.

### Per-frame Metadata

Each frame in `transforms_train.json` includes:

| Field | Description |
|-------|-------------|
| `file_path` | Path to rendered image |
| `transform_matrix` | Camera-to-world 4×4 matrix |
| `envmap_name` | Poly Haven HDRI asset name (e.g. `abandoned_parking`) |
| `envmap_url` | Direct Poly Haven link |
| `envmap_rotation` | Euler XYZ rotation in radians applied to envmap `[0, 0, z]` |

### Output Passes (EXR mode)

| Pass | Format | Description |
|------|--------|-------------|
| RGB | RGBA float32 | Linear HDR colour |
| Depth | float32 | Z-depth from camera |
| Normal | XYZ float32 | World-space surface normals |
| Albedo | RGBA float32 | Diffuse colour (Cycles "Diffuse Color" pass) |
| Alpha | float32 | Object mask (0/1, via transparent film) |

---

## Notes

- Camera and focal-length sampling varies by accepted render profile.
- The ground plane has terrain deformation (hills rising beyond r=80) plus 5 rings of background vegetation to completely hide the horizon.
- Each frame uses a different HDRI with a random rotation, so models must disentangle materials from illumination.
- Point clouds can be extracted from depth maps for initialising Gaussian Splatting.
- For NeuSky training, images are downscaled 4× (`camera_res_scale_factor=0.25`) to fit in 24GB GPU memory.
