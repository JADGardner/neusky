# NeRF-OSR Overlay Release

This directory defines the public release of the additional files NeuSky uses
with the official NeRF-OSR dataset. The builder has an explicit allowlist:

- `Data/<scene>/final/{train,validation,test}/cityscapes_mask/*.png`
- `Data/<scene>/final/points3d.ply`
- `Data/<scene>/final/envmap_rotations.json`

It does not stage the upstream images, environment maps, poses or COLMAP
models.

Build the release:

```bash
python scripts/nerfosr_overlay/build_hf_release.py \
  --data-root ~/data/NeRF-OSR \
  --output /path/to/neusky-nerfosr-overlay-v1
```

The output contains one independently downloadable archive per scene, a
complete extracted-file manifest, release checksums, the rendered Hugging
Face dataset card and the overlay licence.
