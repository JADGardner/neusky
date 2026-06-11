"""Ambient occlusion and single-direction shadow (thesis Fig: ao_and_shadow ->
ao_and_shadow.pdf).

Two visualisations of the soft DDF visibility for one training view:

- left: ambient occlusion — visibility averaged over all (upper-hemisphere)
  illumination directions, via the model's render_ambient_light path
  ("shading" output; the notebook's ambient-lighting cell);
- right: a sharp shadow from a single direction, via the model's shadow-map
  path (compute_visibility with one light direction; the notebook's
  point-light shadow-map cell, azimuth 45 / elevation 20 / sigmoid scale 10).

Sky pixels are filled from the dataset sky mask as in the notebook.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_ao_shadow.py --scene lk2
"""

import argparse

import matplotlib.pyplot as plt

from _common import (SCENE_DEFAULT_VIEWS, add_common_args, load_model,
                     render_view, save_figure, seed_all)


def render_ambient_occlusion(model, datamanager, view_idx, device, gain: float):
    """AO = mean DDF visibility over illumination directions ("shading")."""
    import numpy as np

    model.config.render_ambient_light = True
    model.config.lower_hermisphere_visibility = False
    try:
        outputs = render_view(model, datamanager.train_dataset.cameras, view_idx, device)
    finally:
        model.config.render_ambient_light = False
        model.config.lower_hermisphere_visibility = True

    shading = outputs["shading"].cpu().numpy()
    if shading.ndim == 1:
        H, W = outputs["rgb"].shape[:2]
        shading = shading.reshape(H, W, 1)
    sky_mask = datamanager.train_dataset[view_idx]["mask"][:, :, 3:4].cpu().numpy()
    shading = shading * gain
    shading[sky_mask == 1] = 1
    return np.clip(shading[:, :, 0], 0, 1)


def render_shadow_map(model, datamanager, view_idx, device,
                      azimuth: float, elevation: float, sigmoid_scale: float):
    """Single-direction soft shadow via the model's shadow-map render path."""
    import numpy as np

    model.render_shadow_map_flag = True
    model.shadow_map_azimuth.value = azimuth
    model.shadow_map_elevation.value = elevation
    model.shadow_map_threshold.value = float(model.visibility_threshold)
    model.shadow_map_sigmoid_scale.value = sigmoid_scale
    model.config.lower_hermisphere_visibility = False
    try:
        outputs = render_view(model, datamanager.train_dataset.cameras, view_idx, device)
    finally:
        model.render_shadow_map_flag = False
        model.config.lower_hermisphere_visibility = True

    shadow_map = outputs["shadow_map"].cpu().numpy()
    sky_mask = datamanager.train_dataset[view_idx]["mask"][:, :, 3:4].cpu().numpy()
    shadow_map[sky_mask == 1] = 0
    return np.clip(shadow_map[:, :, 0], 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "ao_and_shadow")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--view", type=int, default=None,
                        help="Train view index (default: per-scene notebook view)")
    parser.add_argument("--azimuth", type=float, default=45.0,
                        help="Shadow light azimuth in degrees (notebook: 45)")
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Shadow light elevation in degrees (notebook: 20)")
    parser.add_argument("--sigmoid-scale", type=float, default=10.0,
                        help="Shadow sigmoid sharpness (notebook: 10)")
    parser.add_argument("--ao-gain", type=float, default=2.5,
                        help="AO display gain (notebook scaled by ~2.5-3)")
    args = parser.parse_args()
    seed_all(args.seed)

    _, pipeline, _, _ = load_model(args.scene, device=args.device, step=args.step)
    datamanager = pipeline.datamanager
    model = pipeline.model
    model.eval()
    assert model.visibility_field is not None, "Run has no DDF visibility field"

    view_idx = args.view if args.view is not None else SCENE_DEFAULT_VIEWS.get(args.scene, 0)

    ao = render_ambient_occlusion(model, datamanager, view_idx, args.device, args.ao_gain)
    shadow = render_shadow_map(model, datamanager, view_idx, args.device,
                               args.azimuth, args.elevation, args.sigmoid_scale)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(ao, cmap="gray", vmin=0, vmax=1)
    axs[1].imshow(shadow, cmap="gray", vmin=0, vmax=1)
    for ax in axs:
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
