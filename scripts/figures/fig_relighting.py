"""Relighting under novel illumination (thesis Fig: relighting_examples ->
further_relighting_examples.png).

For each scene view, renders a row of:

- the original (fitted) illumination;
- the same view lit by *other training sessions'* RENI++ latents (the
  notebook's novel_illumination recipe: keep the camera, override the ray
  bundle's camera_indices with another session's illumination index);
- the same view lit by fixed environment maps (the notebook's dam_wall /
  point-light recipe: swap the illumination field for an
  EnvironmentMapField built from an EXR).

Each render also dumps its illumination envmap when --envmaps-strip is given.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_relighting.py --scene lk2
    PYTHONPATH=. python scripts/figures/fig_relighting.py --scene lk2 \
        --illum-indices 53 90 --envmaps publication/point_light.exr
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from _common import (REPO_ROOT, SCENE_DEFAULT_VIEWS, add_common_args, load_model,
                     load_envmap_preview, render_envmap, render_view,
                     restore_illumination, save_figure,
                     seed_all, swap_in_environment_map)

DAM_WALL_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/dam_wall_1k.exr"
DEFAULT_ENVMAPS = [
    REPO_ROOT / "publication" / "dam_wall_1k.exr",
    REPO_ROOT / "publication" / "point_light.exr",
]


def ensure_dam_wall(path: Path):
    """Download the dam_wall HDRI if missing (notebook recipe)."""
    if path.name == "dam_wall_1k.exr" and not path.exists():
        print(f"[download] {DAM_WALL_URL} -> {path}")
        os.system(f"wget -q {DAM_WALL_URL} -O {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "further_relighting_examples")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--view", type=int, default=None,
                        help="Train view index (default: per-scene notebook view)")
    parser.add_argument("--illum-indices", type=int, nargs="*", default=[53],
                        help="Other train image indices whose fitted latents "
                             "relight the view (notebook used 53)")
    parser.add_argument("--envmaps", type=Path, nargs="*", default=DEFAULT_ENVMAPS,
                        help="EXR environment maps to relight with")
    parser.add_argument("--envmap-strip", action="store_true",
                        help="Add a second row showing each illumination envmap")
    args = parser.parse_args()
    seed_all(args.seed)

    import numpy as np

    _, pipeline, _, _ = load_model(args.scene, device=args.device, step=args.step)
    datamanager = pipeline.datamanager
    model = pipeline.model
    model.eval()
    cameras = datamanager.train_dataset.cameras

    view_idx = args.view if args.view is not None else SCENE_DEFAULT_VIEWS.get(args.scene, 0)

    renders, envmaps, labels = [], [], []

    # Original fitted illumination.
    outputs = render_view(model, cameras, view_idx, args.device)
    renders.append(outputs["rgb"].cpu().numpy())
    envmaps.append(render_envmap(model, view_idx).numpy())
    labels.append("fitted")

    # Latent swaps: same camera, another session's illumination latent.
    for illum_idx in args.illum_indices:
        outputs = render_view(model, cameras, view_idx, args.device,
                              illumination_idx=illum_idx)
        renders.append(outputs["rgb"].cpu().numpy())
        envmaps.append(render_envmap(model, illum_idx).numpy())
        labels.append(f"latent {illum_idx}")

    # Fixed environment maps.
    for envmap_path in args.envmaps:
        envmap_path = ensure_dam_wall(Path(envmap_path))
        if not envmap_path.exists():
            print(f"[skip] envmap not found: {envmap_path}")
            continue
        original = swap_in_environment_map(model, envmap_path, datamanager)
        try:
            outputs = render_view(model, cameras, view_idx, args.device)
            renders.append(outputs["rgb"].cpu().numpy())
            # Decoding through the illumination field would index the full
            # envmap texture per direction sample (OOM); the preview is just
            # the HDRI itself.
            envmaps.append(load_envmap_preview(envmap_path))
            labels.append(envmap_path.stem)
        finally:
            restore_illumination(model, original)

    n = len(renders)
    n_rows = 2 if args.envmap_strip else 1
    fig, axs = plt.subplots(n_rows, n, figsize=(4 * n, 3.2 * n_rows), squeeze=False)
    for c, (render, envmap, label) in enumerate(zip(renders, envmaps, labels)):
        axs[0, c].imshow(np.clip(render, 0, 1))
        axs[0, c].axis("off")
        axs[0, c].set_title(label, fontsize=8)
        if args.envmap_strip:
            axs[1, c].imshow(np.clip(envmap, 0, 1))
            axs[1, c].axis("off")
    plt.tight_layout(pad=0.3)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
