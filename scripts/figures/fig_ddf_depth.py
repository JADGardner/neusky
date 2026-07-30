"""DDF predicted depth vs SDF pseudo-GT (thesis Fig: ddf_depth ->
ddf_sdf_comparison.png).

Places cameras on the DDF sphere looking at the origin (the construction used
by DDFDataset._generate_images and notebooks/ddf.ipynb). For each viewpoint:

- top row: the spherical DDF's expected termination distance in a single
  forward pass (model.visibility_field.get_outputs_for_camera_ray_bundle);
- bottom row: the pseudo ground truth rendered from the scene representation
  (model.generate_ddf_ground_truth), masked by its accumulation.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_ddf_depth.py --scene lk2
    PYTHONPATH=. python scripts/figures/fig_ddf_depth.py --scene lk2 --azimuths 200 245 290
"""

import argparse
import math

import matplotlib.pyplot as plt

from _common import (add_common_args, load_model, save_figure, seed_all,
                     sphere_camera_ray_bundle)


def render_pair(model, position, device, mask_threshold: float, chunk: int = 4096):
    """(ddf_depth, sdf_depth, mask) [H, W] for a camera at `position`."""
    import torch

    ray_bundle = sphere_camera_ray_bundle(position, device)
    H, W = ray_bundle.origins.shape[:2]

    with torch.no_grad():
        ddf_outputs = model.visibility_field.get_outputs_for_camera_ray_bundle(
            ray_bundle, neusky=None, show_progress=True
        )
        ddf_depth = ddf_outputs["expected_termination_dist"].reshape(H, W)

        # SDF pseudo-GT, chunked exactly as DDFDataset._generate_images.
        num_rays = len(ray_bundle)
        dists, masks = [], []
        for i in range(0, num_rays, chunk):
            sliced = ray_bundle.get_row_major_sliced_ray_bundle(i, i + chunk)
            data = model.generate_ddf_ground_truth(
                sliced, mask_threshold=mask_threshold)
            dists.append(data["termination_dist"].cpu())
            masks.append(data["mask"].cpu())
        sdf_depth = torch.cat(dists).reshape(H, W)
        mask = torch.cat(masks).reshape(H, W)
        sdf_depth = torch.clamp(sdf_depth, max=2 * model.visibility_field.ddf_radius)

    return ddf_depth.cpu() * mask, sdf_depth * mask, mask


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "ddf_sdf_comparison")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--labels", action="store_true",
                        help="Bake the row labels (formerly TikZ)")
    parser.add_argument("--azimuths", type=float, nargs="+", default=[200.0, 245.0, 290.0],
                        help="Camera azimuths (deg) on the DDF sphere")
    parser.add_argument("--elevation", type=float, default=30.0,
                        help="Camera elevation (deg) above the horizon")
    parser.add_argument("--mask-threshold", type=float, default=0.7,
                        help="Accumulation threshold for the SDF pseudo-GT mask")
    args = parser.parse_args()
    seed_all(args.seed)

    _, pipeline, _, _ = load_model(args.scene, device=args.device, step=args.step)
    model = pipeline.model
    model.eval()
    assert model.visibility_field is not None, "Run has no DDF visibility field"
    radius = float(model.visibility_field.ddf_radius)

    pairs = []
    for az_deg in args.azimuths:
        az = math.radians(az_deg)
        el = math.radians(args.elevation)
        position = [
            radius * math.cos(az) * math.cos(el),
            radius * math.sin(az) * math.cos(el),
            radius * math.sin(el),
        ]
        pairs.append(render_pair(model, position, args.device, args.mask_threshold))

    vmax = max(float(p[1].max()) for p in pairs)
    n = len(pairs)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 5.2), squeeze=False)
    for c, (ddf_depth, sdf_depth, _) in enumerate(pairs):
        axs[0, c].imshow(ddf_depth.numpy(), cmap="viridis", vmin=0, vmax=vmax)
        axs[1, c].imshow(sdf_depth.numpy(), cmap="viridis", vmin=0, vmax=vmax)
        axs[0, c].axis("off")
        axs[1, c].axis("off")
    plt.tight_layout(pad=0.3)
    if getattr(args, "labels", False):
        for row, text in ((0, "DDF Prediction"), (1, "SDF Target")):
            bb = axs[row, 0].get_position()
            fig.text(bb.x0 - 0.008, (bb.y0 + bb.y1) / 2, text, ha="right",
                     va="center", rotation=90, fontsize=11, fontfamily="serif")
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
