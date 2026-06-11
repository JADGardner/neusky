"""Scene contraction visualisation (thesis Fig -> scene_contraction.png).

Two rays sampled (left) with uniform bins in [0.1, 2] in the uncontracted
scene, and (right) with linear bins in [0, 1] plus quadratic bins out to the
far plane, mapped through nerfstudio's SceneContraction — showing how the
unbounded background collapses into the radius-2 shell around the unit sphere.

Ported from publication/figures_and_tables.ipynb (the plotly scene-contraction
cell), re-rendered with matplotlib 3D so it runs headless without kaleido. The
contraction itself is nerfstudio's SceneContraction (the exact module NeuSky
uses; see NeuSkyFactoModelConfig.scene_contraction_order), not a re-implementation.

CPU-only; runnable while a training job owns the GPU:

    PYTHONPATH=. python scripts/figures/fig_scene_contraction.py
"""

import argparse
import os

# Diagram-only: keep CUDA fully out of the picture before torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib.pyplot as plt
import numpy as np

from _common import add_common_args, save_figure


def make_frustums(bins, origins, directions, pixel_area: float):
    from nerfstudio.cameras.rays import Frustums

    return Frustums(
        origins=origins, directions=directions,
        starts=bins[:, :-1, :], ends=bins[:, 1:, :], pixel_area=pixel_area,
    )


def draw_sphere(ax, radius: float, alpha: float = 0.08, n: int = 30):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="0.2", alpha=alpha, linewidth=0, shade=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "scene_contraction", needs_device=False)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--order", choices=["L2", "Linf"], default="L2",
                        help="Contraction norm (NeuSky config uses L2)")
    args = parser.parse_args()

    import torch
    from nerfstudio.field_components.spatial_distortions import SceneContraction

    num_samples = args.num_samples
    pixel_area = 0.12

    # Notebook constants: two rays inside the unit sphere pointing along +y.
    origins = torch.Tensor([[0.0, 0.0, 0.4], [0.0, 0.2, -0.4]]).unsqueeze(1)
    directions = torch.tensor([[0, 1, 0]]).unsqueeze(0).repeat(2, 1, 1).float()
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Left panel: uniform bins in [0.1, 2], positions kept where ||p|| < 2.
    bins_uniform = torch.linspace(0.1, 2, num_samples + 1)[None, ..., None]
    frustums_one = make_frustums(bins_uniform, origins, directions, pixel_area)

    # Right panel: linear bins in [0, 1] + quadratic bins out to far^2,
    # mapped through the scene contraction.
    linear_bins = torch.linspace(0, 1, num_samples // 2 + 1)
    quadratic_bins = 1 + (torch.linspace(0, args.far, num_samples // 2 + 1) ** 2)
    bins = torch.cat((linear_bins, quadratic_bins[1:]), dim=0)[None, ..., None]
    frustums_two = make_frustums(bins, origins, directions, pixel_area)

    contraction = SceneContraction(order=float("inf") if args.order == "Linf" else None)

    fig = plt.figure(figsize=(12, 6))
    colours = ["tab:red", "tab:blue"]

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for i, frustum in enumerate(frustums_one):
        positions = frustum.get_positions().squeeze()
        positions = positions[positions.norm(dim=-1) < 2]
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                 color=colours[i % 2], linewidth=4)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for i, frustum in enumerate(frustums_two):
        contracted = contraction(frustum.get_positions()).squeeze()
        ax2.plot(contracted[:, 0], contracted[:, 1], contracted[:, 2],
                 color=colours[i % 2], linewidth=4)

    for ax in (ax1, ax2):
        draw_sphere(ax, 1.0, alpha=0.10)
        draw_sphere(ax, 2.0, alpha=0.04)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        lim = 1.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        # Side-on view (rays run along +y, separated in z): camera near +x.
        ax.view_init(elev=18, azim=-15)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
