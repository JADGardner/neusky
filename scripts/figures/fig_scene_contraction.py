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

import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman", "Times New Roman", "Times",
        "Liberation Serif", "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

import matplotlib.pyplot as plt
import numpy as np

from _common import add_common_args, save_figure


def make_frustums(bins, origins, directions, pixel_area: float):
    from nerfstudio.cameras.rays import Frustums

    return Frustums(
        origins=origins, directions=directions,
        starts=bins[:, :-1, :], ends=bins[:, 1:, :], pixel_area=pixel_area,
    )


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

    # Two vertical rays starting inside the unit sphere (matching the
    # original figure's layout: rays rise from within r=1).
    origins = torch.Tensor([[-0.18, -0.30, 0.0], [0.18, -0.10, 0.0]]).unsqueeze(1)
    directions = torch.tensor([[0.0, 1.0, 0.0]]).unsqueeze(0).repeat(2, 1, 1).float()

    bins_uniform = torch.linspace(0.1, 3, num_samples + 1)[None, ..., None]
    frustums_one = make_frustums(bins_uniform, origins, directions, pixel_area)

    linear_bins = torch.linspace(0, 1, num_samples // 2 + 1)
    quadratic_bins = 1 + (torch.linspace(0, args.far, num_samples // 2 + 1) ** 2)
    bins = torch.cat((linear_bins, quadratic_bins[1:]), dim=0)[None, ..., None]
    frustums_two = make_frustums(bins, origins, directions, pixel_area)

    contraction = SceneContraction(order=float("inf") if args.order == "Linf" else None)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    colours = ["red", "blue"]
    CX = {"left": -2.7, "right": 2.7}

    for cx in CX.values():
        ax.add_patch(plt.Circle((cx, 0), 2.0, color="0.93", zorder=0))
        ax.add_patch(plt.Circle((cx, 0), 1.0, color="0.82", zorder=1))

    # left: uncontracted straight rays, clipped at radius 2
    for i, frustum in enumerate(frustums_one):
        pos = frustum.get_positions().squeeze()
        pos = pos[pos.norm(dim=-1) < 2]
        ax.plot(pos[:, 0] + CX["left"], pos[:, 1], color=colours[i % 2],
                linewidth=1.8, zorder=3, solid_capstyle="round")

    # right: contracted rays converge at the radius-2 shell
    for i, frustum in enumerate(frustums_two):
        con = contraction(frustum.get_positions()).squeeze()
        ax.plot(con[:, 0] + CX["right"], con[:, 1], color=colours[i % 2],
                linewidth=1.8, zorder=3, solid_capstyle="round")

    # r = 2 annotation: central label, arrows out to the shell / convergence
    y2 = 1.95
    ax.annotate("", xy=(CX["left"] + 0.55, y2), xytext=(-0.62, y2),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    ax.annotate("", xy=(CX["right"] - 0.35, y2), xytext=(0.62, y2),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    ax.text(0, y2, r"$r = 2$", ha="center", va="center", fontsize=12)

    # r = 1 annotation
    y1 = 0.30
    x1l = CX["left"] + float(np.sqrt(1 - y1 ** 2))
    x1r = CX["right"] - float(np.sqrt(1 - y1 ** 2))
    ax.annotate("", xy=(x1l, y1), xytext=(-0.62, y1),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    ax.annotate("", xy=(x1r, y1), xytext=(0.62, y1),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"))
    ax.text(0, y1, r"$r = 1$", ha="center", va="center", fontsize=12)

    for name, cx in (("No contraction", CX["left"]), ("Contraction", CX["right"])):
        ax.text(cx, -2.45, name, ha="center", va="top", fontsize=12)

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-2.9, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
