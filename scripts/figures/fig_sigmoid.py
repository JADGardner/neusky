"""Soft visibility threshold curve (thesis Fig: visibilitysigmoid -> sigmoid.pdf).

Plots the softened visibility test used by NeuSky's outside-in DDF visibility:

    V(x, d) = 1 - sigmoid(eta * (||s - x|| - f_DDF(s, -d) - epsilon))

i.e. the curve implemented in neusky/models/neusky_model.py
(NeuSkyFactoModel.compute_visibility):

    occlusion = torch.sigmoid(sigmoid_scale * (difference - threshold_distance))
    visibility = 1.0 - occlusion

epsilon (the threshold / learnable sigmoid bias) and eta (the sigmoid scale)
default to the model config's converged targets (target_min_bias /
target_max_scale in neusky/configs/neusky_config.py, parsed at runtime so the
figure tracks the config). `--scene lk2` instead reads the *learned* values
from the latest checkpoint (CPU torch.load of the state dict; no model build).

CPU-only; runnable while a training job owns the GPU:

    PYTHONPATH=. python scripts/figures/fig_sigmoid.py
"""

import argparse
import re

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

from _common import REPO_ROOT, add_common_args, latest_checkpoint_step, resolve_run_dir, save_figure


def config_defaults():
    """Parse target_min_bias / target_max_scale from the model config source.

    Regex parse (not import): importing neusky.configs hard-requires
    tinycudann/CUDA, which this diagram-only script must not touch.
    """
    src = (REPO_ROOT / "neusky" / "configs" / "neusky_config.py").read_text()
    epsilon = float(re.search(r'"target_min_bias":\s*([0-9.eE+-]+)', src).group(1))
    eta = float(re.search(r'"target_max_scale":\s*([0-9.eE+-]+)', src).group(1))
    return epsilon, eta


def checkpoint_values(scene: str):
    """Learned (visibility_threshold, sigmoid_scale) from the latest ckpt."""
    import torch

    run_dir = resolve_run_dir(scene)
    step = latest_checkpoint_step(run_dir)
    ckpt = torch.load(
        run_dir / "nerfstudio_models" / f"step-{step:09d}.ckpt",
        map_location="cpu", weights_only=False,
    )
    pipeline = ckpt["pipeline"]
    epsilon = float(pipeline["_model.visibility_threshold"])
    eta_key = "_model.sigmoid_scale"
    eta = float(pipeline[eta_key]) if eta_key in pipeline else None
    return epsilon, eta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "sigmoid", needs_device=False)
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Threshold epsilon (default: config target_min_bias)")
    parser.add_argument("--eta", type=float, default=None,
                        help="Sigmoid scale eta (default: config target_max_scale)")
    parser.add_argument("--config-values", action="store_true",
                        help="Use the config's converged epsilon/eta instead "
                             "of the schematic defaults")
    parser.add_argument("--scene", default=None,
                        help="Read learned epsilon/eta from this scene's latest "
                             "checkpoint instead of the config defaults")
    args = parser.parse_args()

    # schematic proportions matching the paper's original figure: the
    # curve's bend width is comparable to the 0-to-epsilon spacing. Pass
    # --config-values (or --scene/--epsilon/--eta) for the real parameters.
    epsilon, eta = (0.3, 13.0)
    if args.config_values:
        epsilon, eta = config_defaults()
    if args.scene:
        ckpt_eps, ckpt_eta = checkpoint_values(args.scene)
        epsilon = ckpt_eps
        eta = ckpt_eta if ckpt_eta is not None else eta
        print(f"[ckpt] {args.scene}: epsilon={epsilon:.4f}, eta={eta:.2f}")
    if args.epsilon is not None:
        epsilon = args.epsilon
    if args.eta is not None:
        eta = args.eta

    # x-range proportioned like the paper figure: 0 near the left quarter,
    # epsilon around the centre, transition fully resolved
    span = max(abs(epsilon), 4.0 / eta)
    x = np.linspace(-3.0 * span, 7.0 * span, 1000)
    # visibility = 1 - sigmoid(eta * (x - epsilon)); x = ||s-x|| - f_DDF(s,-d)
    visibility = 1.0 - 1.0 / (1.0 + np.exp(-eta * (x - epsilon)))

    # layout matches the paper's original sigmoid.pdf: wide-short boxed
    # axes, ticks only at {0, epsilon} and {0, 1}, no legend or annotations
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(x, visibility, color="#3A5FAD", linewidth=1.6)  # SphereJEPA blue
    ax.set_xlabel(r"GT $-$ Predicted", fontsize=10)
    ax.set_ylabel("Visibility", fontsize=10)
    ax.set_xticks([0.0, epsilon])
    ax.set_xticklabels(["$0$", r"$\epsilon$"], fontsize=11)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["$0$", "$1$"], fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(direction="in", length=3)
    fig.tight_layout()

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
