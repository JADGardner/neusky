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
    parser.add_argument("--scene", default=None,
                        help="Read learned epsilon/eta from this scene's latest "
                             "checkpoint instead of the config defaults")
    args = parser.parse_args()

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

    x = np.linspace(epsilon - 1.0, epsilon + 1.0, 1000)
    # visibility = 1 - sigmoid(eta * (x - epsilon)); x = ||s-x|| - f_DDF(s,-d)
    visibility = 1.0 - 1.0 / (1.0 + np.exp(-eta * (x - epsilon)))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(x, visibility, color="tab:blue", linewidth=2.0,
            label=rf"$\eta = {eta:g}$")
    ax.axvline(epsilon, color="tab:red", linestyle="--", linewidth=1.0)
    ax.annotate(rf"$\epsilon = {epsilon:g}$", xy=(epsilon, 0.5),
                xytext=(epsilon + 0.12, 0.62), color="tab:red")
    ax.set_xlabel(r"$\|\mathbf{s}-\mathbf{x}\| - f_\mathrm{DDF}(\mathbf{s},-\mathbf{d})$")
    ax.set_ylabel(r"$V(\mathbf{x},\mathbf{d})$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
