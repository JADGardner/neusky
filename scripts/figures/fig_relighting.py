"""Relighting under novel illumination (thesis Fig: relighting_examples ->
further_relighting_examples.png).

Renders a row of views of the scene:

- the original (fitted) illumination;
- the same view relit by NOVEL RENI++ latents taken from the PRIOR's
  training set (real HDRI latents, not scene fits) — pick them with
  --reni-latents;
- optionally, other scene sessions' fitted latents (--illum-indices) and
  fixed EXR environment maps (--envmaps).

Every panel gets its illumination's equirectangular envmap composited
into the top-right corner (disable with --no-insets; --envmap-strip still
gives the separate second row instead).

Checkpoint-dependent (GPU). Pin the run + prior:

    NEUSKY_RUNS="lk2=outputs/lk2_rngfix_w32cyc_hash21/neusky/2026-07-05_082759" \\
    NEUSKY_RENI_PRIOR=/workspace/phd/code/ns_reni/outputs/_neusky_prior_two_bracket_w3_2cyc \\
    PYTHONPATH=. python scripts/figures/fig_relighting.py --scene lk2 \\
        --reni-latents 10 25 42
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


def ensure_dam_wall(path: Path):
    """Download the dam_wall HDRI if missing (notebook recipe)."""
    if path.name == "dam_wall_1k.exr" and not path.exists():
        print(f"[download] {DAM_WALL_URL} -> {path}")
        os.system(f"wget -q {DAM_WALL_URL} -O {path}")
    return path


def default_prior_ckpt() -> Path | None:
    prior = os.environ.get("NEUSKY_RENI_PRIOR")
    if not prior:
        return None
    ckpts = sorted((Path(prior) / "nerfstudio_models").glob("step-*.ckpt"))
    return ckpts[-1] if ckpts else None


def load_prior_train_latents(ckpt_path: Path):
    """RENI++ prior training-set latents (real HDRI latents) from a
    nerfstudio checkpoint: pipeline key ``_model.field.train_mu``."""
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["pipeline"]
    for key in ("_model.field.train_mu", "_model.field.mu"):
        if key in state:
            print(f"[prior] {ckpt_path.name}: {key} {tuple(state[key].shape)}")
            return state[key]
    raise KeyError(f"no train_mu key in {ckpt_path} (keys like: "
                   f"{[k for k in state if 'field' in k][:5]})")


def add_envmap_inset(render, envmap, frac=0.30, border=2):
    """Composite the (equirect, 2:1) envmap into the render's top-right corner."""
    import numpy as np
    from PIL import Image

    h, w = render.shape[:2]
    iw = max(int(w * frac), 8)
    ih = max(iw // 2, 4)
    env = Image.fromarray(
        (np.clip(envmap, 0, 1) * 255).astype("uint8")).resize((iw, ih))
    out = np.clip(render, 0, 1).copy()
    y0, x0 = border, w - iw - border
    # white frame
    out[y0 - border:y0 + ih + border, x0 - border:x0 + iw + border] = 1.0
    out[y0:y0 + ih, x0:x0 + iw] = np.asarray(env, dtype="float32") / 255.0
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "further_relighting_examples")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--view", type=int, default=None,
                        help="Train view index (default: per-scene notebook view)")
    parser.add_argument("--reni-latents", type=int, nargs="*", default=[10, 25, 42],
                        help="RENI++ PRIOR training-latent indices to relight "
                             "with (novel real-HDRI illuminations)")
    parser.add_argument("--reni-rotations", type=float, nargs="*", default=[],
                        help="Yaw rotation (degrees, about the vertical axis) "
                             "applied to each --reni-latents illumination; "
                             "missing entries default to 0")
    parser.add_argument("--reni-exposures", type=float, nargs="*", default=[],
                        help="Exposure multiplier for each --reni-latents "
                             "render, relative to the view's fitted exposure "
                             "(1.0 = as fitted; applied in exp-scale space); "
                             "missing entries default to 1.0")
    parser.add_argument("--reni-prior-ckpt", type=Path, default=None,
                        help="RENI++ prior checkpoint holding field.train_mu "
                             "(default: from $NEUSKY_RENI_PRIOR)")
    parser.add_argument("--illum-indices", type=int, nargs="*", default=[],
                        help="Other train image indices whose fitted latents "
                             "relight the view")
    parser.add_argument("--envmaps", type=Path, nargs="*", default=[],
                        help="EXR environment maps to relight with")
    parser.add_argument("--inset-frac", type=float, default=0.30,
                        help="Corner-envmap width as a fraction of the render")
    parser.add_argument("--no-insets", action="store_true",
                        help="Disable the corner envmap insets")
    parser.add_argument("--envmap-strip", action="store_true",
                        help="Add a second row showing each illumination envmap")
    args = parser.parse_args()
    seed_all(args.seed)

    import numpy as np
    import torch

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

    # Novel RENI++ prior latents: write each into the latent bank at the
    # rendered view's own slot (so its per-image exposure scale applies),
    # render, then restore the fitted latent.
    if args.reni_latents:
        ckpt = args.reni_prior_ckpt or default_prior_ckpt()
        if ckpt is None:
            raise SystemExit("--reni-latents given but no prior checkpoint "
                             "(set --reni-prior-ckpt or $NEUSKY_RENI_PRIOR)")
        prior_mu = load_prior_train_latents(Path(ckpt))
        from reni.utils.utils import rot_y
        rotations_deg = list(args.reni_rotations)
        rotations_deg += [0.0] * (len(args.reni_latents) - len(rotations_deg))
        exposures = list(args.reni_exposures)
        exposures += [1.0] * (len(args.reni_latents) - len(exposures))
        bank = model.train_illumination_latents
        scale_bank = model.train_scale
        saved = bank.data[view_idx].clone()
        saved_scale = scale_bank.data[view_idx].clone()
        try:
            for z_idx, rot_deg, exposure in zip(args.reni_latents, rotations_deg,
                                                exposures):
                z = prior_mu[z_idx].to(bank.device, bank.dtype)
                if z.shape != saved.shape:
                    raise SystemExit(f"latent {z_idx} shape {tuple(z.shape)} != "
                                     f"bank slot {tuple(saved.shape)}")
                rotation = None
                if rot_deg:
                    rotation = rot_y(torch.deg2rad(
                        torch.tensor(rot_deg, dtype=torch.float32))).to(args.device)
                with torch.no_grad():
                    bank.data[view_idx] = z
                    # exp(scale) semantics: multiplier m -> scale + ln(m)
                    scale_bank.data[view_idx] = saved_scale + float(np.log(exposure))
                outputs = render_view(model, cameras, view_idx, args.device,
                                      rotation=rotation)
                renders.append(outputs["rgb"].cpu().numpy())
                envmaps.append(render_envmap(model, view_idx,
                                             rotation=rotation).numpy())
                label = f"RENI latent {z_idx}"
                if rot_deg:
                    label += f" ({rot_deg:g}°)"
                if exposure != 1.0:
                    label += f" ×{exposure:g}"
                labels.append(label)
        finally:
            with torch.no_grad():
                bank.data[view_idx] = saved
                scale_bank.data[view_idx] = saved_scale

    # Latent swaps: same camera, another session's fitted illumination.
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
            envmaps.append(load_envmap_preview(envmap_path))
            labels.append(envmap_path.stem)
        finally:
            restore_illumination(model, original)

    if not args.no_insets:
        renders = [add_envmap_inset(r, e, frac=args.inset_frac)
                   for r, e in zip(renders, envmaps)]

    # Per-panel dumps alongside the combined figure.
    panels_dir = Path(f"{args.output}_panels")
    panels_dir.mkdir(parents=True, exist_ok=True)
    for render, label in zip(renders, labels):
        panel_path = panels_dir / f"{label.replace(' ', '_')}.png"
        plt.imsave(panel_path, np.clip(render, 0, 1))
        print(f"[panel] {panel_path}")

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
