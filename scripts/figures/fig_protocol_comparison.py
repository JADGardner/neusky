"""Protocol comparison figure (thesis Fig: protocol_comparison, lk2).

Per session-row: the GT compare photograph, NeRF-OSR rendered under its
OWN evaluation protocol as shipped (raw per-session SH envmaps, no
alignment rotation - the released code contains none), NeRF-OSR under
OUR holdout protocol (SH fit on the held-out view), and NeuSky under our
protocol. Demonstrates why the thesis evaluates by fitting illumination
through the model rather than trusting unaligned captured envmaps.

Two stages:

    # 1. GPU: fit + render NeuSky's compare views (saves PNGs)
    PYTHONPATH=. python scripts/figures/fig_protocol_comparison.py render-ours

    # 2. CPU: assemble the grid from the four sources
    PYTHONPATH=. python scripts/figures/fig_protocol_comparison.py assemble \
        --nerfosr-theirs <dir> --nerfosr-ours <dir>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (SESSION_HOLDOUT_INDICES, load_model, save_figure, seed_all)

OUR_RENDERS = Path(__file__).resolve().parents[2] / "publication" / "figures" / "protocol_renders" / "neusky"


def render_ours(args):
    import torch
    from reni.utils.colourspace import linear_to_sRGB  # noqa: F401  (parity import)

    def hook(config):
        config.pipeline.model.eval_latent_optimise_method = "nerf_osr_holdout"
        config.pipeline.datamanager.dataparser.session_holdout_indices = \
            SESSION_HOLDOUT_INDICES["lk2"]
        # hardened_v2_noprior fit recipe (the thesis scoring recipe)
        config.pipeline.model.eval_latent_prior_weight = 0.0
        config.pipeline.model.eval_sky_loss_unclamped = True

    _, pipeline, _, _ = load_model("lk2", device=args.device, config_hook=hook)
    model = pipeline.model
    datamanager = pipeline.datamanager
    model.eval()

    print("[fit] optimising eval latents on holdout views (600 steps/session)")
    model.fit_latent_codes_for_eval(datamanager, global_step=0)

    OUR_RENDERS.mkdir(parents=True, exist_ok=True)
    filenames = [Path(str(f)).stem for f in
                 datamanager.eval_dataset._dataparser_outputs.image_filenames]
    wanted = None
    if args.match_dir is not None:
        wanted = {q.stem.replace("_rgb", "") for q in Path(args.match_dir).iterdir()
                  if q.suffix.lower() in (".png", ".jpg")}
        print(f"[render] restricting to {len(wanted)} stems from {args.match_dir}")
    n = len(datamanager.eval_dataset)
    for i in range(n):
        if wanted is not None and not any(w in filenames[i] or filenames[i] in w
                                          for w in wanted):
            continue
        idx, ray_bundle, batch = datamanager.next_eval_image(i)
        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle)
        rgb = outputs["rgb"].cpu().numpy().clip(0, 1)
        out = OUR_RENDERS / f"{filenames[i]}.png"
        plt.imsave(out, rgb)
        print(f"[render] {out}")


def _load(path):
    img = plt.imread(str(path))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[..., :3]


def assemble(args):
    seed_all(args.seed if hasattr(args, "seed") else 42)
    gt_dir = Path("/home/james/data/NeRF-OSR/Data/lk2/final/test/rgb")
    ours_dir = OUR_RENDERS
    theirs_dir = Path(args.nerfosr_theirs)
    ours_osr_dir = Path(args.nerfosr_ours)

    stems = sorted(p.stem for p in ours_dir.glob("*.png"))
    if not stems:
        raise SystemExit(f"no NeuSky renders in {ours_dir} - run render-ours first")

    cols = ("Ground Truth", "NeRF-OSR (envmap protocol)",
            "NeRF-OSR (our protocol)", "NeuSky (our protocol)")
    rows = []
    for stem in stems:
        def find(d, exts=(".png", ".jpg", ".JPG")):
            for e in exts:
                cand = d / f"{stem}{e}"
                if cand.exists():
                    return cand
            return None
        paths = [find(gt_dir), find(theirs_dir), find(ours_osr_dir),
                 ours_dir / f"{stem}.png"]
        if any(p is None for p in paths):
            print(f"[skip] {stem}: missing {[c for c, p in zip(cols, paths) if p is None]}")
            continue
        rows.append((stem, [_load(p) for p in paths]))

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 2.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (stem, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=12, fontfamily="serif")
    plt.tight_layout(pad=0.4)
    save_figure(fig, args.output, svg=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render-ours")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--match-dir", type=Path, default=None,
                   help="Render only eval views whose stems appear in this "
                        "dir (the scored compare views)")
    a = sub.add_parser("assemble")
    a.add_argument("--nerfosr-theirs", required=True)
    a.add_argument("--nerfosr-ours", required=True)
    a.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parents[2] / "publication"
                   / "figures" / "protocol_comparison_thesis")
    args = parser.parse_args()
    if args.cmd == "render-ours":
        render_ours(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
