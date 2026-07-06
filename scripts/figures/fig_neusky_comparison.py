"""Intrinsics comparison figure (thesis Fig: neusky_comparison).

Two blocks, one per NeRF-OSR site, each Albedo + Normal rows x 3 methods:

  top:    lk2  - NeRF-OSR | FEGR     | Ours
  bottom: lwp  - NeRF-OSR | SOL-NeRF | Ours

"Ours" renders live from the registered best NeuSky checkpoints (albedo +
normal_vis outputs). NeRF-OSR columns come from renders of the official
checkpoints (thirdparty/nerf-osr-official.docker machinery). FEGR and
SOL-NeRF released no code or weights, so their cells are pulled from the
papers (scripts/figures/assets/*_from_paper.*) - flagged in the caption.

Labels are baked (previously TikZ overlay nodes).

    NEUSKY_RENI_PRIOR=... PYTHONPATH=. python scripts/figures/fig_neusky_comparison.py \
        --nerfosr-lk2-albedo <path> --nerfosr-lk2-normal <path> \
        --nerfosr-lwp-albedo <path> --nerfosr-lwp-normal <path>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (SCENE_DEFAULT_VIEWS, add_common_args, load_model,
                     render_view, save_figure, seed_all)

ASSETS = Path(__file__).resolve().parent / "assets"

PAPER_ASSETS = {
    ("lk2", "albedo"): ASSETS / "fegr_site1_lk2_albedo_from_paper.png",
    ("lk2", "normal"): ASSETS / "fegr_site1_lk2_normals_from_paper.jpg",
    ("lwp", "albedo"): ASSETS / "sol_nerf_site3_lwp_albedo_from_paper.jpg",
    ("lwp", "normal"): ASSETS / "sol_nerf_site3_lwp_normals_from_paper.jpg",
}
MIDDLE_METHOD = {"lk2": "FEGR", "lwp": "SOL-NeRF"}

# Paper normal maps use different axis conventions; remap each into OUR
# convention (neusky normal_vis, world-space (n+1)/2). Signed permutation
# rows = which source axis (+/-x, +/-y, +/-z) lands in each of ours.
# Identity until calibrated against our render (--debug-normals emits a
# side-by-side hue sheet for choosing these).
NORMAL_REMAPS = {
    # FEGR renders world y-up, z-toward-camera; ours is z-up, camera ~ -y.
    "lk2": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    # SOL-NeRF's convention already matches ours closely.
    "lwp": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}
# The official NeRF-OSR renders use the same y-up convention as FEGR.
NERFOSR_REMAP = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
# Sky masks for the exact figure views: the cityscapes semantic masks
# (sky class = RGB 70,130,180). The plain mask/ dir is a validity mask.
NERFOSR_VIEW_MASKS = {
    "lk2": Path("/home/james/data/NeRF-OSR/Data/lk2/final/train/cityscapes_mask/07-04_17_30_DSC_0049.png"),
    "lwp": Path("/home/james/data/NeRF-OSR/Data/lwp/final/train/cityscapes_mask/26-04_17_50_DSC_2355.png"),
}
CITYSCAPES_SKY = np.array([70, 130, 180]) / 255.0


def remap_normal_image(img, M):
    """rgb -> n -> M @ n -> rgb, preserving mask/invalid regions."""
    n = img * 2.0 - 1.0
    n = n @ np.asarray(M, dtype=np.float32).T
    return ((n + 1.0) / 2.0).clip(0, 1)


def sky_mask_from_albedo(albedo, lum_thresh=0.90):
    """Estimate a sky mask from a paper albedo crop: near-white regions
    connected to the top edge (FEGR crops show blown-out sky)."""
    lum = albedo.mean(-1)
    bright = lum > lum_thresh
    mask = np.zeros_like(bright)
    # simple flood from the top rows through bright pixels
    frontier = bright[0].copy()
    mask[0] = frontier
    for r in range(1, bright.shape[0]):
        frontier = bright[r] & (
            mask[r - 1]
            | np.roll(mask[r - 1], 1)
            | np.roll(mask[r - 1], -1))
        # a couple of lateral passes to spread along the row
        for _ in range(3):
            frontier = frontier | (bright[r] & (
                np.roll(frontier, 1) | np.roll(frontier, -1)))
        mask[r] = frontier
        if not frontier.any():
            break
    return mask


def apply_sky_mask(img, mask, fill=1.0):
    out = img.copy()
    out[mask] = fill
    return out


def _load_image(path: Path) -> np.ndarray:
    img = plt.imread(str(path))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[..., :3]


def _render_ours(scene: str, view: int, device: str):
    _, pipeline, _, _ = load_model(scene, device=device)
    model = pipeline.model
    model.eval()
    cameras = pipeline.datamanager.train_dataset.cameras
    outputs = render_view(model, cameras, view, device)
    albedo = outputs["albedo"].cpu().numpy().clip(0, 1)
    normal = outputs["normal_vis"].cpu().numpy().clip(0, 1)
    # white sky in the normal cell (match the masked baseline cells)
    if "accumulation" in outputs:
        acc = outputs["accumulation"].cpu().numpy()
        normal = normal.copy()
        normal[acc[..., 0] < 0.5] = 1.0
    del pipeline, model
    import torch
    torch.cuda.empty_cache()
    return albedo, normal


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "neusky_comparison")
    # defaults match the original comparisons_full figure / paper crops:
    # lk2 train 0 = 07-04_17_30_DSC_0049.jpg, lwp train 234 = 26-04_17_50_DSC_2355.jpg
    parser.add_argument("--lk2-view", type=int, default=0)
    parser.add_argument("--lwp-view", type=int, default=234)
    for scene in ("lk2", "lwp"):
        for kind in ("albedo", "normal"):
            parser.add_argument(f"--nerfosr-{scene}-{kind}", type=Path,
                                default=None,
                                help=f"NeRF-OSR {kind} render for {scene} "
                                     "(official-checkpoint render)")
    parser.add_argument("--label-fontsize", type=float, default=11.0)
    args = parser.parse_args()
    seed_all(args.seed)

    views = {"lk2": args.lk2_view, "lwp": args.lwp_view}
    grid = {}   # (scene, kind, col) -> image
    for scene in ("lk2", "lwp"):
        albedo, normal = _render_ours(scene, views[scene], args.device)
        grid[(scene, "albedo", 2)] = albedo
        grid[(scene, "normal", 2)] = normal
        paper_albedo = _load_image(PAPER_ASSETS[(scene, "albedo")])
        paper_normal = remap_normal_image(
            _load_image(PAPER_ASSETS[(scene, "normal")]), NORMAL_REMAPS[scene])
        if scene == "lk2":
            # FEGR crops include blown-out sky; mask it (white) in both cells
            sky = sky_mask_from_albedo(paper_albedo)
            paper_albedo = apply_sky_mask(paper_albedo, sky)
            paper_normal = apply_sky_mask(paper_normal, sky)
        grid[(scene, "albedo", 1)] = paper_albedo
        grid[(scene, "normal", 1)] = paper_normal
        view_mask = None
        mask_path = NERFOSR_VIEW_MASKS.get(scene)
        if mask_path is not None and mask_path.exists():
            seg = _load_image(mask_path)
            view_mask = (np.abs(seg - CITYSCAPES_SKY) < 0.02).all(-1)  # True = sky
        for kind in ("albedo", "normal"):
            p = getattr(args, f"nerfosr_{scene}_{kind}")
            if p is None or not Path(p).exists():
                print(f"[warn] NeRF-OSR {scene} {kind} render missing - grey placeholder")
                grid[(scene, kind, 0)] = np.full((400, 600, 3), 0.85, dtype=np.float32)
                continue
            img = _load_image(Path(p))
            if kind == "normal":
                img = remap_normal_image(img, NERFOSR_REMAP)
            if view_mask is not None:
                m = view_mask
                if m.shape != img.shape[:2]:
                    from PIL import Image as _Im
                    m = np.asarray(_Im.fromarray(
                        (m * 255).astype("uint8")).resize(
                        (img.shape[1], img.shape[0]))) > 127
                img = apply_sky_mask(img, m)
            grid[(scene, kind, 0)] = img

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 11.6),
                             gridspec_kw={"hspace": 0.14})
    scenes_rows = [("lk2", "albedo", 0), ("lk2", "normal", 1),
                   ("lwp", "albedo", 2), ("lwp", "normal", 3)]
    for scene, kind, row in scenes_rows:
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(grid[(scene, kind, col)])
            ax.axis("off")

    plt.tight_layout(rect=(0.02, 0, 1, 1), h_pad=2.6)

    # column headers per block + rotated row labels (formerly TikZ)
    fs = args.label_fontsize
    for block_row, scene in ((0, "lk2"), (2, "lwp")):
        names = ("NeRF-OSR", MIDDLE_METHOD[scene], "Ours")
        # one shared header height per block (cells have differing aspects)
        top = max(axes[block_row, c].get_position().y1 for c in range(3))
        for col, name in enumerate(names):
            bb = axes[block_row, col].get_position()
            fig.text((bb.x0 + bb.x1) / 2, top + 0.008, name,
                     ha="center", va="bottom", fontsize=fs, fontfamily="serif")
    for scene, kind, row in scenes_rows:
        bb = axes[row, 0].get_position()
        fig.text(bb.x0 - 0.012, (bb.y0 + bb.y1) / 2, kind.capitalize(),
                 ha="right", va="center", rotation=90, fontsize=fs,
                 fontfamily="serif")

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
