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
GT_IMAGES = {
    "lk2": Path("/home/james/data/NeRF-OSR/Data/lk2/final/train/rgb/09-04_13_00_DSC_0309.jpg"),
    "lwp": Path("/home/james/data/NeRF-OSR/Data/lwp/final/train/rgb/26-04_17_50_DSC_2355.jpg"),
}

# Paper normal maps use different axis conventions; remap each into OUR
# convention (neusky normal_vis, world-space (n+1)/2). Signed permutation
# rows = which source axis (+/-x, +/-y, +/-z) lands in each of ours.
# Identity until calibrated against our render (--debug-normals emits a
# side-by-side hue sheet for choosing these).
# Display convention: the FEGR paper frame (world y-up: ground green,
# left-facing red, right-facing cyan), per James's reference. FEGR's frame
# is opposite-handed to ours, so the conversion includes an x-flip
# (det = -1); sources already in the FEGR frame pass through.
OURS_TO_FEGR = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
NORMAL_REMAPS = {
    "lk2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # FEGR: native target frame
    # SOL-NeRF: solved by Procrustes against our lwp normals at the matched
    # view (outputs/solve_solnerf_frame.py; 660k correspondences, det=-1:
    # an improper convention change, heading ~31 deg + slight tilt).
    "lwp": [[0.8532, 0.1127, -0.5093],
            [-0.1665, 0.9841, -0.0612],
            [0.4943, 0.1371, 0.8584]],
}
# The official NeRF-OSR renders are already y-up like FEGR.
NERFOSR_REMAP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# Sky masks for the exact figure views: the cityscapes semantic masks
# (sky class = RGB 70,130,180). The plain mask/ dir is a validity mask.
NERFOSR_VIEW_MASKS = {
    "lk2": Path("/home/james/data/NeRF-OSR/Data/lk2/final/train/cityscapes_mask/09-04_13_00_DSC_0309.png"),
    "lwp": Path("/home/james/data/NeRF-OSR/Data/lwp/final/train/cityscapes_mask/26-04_17_50_DSC_2355.png"),
}
CITYSCAPES_SKY = np.array([70, 130, 180]) / 255.0
# Crop of the full view corresponding to each paper crop (normalised
# x0, y0, x1, y1 from the template match); applied to our renders and the
# NeRF-OSR renders so every cell shares the paper framing.
CROP_BOXES = {
    "lk2": (0.0078, 0.0118, 0.9922, 0.9905),
    "lwp": (0.0078, 0.0047, 0.9922, 0.8246),
}


def crop_to_box(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


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
    normal = remap_normal_image(normal, OURS_TO_FEGR)
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
    # views matched to the FEGR/SOL-NeRF paper crops by edge-based template
    # search (outputs/match_paper_views.py): lk2 train 81 =
    # 09-04_13_00_DSC_0309.jpg, lwp train 234 = 26-04_17_50_DSC_2355.jpg;
    # both near-full-frame with the CROP_BOXES trims below.
    parser.add_argument("--lk2-view", type=int, default=81)
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
        grid[(scene, "albedo", 2)] = crop_to_box(albedo, CROP_BOXES[scene])
        grid[(scene, "normal", 2)] = crop_to_box(normal, CROP_BOXES[scene])
        paper_albedo = _load_image(PAPER_ASSETS[(scene, "albedo")])
        paper_normal = remap_normal_image(
            _load_image(PAPER_ASSETS[(scene, "normal")]), NORMAL_REMAPS[scene])
        # both paper crops include sky; mask it (white) in both cells.
        # FEGR's sky is blown-out white, SOL-NeRF's a pale blue-grey.
        sky = sky_mask_from_albedo(paper_albedo,
                                   lum_thresh=0.90 if scene == "lk2" else 0.78)
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
            grid[(scene, kind, 0)] = crop_to_box(img, CROP_BOXES[scene])

    fig = plt.figure(figsize=(16.5, 11.6))
    gs = fig.add_gridspec(4, 4, hspace=0.14, wspace=0.06)
    scenes_rows = [("lk2", "albedo", 0), ("lk2", "normal", 1),
                   ("lwp", "albedo", 2), ("lwp", "normal", 3)]
    axes = {}
    gt_axes = {}
    for block_row, scene in ((0, "lk2"), (2, "lwp")):
        ax = fig.add_subplot(gs[block_row:block_row + 2, 0])
        gt = crop_to_box(_load_image(GT_IMAGES[scene]), CROP_BOXES[scene])
        ax.imshow(gt)
        ax.axis("off")
        gt_axes[scene] = ax
    for scene, kind, row in scenes_rows:
        for col in range(3):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(grid[(scene, kind, col)])
            ax.axis("off")
            axes[(row, col)] = ax

    plt.tight_layout(rect=(0.02, 0, 1, 1), h_pad=2.6)

    # column headers per block + rotated row labels (formerly TikZ)
    fs = args.label_fontsize
    for block_row, scene in ((0, "lk2"), (2, "lwp")):
        names = ("NeRF-OSR", MIDDLE_METHOD[scene], "Ours")
        top = max(axes[(block_row, c)].get_position().y1 for c in range(3))
        bbg = gt_axes[scene].get_position()
        fig.text((bbg.x0 + bbg.x1) / 2, top + 0.008, "Ground Truth",
                 ha="center", va="bottom", fontsize=fs, fontfamily="serif")
        for col, name in enumerate(names):
            bb = axes[(block_row, col)].get_position()
            fig.text((bb.x0 + bb.x1) / 2, top + 0.008, name,
                     ha="center", va="bottom", fontsize=fs, fontfamily="serif")
    for scene, kind, row in scenes_rows:
        bb = axes[(row, 0)].get_position()
        gtb = gt_axes[scene].get_position()
        # rotated row labels sit between the GT column and the method grid
        fig.text((gtb.x1 + bb.x0) / 2, (bb.y0 + bb.y1) / 2, kind.capitalize(),
                 ha="center", va="center", rotation=90, fontsize=fs,
                 fontfamily="serif")

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
