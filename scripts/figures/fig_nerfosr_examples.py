"""NeRF-OSR examples figure (thesis fig:examples -> nerfosr_examples_thesis).

2x4 grid: rows Site 1 (lk2) and Site 2 (st); columns GT photo | render with
the view's fitted RENI++ envmap inset | albedo | camera-space normals.
Views match the original figure (template-matched): lk2 C5_DSC_3_4,
st 12-04_18_30_DSC_0553. Column labels are baked (thesis serif).

Two-stage, since lk2 renders locally but st lives on Isambard:

    # per scene, on the host that has the run (GPU)
    PYTHONPATH=.:../ns_reni python scripts/figures/fig_nerfosr_examples.py \
        --panels-only --scene lk2 --chunk 512
    # once both scenes' panels are cached under nerfosr_examples_panels/
    PYTHONPATH=. python scripts/figures/fig_nerfosr_examples.py --compose-only
"""

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman", "Times New Roman", "Times",
        "Liberation Serif", "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

from _common import FIGURES_DIR, load_model, render_view, normal_to_camera_vis, save_figure

PANELS_DIR = FIGURES_DIR / "nerfosr_examples_panels"

VIEWS = {
    "lk2": "C5_DSC_3_4.png",
    "st": "12-04_18_30_DSC_0553.jpg",
}
SCENES = ("lk2", "st")
COLUMNS = ("gt", "render", "albedo", "normal")
LABELS = ("Ground Truth", "Render", "Albedo", "Normals")


def save_png(arr, path):
    import numpy as np
    from PIL import Image

    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8")).save(path)


def stage_panels(args):
    import numpy as np
    import torch
    from PIL import Image

    from fig_teaser import render_envmap_pair

    out = PANELS_DIR / args.scene
    out.mkdir(parents=True, exist_ok=True)

    def _low_vram_hook(config):
        dm = config.pipeline.datamanager
        dm.images_on_gpu = False
        dm.masks_on_gpu = False
        dm.eval_num_images_to_sample_from = 4

    _, pipeline, _, step = load_model(
        args.scene, device=args.device, step=args.step,
        eval_num_rays_per_chunk=args.chunk, config_hook=_low_vram_hook)
    datamanager = pipeline.datamanager
    model = pipeline.model
    model.eval()

    names = [Path(p).name for p in
             datamanager.train_dataset.image_filenames]
    view = names.index(VIEWS[args.scene])
    print(f"[panels] {args.scene}: view {view} ({VIEWS[args.scene]}), step {step}")

    gt = datamanager.train_dataset[view]["image"]
    save_png(gt.cpu().numpy(), out / "gt.png")

    cameras = copy.deepcopy(datamanager.train_dataset.cameras)
    if args.scale != 1.0:
        cameras.rescale_output_resolution(scaling_factor=args.scale)

    outputs = render_view(model, cameras, view, args.device)
    save_png(outputs["rgb"].cpu().numpy(), out / "render.png")
    save_png(outputs["albedo"].cpu().numpy(), out / "albedo.png")
    c2w = cameras.camera_to_worlds[view]
    save_png(normal_to_camera_vis(outputs["normal"], c2w), out / "normal.png")
    del outputs
    torch.cuda.empty_cache()

    ldr, _ = render_envmap_pair(model, view)
    save_png(ldr, out / "envmap.png")
    print(f"[panels] -> {out}")


def compose(args):
    import matplotlib.pyplot as plt
    from PIL import Image

    missing = [s for s in SCENES
               if not (PANELS_DIR / s / "envmap.png").exists()]
    if missing:
        raise SystemExit(f"Missing panels for {missing}; run --panels-only "
                         f"on the host holding those runs first.")

    n_rows, n_cols = len(SCENES), len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 4.7))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.01,
                        wspace=0.06, hspace=0.06)
    for r, scene in enumerate(SCENES):
        for c, name in enumerate(COLUMNS):
            ax = axes[r, c]
            ax.imshow(Image.open(PANELS_DIR / scene / f"{name}.png"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(1.2)
            if r == 0:
                ax.set_title(LABELS[c], fontsize=13, pad=6)
        # envmap inset overflowing the render panel's top-right corner
        rb = axes[r, 1].get_position()
        iw, ih = 0.42 * rb.width, 0.42 * rb.width * 0.5 * (15.0 / 4.7)
        ins = fig.add_axes([rb.x1 - iw * 0.99, rb.y1 - ih * 0.92, iw, ih])
        ins.imshow(Image.open(PANELS_DIR / scene / "envmap.png"))
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_linewidth(1.0)
    save_figure(fig, args.output, svg=args.svg)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=SCENES, default="lk2")
    parser.add_argument("--panels-only", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=FIGURES_DIR / "nerfosr_examples_thesis")
    args = parser.parse_args()

    if not args.compose_only:
        stage_panels(args)
    if not args.panels_only:
        compose(args)


if __name__ == "__main__":
    main()
