"""Per-scene intrinsic decomposition rows (thesis Fig: examples ->
nerfosr_examples.png; also produces the per-scene halves of comparisons_full.png).

For each scene, renders one training view and lays out a row of
GT / render / albedo / camera-space normals — the recipe from
publication/figures_and_tables.ipynb (the "site train view" cells: rays via
generate_rays(keep_shape=True), viewing_training_image=True, normals rotated
into camera space and whited out off-surface).

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_decomposition.py --scenes lk2 st
    PYTHONPATH=. python scripts/figures/fig_decomposition.py --scenes lk2 --views 143
"""

import argparse

import matplotlib.pyplot as plt

from _common import (SCENE_DEFAULT_VIEWS, add_common_args, load_model,
                     normal_to_camera_vis, render_view, save_figure, seed_all)

COLUMNS = ("Ground Truth", "Render", "Albedo", "Normals")


def render_scene_row(scene: str, view_idx, device, step=None):
    import numpy as np

    _, pipeline, _, _ = load_model(scene, device=device, step=step)
    datamanager = pipeline.datamanager
    model = pipeline.model
    model.eval()

    if view_idx is None:
        view_idx = SCENE_DEFAULT_VIEWS.get(scene, 0)

    batch = datamanager.train_dataset[view_idx]
    gt_image = batch["image"].cpu().numpy()

    outputs = render_view(model, datamanager.train_dataset.cameras, view_idx, device)
    rgb = outputs["rgb"].cpu().numpy()
    albedo = outputs["albedo"].cpu().numpy()
    c2w = datamanager.train_dataset.cameras.camera_to_worlds[view_idx]
    normal_vis = normal_to_camera_vis(outputs["normal"], c2w)

    row = [gt_image, rgb, albedo, normal_vis]
    del pipeline, datamanager, model
    import torch
    torch.cuda.empty_cache()
    return [np.clip(img, 0, 1) for img in row]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "nerfosr_examples")
    parser.add_argument("--scenes", nargs="+", default=["lk2", "st"],
                        help="Scenes (lk2/st/lwp or site1/site2/site3 aliases)")
    parser.add_argument("--views", type=int, nargs="+", default=None,
                        help="Train view index per scene (default: notebook-"
                             "documented good views, e.g. lk2=143)")
    parser.add_argument("--labels", action="store_true",
                        help="Draw column titles (omit when LaTeX adds them)")
    args = parser.parse_args()
    seed_all(args.seed)

    views = args.views or [None] * len(args.scenes)
    if len(views) != len(args.scenes):
        raise SystemExit("--views must give one index per scene")

    rows = [render_scene_row(scene, view, args.device, step=args.step)
            for scene, view in zip(args.scenes, views)]

    n_rows, n_cols = len(rows), len(COLUMNS)
    h, w = rows[0][0].shape[:2]
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows * h / w), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            if args.labels and r == 0:
                axs[r, c].set_title(COLUMNS[c])
    plt.tight_layout(pad=0.3)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
