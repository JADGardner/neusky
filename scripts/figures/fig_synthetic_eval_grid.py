"""Synthetic eval grid (thesis Fig: synthetic_eval -> synthetic_eval_grid.png).

5 scenes x 6 columns: GT RGB | pred RGB | GT albedo | pred albedo | GT normal
| pred normal, on the synthetic eval split. Uses the model's own
get_image_metrics_and_images (the same path ns-eval renders through), which
emits the side-by-side "img" / "gt_vs_pred_albedo" / "gt_vs_pred_normal"
panels that are split into the 6 cells here — so the figure matches the
tab:synthetic metrics exactly. Eval illumination latents are fitted first via
pipeline._optimise_evaluation_latents, as in evaluation.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_synthetic_eval_grid.py
    PYTHONPATH=. python scripts/figures/fig_synthetic_eval_grid.py \
        --scenes interstellar_house --frame-indices 12
"""

import argparse

import matplotlib.pyplot as plt

from _common import (SYNTHETIC_SCENES, add_common_args, load_model, save_figure,
                     seed_all)

# Frame index per scene, chosen for good viewpoints (matches
# scripts/eval_and_figures_neusky_synthetic.sh in the phd repo).
DEFAULT_FRAMES = {
    "abandoned_buildings_prepared": 10,
    "apartment_building_prepared": 10,
    "arlanda_uppsala_cathedral_prepared": 10,
    "glass_building_prepared": 0,
    "interstellar_house_prepared": 12,
}

PANELS = ("img", "gt_vs_pred_albedo", "gt_vs_pred_normal")


def split_pair(img):
    """Split a GT|pred side-by-side tensor [H, 2W, 3] into two [H, W, 3]."""
    w = img.shape[1] // 2
    return img[:, :w], img[:, w:]


def render_scene_cells(scene: str, frame_idx: int, device, step=None):
    import torch

    _, pipeline, _, loaded_step = load_model(scene, device=device, step=step)
    pipeline._optimise_evaluation_latents(loaded_step)
    model = pipeline.model
    model.eval()
    if model.visibility_field is not None:
        model.visibility_field.eval()

    datamanager = pipeline.datamanager
    eval_dataset = datamanager.eval_dataset
    batch = eval_dataset[frame_idx]
    batch["image_idx"] = frame_idx
    camera_ray_bundle = eval_dataset.cameras.generate_rays(
        camera_indices=frame_idx, keep_shape=True
    ).to(device)

    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(
            camera_ray_bundle, show_progress=True, step=loaded_step)
    _, images_dict = model.get_image_metrics_and_images(outputs, batch)

    cells = []
    for panel in PANELS:
        if panel not in images_dict:
            raise KeyError(
                f"{scene}: missing '{panel}' (is this a synthetic run with GT layers?)")
        gt, pred = split_pair(images_dict[panel].cpu().numpy())
        cells.extend([gt.clip(0, 1), pred.clip(0, 1)])

    del pipeline, model, datamanager
    torch.cuda.empty_cache()
    return cells


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "synthetic_eval_grid")
    parser.add_argument("--scenes", nargs="+", default=list(SYNTHETIC_SCENES),
                        help="Synthetic scene names (bare stems accepted)")
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None,
                        help="Eval frame per scene (default: curated views)")
    parser.add_argument("--labels", action="store_true",
                        help="Draw row/column labels (omit when LaTeX adds them)")
    args = parser.parse_args()
    seed_all(args.seed)

    from _common import canonical_scene

    scenes = [canonical_scene(s) for s in args.scenes]
    frames = args.frame_indices or [DEFAULT_FRAMES.get(s, 0) for s in scenes]
    if len(frames) != len(scenes):
        raise SystemExit("--frame-indices must give one index per scene")

    rows = [render_scene_cells(scene, frame, args.device, step=args.step)
            for scene, frame in zip(scenes, frames)]

    col_titles = ("GT RGB", "Pred RGB", "GT Albedo", "Pred Albedo",
                  "GT Normal", "Pred Normal")
    n_rows, n_cols = len(rows), 6
    h, w = rows[0][0].shape[:2]
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows * h / w), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            if args.labels and r == 0:
                axs[r, c].set_title(col_titles[c], fontsize=8)
        if args.labels:
            axs[r, 0].set_ylabel(SYNTHETIC_SCENES.get(scenes[r], scenes[r]), fontsize=8)
    plt.tight_layout(pad=0.2)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
