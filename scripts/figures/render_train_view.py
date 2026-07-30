"""Render a train view from a checkpoint with its own train latents (no fit).

Sharpness A/B helper: geometry/appearance comparison between checkpoints of
the same scene without any eval-latent fitting. Pin the run with NEUSKY_RUNS.

    NEUSKY_RUNS='lk2=outputs/<run>' PYTHONPATH=.:scripts/figures \
        python scripts/figures/render_train_view.py --scene lk2 --view 0 \
        --output /tmp/runA_view0 --render-scale 0.5
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _common import add_common_args, load_model, seed_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "train_view")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--view", type=int, default=0, help="train image index")
    parser.add_argument("--render-scale", type=float, default=0.5)
    args = parser.parse_args()
    seed_all(args.seed)

    import numpy as np
    import torch
    from PIL import Image

    _, pipeline, _, step = load_model(args.scene, device=args.device, step=args.step,
                                      eval_num_rays_per_chunk=1024)
    model = pipeline.model
    model.eval()
    dm = pipeline.datamanager

    camera = deepcopy(dm.train_dataset.cameras[args.view : args.view + 1]).to(model.device)
    camera.rescale_output_resolution(args.render_scale)
    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
    ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * args.view
    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, step=step)

    out = Path(str(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = (outputs["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(rgb).save(out.with_suffix(".png"))
    gt = dm.train_dataset[args.view]["image"]
    Image.fromarray((gt.numpy() * 255).astype("uint8")).save(
        out.parent / (out.name + "_gt.png"))
    normal = ((outputs["normal"].cpu().numpy() + 1) / 2 * 255).astype("uint8")
    Image.fromarray(normal).save(out.parent / (out.name + "_normal.png"))
    print(f"[saved] {out}.png (+_gt, +_normal), step {step}, "
          f"size {rgb.shape[1]}x{rgb.shape[0]}")


if __name__ == "__main__":
    main()
