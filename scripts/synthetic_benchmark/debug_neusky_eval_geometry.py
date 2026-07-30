"""Debug NeuSky eval-camera geometry for a trained checkpoint.

This is intentionally a diagnostic script, not a benchmark renderer. It loads a
NeuSky run through the same eval path as the figure scripts, fits eval latents,
and prints compact tensor/SDF stats for selected validation frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.figures import _common


def _stats(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach().float().cpu()
    return (
        f"{name}: shape={tuple(t.shape)} mean={t.mean().item():.6f} "
        f"min={t.min().item():.6f} max={t.max().item():.6f} std={t.std().item():.6f}"
    )


def _zero_crossings(
    sdf: torch.Tensor,
    distances: torch.Tensor,
) -> tuple[float | None, float | None]:
    """Return first negative interval and first sign-change distance."""
    sdf_1d = sdf.detach().flatten()
    distances = distances.detach().flatten()
    negative = torch.nonzero(sdf_1d < 0.0, as_tuple=False).flatten()
    first_negative = float(distances[negative[0]].item()) if len(negative) else None
    signs = torch.signbit(sdf_1d)
    changes = torch.nonzero(signs[1:] != signs[:-1], as_tuple=False).flatten()
    first_change = float(distances[changes[0] + 1].item()) if len(changes) else None
    return first_negative, first_change


def _sample_sdf_along_rays(model, ray_bundle, image_idx: int, device: str) -> None:
    height, width = ray_bundle.shape
    yxs = [
        (height // 2, width // 2),
        (height // 4, width // 4),
        (height // 4, 3 * width // 4),
        (3 * height // 4, width // 4),
        (3 * height // 4, 3 * width // 4),
    ]
    max_dist = float(getattr(model, "ddf_radius", 1.0)) * 2.0
    distances = torch.linspace(0.0, max_dist, 256, device=device)

    origins = []
    dirs = []
    for y, x in yxs:
        origins.append(ray_bundle.origins[y, x])
        dirs.append(ray_bundle.directions[y, x])
    origins = torch.stack(origins).to(device)
    dirs = torch.nn.functional.normalize(torch.stack(dirs).to(device), dim=-1)

    camera_sdf = model.field.get_sdf_at_pos(origins).detach().flatten()
    points = origins[:, None, :] + distances[None, :, None] * dirs[:, None, :]
    sdf = model.field.get_sdf_at_pos(points.reshape(-1, 3)).reshape(len(yxs), -1)

    print(f"  camera_sdf={camera_sdf.detach().cpu().tolist()}")
    for ray_i, (y, x) in enumerate(yxs):
        first_negative, first_change = _zero_crossings(sdf[ray_i], distances)
        print(
            f"  ray({y},{x}) sdf_min={sdf[ray_i].min().item():.6f} "
            f"sdf_max={sdf[ray_i].max().item():.6f} "
            f"first_negative={first_negative} first_sign_change={first_change}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="interstellar_house")
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--test-mode", default="val", choices=["val", "test", "inference"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=256)
    args = parser.parse_args()

    _, pipeline, ckpt_path, step = _common.load_model(
        args.scene,
        device=args.device,
        step=args.step,
        test_mode=args.test_mode,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
    )
    model = pipeline.model
    datamanager = pipeline.datamanager
    eval_dataset = datamanager.eval_dataset
    cameras = eval_dataset.cameras.to(args.device)

    print(f"checkpoint={ckpt_path}")
    print(f"loaded_step={step}")
    print(f"ddf_radius={getattr(model, 'ddf_radius', None)}")
    print("fitting eval latents...")
    model.fit_latent_codes_for_eval(datamanager=datamanager, global_step=step)
    model.eval()

    for split_name, dataset in (("train", datamanager.train_dataset), (args.test_mode, eval_dataset)):
        split_cameras = dataset.cameras.to(args.device)
        origins = split_cameras.camera_to_worlds[:, :3, 3].to(args.device)
        with torch.no_grad():
            sdf_at_origins = model.field.get_sdf_at_pos(origins).detach().flatten()
        radii = origins.norm(dim=-1).detach().flatten()
        neg = int((sdf_at_origins < 0.0).sum().item())
        print(
            f"{split_name}_camera_origin_sdf: neg={neg}/{len(sdf_at_origins)} "
            f"mean={sdf_at_origins.mean().item():.6f} min={sdf_at_origins.min().item():.6f} "
            f"max={sdf_at_origins.max().item():.6f} radius_min={radii.min().item():.6f} "
            f"radius_max={radii.max().item():.6f}"
        )
        if split_name != "train":
            print("eval_camera_origin_sdf_values=" + ",".join(f"{v:.6f}" for v in sdf_at_origins.cpu().tolist()))

    for idx in args.frames:
        stem = Path(eval_dataset.image_filenames[idx]).stem
        ray_bundle = cameras.generate_rays(camera_indices=idx, keep_shape=True).to(args.device)
        height, width = ray_bundle.shape
        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle=ray_bundle,
                show_progress=False,
                step=step,
            )

        print(f"\nframe={idx} stem={stem}")
        for key in ("accumulation", "depth", "rgb", "albedo", "normal", "hdr_background_colours"):
            if key in outputs:
                print("  " + _stats(key, outputs[key]))

        if "weights" in outputs:
            weights = outputs["weights"].detach()
            weights_by_sample = weights[..., 0]
            if weights_by_sample.shape[:2] == (height, width):
                sample_dim = 2
            elif weights_by_sample.dim() >= 3 and weights_by_sample.shape[0] == height and weights_by_sample.shape[2] == width:
                sample_dim = 1
            else:
                sample_dim = weights_by_sample.dim() - 1
            weights_sum = weights_by_sample.sum(dim=sample_dim)
            max_weight, max_idx = weights_by_sample.max(dim=sample_dim)
            print("  " + _stats("weights_sum", weights_sum))
            print("  " + _stats("max_weight", max_weight))
            print("  " + _stats("max_weight_idx", max_idx.float()))

        _sample_sdf_along_rays(model, ray_bundle, idx, args.device)


if __name__ == "__main__":
    main()
