#!/usr/bin/env python
"""Render a trained NeuSky checkpoint into the benchmark prediction contract.

Produces, for each eval frame of a synthetic scene, the files defined in
README.md ("Prediction contract"): <stem>_rgb.png, <stem>_albedo.exr,
<stem>_normal.exr (dataset world frame, Z-up), <stem>_depth.exr (metric
z-depth in dataset units), plus a manifest.json describing the run.

Illumination modes (see README "Tracks"):
  gt          Track 1 (known illumination): per-frame RENI eval latents are
              fitted to the frame's GT HDRI at its known yaw rotation
              (log-domain MSE + cosine similarity, sine-weighted, as in the
              NeRF-OSR GT-envmap protocol in neusky_model
              _fit_eval_latents_to_envmaps). eval scale is frozen at 1.
  estimated   Track 2 (illumination estimation): latents fitted on the LEFT
              image half of each eval view via the existing per_image
              machinery (eval_latent_sample_region="left_image_half"),
              scored on the full frame.
  full-image-diagnostic
              Latents fitted on the FULL eval image. This fits the test
              pixels being scored and is NOT a benchmark track -- diagnostic
              upper bound only.

Usage (GPU required):
    PYTHONPATH=. python scripts/synthetic_benchmark/render_neusky_predictions.py \
        --scene abandoned_buildings --illumination gt \
        --pred-dir outputs/synthetic_benchmark/abandoned_buildings_gt \
        [--split test] [--frames 0 1 2] [--step N] [--device cuda:0]

This module is import-safe on CPU-only hosts: all torch / nerfstudio /
neusky imports are deferred into functions (same pattern as
scripts/figures/_common.py, whose resolve_run_dir/load_model it reuses).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "scripts" / "figures"


def _import_common():
    if str(FIGURES_DIR) not in sys.path:
        sys.path.insert(0, str(FIGURES_DIR))
    import _common

    return _common


# ---------------------------------------------------------------------------
# EXR writing (pyexr is a neusky runtime dep; OpenEXR>=3.2 also supported)
# ---------------------------------------------------------------------------

def write_exr(path, channels: dict) -> None:
    """Write {channel_name: float32 (H, W)} (or one (H, W, 3) under 'RGB')."""
    import numpy as np

    try:
        import OpenEXR  # >=3.2 File API

        OpenEXR.File(
            {}, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in channels.items()}
        ).write(str(path))
        return
    except (ImportError, AttributeError):
        pass
    import pyexr

    names = list(channels.keys())
    if len(names) == 1 and channels[names[0]].ndim == 3:
        arr = channels[names[0]]
        pyexr.write(str(path), arr.astype(np.float32),
                    channel_names=list("RGBA"[: arr.shape[-1]]))
    else:
        arr = np.stack([channels[n] for n in names], axis=-1).astype(np.float32)
        pyexr.write(str(path), arr, channel_names=names)


# ---------------------------------------------------------------------------
# Metric scale recovery
# ---------------------------------------------------------------------------

def compute_metric_scale(scene_data_dir: Path, image_filenames, cameras) -> float:
    """Model-units-per-metre scale of the loaded eval cameras.

    The synthetic dataparser normalises poses (up-orientation + SfM centering
    + scale capped to keep cameras inside the unit sky sphere) but only
    records the configured scale_factor, not the SfM-derived scale it
    actually applied. Rotations and uniform scaling preserve distance ratios,
    so the applied scale is recovered from pairwise camera distances:

        s = median_{i<j} ||t_model_i - t_model_j|| / ||t_metric_i - t_metric_j||

    where t_metric are the transforms.json camera positions (metres) and
    t_model the dataparser-normalised positions. Model depths are divided by
    s to obtain metric depths.
    """
    import numpy as np

    with open(Path(scene_data_dir) / "transforms.json") as f:
        meta = json.load(f)
    by_path = {fr["file_path"]: np.array(fr["transform_matrix"], dtype=np.float64)[:3, 3]
               for fr in meta["frames"]}

    metric, model = [], []
    c2ws = cameras.camera_to_worlds.cpu().numpy()
    for i, img_path in enumerate(image_filenames):
        rel = str(Path(img_path).relative_to(scene_data_dir))
        if rel in by_path:
            metric.append(by_path[rel])
            model.append(c2ws[i][:3, 3].astype(np.float64))
    if len(metric) < 2:
        raise RuntimeError(f"Need >=2 frames matched to transforms.json under {scene_data_dir}")
    metric, model = np.stack(metric), np.stack(model)

    ratios = []
    for i in range(len(metric)):
        for j in range(i + 1, len(metric)):
            d_metric = np.linalg.norm(metric[i] - metric[j])
            if d_metric > 1e-6:
                ratios.append(np.linalg.norm(model[i] - model[j]) / d_metric)
    scale = float(np.median(ratios))
    spread = float(np.std(ratios) / max(scale, 1e-12))
    if spread > 0.01:
        print(f"[render] WARNING: pairwise scale ratios vary by {spread:.2%}; "
              "poses may not be a similarity transform of transforms.json")
    return scale


# ---------------------------------------------------------------------------
# Track 1: fit eval latents to GT HDRIs at known rotations
# ---------------------------------------------------------------------------

def fit_eval_latents_to_gt_hdris(model, gt_envmap_info, frame_indices, rays_per_chunk: int = 262144):
    """Fit per-frame RENI eval latents to the GT HDRIs (known yaw).

    Mirrors neusky_model._fit_eval_latents_to_envmaps (the NeRF-OSR
    GT-envmap protocol) for the synthetic data:

    - The frame's HDRI (gt_envmap_info[idx]["path"]) is rolled horizontally
      by -yaw/(2*pi)*width (the dataset convention: a world direction
      d=(x,y,z) sees HDRI pixel lon = atan2(y,-x) - yaw; same convention as
      neusky_model's gt_vs_pred_envmap comparison) and area-resampled to the
      model's equirectangular sampler resolution, giving per-pixel HDR
      targets aligned with the sampler's model-frame directions.
    - Latents start from zero and are optimised with sine-weighted
      log-domain MSE + cosine similarity under config.eval_latent_optimizer.
    - eval scale is FROZEN at 1: the HDRIs are absolute HDR and test frames
      have exposure_ev = 0 (brightness gauge is handled by the scorer's
      exposure-aligned PSNR, not by fitting scale to test pixels).
    """
    import numpy as np
    import pyexr
    import torch
    from nerfstudio.cameras.rays import Frustums, RaySamples
    from nerfstudio.engine.optimizers import Optimizers
    from reni.field_components.field_heads import RENIFieldHeadNames

    device = model.device
    sampler = model.equirectangular_sampler
    H, W = sampler.height, sampler.width

    # Model-frame ERP directions, row-major (H, W) like render_envmap.
    ray_samples = sampler.generate_direction_samples().to(device)
    directions = ray_samples.frustums.directions.reshape(-1, 3)

    # Solid-angle weight of each ERP row (sin of polar angle).
    polar = torch.pi * (torch.arange(H, dtype=torch.float32) + 0.5) / H
    sineweight = torch.sin(polar)[:, None].repeat(1, W).reshape(-1, 1).to(device)

    targets, cam_idx_chunks, dir_chunks, weight_chunks = [], [], [], []
    for idx in frame_indices:
        info = gt_envmap_info[idx] if idx < len(gt_envmap_info) else None
        if info is None:
            raise RuntimeError(
                f"No GT envmap metadata for eval frame {idx}; --illumination gt "
                "needs hdris/<envmap_name>.exr resolvable by the dataparser."
            )
        hdri = pyexr.open(info["path"]).get().astype(np.float32)[..., :3]
        yaw = float(info["rotation"][2]) if info.get("rotation") else 0.0
        shift = int(round(yaw / (2.0 * np.pi) * hdri.shape[1]))
        if shift:
            hdri = np.roll(hdri, -shift, axis=1)
        t = torch.from_numpy(hdri).permute(2, 0, 1)[None]
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="area")
        targets.append(t[0].permute(1, 2, 0).reshape(-1, 3).to(device))
        dir_chunks.append(directions)
        weight_chunks.append(sineweight)
        cam_idx_chunks.append(torch.full((H * W,), idx, dtype=torch.long, device=device))

    rgb = torch.cat(targets).clamp(min=0.0)
    dirs = torch.cat(dir_chunks)
    weights = torch.cat(weight_chunks)
    camera_indices = torch.cat(cam_idx_chunks)

    assert model.eval_scale is not None, "--illumination gt requires a RENI illumination field"
    model.eval_illumination_latents.data.zero_()
    model.eval_scale.data.fill_(0.0)  # exp(scale) semantics: 0 = neutral; 1.0 was ~2.72x over-bright

    optimizer = Optimizers(
        model.config.eval_latent_optimizer,
        {"eval_latents": [model.eval_illumination_latents]},
    )
    steps = optimizer.config["eval_latents"]["scheduler"].max_steps
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
    total_rays = int(rgb.shape[0])
    chunk_size = max(1, min(int(rays_per_chunk), total_rays))
    print(f"[render] fitting {len(frame_indices)} HDRIs with {total_rays} rays in chunks of {chunk_size}")

    with model.illumination_field.hold_decoder_fixed():
        for step in range(steps):
            optimizer.zero_grad_all()
            loss_value = 0.0
            for start in range(0, total_rays, chunk_size):
                end = min(start + chunk_size, total_rays)
                chunk_dirs = dirs[start:end]
                chunk_rgb = rgb[start:end]
                chunk_weights = weights[start:end]
                chunk_camera_indices = camera_indices[start:end]
                ray_samples = RaySamples(
                    frustums=Frustums(
                        origins=torch.zeros_like(chunk_dirs),
                        directions=chunk_dirs,
                        starts=torch.zeros_like(chunk_dirs[..., :1]),
                        ends=torch.zeros_like(chunk_dirs[..., :1]),
                        pixel_area=torch.ones_like(chunk_dirs[..., :1]),
                    ),
                    camera_indices=chunk_camera_indices[:, None],
                )
                outputs = model.illumination_field.forward(
                    ray_samples=ray_samples,
                    latent_codes=model.eval_illumination_latents[chunk_camera_indices],
                    scale=model.eval_scale[chunk_camera_indices].detach(),
                )
                # two-bracket priors emit 6ch brackets; decode to linear HDR
                # (3ch path defers to unnormalise, byte-identical to before)
                pred_hdr = model.illumination_hdr_decode.to_linear_hdr(
                    model.illumination_field, outputs[RENIFieldHeadNames.RGB])
                log_mse = torch.sum(
                    chunk_weights * (torch.log(pred_hdr + 1e-8) - torch.log(chunk_rgb + 1e-8)) ** 2
                ) / (total_rays * 3)
                cosine_loss = torch.sum(
                    chunk_weights * (1 - cosine(pred_hdr, chunk_rgb)).unsqueeze(-1)
                ) / total_rays
                loss = log_mse + cosine_loss
                loss.backward()
                loss_value += float(loss.detach().cpu())
            optimizer.optimizer_step("eval_latents")
            optimizer.scheduler_step("eval_latents")
            if step % 50 == 0 or step == steps - 1:
                print(f"[render] HDRI latent fit step {step+1}/{steps} loss {loss_value:.4f}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frames(args):
    import numpy as np
    import torch
    from PIL import Image

    _common = _import_common()

    def config_hook(config):
        config.pipeline.datamanager.camera_res_scale_factor = args.render_scale
        m = config.pipeline.model
        if args.illumination == "estimated":
            m.eval_latent_optimise_method = "per_image"
            m.eval_latent_sample_region = "left_image_half"
        elif args.illumination == "full-image-diagnostic":
            m.eval_latent_optimise_method = "per_image"
            m.eval_latent_sample_region = "full_image"
        if args.eval_num_rays_per_chunk:
            m.eval_num_rays_per_chunk = args.eval_num_rays_per_chunk

    config, pipeline, ckpt_path, step = _common.load_model(
        args.scene, device=args.device, test_mode="test", step=args.step,
        config_hook=config_hook,
    )
    model = pipeline.model
    datamanager = pipeline.datamanager
    eval_dataset = datamanager.eval_dataset
    cameras = eval_dataset.cameras
    metadata = eval_dataset.metadata
    scene_data_dir = Path(config.pipeline.datamanager.dataparser.data)

    n = len(eval_dataset.image_filenames)
    frame_indices = [int(f) for f in args.frames] if args.frames else list(range(n))

    scale = compute_metric_scale(scene_data_dir, eval_dataset.image_filenames, cameras)
    print(f"[render] model-units-per-metre scale: {scale:.6f}")

    # Dataset-world <- model-world rotation (dataparser up-orientation).
    orientation = metadata.get("orientation_rotation")
    R = (torch.eye(3) if orientation is None else torch.as_tensor(orientation).float()).to(args.device)

    if args.illumination == "gt":
        fit_eval_latents_to_gt_hdris(
            model, metadata.get("gt_envmap_info") or [], frame_indices, args.hdri_fit_rays_per_chunk
        )
    else:
        # Existing per-image machinery (left half / full image per config_hook).
        model.fit_latent_codes_for_eval(datamanager=datamanager, global_step=step)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for k, idx in enumerate(frame_indices):
        stem = Path(eval_dataset.image_filenames[idx]).stem
        ray_bundle = cameras.generate_rays(camera_indices=idx, keep_shape=True).to(args.device)
        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle=ray_bundle, show_progress=True
            )

        rgb = outputs["rgb"].clamp(0, 1).cpu().numpy()  # already sRGB-encoded
        Image.fromarray((rgb * 255).round().astype(np.uint8)).save(pred_dir / f"{stem}_rgb.png")

        albedo = outputs["albedo"].clamp(0, 1).cpu().numpy().astype(np.float32)  # linear
        write_exr(pred_dir / f"{stem}_albedo.exr", {"RGB": albedo})

        # Model-frame normals -> dataset world frame (Z-up), renormalised.
        normal = outputs["normal"].cpu() @ R.cpu()  # n_dataset = R^T @ n_model, row-vector form
        norm = normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normal = (normal / norm).numpy().astype(np.float32)
        write_exr(
            pred_dir / f"{stem}_normal.exr",
            {"X": normal[..., 0], "Y": normal[..., 1], "Z": normal[..., 2]},
        )

        # Model "depth" is already z-depth (p2p_dist / directions_norm) in
        # model units; divide by the recovered scale for metric z-depth.
        depth = (outputs["depth"].cpu().numpy().astype(np.float64) / scale)[..., 0]
        depth = np.nan_to_num(depth, nan=1.0e10, posinf=1.0e10).astype(np.float32)
        write_exr(pred_dir / f"{stem}_depth.exr", {"Y": depth})

        print(f"[render] wrote frame {stem} ({k + 1}/{len(frame_indices)})")

    manifest = {
        "method": "neusky",
        "scene": args.scene,
        "scene_data_dir": str(scene_data_dir),
        "split": "test",
        "illumination": args.illumination,
        "track": {"gt": "nvs", "estimated": "estimation"}.get(args.illumination, "diagnostic"),
        "checkpoint": str(ckpt_path),
        "step": step,
        "frames": [Path(eval_dataset.image_filenames[i]).stem for i in frame_indices],
        "model_units_per_metre": scale,
        "notes": "normals in dataset world frame (Z-up); depth is metric z-depth; "
                 "albedo linear; rgb sRGB-encoded PNG",
    }
    with open(pred_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[render] wrote {pred_dir}/manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True,
                    help="synthetic scene (e.g. abandoned_buildings or *_prepared)")
    ap.add_argument("--pred-dir", required=True, help="output prediction directory")
    ap.add_argument("--illumination", required=True,
                    choices=["gt", "estimated", "full-image-diagnostic"])
    ap.add_argument("--split", default="test", choices=["test"],
                    help="only 'test' is supported (the model's eval split)")
    ap.add_argument("--frames", nargs="*", default=None, help="eval frame indices (default: all)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-num-rays-per-chunk", type=int, default=None)
    ap.add_argument("--hdri-fit-rays-per-chunk", type=int, default=262144,
                    help="Number of HDRI rays per optimiser sub-batch for --illumination gt.")
    ap.add_argument("--render-scale", type=float, default=1.0,
                    help="Eval image scale for predictions. Public benchmark scoring requires 1.0; use 0.25 only for previews.")
    args = ap.parse_args(argv)
    render_frames(args)


if __name__ == "__main__":
    main()
