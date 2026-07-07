"""Qualitative synthetic-benchmark comparison figure (per scene).

For one "good" evaluation frame per scene (default: the frame with the best
NeuSky exposure-aligned NVS PSNR), builds a grid of GT | NeuSky | NeRF-OSR |
GS-IR for RGB / albedo / normals, plus an illumination column: the GT HDR
environment map (rolled to the frame's rotation) above NeuSky's illumination
fitted from the left half of the frame (the estimated track). All panels are
cached as individual PNGs so the figure recomposes instantly.

Stages:

    # 1. file-derived panels (CPU, local; needs prediction dirs + dataset)
    PYTHONPATH=. python scripts/figures/fig_synthetic_comparison.py --stage panels
    # 2. NeuSky fitted-envmap panels (GPU, on the host with the wave-3 runs)
    PYTHONPATH=.:../ns_reni python scripts/figures/fig_synthetic_comparison.py --stage envmap
    # 3. compose (CPU)
    PYTHONPATH=. python scripts/figures/fig_synthetic_comparison.py --stage compose
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman", "Times New Roman", "Times",
        "Liberation Serif", "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

import numpy as np

from _common import FIGURES_DIR, OUTPUTS_ROOT, SYNTHETIC_SCENES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synthetic_benchmark"))
from evaluate import read_exr  # noqa: E402

PANELS_DIR = FIGURES_DIR / "synthetic_comparison_panels"
BENCH = OUTPUTS_ROOT / "synthetic_benchmark"
DATA_ROOT = Path("/home/james/data/neusky_synthetic_data")
METHODS = ("neusky", "nerf_osr", "gs_ir")
METHOD_LABELS = {"neusky": "NeuSky (Ours)", "nerf_osr": "NeRF-OSR", "gs_ir": "GS-IR"}


def save_png(arr, path):
    from PIL import Image

    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8")).save(path)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def stack_exr_rgb(path):
    data = read_exr(path)
    for key in ("RGBA", "RGB"):
        if key in data:
            return data[key][..., :3].astype(np.float32)
    chans = [data[c] for c in ("R", "G", "B") if c in data]
    if len(chans) == 3:
        return np.stack(chans, -1).astype(np.float32)
    if {"X", "Y", "Z"} <= set(data):
        return np.stack([data["X"], data["Y"], data["Z"]], -1).astype(np.float32)
    raise ValueError(f"unrecognised EXR channels {sorted(data)} in {path}")


def normal_to_vis(n):
    """World-frame normal -> colour vis; zero-length background -> white."""
    vis = (n + 1.0) / 2.0
    vis[np.linalg.norm(n, axis=-1) < 0.5] = 1.0
    return vis


def good_frame(scene_stem):
    """Frame with the best NeuSky exposure-aligned NVS PSNR."""
    metrics = json.loads((BENCH / "neusky" / f"{scene_stem}_gt" / "metrics.json").read_text())
    per_frame = metrics["per_frame"]  # {"0000": {metric: value}}
    key = "nvs/psnr_masked_ea"
    frames = [(v[key], int(k)) for k, v in per_frame.items() if v.get(key) is not None]
    return max(frames)[1]


def frame_meta(scene_stem, frame):
    tj = json.loads((DATA_ROOT / "renders" / f"{scene_stem}_prepared" / "transforms.json").read_text())
    for split_key in ("test_frames", "frames"):
        pass
    frames = tj.get("test_frames") or tj["frames"]
    for fr in frames:
        fp = fr.get("file_path", "")
        if fp.split("/")[-1].split(".")[0] == f"{frame:04d}" and ("test" in fp or "test_frames" in tj):
            return fr
    # fall back: index into test split ordering
    test_frames = [fr for fr in frames if "test" in fr.get("file_path", "")]
    return test_frames[frame] if test_frames else frames[frame]


def stage_panels(args):
    from PIL import Image

    manifest = {}
    for scene, label in SYNTHETIC_SCENES.items():
        stem = scene.removesuffix("_prepared")
        frame = args.frame if args.frame is not None else good_frame(stem)
        out = PANELS_DIR / stem
        out.mkdir(parents=True, exist_ok=True)
        test_dir = DATA_ROOT / "renders" / scene / "test"

        # GT layers
        gt_rgb = np.asarray(Image.open(test_dir / "rgb" / f"{frame:04d}.png").convert("RGB"), np.float32) / 255
        save_png(gt_rgb, out / "gt_rgb.png")
        albedo_path = test_dir / "albedo" / f"{frame:04d}.png"
        if albedo_path.exists():
            gt_albedo = np.asarray(Image.open(albedo_path).convert("RGB"), np.float32) / 255
        else:
            gt_albedo = linear_to_srgb(stack_exr_rgb(test_dir / "albedo" / f"{frame:04d}.exr"))
        save_png(gt_albedo, out / "gt_albedo.png")
        normal_png = test_dir / "normal" / f"{frame:04d}.png"
        if normal_png.exists():
            gt_normal = np.asarray(Image.open(normal_png).convert("RGB"), np.float32) / 255
        else:
            gt_normal = normal_to_vis(stack_exr_rgb(test_dir / "normal" / f"{frame:04d}.exr"))
        save_png(gt_normal, out / "gt_normal.png")

        # method panels
        for method in METHODS:
            pred = BENCH / method / f"{stem}_gt"
            frames_dir = pred / "frames" if (pred / "frames").exists() else pred
            rgb_p = frames_dir / f"{frame:04d}_rgb.png"
            if rgb_p.exists():
                arr = np.asarray(Image.open(rgb_p).convert("RGB"), np.float32) / 255
                save_png(arr, out / f"{method}_rgb.png")
            alb_png = frames_dir / f"{frame:04d}_albedo.png"
            alb_exr = frames_dir / f"{frame:04d}_albedo.exr"
            if alb_png.exists():
                arr = np.asarray(Image.open(alb_png).convert("RGB"), np.float32) / 255
                save_png(arr, out / f"{method}_albedo.png")
            elif alb_exr.exists():
                # benchmark albedo is scale-invariant; align the prediction's
                # scale to GT (median ratio over lit pixels) before display
                pred_lin = np.clip(stack_exr_rgb(alb_exr), 0.0, None)
                gt_lin = np.power(np.clip(gt_albedo, 1e-4, 1.0), 2.4)
                m = (pred_lin.mean(-1) > 1e-3) & (gt_lin.mean(-1) > 1e-3)
                if m.sum() > 100:
                    scale = float(np.median(gt_lin[m] / np.clip(pred_lin[m], 1e-6, None)))
                    pred_lin = pred_lin * scale
                save_png(linear_to_srgb(pred_lin), out / f"{method}_albedo.png")
            nrm_exr = frames_dir / f"{frame:04d}_normal.exr"
            if nrm_exr.exists():
                save_png(normal_to_vis(stack_exr_rgb(nrm_exr)), out / f"{method}_normal.png")

        # NeuSky estimated-track render (left-half illumination fit)
        est = BENCH / "neusky" / f"{stem}_estimated"
        est_dir = est / "frames" if (est / "frames").exists() else est
        est_p = est_dir / f"{frame:04d}_rgb.png"
        if est_p.exists():
            arr = np.asarray(Image.open(est_p).convert("RGB"), np.float32) / 255
            save_png(arr, out / "neusky_est_rgb.png")

        # GT envmap, rolled to the frame's rotation
        meta = frame_meta(stem, frame)
        name = meta.get("envmap_name")
        rot = meta.get("envmap_rotation")
        hdri = None
        for cand_dir in (DATA_ROOT / "hdris", DATA_ROOT / "hdris_16k"):
            for ext in (".exr", ".hdr"):
                cand = cand_dir / f"{name}{ext}"
                if cand.exists():
                    hdri = cand
                    break
            if hdri:
                break
        if hdri is not None:
            if hdri.suffix == ".exr":
                env = stack_exr_rgb(hdri)
            else:
                import imageio.v3 as iio
                env = iio.imread(hdri).astype(np.float32)[..., :3]
            if isinstance(rot, list):
                # Z-rotation (yaw) in the metadata; roll the ERP horizontally
                import math
                yaw = math.atan2(rot[1][0], rot[0][0]) if isinstance(rot[0], list) else float(rot[-1])
                shift = int(round(-yaw / (2 * math.pi) * env.shape[1])) % env.shape[1]
                env = np.roll(env, shift, axis=1)
            small = env[::max(1, env.shape[0] // 256), ::max(1, env.shape[1] // 512)]
            save_png(linear_to_srgb(small / max(np.percentile(small, 99), 1e-6)),
                     out / "envmap_gt.png")
            np.save(out / "envmap_gt_hdr.npy", small)

        manifest[stem] = {"frame": frame, "label": label}
        print(f"[panels] {stem}: frame {frame} -> {out}")

    (PANELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


def stage_envmap(args):
    """Fit NeuSky's illumination to the LEFT HALF of the manifest frame
    (the estimated track's protocol), decode the fitted latent to an ERP,
    and score it against the GT HDRI (both p99-normalised): log-HDR PSNR
    and sun angular error. GPU; runs on the host with the wave-3 runs."""
    import math

    import torch

    from _common import load_model

    scene = args.scene
    stem = scene.removesuffix("_prepared")
    frame = args.frame
    assert frame is not None, "--frame required for the envmap stage"
    out = PANELS_DIR / stem
    out.mkdir(parents=True, exist_ok=True)

    def config_hook(config):
        m = config.pipeline.model
        m.eval_latent_optimise_method = "per_image"
        m.eval_latent_sample_region = "left_image_half"
        m.eval_num_rays_per_chunk = 512

    config, pipeline, _, step = load_model(
        scene, device=args.device, test_mode="test", config_hook=config_hook)
    model = pipeline.model
    model.eval()
    model.fit_latent_codes_for_eval(datamanager=pipeline.datamanager,
                                    global_step=step)

    # decode the frame's fitted eval latent over an equirect grid
    from nerfstudio.cameras.rays import Frustums, RaySamples
    from reni.field_components.field_heads import RENIFieldHeadNames
    from reni.model_components.illumination_samplers import EquirectangularSamplerConfig

    sampler = EquirectangularSamplerConfig(width=512).setup()
    with torch.no_grad():
        model.viewing_training_image = False
        model.fitting_eval_latents = False
        latents, scales = model.get_illumination_field()
        ray_bundle = sampler.camera.generate_rays(camera_indices=0, keep_shape=False)
        directions = ray_bundle.directions
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=torch.zeros_like(directions),
                directions=directions,
                starts=torch.zeros_like(directions[:, :1]),
                ends=torch.ones_like(directions[:, :1]),
                pixel_area=torch.ones_like(directions[:, :1]),
            ),
            camera_indices=torch.full_like(ray_bundle.camera_indices, frame),
        ).to(model.device)
        chunks = []
        for i in range(0, ray_samples.shape[0], 4096):
            sl = ray_samples[i:i + 4096]
            outputs = model.illumination_field(
                ray_samples=sl,
                latent_codes=latents[sl.camera_indices[:, 0]],
                scale=scales[sl.camera_indices[:, 0]],
                rotation=None,
            )
            chunks.append(outputs[RENIFieldHeadNames.RGB])
        raw = torch.cat(chunks)
        hdr = model.illumination_hdr_decode.to_linear_hdr(
            model.illumination_field, raw)
        H, W = sampler.height, sampler.width
        pred = hdr.reshape(H, W, 3).cpu().numpy()

    # model->dataset world yaw: decode is in model frame; GT ERP was saved in
    # dataset frame (panels stage). Roll pred by the dataparser orientation.
    meta = pipeline.datamanager.eval_dataset.metadata
    orientation = meta.get("orientation_rotation")
    if orientation is not None:
        R = np.asarray(orientation, np.float32)
        yaw = math.atan2(R[1, 0], R[0, 0])
        pred = np.roll(pred, int(round(-yaw / (2 * math.pi) * pred.shape[1])) % pred.shape[1], axis=1)

    pred_n = pred / max(float(np.percentile(pred, 99)), 1e-6)
    save_png(linear_to_srgb(pred_n), out / "envmap_pred.png")

    # score vs GT HDRI if the GT panel exists (both p99-normalised, log-HDR)
    gt_png = out / "envmap_gt_hdr.npy"
    metrics = {"scene": stem, "frame": frame, "step": step}
    if gt_png.exists():
        gt = np.load(gt_png)
        # resize GT to pred grid (area mean)
        fy, fx = gt.shape[0] // pred.shape[0], gt.shape[1] // pred.shape[1]
        if fy >= 1 and fx >= 1:
            gt_small = gt[:fy * pred.shape[0], :fx * pred.shape[1]].reshape(
                pred.shape[0], fy, pred.shape[1], fx, 3).mean((1, 3))
        else:
            gt_small = gt
        gt_n = gt_small / max(float(np.percentile(gt_small, 99)), 1e-6)
        a, b = np.log1p(np.clip(pred_n, 0, None)), np.log1p(np.clip(gt_n, 0, None))
        mse = float(((a - b) ** 2).mean())
        metrics["envmap_psnr_loghdr"] = -10 * math.log10(max(mse, 1e-12))

        def sun_dir(env):
            lum = env.mean(-1)
            iy, ix = np.unravel_index(np.argmax(lum), lum.shape)
            phi = (ix + 0.5) / lum.shape[1] * 2 * math.pi - math.pi
            theta = (iy + 0.5) / lum.shape[0] * math.pi
            return np.array([math.sin(theta) * math.cos(phi),
                             math.sin(theta) * math.sin(phi),
                             math.cos(theta)])
        cosang = float(np.clip(np.dot(sun_dir(pred_n), sun_dir(gt_n)), -1, 1))
        metrics["sun_angle_deg"] = math.degrees(math.acos(cosang))
    (out / "envmap_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[envmap] {stem} frame {frame}: {metrics}")


def stage_compose(args):
    import matplotlib.pyplot as plt
    from PIL import Image

    manifest = json.loads((PANELS_DIR / "manifest.json").read_text())
    rows = ("rgb", "albedo", "normal")
    row_labels = {"rgb": "Render", "albedo": "Albedo", "normal": "Normals"}
    cols = ("gt",) + METHODS

    for stem, info in manifest.items():
        out = PANELS_DIR / stem
        fig, axs = plt.subplots(3, 5, figsize=(15.2, 7.6))
        fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.01,
                            wspace=0.03, hspace=0.05)
        for r, modality in enumerate(rows):
            for c, source in enumerate(cols):
                ax = axs[r, c]
                name = f"{source}_{modality}.png"
                path = out / name
                if path.exists():
                    ax.imshow(Image.open(path))
                else:
                    ax.text(0.5, 0.5, "---", ha="center", va="center",
                            fontsize=16, color="0.6", transform=ax.transAxes)
                    ax.set_facecolor("0.92")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title("Ground Truth" if source == "gt"
                                 else METHOD_LABELS[source], fontsize=13, pad=5)
                if c == 0:
                    ax.set_ylabel(row_labels[modality], fontsize=13)
            # illumination column: GT envmap / NeuSky fitted / estimated render
            ax = axs[r, 4]
            name = {"rgb": "envmap_gt.png", "albedo": "envmap_pred.png",
                    "normal": "neusky_est_rgb.png"}[modality]
            lbl = {"rgb": "GT illumination", "albedo": "NeuSky fitted (left half)",
                   "normal": "NeuSky render, fitted illum."}[modality]
            path = out / name
            if path.exists():
                ax.imshow(Image.open(path))
            else:
                ax.text(0.5, 0.5, "---", ha="center", va="center",
                        fontsize=16, color="0.6", transform=ax.transAxes)
                ax.set_facecolor("0.92")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("Illumination", fontsize=13, pad=5)
            ax.set_xlabel(lbl, fontsize=10, labelpad=3)

        from _common import save_figure
        save_figure(fig, FIGURES_DIR / f"synthetic_comparison_{stem}", svg=False)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["panels", "envmap", "compose"],
                        required=True)
    parser.add_argument("--frame", type=int, default=None,
                        help="Override the auto-selected frame (all scenes)")
    parser.add_argument("--scene", default=None,
                        help="Scene for the envmap stage")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "panels":
        stage_panels(args)
    elif args.stage == "compose":
        stage_compose(args)
    else:
        stage_envmap(args)


if __name__ == "__main__":
    main()
