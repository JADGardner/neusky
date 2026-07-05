"""Visualise the NeRF-OSR holdout relighting protocol (thesis Fig:
holdout_protocol).

One row per test session, walking the protocol left to right:

- Holdout: the GT photo the session's RENI++ eval latent is fitted on
  (canonical holdout indices, 250-step seeded fit);
- Fit: the holdout viewpoint rendered under the fitted illumination;
- Fitted env.: the fitted RENI++ envmap decoded on the model-frame
  equirectangular grid (LDR);
- GT ERP: the session's ENV_MAP_CC panorama resampled into the SAME frame
  using the recovered rotation (envmap_rotations.json: d_world = R @ d_pano,
  composed with the dataparser orientation_transform exactly as the
  nerf_osr_envmap eval path does) -- the sun in this column should line up
  with the previous one;
- Target: the session's compare (test) view GT photo;
- Relit: the compare view rendered under the fitted illumination (the
  protocol's scored render, shown unmasked).

Prints a per-session sun-alignment diagnostic (angle between the brightest
fitted and GT ERP directions) so convention slips show up as a consistent
yaw offset.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/fig_holdout_protocol.py --scene lk2
"""

import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from _common import (SESSION_HOLDOUT_INDICES, add_common_args, load_model,
                     save_figure, seed_all)


def session_label(name: str) -> str:
    """'01-08_07_30' -> '01-08 07:30' (session folder names are timestamps)."""
    parts = str(name).split("_")
    if len(parts) == 3:
        return f"{parts[0]} {parts[1]}:{parts[2]}"
    return str(name)


def load_photo(path, max_width: int = 640):
    """GT photo as float [H, W, 3], downsampled for the figure."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)),
                         Image.BILINEAR)
    return np.asarray(img, dtype="float32") / 255.0


def render_view(model, dm, image_idx: int, session_idx: int, step, scale: float = 0.5,
                components: bool = False):
    """Render an eval view under the fitted session illumination [H, W, 3].

    With components=True, returns a dict decomposing the render: the composite
    rgb, the RENI-only sky decode (sRGB of hdr_background_colours, i.e. what the
    illumination fit produced along these rays with no geometry in front), and
    the accumulation map (geometry opacity: where floaters/fog occlude the sky).
    """
    import torch
    from reni.utils.colourspace import linear_to_sRGB

    camera, _ = dm.eval_dataloader.get_camera(image_idx)
    camera = deepcopy(camera)
    camera.rescale_output_resolution(scale)
    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
    ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * session_idx
    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, step=step)
    rgb = outputs["rgb"].clamp(0, 1).cpu().numpy()
    if not components:
        return rgb
    sky_srgb_raw = linear_to_sRGB(outputs["hdr_background_colours"], clamp=False)
    return {
        "rgb": rgb,
        "sky_srgb": sky_srgb_raw.clamp(0, 1).cpu().numpy(),
        "sky_saturated_frac": float((sky_srgb_raw.max(dim=-1).values >= 1.0).float().mean()),
        "accumulation": outputs["accumulation"].clamp(0, 1).cpu().numpy(),
    }


def render_fitted_envmap(model, session_idx: int):
    """Fitted eval RENI++ envmap on the equirectangular sampler grid.

    Returns (ldr [H, W, 3], hdr_mean [H, W], directions [H, W, 3] world frame).
    Unlike _common.render_envmap this reads the EVAL latent bank (the model is
    in eval mode and not fitting, so get_illumination_field returns it).
    """
    import torch
    from reni.field_components.field_heads import RENIFieldHeadNames
    from reni.utils.colourspace import linear_to_sRGB

    with torch.no_grad():
        ray_samples = model.equirectangular_sampler.generate_direction_samples()
        ray_samples = ray_samples.to(model.device)
        ray_samples.camera_indices = (
            torch.ones_like(ray_samples.camera_indices) * session_idx
        )
        latents, scales = model.get_illumination_field()
        env_scales = scales[ray_samples.camera_indices[:, 0]]
        two_bracket = getattr(model.illumination_hdr_decode, "two_bracket", False)
        outputs = model.illumination_field(
            ray_samples=ray_samples,
            latent_codes=latents[ray_samples.camera_indices[:, 0]],
            # two-bracket priors take exposure post-blend (zeros = exp-neutral in-field)
            scale=torch.zeros_like(env_scales) if two_bracket else env_scales,
        )
        hdr = model.illumination_hdr_decode.to_linear_hdr(
            model.illumination_field, outputs[RENIFieldHeadNames.RGB],
            scale=env_scales if two_bracket else None,
        )
        ldr = linear_to_sRGB(hdr).clamp(0, 1)
    height = model.equirectangular_sampler.height
    width = model.equirectangular_sampler.width
    directions = ray_samples.frustums.directions.reshape(height, width, 3).cpu()
    return (ldr.reshape(height, width, 3).cpu().numpy(),
            hdr.mean(dim=-1).reshape(height, width).cpu(),
            directions)


def session_rotation(metadata, session_name: str):
    """R_total mapping panorama dirs -> model world frame, plus the pano path.

    Mirrors NeuSkyDataManager.get_nerfosr_envmap_lighting_optimisation_bundle:
    d_model = orientation_transform[:3, :3] @ R_json @ d_pano, with R_json from
    <scene_dir>/envmap_rotations.json (scripts/register_envmaps.py output).
    """
    import torch

    scene_dir = metadata["scene_dir"]
    rotations_path = Path(scene_dir) / "envmap_rotations.json"
    entry = json.loads(rotations_path.read_text())[session_name]
    r_json = torch.tensor(entry["rotation"], dtype=torch.float32)
    transform = metadata.get("orientation_transform")
    r_orient = (torch.eye(3) if transform is None
                else torch.as_tensor(transform)[:3, :3].float())

    filenames = sorted(
        f for f in metadata["envmap_filenames"] if f"{os.sep}{session_name}{os.sep}" in f
    )
    if entry.get("envmap_image") is not None:
        pinned = os.path.basename(entry["envmap_image"])
        filenames = [f for f in filenames if os.path.basename(f) == pinned] or filenames
    return r_orient @ r_json, filenames[0]


def resample_gt_erp(pano_path, directions, r_total, saturation_scale: float = 10.0,
                    saturation_threshold: float = 0.95):
    """Sample the session panorama at the fitted-envmap grid directions.

    directions: [H, W, 3] model-frame grid from render_fitted_envmap, so this
    column is pixel-aligned with the fitted one by construction. Pano pixel
    lookup follows scripts/register_envmaps.py: d_pano = R_total^T d_world,
    lon = atan2(x, z), lat = asin(y). Tonemapped like the fitted column
    (NeRF-OSR pseudo-HDR scaling then linear->sRGB).

    Returns (ldr [H, W, 3] numpy, hdr_mean [H, W] tensor).
    """
    import numpy as np
    import torch
    from PIL import Image
    from reni.utils.colourspace import linear_to_sRGB

    h, w = directions.shape[:2]
    pano = torch.from_numpy(
        np.asarray(Image.open(pano_path).convert("RGB"), dtype="float32") / 255.0
    ).permute(2, 0, 1)[None]  # [1, 3, Hp, Wp]
    # Antialias to ~4x the output grid before the bilinear lookup.
    pano = torch.nn.functional.interpolate(pano, size=(4 * h, 8 * w), mode="area")

    d_pano = directions.reshape(-1, 3) @ r_total  # rows: R_total^T @ d_world
    lon = torch.atan2(d_pano[:, 0], d_pano[:, 2])
    lat = torch.asin(d_pano[:, 1].clamp(-1.0, 1.0))
    gx = (lon / math.pi).reshape(1, h, w, 1)            # [-1, 1] across width
    gy = (lat / (math.pi / 2)).reshape(1, h, w, 1)      # [-1, 1] down height
    grid = torch.cat([gx, gy], dim=-1)
    sampled = torch.nn.functional.grid_sample(
        pano, grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0].permute(1, 2, 0)  # [H, W, 3]

    sampled[sampled > saturation_threshold] *= saturation_scale
    return (linear_to_sRGB(sampled).clamp(0, 1).numpy(),
            sampled.mean(dim=-1))


def bright_region_azimuth(hdr_mean, directions, top_frac: float = 0.02):
    """Luminance-weighted mean azimuth (deg) of the brightest upper-hemisphere
    pixels -- a sun proxy robust to saturated clouds and soft RENI maxima."""
    import torch

    flat = hdr_mean.flatten()
    dirs = directions.reshape(-1, 3)
    vals = flat.clone()
    vals[dirs[:, 2] <= 0] = -torch.inf  # upper hemisphere only
    idx = vals.topk(max(1, int(top_frac * flat.numel()))).indices
    weights = flat[idx].clamp(min=0)[:, None]
    centroid = (dirs[idx] * weights).sum(0)
    return math.degrees(math.atan2(float(centroid[0]), float(centroid[1])))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "holdout_protocol_lk2")
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--render-scale", type=float, default=0.5,
                        help="Resolution scale for the view renders")
    from make_tables import EVAL_FITS, apply_eval_fit
    parser.add_argument("--eval-fit", choices=list(EVAL_FITS), default="eccv",
                        help="Eval-latent fit recipe (F11: 'hardened' adds the latent "
                             "prior + gentler lr schedule). Pass a distinct --output "
                             "so the default figure is not overwritten.")
    parser.add_argument("--components", type=Path, default=None,
                        help="Also write per-session holdout decomposition strips "
                             "(GT | fit render | RENI sky only | accumulation) to "
                             "this directory — separates illumination-fit error "
                             "from geometry/floater occlusion.")
    args = parser.parse_args()
    seed_all(args.seed)

    import numpy as np
    import torch

    def hook(config):
        config.pipeline.model.eval_latent_optimise_method = "nerf_osr_holdout"
        apply_eval_fit(config, args.eval_fit)
        config.pipeline.datamanager.dataparser.session_holdout_indices = \
            SESSION_HOLDOUT_INDICES[args.scene]

    _, pipeline, _, step = load_model(args.scene, device=args.device, step=args.step,
                                      config_hook=hook, eval_num_rays_per_chunk=1024)
    dm = pipeline.datamanager
    model = pipeline.model
    model.eval()

    # The protocol's latent fit (250 steps, seeded, canonical holdouts).
    model.fit_latent_codes_for_eval(dm, step)

    meta = dm.eval_dataset.metadata
    session_to_indices = meta["session_to_indices"]
    session_names = meta["session_names"]
    canonical = SESSION_HOLDOUT_INDICES[args.scene]
    compare_of = {dm.indices_to_session[i]: i for i in dm.eval_dataset.test_eval_mask_dict}
    image_filenames = dm.eval_dataset._dataparser_outputs.image_filenames

    sessions = sorted(session_to_indices)
    columns = ["Holdout", "Fit", "Fitted env.", "GT ERP", "Target", "Relit"]
    fig, axs = plt.subplots(len(sessions), len(columns),
                            figsize=(3.2 * len(columns), 1.9 * len(sessions)),
                            constrained_layout=True)

    for r, s in enumerate(sessions):
        name = session_names[s]
        # canonical[s] may be list-form (two-view holdout); the protocol figure
        # shows one holdout column, so display the primary (first) holdout view.
        first_holdout = canonical[s]
        first_holdout = first_holdout[0] if isinstance(first_holdout, (list, tuple)) else first_holdout
        holdout_idx = session_to_indices[s][first_holdout]
        compare_idx = compare_of[s]

        fitted_ldr, fitted_hdr, directions = render_fitted_envmap(model, s)
        r_total, pano_path = session_rotation(meta, name)
        gt_ldr, gt_hdr = resample_gt_erp(pano_path, directions, r_total)

        yaw_fit = bright_region_azimuth(fitted_hdr, directions)
        yaw_gt = bright_region_azimuth(gt_hdr, directions)
        dyaw = (yaw_fit - yaw_gt + 180.0) % 360.0 - 180.0
        print(f"[sun] {name}: bright-region azimuth fitted {yaw_fit:.0f} deg, "
              f"GT ERP {yaw_gt:.0f} deg, yaw offset {dyaw:+.1f} deg")

        cells = [
            load_photo(image_filenames[holdout_idx]),
            render_view(model, dm, holdout_idx, s, step, args.render_scale),
            fitted_ldr,
            gt_ldr,
            load_photo(image_filenames[compare_idx]),
            render_view(model, dm, compare_idx, s, step, args.render_scale),
        ]
        for c, img in enumerate(cells):
            axs[r, c].imshow(np.clip(img, 0, 1))
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(columns[c], fontsize=11)
        axs[r, 0].set_ylabel(session_label(name), fontsize=9)
        print(f"[row] {name}: holdout={Path(image_filenames[holdout_idx]).stem} "
              f"compare={Path(image_filenames[compare_idx]).stem}")

        if args.components is not None:
            comp = render_view(model, dm, holdout_idx, s, step, args.render_scale,
                               components=True)
            print(f"[components] {name}: sky sRGB saturated on "
                  f"{100 * comp['sky_saturated_frac']:.1f}% of rays")
            strip_cols = ["Holdout", "Fit render", "RENI sky only", "Accumulation"]
            strip_imgs = [cells[0], comp["rgb"], comp["sky_srgb"],
                          np.repeat(comp["accumulation"], 3, axis=-1)]
            sfig, saxs = plt.subplots(1, 4, figsize=(3.2 * 4, 2.1),
                                      constrained_layout=True)
            for c, (title, img) in enumerate(zip(strip_cols, strip_imgs)):
                saxs[c].imshow(np.clip(img, 0, 1))
                saxs[c].set_xticks([]); saxs[c].set_yticks([])
                saxs[c].set_title(title, fontsize=10)
            args.components.mkdir(parents=True, exist_ok=True)
            save_figure(sfig, args.components / f"holdout_components_{name}", svg=False)
            plt.close(sfig)

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
