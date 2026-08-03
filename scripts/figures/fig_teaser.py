"""NeuSky teaser (working output -> outputs/figures/teaser.{svg,png,pdf}).

Programmatic reproduction of the TPAMI/thesis teaser composite
(latex/9_Chapter3/figures/teaser.pdf), left to right:

1. a collage of slightly-rotated training photos (one per capture session, so
   the illumination visibly varies);
2. the scene render sitting inside two concentric hemisphere arcs (inner =
   DDF sphere, outer = distant illumination sphere), a camera frustum at the
   lower left, a blue camera ray with orange sample dots reaching the outer
   sphere, and the two latent squares (DDF latent <-> RENI++ latent) joined
   by a red double-headed arrow;
3. a middle column of model outputs: a 3x2 grid of spherical DDF depth maps
   (turbo colormap), albedo + camera-space normal renders, and the fitted
   RENI++ environment map next to its HDR-luminance "sun" heatmap;
4. two renders of the scene relit with *other training sessions'* RENI++
   latents (the notebook's latent-swap recipe -- the June figures showed the
   fixed-envmap swap path renders blown out on the lk2 refit, so latent swaps
   are used for both relit panels, matching the RENI-decoded insets of the
   original teaser), each with its decoded envmap inset, plus a small
   camera-pair icon (novel view + relighting).

The figure is emitted as an SVG that embeds the rendered panels as base64
PNGs with all annotation (arcs, arrows, frustum, latent squares) as native
vector elements, then rasterised to PNG/PDF via rsvg-convert (or cairosvg).

Two-stage workflow so composition can iterate without re-rendering:

    # render panel PNGs into outputs/figures/teaser_panels/ (GPU)
    PYTHONPATH=.:../ns_reni python scripts/figures/fig_teaser.py --panels-only \
        --scale 0.5 --chunk 512
    # compose the SVG/PNG/PDF from cached panels (CPU only)
    PYTHONPATH=. python scripts/figures/fig_teaser.py --compose-only

Running with neither flag does both stages. --scale 0.5 renders at half
resolution (smoke test); --scale 1.0 for the final pass. The GPU may be
shared with a training job: keep --chunk small (the training configs eval at
256 rays/chunk because every camera ray spawns 512 illumination-direction
DDF queries) and reduce it further on CUDA OOM rather than retrying.
"""

import argparse
import base64
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from _common import (FIGURES_DIR, NERF_OSR_ROOT, SCENE_DEFAULT_VIEWS,
                     load_model, normal_to_camera_vis, render_view,
                     resolve_run_dir, seed_all, sphere_camera_ray_bundle)

# Per-scene teaser choices (train view indices / illumination latent indices).
# lk2: view 143 is the notebook-documented "good" view; latent 53 is the
# notebook's bright-overcast relighting session (verified in the June
# further_relighting_examples figure); latent 70 sits in the sunny midday
# 09-04_13_00 session. View 120 gives a second, different viewpoint.
TEASER_DEFAULTS = {
    "lk2": {
        # top = 12-04_10_00_DSC_0374 relit with another session's latent;
        # bottom = C5_DSC_8_4 under its own training latent (novel view)
        "relight_views": (92, 148),
        "relight_latents": (20, 148),
        "view": 92,   # decomposition view = the top-right relit view
        "hero_latent": 70,
    },
}

SESSION_RE = re.compile(r"^(\d{2}-\d{2}_\d{2}_\d{2})")

# Base (scale=1.0) render sizes. Views render at the training-camera
# resolution scaled by --scale; the hero / DDF sphere cameras use the
# DDFDataset eval intrinsics (fx=fy=1007 at 1280x822).
HERO_BASE = dict(height=822, width=1280, focal=1007.0)
DDF_TILE_BASE = dict(height=411, width=640, focal=503.5)


# ---------------------------------------------------------------------------
# Stage 1: panels (GPU)
# ---------------------------------------------------------------------------

def save_png(arr01, path: Path):
    """Save a float [0,1] HxWx3 array as PNG."""
    import numpy as np
    from PIL import Image

    arr = (np.clip(arr01, 0.0, 1.0) * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)
    print(f"[panel] {path}")


def save_collage_panels(scene: str, panels_dir: Path, width: int = 480):
    """One training photo per dated capture session (varying illumination)."""
    from PIL import Image

    rgb_dir = NERF_OSR_ROOT / scene / "final" / "train" / "rgb"
    sessions = {}
    for f in sorted(rgb_dir.iterdir()):
        m = SESSION_RE.match(f.name)
        if m:
            sessions.setdefault(m.group(1), []).append(f)
    if not sessions:
        raise FileNotFoundError(f"No session-named training images in {rgb_dir}")
    picks = [files[len(files) // 2] for _, files in sorted(sessions.items())][:5]
    names = []
    for i, f in enumerate(picks):
        img = Image.open(f).convert("RGB")
        h = round(width * img.height / img.width)
        img = img.resize((width, h), Image.LANCZOS)
        out = panels_dir / f"collage_{i}.png"
        img.save(out)
        print(f"[panel] {out}  (source {f.name})")
        names.append(f.name)
    return names


def sphere_position(radius: float, azimuth_deg: float, elevation_deg: float):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return [radius * math.cos(az) * math.cos(el),
            radius * math.sin(az) * math.cos(el),
            radius * math.sin(el)]


def render_hero(model, device, latent_idx: int, azimuth: float, elevation: float,
                scale: float, acc_threshold: float, depth_trim: float):
    """Whole-scene render from the DDF sphere, floated on white.

    Sky pixels (low accumulation) and far ground (termination distance beyond
    depth_trim * ddf_radius) render as white, giving the teaser's
    scene-inside-the-spheres look.
    """
    import numpy as np
    import torch

    radius = float(model.visibility_field.ddf_radius)
    H = max(8, round(HERO_BASE["height"] * scale))
    W = max(8, round(HERO_BASE["width"] * scale))
    ray_bundle = sphere_camera_ray_bundle(
        sphere_position(radius, azimuth, elevation), device,
        height=H, width=W, focal=HERO_BASE["focal"] * (W / HERO_BASE["width"]))
    ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * latent_idx
    model.viewing_training_image = True
    try:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle=ray_bundle, show_progress=True)
    finally:
        model.viewing_training_image = False
    rgb = outputs["rgb"].clamp(0, 1).cpu().numpy()
    acc = outputs["accumulation"].reshape(H, W).cpu().numpy()
    white = acc < acc_threshold
    depth_key = "p2p_dist" if "p2p_dist" in outputs else "depth"
    if depth_trim > 0 and depth_key in outputs:
        depth = outputs[depth_key].reshape(H, W).cpu().numpy()
        white |= depth > depth_trim * radius
    rgb[white] = 1.0
    return np.clip(rgb, 0, 1)


def render_ddf_tile(model, device, azimuth: float, elevation: float, scale: float,
                    sky_height: float = 0.4):
    """Spherical DDF expected termination distance -> turbo colormap tile.

    Sky rays are pushed to the top of the colormap (red, "far") and geometry
    depth is normalised over the remaining pixels -- the qualitative
    near-blue building / red sky look of the original teaser tiles. A ray
    counts as sky if it terminates near its sphere-exit chord (for |origin|
    = r and unit direction d the exit is at t = -2 origin . d) or if its
    termination point sits above sky_height * radius (steep upward rays,
    where the DDF under-predicts the short chord).
    """
    import matplotlib
    import numpy as np
    import torch

    radius = float(model.visibility_field.ddf_radius)
    H = max(8, round(DDF_TILE_BASE["height"] * scale))
    W = max(8, round(DDF_TILE_BASE["width"] * scale))
    position = sphere_position(radius, azimuth, elevation)
    ray_bundle = sphere_camera_ray_bundle(
        position, device,
        height=H, width=W, focal=DDF_TILE_BASE["focal"] * (W / DDF_TILE_BASE["width"]))
    with torch.no_grad():
        outputs = model.visibility_field.get_outputs_for_camera_ray_bundle(
            ray_bundle, neusky=None, show_progress=True)
    depth = outputs["expected_termination_dist"].reshape(H, W).cpu().numpy()

    dirs = ray_bundle.directions.reshape(H, W, 3).cpu().numpy()
    origin = np.asarray(position, dtype=np.float32)
    chord = np.clip(-2.0 * (dirs @ origin), 0.0, None)
    termination_z = origin[2] + depth * dirs[..., 2]
    sky = (depth > 0.92 * chord) | (termination_z > sky_height * radius)

    geometry = depth[~sky]
    if geometry.size > 32:
        vmin, vmax = np.percentile(geometry, [2.0, 98.0])
    else:
        vmin, vmax = 0.0, 2.0 * radius
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 0.92)
    norm[sky] = 1.0
    return matplotlib.colormaps["turbo"](norm)[..., :3]


def render_envmap_pair(model, illumination_idx: int, width: int = 256,
                       chunk: int = 4096):
    """(ldr, hdr) [H, W, 3] decode of a fitted RENI++ latent.

    Same recipe as _common.render_envmap but at a figure-friendly resolution
    (the model's own equirectangular sampler is only 64x128), decoded in
    chunks and also returning the HDR values for the sun-visibility heatmap.

    NOTE: does not call sampler.generate_direction_samples() -- that helper
    passes starts/ends/pixel_area as [N] instead of [N, 1], so the frustum
    TensorDataclass broadcasts them to [N, N] (4 GiB at width=256). The lean
    RaySamples below carries the same information in O(N).
    """
    import torch
    from nerfstudio.cameras.rays import Frustums, RaySamples
    from reni.field_components.field_heads import RENIFieldHeadNames
    from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
    from reni.utils.colourspace import linear_to_sRGB

    sampler = EquirectangularSamplerConfig(width=width).setup()
    with torch.no_grad():
        model.viewing_training_image = True
        try:
            latents, scales = model.get_illumination_field()
            ray_bundle = sampler.camera.generate_rays(camera_indices=0,
                                                      keep_shape=False)
            directions = ray_bundle.directions
            ray_samples = RaySamples(
                frustums=Frustums(
                    origins=torch.zeros_like(directions),
                    directions=directions,
                    starts=torch.zeros_like(directions[:, :1]),
                    ends=torch.ones_like(directions[:, :1]),
                    pixel_area=torch.ones_like(directions[:, :1]),
                ),
                camera_indices=torch.full_like(
                    ray_bundle.camera_indices, illumination_idx),
            ).to(model.device)
            rgb_chunks = []
            for i in range(0, ray_samples.shape[0], chunk):
                sliced = ray_samples[i:i + chunk]
                outputs = model.illumination_field(
                    ray_samples=sliced,
                    latent_codes=latents[sliced.camera_indices[:, 0]],
                    scale=scales[sliced.camera_indices[:, 0]],
                    rotation=None,
                )
                rgb_chunks.append(outputs[RENIFieldHeadNames.RGB])
        finally:
            model.viewing_training_image = False
        raw = torch.cat(rgb_chunks)
        # two-bracket priors emit 6ch brackets; decode to linear HDR (the
        # 3ch path defers to unnormalise, unchanged)
        hdr = model.illumination_hdr_decode.to_linear_hdr(
            model.illumination_field, raw)
        ldr = linear_to_sRGB(hdr)
        H, W = sampler.height, sampler.width
        return (ldr.reshape(H, W, 3).cpu().numpy(),
                hdr.reshape(H, W, 3).cpu().numpy())


def envmap_heatmap(hdr):
    """HDR luminance -> viridis; dark sky, bright sun blob (teaser style)."""
    import matplotlib
    import numpy as np

    # log-HDR luminance (matches the corrected Ch2 comparison-figure cells)
    lum = np.log1p(np.clip(hdr, 0.0, None).mean(axis=-1))
    vmax = max(float(np.percentile(lum, 99.9)), 1e-6)
    return matplotlib.colormaps["viridis"](np.clip(lum / vmax, 0.0, 1.0))[..., :3]


def stage_panels(args):
    import copy

    import numpy as np
    import torch

    seed_all(args.seed)
    panels_dir = args.panels_dir
    panels_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene": args.scene,
        "scale": args.scale,
        "run_dir": str(resolve_run_dir(args.scene)),
        "collage_sources": save_collage_panels(args.scene, panels_dir),
    }

    def _low_vram_hook(config):
        # The training configs cache every train/eval image on the GPU
        # (images_on_gpu / masks_on_gpu true, eval_num_images_to_sample_from
        # inf) -- several GB we cannot spare while a training job shares the
        # card, and the figure only ever renders train views. Keep image
        # caches on the CPU and cache only a handful of eval images.
        dm = config.pipeline.datamanager
        dm.images_on_gpu = False
        dm.masks_on_gpu = False
        dm.eval_num_images_to_sample_from = 4

    _, pipeline, _, step = load_model(
        args.scene, device=args.device, step=args.step,
        eval_num_rays_per_chunk=args.chunk, config_hook=_low_vram_hook)
    manifest["step"] = step
    datamanager = pipeline.datamanager
    model = pipeline.model
    model.eval()
    assert model.visibility_field is not None, "Run has no DDF visibility field"

    defaults = TEASER_DEFAULTS.get(args.scene, {})
    dec_view = args.view if args.view is not None else defaults.get(
        "view", SCENE_DEFAULT_VIEWS.get(args.scene, 0))
    relight_views = args.relight_views or defaults.get("relight_views", (dec_view, dec_view))
    n_train = len(datamanager.train_dataset)
    relight_latents = args.relight_latents or defaults.get(
        "relight_latents", (0, n_train // 2))
    hero_latent = args.hero_latent if args.hero_latent is not None else defaults.get(
        "hero_latent", dec_view)
    # The envmap/heatmap panel defaults to the decomposition view's own
    # fitted illumination: the middle column then reads as one view's full
    # decomposition (albedo / normals / illumination), and the panel stays
    # distinct from the relit-render insets.
    envmap_latent = args.envmap_latent if args.envmap_latent is not None else dec_view
    manifest.update(view=dec_view, relight_views=list(relight_views),
                    relight_latents=list(relight_latents),
                    hero_latent=hero_latent, envmap_latent=envmap_latent)

    cameras = copy.deepcopy(datamanager.train_dataset.cameras)
    if args.scale != 1.0:
        cameras.rescale_output_resolution(scaling_factor=args.scale)

    def fresh(*names):
        if not args.skip_existing:
            return True
        if all((panels_dir / n).exists() for n in names):
            print(f"[skip] {', '.join(names)} (cached)")
            return False
        return True

    # -- hero: the scene inside the spheres ---------------------------------
    # James's pre-rendered 3D view (assets/teaser_3d_view.png) replaces the
    # sphere-camera render: exact camera alignment without 3D fiddling.
    hero_asset = Path(__file__).resolve().parent / "assets" / "teaser_3d_view.png"
    if fresh("hero.png"):
        if hero_asset.exists():
            print(f"[asset] hero <- {hero_asset.name}")
            from PIL import Image as _Im
            import numpy as _np
            from scipy import ndimage as _ndi
            im = _np.asarray(_Im.open(hero_asset).convert("RGB")).copy()
            h_, w_ = im.shape[:2]
            corner = im[: int(0.14 * h_), : int(0.10 * w_)]
            corner[corner.mean(-1) < 80] = 255   # viewport wedge -> white
            # white background -> transparent so the arcs show through:
            # near-white region connected to the image border
            near_white = im.min(-1) > 235
            lab, n = _ndi.label(near_white)
            border_labels = set(lab[0].tolist()) | set(lab[-1].tolist()) \
                | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
            border_labels.discard(0)
            bg = _np.isin(lab, list(border_labels))
            rgba = _np.dstack([im, _np.where(bg, 0, 255).astype(_np.uint8)])
            _Im.fromarray(rgba, "RGBA").save(panels_dir / "hero.png")
        else:
            print(f"[render] hero (azimuth {args.hero_azimuth}, latent {hero_latent})")
            hero = render_hero(model, args.device, hero_latent, args.hero_azimuth,
                               args.hero_elevation, args.scale,
                               args.hero_acc_threshold, args.hero_depth_trim)
            save_png(hero, panels_dir / "hero.png")
            torch.cuda.empty_cache()

    # -- frustum image plane: fixed training photo (original teaser view) ----
    if fresh("frustum.png"):
        from PIL import Image as _Im
        src = Path("/home/james/data/NeRF-OSR/Data/lk2/final/train/rgb/C5_DSC_13_5.png")
        im = _Im.open(src).convert("RGB")
        im.thumbnail((480, 480))
        im.save(panels_dir / "frustum.png")
        print(f"[asset] frustum <- {src.name}")

    # -- DDF spherical depth tiles ------------------------------------------
    for i, azimuth in enumerate(args.ddf_azimuths):
        if not fresh(f"ddf_{i}.png"):
            continue
        print(f"[render] ddf tile {i} (azimuth {azimuth})")
        tile = render_ddf_tile(model, args.device, azimuth, args.ddf_elevation,
                               args.scale)
        save_png(tile, panels_dir / f"ddf_{i}.png")
    torch.cuda.empty_cache()

    # -- decomposition view: albedo + camera-space normals ------------------
    if fresh("albedo.png", "normal.png"):
        print(f"[render] decomposition view {dec_view}")
        outputs = render_view(model, cameras, dec_view, args.device)
        save_png(outputs["albedo"].cpu().numpy(), panels_dir / "albedo.png")
        c2w = cameras.camera_to_worlds[dec_view]
        save_png(normal_to_camera_vis(outputs["normal"], c2w),
                 panels_dir / "normal.png")
        del outputs
        torch.cuda.empty_cache()

    # -- fitted environment map + sun heatmap -------------------------------
    if fresh("envmap.png", "envmap_heat.png"):
        print(f"[render] envmap (latent {envmap_latent})")
        ldr, hdr = render_envmap_pair(model, envmap_latent)
        save_png(ldr, panels_dir / "envmap.png")
        save_png(envmap_heatmap(hdr), panels_dir / "envmap_heat.png")

    # -- relit renders (latent swaps) + their decoded envmap insets ---------
    for name, view_idx, latent_idx in zip(
            ("a", "b"), relight_views, relight_latents):
        if not fresh(f"relit_{name}.png", f"envmap_{name}.png"):
            continue
        print(f"[render] relit_{name}: view {view_idx}, latent {latent_idx}")
        outputs = render_view(model, cameras, view_idx, args.device,
                              illumination_idx=latent_idx)
        save_png(outputs["rgb"].cpu().numpy(), panels_dir / f"relit_{name}.png")
        ldr, _ = render_envmap_pair(model, latent_idx)
        save_png(ldr, panels_dir / f"envmap_{name}.png")
        del outputs
        torch.cuda.empty_cache()

    with open(panels_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[panels] manifest -> {panels_dir / 'manifest.json'}")


# ---------------------------------------------------------------------------
# Stage 2: compose (CPU only -- no torch)
# ---------------------------------------------------------------------------

RAY_BLUE = "#1533cc"
DOT_ORANGE = "#ff8c00"
SQUARE_FILL = "#5878a8"
ARROW_RED = "#e01212"
ARC_GREY = "#7a7a7a"
INK = "#1a1a1a"
LABEL_FONT = "Helvetica, Arial, sans-serif"


def _pt(cx, cy, r, deg):
    """Point on a circle, math convention (deg CCW from +x, y up on screen)."""
    t = math.radians(deg)
    return cx + r * math.cos(t), cy - r * math.sin(t)


def arc_path(cx, cy, r, deg0, deg1, stroke=ARC_GREY, width=2.5):
    """Arc from deg0 to deg1 travelling over the top (deg0 > 90 > deg1)."""
    x0, y0 = _pt(cx, cy, r, deg0)
    x1, y1 = _pt(cx, cy, r, deg1)
    large = 1 if (deg0 - deg1) > 180 else 0
    return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 {large} 1 '
            f'{x1:.1f} {y1:.1f}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}"/>')


def arrow(x1, y1, x2, y2, color=INK, width=2.2, marker="arrow", dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{dash} '
            f'marker-end="url(#{marker})"/>')


def curve_arrow(d, color=INK, width=2.2, marker="arrow", dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{dash} marker-end="url(#{marker})"/>')


def double_arrow_red(x1, x2, y, width=4.0, head=11.0):
    """Horizontal red double-headed arrow with manual triangle heads."""
    parts = [
        f'<line x1="{x1 + head:.1f}" y1="{y:.1f}" x2="{x2 - head:.1f}" '
        f'y2="{y:.1f}" stroke="{ARROW_RED}" stroke-width="{width}"/>',
        f'<polygon points="{x1:.1f},{y:.1f} {x1 + head:.1f},{y - head * 0.62:.1f} '
        f'{x1 + head:.1f},{y + head * 0.62:.1f}" fill="{ARROW_RED}"/>',
        f'<polygon points="{x2:.1f},{y:.1f} {x2 - head:.1f},{y - head * 0.62:.1f} '
        f'{x2 - head:.1f},{y + head * 0.62:.1f}" fill="{ARROW_RED}"/>',
    ]
    return "\n".join(parts)


def camera_glyph(x, y, s=1.0, stroke=INK):
    """Small "camera" glyph (body + lens trapezoid), top-left at (x, y)."""
    bw, bh = 20 * s, 13 * s
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
        f'rx="{2 * s:.1f}" fill="#fff" stroke="{stroke}" stroke-width="{1.6 * s:.1f}"/>'
        f'<polygon points="{x + bw:.1f},{y + 2.5 * s:.1f} '
        f'{x + bw + 7 * s:.1f},{y:.1f} {x + bw + 7 * s:.1f},{y + bh:.1f} '
        f'{x + bw:.1f},{y + bh - 2.5 * s:.1f}" fill="#fff" stroke="{stroke}" '
        f'stroke-width="{1.6 * s:.1f}"/>')


def label_text(x, y, text, size=15, anchor="middle", color="#444"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="{LABEL_FONT}" font-size="{size}" '
            f'fill="{color}">{text}</text>')


def stage_compose(args):
    from PIL import Image

    panels_dir = args.panels_dir
    if not (panels_dir / "hero.png").exists():
        raise SystemExit(
            f"No cached panels under {panels_dir}; run the panels stage first "
            f"(drop --compose-only).")

    sizes, uris = {}, {}

    def load(name):
        path = panels_dir / name
        with Image.open(path) as img:
            sizes[name] = img.size
        uris[name] = ("data:image/png;base64,"
                      + base64.b64encode(path.read_bytes()).decode("ascii"))

    panel_names = ([f"collage_{i}.png" for i in range(5)]
                   + [f"ddf_{i}.png" for i in range(len(args.ddf_azimuths))]
                   + ["hero.png", "frustum.png", "albedo.png", "normal.png",
                      "envmap.png", "envmap_heat.png", "relit_a.png",
                      "relit_b.png", "envmap_a.png", "envmap_b.png"])
    for name in panel_names:
        load(name)

    defs, body = [], []
    clip_n = [0]

    def panel(name, x, y, w, rot=0.0, rx=5.0, border="#fff", border_w=0.0):
        """Embed a panel PNG at width w (height from aspect). Returns h."""
        iw, ih = sizes[name]
        h = w * ih / iw
        clip_n[0] += 1
        cid = f"clip{clip_n[0]}"
        defs.append(f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" '
                    f'width="{w:.1f}" height="{h:.1f}" rx="{rx}" ry="{rx}"/>'
                    f'</clipPath>')
        transform = (f' transform="rotate({rot:.1f} {x + w / 2:.1f} '
                     f'{y + h / 2:.1f})"' if rot else "")
        elems = [f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" clip-path="url(#{cid})" '
                 f'href="{uris[name]}"/>']
        if border_w > 0:
            elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                         f'height="{h:.1f}" rx="{rx}" ry="{rx}" fill="none" '
                         f'stroke="{border}" stroke-width="{border_w}"/>')
        body.append(f'<g{transform}>' + "".join(elems) + "</g>")
        return h

    # Arrow markers (triangle tip pulled back so lines never poke past it).
    defs.append(
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="7" '
        'refY="3.2" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L9,3.2 L0,6.4 z" fill="{INK}"/></marker>'
        '<marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="6" '
        'refY="2.6" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L7.2,2.6 L0,5.2 z" fill="{RAY_BLUE}"/></marker>'
        '<marker id="arrow-thin" markerWidth="12" markerHeight="12" refX="8" '
        'refY="3.2" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L9,3.2 L0,6.4 z" fill="{INK}"/></marker>')

    CANVAS_W, CANVAS_H = 2380, 800
    body.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" '
                f'fill="#ffffff"/>')

    # ---- 1. training-photo collage -----------------------------------------
    collage_w = 250
    collage_spots = [  # (x, y, rotation) -- centre photo drawn last, on top
        (55, 267, -8.0), (215, 232, 6.0), (30, 442, 5.0), (215, 432, -5.0),
        (120, 342, 1.5),
    ]
    for i, (cx_, cy_, rot) in enumerate(collage_spots):
        panel(f"collage_{i}.png", cx_, cy_, collage_w, rot=rot, rx=3,
              border="#fff", border_w=5)
    if args.labels:
        body.append(label_text(270, 585, "training images, varying illumination"))

    body.append(arrow(500, 420, 555, 420, width=3.0))

    # ---- 2. scene + spheres diagram ----------------------------------------
    # James's complete central asset: 3D view, arcs, frustum with warped
    # image, blue camera ray, and the two latent squares with their arrow.
    import re as _re
    asset = (Path(__file__).resolve().parent / "assets"
             / "teaser_3d_view_with_frustum.svg").read_text()
    asset = asset[asset.index("<svg"):]
    vb = asset[asset.index('viewBox="') + 9:]
    vb_w2, vb_h2 = float(vb.split()[2]), float(vb.split()[3].split('"')[0])
    A_X, A_W = 470, 690
    A_H = A_W * vb_h2 / vb_w2
    A_Y = 470 - A_H / 2
    open_end = asset.index(">")
    open_tag = _re.sub(r'\s(width|height|x|y)="[^"]*"', "", asset[:open_end + 1])
    open_tag = open_tag[:-1] + (
        f' x="{A_X}" y="{A_Y:.1f}" width="{A_W}" height="{A_H:.1f}" '
        f'preserveAspectRatio="xMidYMid meet">')
    body.append(open_tag + asset[open_end + 1:])

    # anchor points on the asset (fractions of its box)
    def apt(fx, fy):
        return A_X + fx * A_W, A_Y + fy * A_H
    tipx, tipy = apt(0.885, 0.20)      # blue-ray tip (curve to DDF grid)
    dsx, dsy = apt(0.900, 0.44)        # outer-arc right (dashed curve start)
    sq_rx, sq_ry = apt(0.960, 0.905)   # right latent square (arrow target)

    body.append(arrow(1160, 420, 1215, 420, width=3.0))

    # ---- 3. middle column: DDF depths / albedo+normals / envmap+heat -------
    col_x = 1250
    # (a) DDF spherical depth grid, 3x2
    tile_w, gap = 158, 6
    n_tiles = len(args.ddf_azimuths)
    tile_h = tile_w * sizes["ddf_0.png"][1] / sizes["ddf_0.png"][0]
    ddf_y = 62
    for i in range(n_tiles):
        r, c = divmod(i, 3)
        panel(f"ddf_{i}.png", col_x + c * (tile_w + gap),
              ddf_y + r * (tile_h + gap), tile_w, rx=0)
    ddf_h = 2 * tile_h + gap
    grid_w = 3 * tile_w + 2 * gap
    if args.labels:
        body.append(label_text(col_x + grid_w / 2, ddf_y + ddf_h + 22,
                               "spherical directional distance (DDF)", 13))

    # Curved arrow: blue-ray tip -> depth grid.
    body.append(curve_arrow(
        f"M {tipx:.1f} {tipy:.1f} C {tipx + 90:.1f} {tipy - 60:.1f}, "
        f"{col_x - 170} {ddf_y + ddf_h / 2 - 20:.1f}, "
        f"{col_x - 12} {ddf_y + ddf_h / 2:.1f}",
        width=2.6))

    # (b) albedo + camera-space normals
    dec_w = 242
    dec_y = ddf_y + ddf_h + 62
    dec_h = panel("albedo.png", col_x, dec_y, dec_w, rx=4)
    panel("normal.png", col_x + dec_w + 8, dec_y, dec_w, rx=4)
    if args.labels:
        body.append(label_text(col_x + dec_w + 4, dec_y + dec_h + 22,
                               "albedo / normals", 13))

    # (c) fitted RENI++ envmap + HDR-luminance sun heatmap
    env_w = 242
    env_y = dec_y + dec_h + 55
    env_h = panel("envmap.png", col_x, env_y, env_w, rx=4)
    panel("envmap_heat.png", col_x + env_w + 8, env_y, env_w, rx=4)
    if args.labels:
        body.append(label_text(col_x + env_w + 4, env_y + env_h + 22,
                               "environment map / sun visibility", 13))

    # Dashed curve: outer illumination sphere curving DOWN to the RENI
    # envmap image (the sphere is what the envmap depicts); thin arrow:
    # envmap panel -> RENI latent square.
    body.append(curve_arrow(
        f"M {dsx:.1f} {dsy:.1f} C {dsx + 80:.1f} {dsy + 85:.1f}, "
        f"{col_x - 150} {env_y + env_h * 0.5:.1f}, "
        f"{col_x - 12} {env_y + env_h * 0.5:.1f}",
        width=1.8, dashed=True, marker="arrow-thin"))
    body.append(arrow(col_x - 6, env_y + env_h * 0.75, sq_rx + 8, sq_ry,
                      width=1.6, marker="arrow-thin"))

    body.append(arrow(1790, 420, 1845, 420, width=3.0))

    # ---- 4. relit renders under novel illumination -------------------------
    rel_x, rel_w = 1900, 460
    rel_a_y = 40
    rel_a_h = panel("relit_a.png", rel_x, rel_a_y, rel_w, rx=4)
    rel_b_y = rel_a_y + rel_a_h + 40
    rel_b_h = panel("relit_b.png", rel_x, rel_b_y, rel_w, rx=4)

    # Relighting panel (top): decoded-envmap inset. Novel-view panel
    # (bottom): camera-pair icon inset, matching the original teaser.
    inset_w = 128
    panel("envmap_a.png", rel_x - 14, rel_a_y - 16, inset_w, rx=3,
          border="#fff", border_w=3)

    # camera-transform icon: James's SVG asset, nested with a white card
    box_w = 96
    icon = (Path(__file__).resolve().parent / "assets"
            / "camera_transform.svg").read_text()
    icon = icon[icon.index("<svg"):]
    vb = icon[icon.index('viewBox="') + 9:]
    vb_w, vb_h = float(vb.split()[2]), float(vb.split()[3].split('"')[0])
    box_h = box_w * vb_h / vb_w + 14
    box_x, box_y = rel_x - 14, rel_b_y - 16
    body.append(f'<rect x="{box_x}" y="{box_y:.1f}" width="{box_w + 14}" '
                f'height="{box_h:.1f}" rx="5" fill="#fff" stroke="{INK}" '
                f'stroke-width="2"/>')
    # keep the asset's own <svg> tag (it declares the inkscape/sodipodi
    # namespaces its content uses); override placement and size on it
    import re as _re
    open_tag_end = icon.index(">")
    open_tag = icon[:open_tag_end + 1]
    rest = icon[open_tag_end + 1:]
    open_tag = _re.sub(r'\s(width|height|x|y)="[^"]*"', "", open_tag)
    open_tag = open_tag[:-1] + (
        f' x="{box_x + 7}" y="{box_y + 7:.1f}" width="{box_w}" '
        f'height="{box_h - 14:.1f}" preserveAspectRatio="xMidYMid meet">')
    body.append(open_tag + rest)
    if args.labels:
        body.append(label_text(rel_x + rel_w / 2, rel_b_y + rel_b_h + 24,
                               "novel view + relighting", 13))

    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        "<defs>", *defs, "</defs>", *body, "</svg>",
    ])

    out_stem = args.output
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = Path(f"{out_stem}.svg")
    svg_path.write_text(svg)
    print(f"[saved] {svg_path}")
    rasterize(svg_path, out_stem)


def rasterize(svg_path: Path, out_stem: Path, zoom: float = 1.0):
    """SVG -> PNG (+PDF) via rsvg-convert, falling back to cairosvg."""
    png_path, pdf_path = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf")
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run([rsvg, "-f", "png", "--zoom", str(zoom),
                        "-o", str(png_path), str(svg_path)], check=True)
        subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
                       check=True)
        print(f"[saved] {png_path} / {pdf_path}")
        return
    try:
        import cairosvg
    except ImportError:
        print("[warn] neither rsvg-convert nor cairosvg available; "
              "SVG written but not rasterised")
        return
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=zoom)
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    print(f"[saved] {png_path} / {pdf_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Render-resolution scale (0.5 for smoke tests)")
    parser.add_argument("--chunk", type=int, default=512,
                        help="eval_num_rays_per_chunk (reduce on CUDA OOM; "
                             "each camera ray spawns 512 illumination-"
                             "direction DDF queries, so the training configs "
                             "themselves eval at 256)")
    parser.add_argument("--panels-only", action="store_true",
                        help="Only render/cache the panel PNGs (GPU stage)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip renders whose panel PNGs are already "
                             "cached (resume an interrupted panels stage)")
    parser.add_argument("--compose-only", action="store_true",
                        help="Only compose the SVG from cached panels (CPU)")
    parser.add_argument("--panels-dir", type=Path,
                        default=FIGURES_DIR / "teaser_panels")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "teaser",
                        help="Output stem (no extension)")
    parser.add_argument("--view", type=int, default=None,
                        help="Decomposition train view (default: notebook view)")
    parser.add_argument("--relight-views", type=int, nargs=2, default=None,
                        help="Train views for the two relit panels")
    parser.add_argument("--relight-latents", type=int, nargs=2, default=None,
                        help="Train illumination indices for the relit panels")
    parser.add_argument("--hero-latent", type=int, default=None,
                        help="Illumination index for the central scene render")
    parser.add_argument("--envmap-latent", type=int, default=None,
                        help="Illumination index for the envmap/heatmap panel "
                             "(default: the decomposition view, so the middle "
                             "column is one view's full decomposition)")
    parser.add_argument("--hero-azimuth", type=float, default=245.0)
    parser.add_argument("--hero-elevation", type=float, default=30.0)
    parser.add_argument("--hero-acc-threshold", type=float, default=0.7,
                        help="Accumulation below this renders as white sky")
    parser.add_argument("--hero-depth-trim", type=float, default=1.35,
                        help="White out hero pixels with termination distance "
                             "beyond this multiple of the DDF radius "
                             "(floats the scene on white); <=0 disables")
    parser.add_argument("--ddf-azimuths", type=float, nargs="+",
                        default=[180.0, 225.0, 270.0, 315.0, 0.0, 45.0])
    parser.add_argument("--ddf-elevation", type=float, default=30.0)
    parser.add_argument("--labels", action="store_true",
                        help="Bake small text labels under each region "
                             "(the original teaser is label-free)")
    args = parser.parse_args()

    if args.panels_only and args.compose_only:
        raise SystemExit("--panels-only and --compose-only are exclusive")
    args.panels_dir = args.panels_dir.resolve()
    args.output = args.output.resolve()

    if not args.compose_only:
        stage_panels(args)
    if not args.panels_only:
        stage_compose(args)


if __name__ == "__main__":
    main()
