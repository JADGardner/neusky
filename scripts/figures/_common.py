"""Shared helpers for the NeuSky figure scripts.

Headless, deterministic, inference-driven figure generation (same philosophy
as ns_reni/scripts/figures and 360anything/scripts/sphere_jepa/figures): each
fig_*.py loads trained checkpoints, runs inference, and writes PNG+PDF
(+optional SVG) without any notebook or display dependency. Recipes are ported
from publication/figures_and_tables.ipynb and publication/render_animation.py
with the container-absolute /workspace/... paths replaced by the resolvers
below.

Run from the neusky repo root, e.g.:

    PYTHONPATH=. python scripts/figures/fig_sigmoid.py

Everything defaults sensibly for a standalone clone of this repo (repo-local
`data` / `model-storage` symlinks plus `outputs/`); env vars only override:

    NEUSKY_OUTPUTS          outputs root           (default <repo>/outputs)
    NERF_OSR_ROOT           NeRF-OSR data root     (default ~/data/NeRF-OSR/Data)
    NEUSKY_SYNTHETIC_ROOT   synthetic renders root (default ~/data/neusky_synthetic_data/renders)
    NEUSKY_RENI_PRIOR       RENI++ prior ckpt dir  (default <repo>/model-storage/
                            reni_paper_models/reni_plus_plus_models/latent_dim_100)
    NEUSKY_RUNS             pin scene -> run dir, e.g.
                            "lk2=outputs/lk2_refit_optimised/neusky/2026-06-11_121134;st=..."

Heavy imports (torch / nerfstudio / neusky) are deferred into functions so
this module — and the diagram-only scripts — import and run on CPU-only hosts.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = REPO_ROOT / "publication" / "figures"
TABLES_DIR = REPO_ROOT / "publication" / "tables"


def _env_path(name: str, *fallbacks: Path) -> Path:
    """First of $name, then the first existing fallback, then fallbacks[0]."""
    if os.environ.get(name):
        return Path(os.environ[name]).expanduser()
    for p in fallbacks:
        if Path(p).expanduser().exists():
            return Path(p).expanduser()
    return Path(fallbacks[0]).expanduser()


OUTPUTS_ROOT = _env_path("NEUSKY_OUTPUTS", REPO_ROOT / "outputs")
NERF_OSR_ROOT = _env_path(
    "NERF_OSR_ROOT",
    Path("~/data/NeRF-OSR/Data"),
    REPO_ROOT / "data" / "NeRF-OSR" / "Data",
)
SYNTHETIC_ROOT = _env_path(
    "NEUSKY_SYNTHETIC_ROOT",
    Path("~/data/neusky_synthetic_data/renders"),
    REPO_ROOT / "data" / "neusky_synthetic_data" / "renders",
)

# The NeRF-OSR dataparser maps the benchmark site names onto dataset dirs.
SITE_TO_SCENE = {"site1": "lk2", "site2": "st", "site3": "lwp"}
SCENE_TO_SITE = {v: k for k, v in SITE_TO_SCENE.items()}

# Per-session eval holdout indices used for the NeRF-OSR relighting protocol
# (from publication/figures_and_tables.ipynb / render_animation.py).
SESSION_HOLDOUT_INDICES = {
    "lk2": [0, 0, 0, 0, 0],
    "st": [1, 2, 2, 7, 9],
    "lwp": [0, 6, 6, 2, 11],
    "stjacob": [0, 0, 0],
}

# Synthetic eval scenes: dataset dir stem -> figure/table label.
SYNTHETIC_SCENES = {
    "abandoned_buildings_prepared": "Abandoned Buildings",
    "apartment_building_prepared": "Apartment Building",
    "arlanda_uppsala_cathedral_prepared": "Arlanda Cathedral",
    "glass_building_prepared": "Glass Building",
    "interstellar_house_prepared": "Interstellar House",
}

_SCENE_ALIASES = {
    **SITE_TO_SCENE,
    "arlanda_cathedral": "arlanda_uppsala_cathedral_prepared",
    "arlanda_uppsala_cathedral": "arlanda_uppsala_cathedral_prepared",
    **{s.removesuffix("_prepared"): s for s in SYNTHETIC_SCENES},
}

# Notebook-documented "good" train view indices per NeRF-OSR scene
# ("Site 1: Train: 143", "Site 3: Train: 63 / 90"; idx 5 used for st renders).
SCENE_DEFAULT_VIEWS = {"lk2": 143, "st": 5, "lwp": 90}


def canonical_scene(name: str) -> str:
    return _SCENE_ALIASES.get(name, name)


def _runs_map() -> dict:
    """Parse $NEUSKY_RUNS ("scene=path;scene2=path2", ';' or ',' separated)."""
    raw = os.environ.get("NEUSKY_RUNS", "")
    out = {}
    for entry in raw.replace(",", ";").split(";"):
        if "=" in entry:
            scene, path = entry.split("=", 1)
            out[canonical_scene(scene.strip())] = Path(path.strip()).expanduser()
    return out


def _timestamp_runs(method_root: Path):
    """Timestamped run dirs (containing checkpoints) under <exp>/<method>/."""
    for ts_dir in sorted(method_root.iterdir()) if method_root.is_dir() else []:
        if (ts_dir / "nerfstudio_models").is_dir() and list(
            (ts_dir / "nerfstudio_models").glob("step-*.ckpt")
        ):
            yield ts_dir


def resolve_run_dir(scene: str) -> Path:
    """Latest timestamped run dir for a scene.

    Handles both bare scene names (outputs/lk2/neusky/<ts>) and derived
    experiment names (outputs/lk2_refit_optimised/neusky/<ts>), plus the
    synthetic layout (outputs/synthetic/<scene>_prepared/neusky-synthetic/<ts>).
    $NEUSKY_RUNS pins a scene to an explicit run (or experiment) dir.
    """
    scene = canonical_scene(scene)

    pinned = _runs_map().get(scene)
    if pinned is not None:
        pinned = pinned if pinned.is_absolute() else REPO_ROOT / pinned
        if (pinned / "nerfstudio_models").is_dir():
            return pinned
        runs = [ts for m in sorted(pinned.iterdir()) for ts in _timestamp_runs(m)] \
            if pinned.is_dir() else []
        if runs:
            return sorted(runs, key=lambda p: p.name)[-1]
        raise FileNotFoundError(f"NEUSKY_RUNS pin for {scene} has no checkpoints: {pinned}")

    candidates = []
    for root in (OUTPUTS_ROOT, OUTPUTS_ROOT / "synthetic"):
        if not root.is_dir():
            continue
        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir():
                continue
            if exp_dir.name == scene or exp_dir.name.startswith(scene + "_"):
                for method_dir in sorted(exp_dir.iterdir()):
                    candidates.extend(_timestamp_runs(method_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No runs with checkpoints for scene '{scene}' under {OUTPUTS_ROOT} "
            f"(searched <root>/[synthetic/]{scene}*/<method>/<timestamp>/nerfstudio_models)"
        )
    # Latest timestamp wins (timestamps sort lexicographically).
    return sorted(candidates, key=lambda p: p.name)[-1]


def latest_checkpoint_step(run_dir: Path) -> int:
    steps = sorted(
        int(p.name[len("step-"):-len(".ckpt")])
        for p in (Path(run_dir) / "nerfstudio_models").glob("step-*.ckpt")
    )
    if not steps:
        raise FileNotFoundError(f"No step-*.ckpt under {run_dir}/nerfstudio_models")
    return steps[-1]


def resolve_reni_prior() -> Path:
    """RENI++ illumination prior checkpoint dir (latent_dim_100)."""
    candidates = []
    if os.environ.get("NEUSKY_RENI_PRIOR"):
        candidates.append(Path(os.environ["NEUSKY_RENI_PRIOR"]).expanduser())
    if os.environ.get("RENI_CKPT_PATH"):  # docker-compose convention
        candidates.append(Path(os.environ["RENI_CKPT_PATH"]).expanduser() / "latent_dim_100")
    candidates.append(
        REPO_ROOT / "model-storage" / "reni_paper_models" / "reni_plus_plus_models" / "latent_dim_100"
    )
    candidates.append(REPO_ROOT / "ns_reni" / "checkpoints" / "reni_plus_plus_models" / "latent_dim_100")
    for c in candidates:
        if (c / "nerfstudio_models").is_dir():
            return c
    return candidates[-2]  # default; populate_modules raises a clear error if absent


def _remap_data_path(saved: Path) -> Path:
    """Remap a saved dataparser data path onto this machine's data roots.

    Saved configs may hold container-absolute (/workspace/...) or other-host
    paths; resolve NeRF-OSR and synthetic dirs against the env-overridable
    roots, leaving valid paths untouched.
    """
    saved = Path(saved)
    probe = saved if saved.is_absolute() else REPO_ROOT / saved
    if probe.exists():
        return saved
    parts = saved.parts
    if "NeRF-OSR" in parts:
        return NERF_OSR_ROOT
    for i, part in enumerate(parts):
        if part in SYNTHETIC_SCENES or part.endswith("_prepared"):
            return SYNTHETIC_ROOT / Path(*parts[i:])
    return saved


def load_model(
    scene: str,
    device: str = "cuda:0",
    step=None,
    test_mode: str = "test",
    eval_num_rays_per_chunk=None,
    config_hook=None,
):
    """Load a trained NeuSky pipeline for a scene via nerfstudio eval_setup.

    Resolves the latest run/checkpoint (resolve_run_dir), performs config
    surgery for relocated paths (data roots + the RENI++ prior the configs
    reference at model-storage/reni_paper_models/...), and returns
    (config, pipeline, checkpoint_path, step). `config_hook(config)` allows
    per-figure overrides (e.g. eval_latent_optimise_method) before setup.

    NOTE: builds the model on GPU; do not call while a training job owns it.
    """
    # cwd must be the repo root: saved configs hold repo-relative paths.
    os.chdir(REPO_ROOT)

    from nerfstudio.utils.eval_utils import eval_setup  # heavy import, deferred

    run_dir = resolve_run_dir(scene)
    load_step = step if step is not None else latest_checkpoint_step(run_dir)

    def _callback(config):
        # Point checkpoint resolution at the resolved run dir regardless of
        # where the run was trained (get_checkpoint_dir() recomposes from
        # output_dir/experiment_name/method_name/timestamp).
        config.output_dir = run_dir.parents[2]
        config.experiment_name = run_dir.parents[1].name
        config.method_name = run_dir.parents[0].name
        config.timestamp = run_dir.name
        config.load_step = load_step
        config.vis = "tensorboard"  # never spawn a viewer from figure scripts

        dataparser = config.pipeline.datamanager.dataparser
        dataparser.data = _remap_data_path(dataparser.data)
        config.pipeline.datamanager.data = dataparser.data
        config.data = dataparser.data

        config.pipeline.model.illumination_field_ckpt_path = resolve_reni_prior()

        if eval_num_rays_per_chunk is not None:
            config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
        if config_hook is not None:
            config_hook(config)
        return config

    # torch>=2.6 defaults torch.load to weights_only=True, which rejects
    # nerfstudio checkpoints (they pickle numpy scalars). These are our own
    # trusted checkpoints, so force the legacy behaviour for eval_setup.
    import torch

    _orig_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _trusted_load
    try:
        config, pipeline, ckpt_path, loaded_step = eval_setup(
            run_dir / "config.yml",
            test_mode=test_mode,
            update_config_callback=_callback,
        )
    finally:
        torch.load = _orig_load
    pipeline.to(device)
    pipeline.eval()
    return config, pipeline, ckpt_path, loaded_step


# ---------------------------------------------------------------------------
# Rendering helpers (ported from publication/figures_and_tables.ipynb)
# ---------------------------------------------------------------------------

def render_view(model, cameras, view_idx: int, device, illumination_idx=None,
                rotation=None, show_progress: bool = True):
    """Render a full training/eval view with the fitted train illumination.

    Ports the notebook recipe: generate_rays(keep_shape=True), override
    camera_indices with the illumination index, render with
    viewing_training_image=True so train latents are used.
    """
    import torch

    illumination_idx = view_idx if illumination_idx is None else illumination_idx
    ray_bundle = cameras.generate_rays(camera_indices=view_idx, keep_shape=True)
    ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * illumination_idx
    ray_bundle = ray_bundle.to(device)
    model.viewing_training_image = True
    try:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle=ray_bundle, rotation=rotation, show_progress=show_progress
            )
    finally:
        model.viewing_training_image = False
    return outputs


def normal_to_camera_vis(normal, c2w):
    """World-space normals -> camera-space visualisation in [0, 1].

    normal: [H, W, 3] tensor (model output); c2w: [3|4, 4] camera-to-world.
    Background (short normals) is whited out, as in the paper figures.
    """
    normal = normal.cpu()
    normal_cam = normal @ c2w[:3, :3].cpu()
    vis = (normal_cam + 1) / 2
    vis[normal.norm(dim=-1) < 0.5] = 1
    return vis.numpy()


def render_envmap(model, illumination_idx: int, rotation=None):
    """Decode the fitted illumination for a view to an LDR envmap [H, W, 3].

    Ported from publication/render_animation.py:get_envmap.
    """
    import torch
    from reni.field_components.field_heads import RENIFieldHeadNames
    from reni.utils.colourspace import linear_to_sRGB

    with torch.no_grad():
        model.viewing_training_image = True
        try:
            illumination_latents, scales = model.get_illumination_field()
            ray_samples = model.equirectangular_sampler.generate_direction_samples()
            ray_samples = ray_samples.to(model.device)
            ray_samples.camera_indices = (
                torch.ones_like(ray_samples.camera_indices) * illumination_idx
            )
            outputs = model.illumination_field(
                ray_samples=ray_samples,
                latent_codes=illumination_latents[ray_samples.camera_indices[:, 0]],
                scale=scales[ray_samples.camera_indices[:, 0]],
                rotation=rotation,
            )
        finally:
            model.viewing_training_image = False
        hdr = model.illumination_field.unnormalise(outputs[RENIFieldHeadNames.RGB])
        ldr = linear_to_sRGB(hdr)
        H = model.equirectangular_sampler.height
        W = model.equirectangular_sampler.width
        return ldr.reshape(H, W, 3).cpu()


def swap_in_environment_map(model, envmap_path: Path, datamanager,
                            resolution=(512, 1024)):
    """Replace the model's illumination field with a fixed environment map.

    Ported from the notebook's dam_wall/point-light relighting cells. Returns
    (original_field, original_latents, original_scale) so callers can restore.
    """
    import torch
    from reni.illumination_fields.environment_map_field import EnvironmentMapFieldConfig

    original = (model.illumination_field, model.train_illumination_latents, model.train_scale)
    env_config = EnvironmentMapFieldConfig(
        path=Path(envmap_path), resolution=resolution, trainable=False,
        apply_padding=True, fixed_decoder=False,
    )
    envmap_field = env_config.setup(
        num_train_data=1, num_eval_data=1,
        normalisations={"min_max": None, "log_domain": False},
    )
    model.illumination_field = envmap_field.to(model.device)
    train_mu = envmap_field.train_mu.repeat(
        len(datamanager.train_dataset), 1, 1, 1
    ).to(model.device)
    model.train_illumination_latents = torch.nn.Parameter(train_mu)
    model.train_scale = None
    return original


def restore_illumination(model, original):
    field, latents, scale = original
    model.illumination_field = field
    model.train_illumination_latents = latents
    model.train_scale = scale


def sphere_camera_ray_bundle(position, device, height: int = 411, width: int = 640,
                             focal: float = 503.5):
    """Perspective camera at `position` on the DDF sphere looking at origin.

    Intrinsics are half-resolution versions of DDFDataset's eval cameras
    (fx=fy=1007, 1280x823 full-res).
    """
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from neusky.utils.utils import look_at_target

    position = torch.as_tensor(position, dtype=torch.float32).reshape(1, 3)
    c2w = look_at_target(position, torch.zeros_like(position))[..., :3, :4]
    camera = Cameras(
        camera_to_worlds=c2w,
        fx=torch.tensor([[focal]]),
        fy=torch.tensor([[focal]]),
        cx=torch.tensor([[width / 2]]),
        cy=torch.tensor([[height / 2]]),
        camera_type=CameraType.PERSPECTIVE,
    )
    return camera.generate_rays(0).to(device)


# ---------------------------------------------------------------------------
# Figure I/O
# ---------------------------------------------------------------------------

def save_figure(fig, out_stem: Path, svg: bool = False, dpi: int = 200):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    if svg:
        fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    print(f"[saved] {out_stem}.png / .pdf" + (" / .svg" if svg else ""))


def add_common_args(parser: argparse.ArgumentParser, default_output: str,
                    needs_device: bool = True):
    if needs_device:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--step", type=int, default=None,
                            help="Checkpoint step (default: latest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svg", action="store_true", help="Also write SVG")
    parser.add_argument("--output", type=Path,
                        default=FIGURES_DIR / default_output,
                        help="Output stem (no extension)")
    return parser


def seed_all(seed: int):
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
