# NeuSky programmatic figures and tables

Headless, deterministic regeneration of the NeuSky thesis/TPAMI figures and
tables (replaces `publication/figures_and_tables.ipynb`). Same philosophy as
`ns_reni/scripts/figures/`: one script per figure, a shared `_common.py`
loader, env-overridable roots with repo-relative defaults, argparse CLIs
(`--help` on every script).

Synthetic-benchmark scoring/rendering lives in `scripts/synthetic_benchmark/` (see its README).

All commands run from the repo root:

```bash
PYTHONPATH=. python scripts/figures/<script>.py [--help]
```

Outputs default to `publication/figures/<name>.{png,pdf}` and
`publication/tables/<name>.{tex,csv}` (`--output` / `--output-dir` override;
`--svg` adds SVG).

## Inventory

| Script | Output (thesis asset) | Needs | Status |
|--------|----------------------|-------|--------|
| `fig_sigmoid.py` | `sigmoid.pdf` — soft visibility threshold curve | nothing (CPU; optional `--scene` reads learned eps/eta from a ckpt) | runnable now |
| `fig_scene_contraction.py` | `scene_contraction.png` — linear 0-1 / quadratic 1-2 contraction | nothing (CPU, nerfstudio import only) | runnable now |
| `fig_decomposition.py` | `nerfosr_examples.png` (+ per-scene halves of `comparisons_full.png`) — GT / render / albedo / normals rows | NeRF-OSR data + scene ckpts | lk2 runnable; st/lwp blocked on refits |
| `fig_ddf_depth.py` | `ddf_sdf_comparison.png` — DDF depth vs SDF pseudo-GT from sphere cameras | scene ckpt with DDF | lk2 runnable |
| `fig_ao_shadow.py` | `ao_and_shadow.pdf` — direction-averaged AO + single-direction shadow | scene ckpt with DDF | lk2 runnable |
| `fig_relighting.py` | `further_relighting_examples.png` — latent swaps + fixed envmaps | scene ckpt (+ `publication/point_light.exr`; dam_wall auto-downloads) | lk2 runnable |
| `fig_synthetic_eval_grid.py` | `synthetic_eval_grid.png` — 5 scenes x 6 cols (GT/pred RGB, albedo, normal) | synthetic data + `outputs/synthetic_benchmark/neusky/*_gt` predictions | partial as benchmark renders finish |
| `make_tables.py` | `nerf_osr.{tex,csv}` (tab:nerf_osr) and `synthetic.{tex,csv}` (tab:synthetic) | NeRF-OSR ckpts; synthetic benchmark metrics JSON | synthetic table reads saved benchmark metrics |

"Runnable now" for checkpoint scripts assumes the GPU is free — do **not**
run them while a training job owns the local 4090. The diagram-only scripts
(`fig_sigmoid`, `fig_scene_contraction`) are pure CPU/matplotlib and always
safe.

## Invocation examples

```bash
# Diagram-only (CPU, safe anytime)
PYTHONPATH=. python scripts/figures/fig_sigmoid.py
PYTHONPATH=. python scripts/figures/fig_scene_contraction.py

# Checkpoint-dependent (GPU)
PYTHONPATH=. python scripts/figures/fig_decomposition.py --scenes lk2 st
PYTHONPATH=. python scripts/figures/fig_ddf_depth.py --scene lk2
PYTHONPATH=. python scripts/figures/fig_ao_shadow.py --scene lk2
PYTHONPATH=. python scripts/figures/fig_relighting.py --scene lk2 --envmap-strip
PYTHONPATH=. python scripts/figures/fig_synthetic_eval_grid.py

# Tables (partial run while only lk2 is trained)
PYTHONPATH=. python scripts/figures/make_tables.py --tables nerf_osr --scenes lk2
PYTHONPATH=. python scripts/figures/make_tables.py             # everything
```

Metrics are cached in `publication/tables/neusky_metrics.json`; delete an
entry (or the file) to force re-evaluation. Re-running `make_tables.py` after
more scenes finish training fills in the `---` cells.

## Inputs and resolution

- **Runs/checkpoints**: `_common.resolve_run_dir(scene)` finds the latest
  timestamped run under `outputs/<scene*>/<method>/<timestamp>/` (also
  `outputs/synthetic/...`), accepting both bare scene names (`lk2`) and
  derived experiment names (`lk2_refit_optimised`). The latest `step-*.ckpt`
  is used unless `--step` is given. Pin runs explicitly with
  `NEUSKY_RUNS="lk2=outputs/lk2_refit_optimised/neusky/2026-06-11_121134;st=..."`.
- **RENI++ prior**: saved configs reference
  `model-storage/reni_paper_models/reni_plus_plus_models/latent_dim_100`;
  `_common.load_model` remaps this via `$NEUSKY_RENI_PRIOR`, then
  `$RENI_CKPT_PATH/latent_dim_100`, then the repo-relative `model-storage`
  symlink.
- **Data**: NeRF-OSR sites live under `$NERF_OSR_ROOT` (the dataparser maps
  site1→lk2, site2→st, site3→lwp); synthetic scenes under
  `$NEUSKY_SYNTHETIC_ROOT` (`*_prepared` dirs with train/val/test splits and
  GT albedo/normal layers). Saved container paths (`/workspace/...`) are
  remapped automatically.

## Environment variables (all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `NEUSKY_OUTPUTS` | `<repo>/outputs` | training outputs root |
| `NERF_OSR_ROOT` | `~/data/NeRF-OSR/Data` (else `<repo>/data/NeRF-OSR/Data`) | NeRF-OSR dataset root |
| `NEUSKY_SYNTHETIC_ROOT` | `~/data/neusky_synthetic_data/renders` (else repo-relative) | synthetic renders root |
| `NEUSKY_RENI_PRIOR` | `<repo>/model-storage/reni_paper_models/reni_plus_plus_models/latent_dim_100` | RENI++ prior ckpt dir |
| `NEUSKY_RUNS` | unset | pin scene→run dir (`;`/`,`-separated `scene=path`) |

No env var is required — a standalone clone with the repo-local `data` /
`model-storage` symlinks and `outputs/` works out of the box.

## Provenance

Recipes ported from `publication/figures_and_tables.ipynb` (decomposition
rows, relighting latent/envmap swaps, scene contraction, AO/shadow render
paths), `publication/render_animation.py` (envmap decode), and
`notebooks/ddf.ipynb` + `neusky/data/datasets/ddf_dataset.py` (sphere-camera
DDF vs SDF depth). Table layouts match the hardcoded tables in the thesis
(`latex/9_Chapter3/paper_sections/07_Evaluation.tex`); baseline numbers in
`make_tables.py` are the published NeRF-OSR / FEGR / SOL-NeRF results.
