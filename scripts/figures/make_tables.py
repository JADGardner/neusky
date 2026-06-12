"""Quantitative tables (thesis Tables: tab:nerf_osr and tab:synthetic).

- nerf_osr: outdoor relighting on the NeRF-OSR benchmark (PSNR / MSE per
  site). Baseline rows (NeRF-OSR, FEGR, SOL-NeRF) are the published numbers;
  the NeuSky row is computed by evaluating each site's checkpoint with the
  benchmark protocol (eval_latent_optimise_method="nerf_osr_holdout" with the
  per-session holdout indices, masked metrics via the provided test masks).
- synthetic: novel view + intrinsic decomposition on the five synthetic
  scenes (PSNR / SSIM / LPIPS / Albedo PSNR / Normal MAE; roughness and
  metallic are not predicted by NeuSky and emit ---).
- protocol_sensitivity: NeuSky under the three relighting eval protocols
  (holdout-view latent fit; GT envmap with x10 / x30 saturation scaling and
  KNOWN rotations from <scene>/final/envmap_rotations.json). --protocols
  restricts the rows (e.g. --protocols holdout while rotation JSONs are
  still being generated). If a holdout_sensitivity sweep is cached, the
  holdout row reports mean +/- std over holdout-image choice.
- holdout_sensitivity: error bars over the choice of per-session holdout
  image for the nerf_osr_holdout protocol. Session latents only receive
  gradients from their own holdout image, so the sweep varies every session
  to its rank-k candidate simultaneously (one 250-step joint fit per rank)
  and scores only each session's own compare view. A session's compare
  image is excluded from its candidates (fitting on the scored view would
  be test-set fitting; the dataparser rejects the overlap outright).
  Results are cached per (session, holdout) under
  protocol/holdout_sweep/<scene>/<session>/<rel_idx> and emitted as a
  per-combination csv plus scene-level mean +/- std.

Both emit .tex and .csv next to a JSON metrics cache; pass --scenes to run a
partial set (e.g. only lk2 while st/lwp refits are pending) — missing scenes
render as --- and can be filled in by re-running once their runs exist.

Checkpoint-dependent (GPU):

    PYTHONPATH=. python scripts/figures/make_tables.py
    PYTHONPATH=. python scripts/figures/make_tables.py --tables nerf_osr --scenes lk2
"""

import argparse
import csv
import json
from pathlib import Path

from _common import (SCENE_TO_SITE, SESSION_HOLDOUT_INDICES, SYNTHETIC_SCENES,
                     TABLES_DIR, canonical_scene, load_model, resolve_run_dir)

NERF_OSR_SCENES = ("lk2", "st", "lwp")  # site1, site2, site3

# Published baselines for tab:nerf_osr (PSNR up, MSE down), per site.
NERF_OSR_BASELINES = {
    "NeRF-OSR~\\cite{rudnev_nerf_2022}": {
        "lk2": (19.34, 0.012), "st": (16.35, 0.027), "lwp": (15.66, 0.029)},
    "FEGR~\\cite{wang_neural_2023}": {
        "lk2": (21.53, 0.007), "st": (17.00, 0.023), "lwp": (17.57, 0.018)},
    "SOL-NeRF~\\cite{sunSOLNeRFSunlightModeling2023}": {
        "lk2": (21.23, 0.0084), "st": (18.18, 0.019), "lwp": (17.58, 0.028)},
}

SYNTHETIC_METRICS = ("psnr", "ssim", "lpips", "albedo_psnr", "normal_mae")

# Relighting protocol-sensitivity rows: latent-fit source and, for the GT
# envmap protocol, the NeRF-OSR pseudo-HDR saturation scaling.
PROTOCOLS = {
    "holdout": {"method": "nerf_osr_holdout", "label": "Holdout view"},
    "envmap10": {"method": "nerf_osr_envmap", "saturation_scale": 10.0,
                 "label": "GT envmap ($\\times$10)"},
    "envmap30": {"method": "nerf_osr_envmap", "saturation_scale": 30.0,
                 "label": "GT envmap ($\\times$30)"},
}

# Matches eval_latent_optimizer in the model config / checkpoints plus the
# eval_latent_optimisation_seed pinned in fit_latent_codes_for_eval.
PROTOCOL_FIT_NOTE = "Latent fit: Adam lr 1e-1 -> 1e-7, 250 steps, seed 42."


def has_run(scene: str) -> bool:
    try:
        resolve_run_dir(scene)
        return True
    except FileNotFoundError:
        return False


def evaluate_nerfosr_relighting(scene: str, device: str, protocol: str = "holdout"):
    """NeuSky relighting metrics for one NeRF-OSR site (psnr, mse)."""
    import torch

    spec = PROTOCOLS[protocol]

    def hook(config):
        config.pipeline.model.eval_latent_optimise_method = spec["method"]
        if "saturation_scale" in spec:
            config.pipeline.model.envmap_saturation_scale = spec["saturation_scale"]
        config.pipeline.datamanager.dataparser.session_holdout_indices = \
            SESSION_HOLDOUT_INDICES[scene]

    _, pipeline, _, step = load_model(scene, device=device, config_hook=hook)
    metrics = pipeline.get_average_eval_image_metrics(step=step)
    result = {"psnr": float(metrics["psnr"]), "mse": float(metrics["mse"])}
    del pipeline
    torch.cuda.empty_cache()
    return result


def evaluate_holdout_sensitivity(scene: str, device: str, cache_path: Path):
    """Sweep the per-session holdout-image choice for the holdout protocol.

    Loads the pipeline once. Sessions' latents are independent in the joint
    250-step fit (each latent only receives gradients from its own holdout
    image), so assignments are batched: rank k varies EVERY session to its
    k-th candidate (exhausted/cached sessions fall back to the canonical
    index) and only the freshly varied sessions' compare views are rendered.
    Results are cached incrementally per (session, holdout index).
    """
    import torch
    from neusky.data.utils.dataloaders import SelectedIndicesCacheDataloader

    def hook(config):
        config.pipeline.model.eval_latent_optimise_method = "nerf_osr_holdout"
        config.pipeline.datamanager.dataparser.session_holdout_indices = \
            SESSION_HOLDOUT_INDICES[scene]

    # Larger render chunks are numerically identical and ~4x faster than the
    # configs' 256; the sweep renders ~90 compare images.
    _, pipeline, _, step = load_model(scene, device=device, config_hook=hook,
                                      eval_num_rays_per_chunk=1024)
    dm = pipeline.datamanager
    model = pipeline.model
    meta = dm.eval_dataset.metadata
    session_to_indices = meta["session_to_indices"]  # {session: [abs test idxs]}
    session_names = meta.get("session_names") or {}
    canonical = SESSION_HOLDOUT_INDICES[scene]
    compare_of = {dm.indices_to_session[i]: i for i in dm.eval_dataset.test_eval_mask_dict}

    def image_name(idx):
        return Path(dm.eval_dataset._dataparser_outputs.image_filenames[idx]).stem

    # Candidate holdout indices (relative to session): every test image of the
    # session except its compare view(s).
    candidates = {
        s: [k for k, i in enumerate(idxs) if i not in compare_of.values()]
        for s, idxs in session_to_indices.items()
    }

    results = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def key(s, rel_idx):
        return f"protocol/holdout_sweep/{scene}/{session_names.get(s, s)}/{rel_idx}"

    sessions = sorted(session_to_indices)
    for rank in range(max(len(c) for c in candidates.values())):
        assignment, fresh = {}, []
        for s in sessions:
            cands = candidates[s]
            if rank < len(cands) and key(s, cands[rank]) not in results:
                assignment[s] = cands[rank]
                fresh.append(s)
            else:
                assignment[s] = canonical[s]
        if not fresh:
            continue

        # Point the optimise bundle at this holdout set. num_workers=0: the
        # cache loader only thread-pools the initial image load, so the ~30
        # rebuilds spawn no worker processes.
        holdout_abs = [session_to_indices[s][assignment[s]] for s in sessions]
        dm.eval_session_holdout_dataloader = SelectedIndicesCacheDataloader(
            dm.eval_dataset,
            num_images_to_sample_from=dm.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=dm.config.eval_num_times_to_repeat_images,
            device=dm.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=dm.config.collate_fn,
            selected_indices=holdout_abs,
        )
        dm.iter_eval_session_holdout_dataloader = iter(dm.eval_session_holdout_dataloader)

        model.fit_latent_codes_for_eval(dm, step)

        for s in fresh:
            cidx = compare_of[s]
            camera, batch = dm.eval_dataloader.get_camera(cidx)
            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * s
            batch["image_idx"] = s
            with torch.no_grad():
                outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, step=step)
            metrics, _ = model.get_image_metrics_and_images(outputs, batch)
            k = key(s, assignment[s])
            results[k] = {
                "psnr": float(metrics["psnr"]),
                "mse": float(metrics["mse"]),
                "holdout_image": image_name(session_to_indices[s][assignment[s]]),
                "compare_image": image_name(cidx),
            }
            print(f"[sweep] {k} holdout={results[k]['holdout_image']} "
                  f"psnr={results[k]['psnr']:.3f} mse={results[k]['mse']:.5f}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(results, indent=2))

    del pipeline
    torch.cuda.empty_cache()
    return results


def holdout_sweep_stats(results, scene: str):
    """Per-session and scene-level stats over the cached holdout sweep.

    The scene metric is the mean of the per-session (compare-image) metrics,
    so across independent per-session holdout choices: scene mean = mean of
    session means and scene var = sum of session vars / S^2.
    """
    prefix = f"protocol/holdout_sweep/{scene}/"
    per_session = {}
    for k, v in results.items():
        if k.startswith(prefix) and v is not None:
            session, rel_idx = k[len(prefix):].split("/")
            per_session.setdefault(session, []).append({"rel_idx": int(rel_idx), **v})
    if not per_session:
        return None

    def mean(xs):
        return sum(xs) / len(xs)

    def var(xs):  # sample variance
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    stats = {"sessions": {}, "num_combinations": 0}
    psnr_means, psnr_vars, mse_means, mse_vars = [], [], [], []
    for session, rows in sorted(per_session.items()):
        psnrs = [r["psnr"] for r in rows]
        mses = [r["mse"] for r in rows]
        stats["sessions"][session] = {
            "n": len(rows),
            "psnr_mean": mean(psnrs), "psnr_std": var(psnrs) ** 0.5,
            "mse_mean": mean(mses), "mse_std": var(mses) ** 0.5,
            "rows": sorted(rows, key=lambda r: r["rel_idx"]),
        }
        stats["num_combinations"] += len(rows)
        psnr_means.append(mean(psnrs)); psnr_vars.append(var(psnrs))
        mse_means.append(mean(mses)); mse_vars.append(var(mses))
    n_sessions = len(per_session)
    stats["psnr_mean"] = mean(psnr_means)
    stats["psnr_std"] = (sum(psnr_vars) / n_sessions**2) ** 0.5
    stats["mse_mean"] = mean(mse_means)
    stats["mse_std"] = (sum(mse_vars) / n_sessions**2) ** 0.5
    return stats


def write_holdout_sensitivity_csv(stats, scene: str, output: Path):
    csv_rows = [("session", "holdout_rel_idx", "holdout_image", "compare_image",
                 "psnr", "mse")]
    for session, s in stats["sessions"].items():
        for r in s["rows"]:
            csv_rows.append((session, r["rel_idx"], r["holdout_image"],
                             r["compare_image"], f"{r['psnr']:.4f}", f"{r['mse']:.6f}"))
    csv_rows.append(())
    csv_rows.append(("# per-session summary", "n", "", "", "psnr mean/std", "mse mean/std"))
    for session, s in stats["sessions"].items():
        csv_rows.append((session, s["n"], "", "",
                         f"{s['psnr_mean']:.4f}/{s['psnr_std']:.4f}",
                         f"{s['mse_mean']:.6f}/{s['mse_std']:.6f}"))
    csv_rows.append((f"# scene ({scene})", stats["num_combinations"], "", "",
                     f"{stats['psnr_mean']:.4f}/{stats['psnr_std']:.4f}",
                     f"{stats['mse_mean']:.6f}/{stats['mse_std']:.6f}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[saved] {output}.csv  ({stats['num_combinations']} combinations; "
          f"scene PSNR {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f})")


def evaluate_synthetic(scene: str, device: str):
    """NeuSky synthetic-benchmark metrics for one scene."""
    import torch

    _, pipeline, _, step = load_model(scene, device=device)
    metrics = pipeline.get_average_eval_image_metrics(step=step)
    result = {m: float(metrics[m]) for m in SYNTHETIC_METRICS if m in metrics}
    del pipeline
    torch.cuda.empty_cache()
    return result


def collect(scenes, evaluator, cache_path: Path, device: str, prefix: str):
    results = {}
    if cache_path.exists():
        results = json.loads(cache_path.read_text())
        print(f"[cache] loaded {len(results)} entries from {cache_path}")
    for scene in scenes:
        key = f"{prefix}/{scene}"
        if key in results:
            continue
        if not has_run(scene):
            print(f"[skip] {key}: no run with checkpoints (cells will show ---)")
            results[key] = None
        else:
            print(f"[eval] {key}")
            results[key] = evaluator(scene, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2))
    return results


def fmt(value, best, decimals, lower_better=False):
    s = f"{value:.{decimals}f}"
    is_best = (value <= best) if lower_better else (value >= best)
    return f"\\textbf{{{s}}}" if is_best else s


def write_nerf_osr_table(results, output: Path):
    rows = dict(NERF_OSR_BASELINES)
    neusky = {s: results.get(f"nerf_osr/{s}") for s in NERF_OSR_SCENES}
    rows["NeuSky (Ours)"] = {
        s: (r["psnr"], r["mse"]) if r else None for s, r in neusky.items()}
    # Baselines store tuples directly; normalise.
    for name in NERF_OSR_BASELINES:
        rows[name] = {s: tuple(v) for s, v in NERF_OSR_BASELINES[name].items()}

    best = {}
    for s in NERF_OSR_SCENES:
        vals = [v[s] for v in rows.values() if v.get(s) is not None]
        best[s] = (max(v[0] for v in vals), min(v[1] for v in vals))

    lines = [
        "% Generated by scripts/figures/make_tables.py -- do not edit by hand",
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Site 1} & \\phantom{ab} & \\multicolumn{2}{c}{Site 2}"
        " & \\phantom{ab} & \\multicolumn{2}{c}{Site 3} \\\\",
        "& PSNR $\\uparrow$ & MSE $\\downarrow$ && PSNR $\\uparrow$ & MSE $\\downarrow$"
        " && PSNR $\\uparrow$ & MSE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    csv_rows = [("method",) + tuple(
        f"{SCENE_TO_SITE[s]}_{m}" for s in NERF_OSR_SCENES for m in ("psnr", "mse"))]
    for name, per_scene in rows.items():
        cells, csv_cells = [], []
        for s in NERF_OSR_SCENES:
            v = per_scene.get(s)
            if v is None:
                cells.append("--- & ---")
                csv_cells.extend(["", ""])
            else:
                psnr, mse = v
                cells.append(f"{fmt(psnr, best[s][0], 2)} & "
                             f"{fmt(mse, best[s][1], 3, lower_better=True)}")
                csv_cells.extend([f"{psnr:.4f}", f"{mse:.4f}"])
        lines.append(f"{name} & " + " && ".join(cells) + " \\\\")
        csv_rows.append((name, *csv_cells))
    lines += ["\\bottomrule", "\\end{tabular}"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    with open(output.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[saved] {output}.tex / .csv")
    print("\n".join(lines))


def write_protocol_sensitivity_table(results, protocols, output: Path):
    lines = [
        "% Generated by scripts/figures/make_tables.py -- do not edit by hand",
        f"% {PROTOCOL_FIT_NOTE}",
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Site 1} & \\phantom{ab} & \\multicolumn{2}{c}{Site 2}"
        " & \\phantom{ab} & \\multicolumn{2}{c}{Site 3} \\\\",
        "Protocol & PSNR $\\uparrow$ & MSE $\\downarrow$ && PSNR $\\uparrow$ & MSE $\\downarrow$"
        " && PSNR $\\uparrow$ & MSE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    csv_rows = [("protocol",) + tuple(
        f"{SCENE_TO_SITE[s]}_{m}" for s in NERF_OSR_SCENES for m in ("psnr", "mse"))]
    footers = []
    for protocol in protocols:
        cells, csv_cells = [], []
        for s in NERF_OSR_SCENES:
            sweep = holdout_sweep_stats(results, s) if protocol == "holdout" else None
            r = results.get(f"protocol/{protocol}/{s}")
            if sweep is not None:
                # Mean +/- std over per-session holdout-image choice.
                cells.append(f"{sweep['psnr_mean']:.2f} $\\pm$ {sweep['psnr_std']:.2f} & "
                             f"{sweep['mse_mean']:.3f} $\\pm$ {sweep['mse_std']:.3f}")
                csv_cells.extend([f"{sweep['psnr_mean']:.4f}", f"{sweep['mse_mean']:.4f}"])
                fixed = (f"; fixed canonical indices give {r['psnr']:.2f} / {r['mse']:.4f}"
                         if r else "")
                footers.append(
                    f"% Holdout {SCENE_TO_SITE[s]}: mean +/- std over the per-session choice of "
                    f"holdout image ({sweep['num_combinations']} combinations, sessions swept "
                    f"independently and combined as mean-of-sessions){fixed}.")
            elif r is None:
                cells.append("--- & ---")
                csv_cells.extend(["", ""])
            else:
                cells.append(f"{r['psnr']:.2f} & {r['mse']:.3f}")
                csv_cells.extend([f"{r['psnr']:.4f}", f"{r['mse']:.4f}"])
        lines.append(f"{PROTOCOLS[protocol]['label']} & " + " && ".join(cells) + " \\\\")
        csv_rows.append((protocol, *csv_cells))
    lines += ["\\bottomrule", "\\end{tabular}"] + footers

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    with open(output.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[saved] {output}.tex / .csv  ({PROTOCOL_FIT_NOTE})")
    print("\n".join(lines))


def write_synthetic_table(results, output: Path):
    decimals = {"psnr": 2, "ssim": 3, "lpips": 3, "albedo_psnr": 2, "normal_mae": 2}
    lines = [
        "% Generated by scripts/figures/make_tables.py -- do not edit by hand",
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Novel View} & \\multicolumn{4}{c}{Intrinsic Decomposition} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-8}",
        "Scene & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & "
        "Albedo PSNR $\\uparrow$ & Normal MAE $\\downarrow$ & "
        "Rough. PSNR $\\uparrow$ & Metal. PSNR $\\uparrow$ \\\\",
        "\\midrule",
    ]
    csv_rows = [("scene", *SYNTHETIC_METRICS)]
    collected = {m: [] for m in SYNTHETIC_METRICS}
    for scene, label in SYNTHETIC_SCENES.items():
        r = results.get(f"synthetic/{scene}")
        if r is None:
            lines.append(f"{label} & " + " & ".join(["---"] * 7) + " \\\\")
            csv_rows.append((scene, "", "", "", "", ""))
            continue
        cells = [f"{r[m]:.{decimals[m]}f}" if m in r else "---"
                 for m in SYNTHETIC_METRICS]
        for m in SYNTHETIC_METRICS:
            if m in r:
                collected[m].append(r[m])
        lines.append(f"{label} & " + " & ".join(cells) + " & --- & --- \\\\")
        csv_rows.append((scene, *(f"{r.get(m, float('nan')):.4f}" for m in SYNTHETIC_METRICS)))

    lines.append("\\midrule")
    if any(collected[m] for m in SYNTHETIC_METRICS):
        mean_cells = [
            f"\\textbf{{{sum(v) / len(v):.{decimals[m]}f}}}" if (v := collected[m]) else "---"
            for m in SYNTHETIC_METRICS
        ]
        lines.append("\\textbf{Mean} & " + " & ".join(mean_cells) + " & --- & --- \\\\")
        csv_rows.append(("mean", *(
            f"{sum(v) / len(v):.4f}" if (v := collected[m]) else "" for m in SYNTHETIC_METRICS)))
    lines += ["\\bottomrule", "\\end{tabular}"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    with open(output.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[saved] {output}.tex / .csv")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tables", nargs="+",
                        choices=["nerf_osr", "synthetic", "protocol_sensitivity",
                                 "holdout_sensitivity"],
                        default=["nerf_osr", "synthetic"])
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Restrict evaluation to these scenes (others show ---). "
                             "E.g. --scenes lk2 for a partial lk2-only run.")
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS),
                        default=list(PROTOCOLS),
                        help="Restrict protocol_sensitivity rows (envmap rows need "
                             "<scene>/final/envmap_rotations.json).")
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--cache", type=Path, default=None,
                        help="JSON metrics cache (default <output-dir>/neusky_metrics.json)")
    args = parser.parse_args()

    wanted = {canonical_scene(s) for s in args.scenes} if args.scenes else None
    cache_path = args.cache or args.output_dir / "neusky_metrics.json"

    if "nerf_osr" in args.tables:
        scenes = [s for s in NERF_OSR_SCENES if wanted is None or s in wanted]
        results = collect(scenes, evaluate_nerfosr_relighting, cache_path,
                          args.device, "nerf_osr")
        write_nerf_osr_table(results, args.output_dir / "nerf_osr")

    if "synthetic" in args.tables:
        scenes = [s for s in SYNTHETIC_SCENES if wanted is None or s in wanted]
        results = collect(scenes, evaluate_synthetic, cache_path,
                          args.device, "synthetic")
        write_synthetic_table(results, args.output_dir / "synthetic")

    if "holdout_sensitivity" in args.tables:
        scenes = [s for s in NERF_OSR_SCENES if wanted is None or s in wanted]
        for scene in scenes:
            if not has_run(scene):
                print(f"[skip] holdout_sensitivity/{scene}: no run with checkpoints")
                continue
            results = evaluate_holdout_sensitivity(scene, args.device, cache_path)
            stats = holdout_sweep_stats(results, scene)
            if stats is not None:
                write_holdout_sensitivity_csv(
                    stats, scene, args.output_dir / f"holdout_sensitivity_{scene}")

    if "protocol_sensitivity" in args.tables:
        scenes = [s for s in NERF_OSR_SCENES if wanted is None or s in wanted]
        results = {}
        for protocol in args.protocols:
            # Cache key includes protocol (and thereby saturation scale).
            results.update(collect(
                scenes,
                lambda scene, device, p=protocol: evaluate_nerfosr_relighting(scene, device, protocol=p),
                cache_path, args.device, f"protocol/{protocol}"))
        write_protocol_sensitivity_table(
            results, args.protocols, args.output_dir / "protocol_sensitivity")


if __name__ == "__main__":
    main()
