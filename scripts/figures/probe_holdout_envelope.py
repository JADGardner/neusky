"""Pick a second in-envelope holdout view per NeRF-OSR session (CPU only).

The nerf_osr_holdout relighting protocol fits each test session's RENI++ latent
(and exposure) on a single held-out view, then scores a separate compare view.
Two held-out views per session constrain the illumination better (especially the
sun position). This probe ranks the candidate second views the way the canonical
first-view choice was made (see scripts/figures/_common.py SESSION_HOLDOUT_INDICES
and the lk2/lwp holdout-replacement commits):

  * radius   = ||camera origin|| in the normalised model frame
  * d_train  = distance to the NEAREST training camera origin (in-envelope if
               d_train <~ 0.05: the holdout then sits inside the region training
               rays actually constrained, so it renders sharply not as mush)

For each session it prints every candidate (all test images of the session except
the canonical first holdout and the compare/eval view) ranked by d_train, and
proposes the smallest-d_train candidate as the second view. Loads only the
dataparser (no model, no GPU), so run it while a GPU job holds the 4090.

    PYTHONPATH=.:scripts/figures python scripts/figures/probe_holdout_envelope.py --scene lk2

Pin the run whose normalised frame you are scoring against:

    NEUSKY_RUNS="lk2=outputs/lk2_rngfix_w32cyc_hash21/neusky/2026-07-05_082759" \
        PYTHONPATH=.:scripts/figures python scripts/figures/probe_holdout_envelope.py --scene lk2
"""

import argparse
import sys
import types
from pathlib import Path


def _stub_tinycudann_if_unavailable() -> None:
    """Let the saved config reconstruct without a GPU.

    yaml.load of a run's config.yml reconstructs the whole pipeline config, which
    transitively imports neusky.fields.directional_distance_field -> tinycudann.
    tinycudann raises OSError at import when no CUDA device is visible
    (CUDA_VISIBLE_DEVICES= keeps this probe off a busy GPU). Only field *setup*
    touches tinycudann; reconstructing config dataclasses and parsing the
    dataparser (all this probe does) never calls it, so a stub whose attributes
    resolve to a dummy is sufficient and keeps the probe strictly CPU-only.
    """
    try:
        import tinycudann  # noqa: F401
        return
    except (ImportError, OSError):
        stub = types.ModuleType("tinycudann")
        stub.__file__ = "<tinycudann-stub>"

        def _getattr(name):
            # Leave dunders (__file__, __path__, ...) to normal lookup so
            # inspect.getmodule / import machinery still works on the stub;
            # only tcnn.<public symbol> resolves to a harmless dummy.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return object

        stub.__getattr__ = _getattr
        sys.modules["tinycudann"] = stub


_stub_tinycudann_if_unavailable()

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402  (chdir to repo root side effect)
    REPO_ROOT,
    SESSION_HOLDOUT_INDICES,
    canonical_scene,
    resolve_run_dir,
    _remap_data_path,
)


def load_test_dataparser(scene: str):
    """Load the pinned run's dataparser and return its test-split outputs + train origins.

    Uses the SAVED run config so radii / d_train are computed in exactly the
    normalised frame the scoring uses (SfM centering, scale caps, orientation).
    """
    import yaml

    run_dir = resolve_run_dir(scene)
    config = yaml.load((run_dir / "config.yml").read_text(), Loader=yaml.Loader)

    dp_cfg = config.pipeline.datamanager.dataparser
    dp_cfg.data = _remap_data_path(dp_cfg.data)
    # Set valid canonical holdout indices so get_dataparser_outputs("test") does
    # not trip the holdout/compare overlap assertion with the training default.
    dp_cfg.session_holdout_indices = SESSION_HOLDOUT_INDICES[scene]

    dataparser = dp_cfg.setup()
    test_outputs = dataparser.get_dataparser_outputs("test")
    train_outputs = dataparser.get_dataparser_outputs("train")
    train_origins = train_outputs.cameras.camera_to_worlds[:, :3, 3].double()
    return test_outputs, train_origins, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="lk2")
    ap.add_argument("--envelope", type=float, default=0.05,
                    help="in-envelope threshold on d_train (default 0.05)")
    args = ap.parse_args()

    import torch

    scene = canonical_scene(args.scene)
    test_outputs, train_origins, run_dir = load_test_dataparser(scene)
    meta = test_outputs.metadata
    session_to_indices = meta["session_to_indices"]        # {session: [abs test idxs]}
    indices_to_session = meta["indices_to_session"]         # {abs test idx: session}
    session_names = meta.get("session_names") or {}
    test_origins = test_outputs.cameras.camera_to_worlds[:, :3, 3].double()

    # Compare/eval view per session: the test image carrying an eval mask
    # (test_eval_mask_dict), mirroring make_tables.evaluate_holdout_sensitivity.
    test_eval_mask_dict = meta["test_eval_mask_dict"]
    compare_of = {indices_to_session[i]: i for i in test_eval_mask_dict}

    canonical = SESSION_HOLDOUT_INDICES[scene]

    def name(idx):
        return Path(test_outputs.image_filenames[idx]).stem

    def d_train(abs_idx):
        return float((train_origins - test_origins[abs_idx]).norm(dim=-1).min())

    def radius(abs_idx):
        return float(test_origins[abs_idx].norm())

    train_r = train_origins.norm(dim=-1)
    print(f"[run] {run_dir}")
    print(f"[train envelope] {len(train_origins)} cameras, radius "
          f"max {train_r.max():.3f} median {train_r.median():.3f}")
    print(f"[criterion] in-envelope if d_train < {args.envelope}\n")

    proposed = {}
    for s in sorted(session_to_indices):
        sname = session_names.get(s, s)
        first_rel = canonical[s]
        first_rels = list(first_rel) if isinstance(first_rel, (list, tuple)) else [first_rel]
        abs_idxs = session_to_indices[s]
        compare_abs = compare_of.get(s)

        print(f"=== session {s} ({sname}) ===")
        print(f"    canonical first holdout rel {first_rels} -> "
              f"{[name(abs_idxs[r]) for r in first_rels]} "
              f"(radius {[round(radius(abs_idxs[r]), 3) for r in first_rels]}, "
              f"d_train {[round(d_train(abs_idxs[r]), 3) for r in first_rels]})")
        if compare_abs is not None:
            print(f"    compare/eval view rel {abs_idxs.index(compare_abs)} -> "
                  f"{name(compare_abs)} (excluded)")

        # Candidates: every test image of the session except the canonical first
        # holdout(s) and the compare view; rank by nearest-train-camera distance.
        excluded = set(first_rels)
        if compare_abs is not None:
            excluded.add(abs_idxs.index(compare_abs))
        cands = [(rel, abs_idxs[rel]) for rel in range(len(abs_idxs)) if rel not in excluded]
        cands.sort(key=lambda ra: d_train(ra[1]))

        for rel, ai in cands:
            flag = "in-env" if d_train(ai) < args.envelope else "  OOD "
            print(f"      rel {rel:2d}  {name(ai):<12s}  radius {radius(ai):.3f}  "
                  f"d_train {d_train(ai):.3f}  [{flag}]")

        in_env = [ra for ra in cands if d_train(ra[1]) < args.envelope]
        pick = (in_env or cands)[0] if cands else None
        if pick is not None:
            rel, ai = pick
            proposed[s] = first_rels + [rel]
            print(f"    -> second view rel {rel} ({name(ai)}), "
                  f"radius {radius(ai):.3f}, d_train {d_train(ai):.3f}")
        else:
            proposed[s] = first_rels
            print("    -> no candidate available; keeping single view")
        print()

    two_view = [proposed[s] for s in sorted(proposed)]
    print(f'SESSION_HOLDOUT_INDICES["{scene}"] two-view form:')
    print(f"    {scene!r}: {two_view},")


if __name__ == "__main__":
    main()
