"""What does the holdout eval-latent fit actually supervise the sky with?

Samples the nerf_osr_holdout optimise bundle exactly as fit_latent_codes_for_eval
does, then reports per session: how many rays are sky (mask channel 3), how many
survive the accumulation<0.5 gate, and the MEAN GT COLOUR of those rays. If the
mean GT colour of the supervised sky pixels is white/grey rather than blue, the
fit is being starved of the blue-sky evidence (sampler mask channel 0 restricts
which pixels can be drawn at all).

    NEUSKY_RUNS='lk2=outputs/<run>' PYTHONPATH=.:scripts/figures \
        python scripts/figures/probe_holdout_sky_supervision.py --scene lk2
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _common import SESSION_HOLDOUT_INDICES, load_model, seed_all  # noqa: E402
from make_tables import EVAL_FITS, apply_eval_fit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="lk2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20,
                        help="Bundle draws to aggregate (each = one fit step's batch)")
    parser.add_argument("--eval-fit", choices=list(EVAL_FITS), default="sky_only")
    parser.add_argument("--fit", action="store_true",
                        help="Run the actual latent fit first, then compare the FIT path "
                             "(fitting_eval_latents=True) against the RENDER path on the "
                             "same rays — separates 'fit failed to converge' from 'fit "
                             "converged but the display/render path reads different "
                             "latents/scale/directions'.")
    parser.add_argument("--fit-steps", type=int, default=300,
                        help="Override fit length when --fit is set (speed).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)

    import torch

    def hook(config):
        config.pipeline.model.eval_latent_optimise_method = "nerf_osr_holdout"
        apply_eval_fit(config, args.eval_fit)
        if args.fit:
            config.pipeline.model.eval_latent_optimizer["eval_latents"][
                "scheduler"].max_steps = args.fit_steps
        config.pipeline.datamanager.dataparser.session_holdout_indices = \
            SESSION_HOLDOUT_INDICES[args.scene]

    _, pipeline, _, step = load_model(args.scene, device=args.device,
                                      config_hook=hook, eval_num_rays_per_chunk=1024)
    dm = pipeline.datamanager
    model = pipeline.model
    model.eval()

    if args.fit:
        from reni.utils.colourspace import linear_to_sRGB

        model.fit_latent_codes_for_eval(dm, step)
        seed_all(args.seed)  # same bundle draws for both paths
        print(f"\nPost-fit path comparison ({args.eval_fit}, {args.fit_steps} steps):")
        session_names = dm.eval_dataset.metadata["session_names"]
        agg = {}
        for _ in range(5):
            ray_bundle, batch = dm.get_nerfosr_lighting_eval_bundle("optimise")
            sky = batch["mask"][..., 3].bool().cpu()
            gt = batch["image"].cpu()
            with torch.no_grad():
                model.fitting_eval_latents = True
                fit_out = model.forward(ray_bundle=ray_bundle, step=step)
                fit_rgb = linear_to_sRGB(fit_out["hdr_background_colours"], clamp=False).cpu()
                model.fitting_eval_latents = False
                ren_out = model.forward(ray_bundle=ray_bundle, step=step)
                ren_rgb = linear_to_sRGB(ren_out["hdr_background_colours"], clamp=False).cpu()
            sess = batch["indices"][:, 0].cpu()
            for s in sess.unique().tolist():
                m = (sess == s) & sky
                if not m.any():
                    continue
                d = agg.setdefault(s, {"n": 0, "gt": 0.0, "fit": 0.0, "ren": 0.0})
                d["n"] += int(m.sum())
                d["gt"] = d["gt"] + gt[m].sum(0)
                d["fit"] = d["fit"] + fit_rgb[m].sum(0)
                d["ren"] = d["ren"] + ren_rgb[m].sum(0)
        for s in sorted(agg):
            d = agg[s]
            f = lambda v: [round(float(x), 3) for x in (v / d["n"])]
            print(f"[{session_names[s]}] sky rays {d['n']:5d} | GT {f(d['gt'])} "
                  f"| FIT-path {f(d['fit'])} | RENDER-path {f(d['ren'])}")
        return

    model.fitting_eval_latents = True

    session_names = dm.eval_dataset.metadata["session_names"]
    stats = {}
    with torch.no_grad():
        for _ in range(args.steps):
            ray_bundle, batch = dm.get_nerfosr_lighting_eval_bundle("optimise")
            outputs = model.forward(ray_bundle=ray_bundle, step=step)
            acc = outputs["accumulation"].squeeze(-1)
            sky = batch["mask"][..., 3].to(acc.device).bool()
            static = batch["mask"][..., 0].to(acc.device).bool()
            gated = sky & (acc < 0.5)
            gt = batch["image"].to(acc.device)
            sess = batch["indices"][:, 0].to(acc.device)
            for s in sess.unique().tolist():
                m = sess == s
                d = stats.setdefault(s, {"rays": 0, "static": 0, "sky": 0, "gated": 0,
                                         "sky_rgb": torch.zeros(3, device=acc.device),
                                         "gated_rgb": torch.zeros(3, device=acc.device)})
                d["rays"] += int(m.sum())
                d["static"] += int((m & static).sum())
                d["sky"] += int((m & sky).sum())
                d["gated"] += int((m & gated).sum())
                if (m & sky).any():
                    d["sky_rgb"] += gt[m & sky].sum(0)
                if (m & gated).any():
                    d["gated_rgb"] += gt[m & gated].sum(0)

    model.fitting_eval_latents = False
    print(f"\nAggregated over {args.steps} bundle draws "
          f"(sampler restricts to mask channel 0 = static):")
    for s in sorted(stats):
        d = stats[s]
        sky_rgb = (d["sky_rgb"] / max(d["sky"], 1)).tolist()
        gated_rgb = (d["gated_rgb"] / max(d["gated"], 1)).tolist()
        print(f"[{session_names[s]}] rays {d['rays']:6d} | static {d['static']:6d} "
              f"| sky {d['sky']:5d} ({100*d['sky']/max(d['rays'],1):.1f}%) "
              f"| sky&acc<0.5 {d['gated']:5d} "
              f"| mean GT sky rgb {[round(v,3) for v in sky_rgb]} "
              f"| gated rgb {[round(v,3) for v in gated_rgb]}")


if __name__ == "__main__":
    main()
