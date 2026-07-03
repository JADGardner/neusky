"""Compare the effective normalised-scene layout of two NeRF-OSR runs (CPU).

The same physical cameras exist in both runs' normalised frames, so a
similarity transform (Procrustes) between the two sets of camera origins
recovers the relative scale and the image of one frame's origin in the other.
Run A (SfM centering) puts the building at its origin with a known p50 point
radius (printed by sfm_utils at load time), so this yields, for BOTH runs:
where the building sits and how large it is in the normalised box — i.e. how
many hash-grid voxels span the facade.

    PYTHONPATH=.:scripts/figures python scripts/figures/probe_scene_scale.py \
        --run-a outputs/lk2_refit_sfm_lrprior/neusky/2026-07-02_183521 \
        --run-b outputs/lk2_refit_optimised/neusky/2026-06-11_121134 \
        --building-p50 0.171
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO_ROOT  # noqa: E402  (chdir to repo root side effect)


def load_train_origins(run_dir: Path):
    import torch
    import yaml

    config = yaml.load((run_dir / "config.yml").read_text(), Loader=yaml.Loader)
    dataparser = config.pipeline.datamanager.dataparser.setup()
    outputs = dataparser.get_dataparser_outputs(split="train")
    return outputs.cameras.camera_to_worlds[:, :3, 3].double()


def procrustes_similarity(src, dst):
    """Least-squares s, R, t with dst ~ s * R @ src + t. src/dst: [N, 3]."""
    import torch

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = torch.linalg.svd(cov)
    d = torch.sign(torch.det(U @ Vt))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=src.dtype))
    R = U @ D @ Vt
    scale = (S * torch.tensor([1.0, 1.0, d], dtype=src.dtype)).sum() / xs.pow(2).sum(1).mean()
    t = mu_d - scale * (R @ mu_s)
    resid = (scale * (R @ src.T).T + t - dst).norm(dim=-1).max()
    return scale.item(), R, t, resid.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", type=Path, required=True,
                    help="SfM-centred run (building at origin)")
    ap.add_argument("--run-b", type=Path, required=True,
                    help="Comparison run (e.g. June pose-centred)")
    ap.add_argument("--building-p50", type=float, required=True,
                    help="p50 building-point radius in run A's frame "
                         "(raw p50 x capped SfM scale, from the load log)")
    args = ap.parse_args()

    a = load_train_origins(REPO_ROOT / args.run_a if not args.run_a.is_absolute() else args.run_a)
    b = load_train_origins(REPO_ROOT / args.run_b if not args.run_b.is_absolute() else args.run_b)
    assert len(a) == len(b), f"camera counts differ: {len(a)} vs {len(b)}"

    scale, R, t, resid = procrustes_similarity(a, b)
    print(f"[procrustes] scale B/A = {scale:.4f}, max residual {resid:.4f} "
          f"(should be ~0 if same cameras)")

    print(f"[run A] cameras: max radius {a.norm(dim=-1).max():.3f}, "
          f"median {a.norm(dim=-1).median():.3f}")
    print(f"[run B] cameras: max radius {b.norm(dim=-1).max():.3f}, "
          f"median {b.norm(dim=-1).median():.3f}")

    building_b = t  # image of run A's origin (the building centre) in B's frame
    print(f"[building] run A: centre at origin, p50 radius {args.building_p50:.3f}")
    print(f"[building] run B: centre at radius {building_b.norm():.3f} "
          f"(pos {[round(float(x), 3) for x in building_b]}), "
          f"p50 radius {args.building_p50 * scale:.3f}")
    print(f"[verdict] building is {scale:.2f}x "
          f"{'LARGER' if scale > 1 else 'smaller'} in run B's box than run A's; "
          f"finest-hash-level voxels across the facade scale by the same factor.")


if __name__ == "__main__":
    main()
