"""CPU parity tests for the two-bracket RENI++ illumination prior decode.

These verify that :class:`neusky.model_components.illumination_hdr.IlluminationHDRDecode`
turns RENI field outputs into linear HDR identically to the ns_reni reference:

* a two-bracket (6-channel) checkpoint decodes via
  ``reni.utils.tonemap.two_bracket_to_linear``;
* a standard 3-channel checkpoint still decodes via ``RENIField.unnormalise``
  (byte-for-byte unchanged).

Everything here is CPU-only and free of the tinycudann / DDF stack, so it runs
without a GPU (the training GPU stays untouched):

    docker compose run --rm -e CUDA_VISIBLE_DEVICES= research bash -c \
        "cd /workspace/phd && PYTHONPATH=/workspace/phd/outputs/neusky_b2:/workspace/phd/code/ns_reni \
         python tests/test_illumination_hdr.py --device cpu"

or under pytest:  python -m pytest tests/test_illumination_hdr.py -v
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure THIS worktree's ``neusky`` package is imported (the research container's
# entrypoint editable-installs the primary code/neusky worktree, which would
# otherwise shadow the two-bracket integration under test). ``reni`` is left to
# resolve via PYTHONPATH (the canonical code/ns_reni).
_WORKTREE_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_WORKTREE_ROOT]:
    sys.path.insert(0, _WORKTREE_ROOT)

import torch

try:  # pytest is optional: the file also runs as a plain CPU script.
    import pytest

    _SkipException = pytest.skip.Exception

    def _skip(msg: str):
        pytest.skip(msg)
except ImportError:  # pragma: no cover - container has no pytest

    class _SkipException(Exception):
        pass

    def _skip(msg: str):
        raise _SkipException(msg)

from nerfstudio.cameras.rays import Frustums, RaySamples

from reni.field_components.field_heads import RENIFieldHeadNames
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.utils.tonemap import two_bracket_to_linear

from neusky.model_components.illumination_hdr import IlluminationHDRDecode

# Checkpoint run dirs (config.yml + nerfstudio_models). Overridable for other
# hosts; defaults are the container mount paths.
TWO_BRACKET_DIR = Path(
    os.environ.get(
        "NEUSKY_TB_CKPT_DIR",
        "model-storage/reni/neusky-prior",
    )
)
PAPER_DIR = Path(
    os.environ.get(
        "NEUSKY_PP_CKPT_DIR",
        "model-storage/reni/published/reni_plus_plus_models/latent_dim_100",
    )
)

# Known ground-truth two-bracket saturation points for the checkpoint above,
# used as an INDEPENDENT reference (not read back from the config) so the test
# also catches a mis-read of tonemap_m_ldr / tonemap_m_log.
KNOWN_M_LDR = 16.0
KNOWN_M_LOG = 10000.0

DEVICE = "cpu"


def _base_reni_config() -> RENIFieldConfig:
    """The RENIField architecture NeuSky uses for its illumination prior.

    Mirrors ``neusky/configs/neusky_config.py``; the two-bracket override
    (out_features / output_activation) is applied exactly as NeuSky's
    ``populate_modules`` does.
    """
    return RENIFieldConfig(
        conditioning="Attention",
        invariant_function="VN",
        equivariance="SO2",
        axis_of_invariance="z",
        positional_encoding="NeRF",
        encoded_input="Directions",
        latent_dim=100,
        hidden_features=128,
        hidden_layers=9,
        mapping_layers=5,
        mapping_features=128,
        num_attention_heads=8,
        num_attention_layers=6,
        output_activation="None",
        last_layer_linear=True,
        fixed_decoder=True,
        trainable_scale=False,
    )


def _latest_ckpt(run_dir: Path) -> Path:
    ckpts = sorted((run_dir / "nerfstudio_models").glob("step-*.ckpt"))
    assert ckpts, f"no checkpoint under {run_dir / 'nerfstudio_models'}"
    return ckpts[-1]


def _build_field_like_neusky(run_dir: Path, decode: IlluminationHDRDecode) -> RENIField:
    """Reproduce NeuSky ``populate_modules``: build the field to match the
    checkpoint architecture and load ONLY the decoder weights."""
    cfg = _base_reni_config()
    if decode.two_bracket:
        cfg.out_features = decode.out_features
        cfg.output_activation = decode.output_activation
    field = cfg.setup(num_train_data=None, num_eval_data=None).to(DEVICE).eval()

    ckpt = torch.load(_latest_ckpt(run_dir), map_location=DEVICE, weights_only=False)
    match_str = "_model.field."
    ignore = [
        "_model.field.train_logvar",
        "_model.field.eval_logvar",
        "_model.field.train_mu",
        "_model.field.eval_mu",
    ]
    state = {
        k[len(match_str):]: v
        for k, v in ckpt["pipeline"].items()
        if k.startswith(match_str) and not any(s in k for s in ignore)
    }
    missing, unexpected = field.load_state_dict(state, strict=False)
    # The only tolerated gaps are the latent tables we deliberately dropped.
    unexpected_real = [k for k in unexpected if not any(s.split(".", 2)[-1] in k for s in ignore)]
    assert not unexpected_real, f"unexpected decoder keys: {unexpected_real}"
    return field


def _forward_raw(field: RENIField, n: int = 64, seed: int = 0) -> torch.Tensor:
    """Decode a small random direction grid with one random latent code."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    ray_samples = RaySamples(
        frustums=Frustums(
            origins=torch.zeros(n, 3),
            directions=dirs,
            starts=torch.zeros(n, 1),
            ends=torch.zeros(n, 1),
            pixel_area=torch.ones(n, 1),
        ),
        camera_indices=torch.zeros(n, 1, dtype=torch.long),
    ).to(DEVICE)
    latent = torch.randn(1, field.latent_dim, 3, generator=g).repeat(n, 1, 1).to(DEVICE)
    with torch.no_grad():
        out = field.forward(ray_samples=ray_samples, latent_codes=latent)
    return out[RENIFieldHeadNames.RGB]


def _require(run_dir: Path):
    if not (run_dir / "config.yml").exists():
        _skip(f"checkpoint config not found: {run_dir / 'config.yml'}")


def test_two_bracket_config_detection():
    _require(TWO_BRACKET_DIR)
    decode = IlluminationHDRDecode.from_reni_run_config(TWO_BRACKET_DIR / "config.yml")
    assert decode.two_bracket is True
    assert decode.out_features == 6
    assert decode.output_activation == "sigmoid"
    assert decode.m_ldr == KNOWN_M_LDR
    assert decode.m_log == KNOWN_M_LOG
    assert decode.fixed_gauge is True


def test_paper_config_detection():
    _require(PAPER_DIR)
    decode = IlluminationHDRDecode.from_reni_run_config(PAPER_DIR / "config.yml")
    assert decode.two_bracket is False
    assert decode.out_features == 3
    assert decode.output_activation == "None"
    assert decode.fixed_gauge is False


def test_two_bracket_helper_matches_reference():
    _require(TWO_BRACKET_DIR)
    decode = IlluminationHDRDecode.from_reni_run_config(TWO_BRACKET_DIR / "config.yml")
    field = _build_field_like_neusky(TWO_BRACKET_DIR, decode)
    raw = _forward_raw(field)

    assert raw.shape[-1] == 6, "two-bracket field must emit six channels"
    assert torch.all((raw >= 0) & (raw <= 1)), "sigmoid brackets must be in [0, 1]"

    neusky_hdr = decode.to_linear_hdr(field, raw)
    # Independent reference with the KNOWN checkpoint saturation points.
    reference = two_bracket_to_linear(raw, m_ldr=KNOWN_M_LDR, m_log=KNOWN_M_LOG)

    assert neusky_hdr.shape[-1] == 3
    assert torch.allclose(neusky_hdr, reference, atol=1e-6, rtol=1e-5)
    assert torch.all(neusky_hdr >= 0)


def test_two_bracket_helper_applies_scale_after_reconstruction():
    _require(TWO_BRACKET_DIR)
    decode = IlluminationHDRDecode.from_reni_run_config(TWO_BRACKET_DIR / "config.yml")
    field = _build_field_like_neusky(TWO_BRACKET_DIR, decode)
    raw = _forward_raw(field)
    scale = torch.linspace(-0.5, 0.5, raw.shape[0], device=raw.device, dtype=raw.dtype)

    neusky_hdr = decode.to_linear_hdr(field, raw, scale=scale)
    reference = two_bracket_to_linear(
        raw, m_ldr=KNOWN_M_LDR, m_log=KNOWN_M_LOG
    ) * torch.exp(scale).unsqueeze(-1)

    assert torch.allclose(neusky_hdr, reference, atol=1e-6, rtol=1e-5)


def test_three_channel_helper_matches_unnormalise():
    _require(PAPER_DIR)
    decode = IlluminationHDRDecode.from_reni_run_config(PAPER_DIR / "config.yml")
    assert decode.two_bracket is False
    field = _build_field_like_neusky(PAPER_DIR, decode)
    raw = _forward_raw(field)

    assert raw.shape[-1] == 3
    neusky_hdr = decode.to_linear_hdr(field, raw)
    reference = field.unnormalise(raw)  # the pre-existing 3-channel path
    # Byte-for-byte identical: the helper simply defers to unnormalise here.
    assert torch.equal(neusky_hdr, reference)


def _run_all() -> int:
    checks = [
        test_two_bracket_config_detection,
        test_paper_config_detection,
        test_two_bracket_helper_matches_reference,
        test_two_bracket_helper_applies_scale_after_reconstruction,
        test_three_channel_helper_matches_unnormalise,
    ]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except _SkipException as exc:
            print(f"SKIP  {check.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed (device={DEVICE})")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    DEVICE = args.device
    raise SystemExit(1 if _run_all() else 0)
