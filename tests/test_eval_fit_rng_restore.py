"""The seeded eval-latent fit must not perturb the global RNG streams.

Regression test for the 2026-07 "speckle" bug: fit_latent_codes_for_eval
seeds python/numpy/torch RNGs for deterministic relighting fits; without
restoring the prior states, every in-training eval reset the training
ray-sampling stream to the same sequence, overfitting the repeated batches
and baking speckle into geometry and the DDF.

Runs on CPU: exercises the snapshot/restore pair directly on an
un-initialised NeuSkyModel instance.
"""
import random

import numpy as np
import torch

from neusky.models.neusky_model import NeuSkyFactoModel as NeuSkyModel


def _snapshot_then_seed(model, seed=42):
    model._pre_fit_rng_states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_rng_streams_continue_across_seeded_fit():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Control: the stream we expect training to see with no eval fit at all.
    _ = random.random(), np.random.rand(), torch.rand(3)
    control = (random.random(), float(np.random.rand()), torch.rand(3))

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    _ = random.random(), np.random.rand(), torch.rand(3)

    model = NeuSkyModel.__new__(NeuSkyModel)  # no __init__: helper is self-contained
    _snapshot_then_seed(model, seed=42)
    # the seeded fit consumes randomness
    _ = random.random(), np.random.rand(), torch.rand(100)
    model._restore_rng_after_eval_fit()

    resumed = (random.random(), float(np.random.rand()), torch.rand(3))
    assert resumed[0] == control[0]
    assert resumed[1] == control[1]
    assert torch.equal(resumed[2], control[2])
    assert model._pre_fit_rng_states is None


def test_restore_is_noop_without_snapshot():
    model = NeuSkyModel.__new__(NeuSkyModel)
    model._restore_rng_after_eval_fit()  # must not raise


def test_all_fit_paths_restore_rng():
    """Every return path of fit_latent_codes_for_eval must restore the RNG.

    Regression for the synthetic_gt_envmap early-return, which skipped the
    restore and reintroduced the speckle mechanism for --fit-gt-envmap-latents
    runs. Static check: each `return` inside the function body (and its end)
    must be preceded by a _restore_rng_after_eval_fit() call within the
    preceding lines of the same branch.
    """
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(NeuSkyModel.fit_latent_codes_for_eval))
    lines = src.splitlines()
    returns = [i for i, l in enumerate(lines) if l.strip() == "return"]
    for i in returns:
        window = "\n".join(lines[max(0, i - 3):i])
        assert "_restore_rng_after_eval_fit()" in window, (
            f"return at source line {i} not preceded by RNG restore:\n{window}")
    assert "_restore_rng_after_eval_fit()" in "\n".join(lines[-6:]), "final path must restore"
