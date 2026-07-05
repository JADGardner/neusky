"""CPU tests for two-view (multi-view) NeRF-OSR holdout designation.

The nerf_osr_holdout relighting protocol fits each test session's RENI++ latent
on held-out view(s) and scores a separate compare view. ``session_holdout_indices``
carries one entry per session, either a single relative index (classic one-view
holdout) or a list of relative indices (two-view / multi-view). These tests cover
the shared resolver + overlap assertion used by both the dataparser and the
datamanager, without touching the GPU or any on-disk NeRF-OSR data.

Runs standalone (the research env has no pytest) and also collects under pytest:

    python tests/test_session_holdout_indices.py
    python -m pytest tests/test_session_holdout_indices.py -v

Imports only the dataparser module's pure helpers (no tinycudann / torch CUDA).
"""

from __future__ import annotations

from contextlib import contextmanager

from neusky.data.dataparsers.nerfosr_cityscapes_dataparser import (
    resolve_session_holdout_indices,
    assert_holdout_not_in_compare,
)


@contextmanager
def raises(exc, match=None):
    """Minimal pytest.raises stand-in so the suite runs without pytest."""
    try:
        yield
    except exc as e:
        if match is not None:
            assert match in str(e), f"expected {match!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


# A stand-in for the dataparser's session_to_indices: session idx -> abs test idxs.
# Sessions have different sizes to catch any exactly-one-per-session assumption.
SESSION_TO_INDICES = {
    0: [10, 11, 12, 13],
    1: [20, 21, 22],
    2: [30, 31, 32, 33, 34],
}


def test_int_form_backward_compat():
    """Plain-int entries resolve to one absolute index per session (unchanged)."""
    holdout = [0, 2, 4]  # rel indices, one per session
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    assert resolved == [10, 22, 34]


def test_list_form_designates_multiple_holdouts():
    """List entries designate several holdout images per session (flattened)."""
    holdout = [[0, 3], [0, 2], [4, 0]]  # two views per session
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    # session 0: rel 0,3 -> 10,13 ; session 1: rel 0,2 -> 20,22 ; session 2: rel 4,0 -> 34,30
    assert resolved == [10, 13, 20, 22, 34, 30]
    # every session still contributes; the total count grows with the list lengths
    assert len(resolved) == 6


def test_mixed_int_and_list_forms():
    """A session may keep a single int while others move to list form."""
    holdout = [[0, 3], 2, [4, 0]]
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    assert resolved == [10, 13, 22, 34, 30]


def test_tuple_entries_supported():
    """Tuple entries behave like list entries."""
    holdout = [(0, 1), 0, 0]
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    assert resolved == [10, 11, 20, 30]


def test_compare_view_overlap_rejected_int_form():
    """A one-view holdout that lands on a compare view is rejected."""
    test_eval_mask_dict = {12: "site/test/mask/0012.png"}  # abs idx 12 is the compare view
    holdout = [2, 0, 0]  # session 0 rel 2 -> abs 12 == compare view
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    with raises(ValueError, match="both a holdout image and an eval image"):
        assert_holdout_not_in_compare(resolved, test_eval_mask_dict)


def test_compare_view_overlap_rejected_in_list_form():
    """A two-view holdout is rejected if EITHER designated view is the compare view."""
    test_eval_mask_dict = {13: "site/test/mask/0013.png"}  # abs idx 13 is the compare view
    holdout = [[0, 3], [0, 2], 0]  # session 0 rel 3 -> abs 13 == compare view
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    with raises(ValueError, match="both a holdout image and an eval image"):
        assert_holdout_not_in_compare(resolved, test_eval_mask_dict)


def test_disjoint_holdout_and_compare_passes():
    """A valid two-view holdout disjoint from all compare views is accepted."""
    test_eval_mask_dict = {12: "m", 21: "m", 31: "m"}  # one compare view per session
    holdout = [[0, 3], [0, 2], [4, 0]]  # abs {10,13,20,22,34,30} disjoint from compare
    resolved = resolve_session_holdout_indices(holdout, SESSION_TO_INDICES)
    assert_holdout_not_in_compare(resolved, test_eval_mask_dict)  # no raise


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
