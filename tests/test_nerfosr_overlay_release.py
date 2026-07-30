"""Release invariants for the NeuSky additions to NeRF-OSR."""

from __future__ import annotations

import importlib.util
from pathlib import Path


GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "nerfosr_overlay"


def _load_release_builder():
    path = GENERATOR / "build_hf_release.py"
    spec = importlib.util.spec_from_file_location("build_nerfosr_overlay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_uses_an_explicit_additions_only_allowlist():
    builder = _load_release_builder()

    assert builder.SCENES == ("lk2", "lwp", "st")
    assert builder.SPLITS == ("train", "validation", "test")
    assert builder.ALLOWED_FINAL_FILES == (
        "points3d.ply",
        "envmap_rotations.json",
    )
    assert sum(
        sum(split_counts.values())
        for split_counts in builder.MASK_COUNTS.values()
    ) == 1021


def test_dataset_card_describes_an_overlay_not_a_dataset_mirror():
    card = (GENERATOR / "DATASET_CARD.md").read_text()

    assert "license: cc-by-4.0" in card
    assert "does **not** contain" in card
    assert "NeRF-OSR RGB images" in card
    assert "COLMAP models" in card
    assert "Data/<scene>/final/" in card
    assert "@@REPO_ID@@" in card
    assert "--revision v@@RELEASE_VERSION@@" in card
    assert "/resolve/v@@RELEASE_VERSION@@/" in card
