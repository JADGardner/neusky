"""Release invariants for the NeuSky synthetic dataset generator."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_dataset"
SCENES = {
    "abandoned_buildings",
    "apartment_building",
    "arlanda_uppsala_cathedral",
    "glass_building",
    "interstellar_house",
}


def _load_release_builder():
    path = GENERATOR / "build_hf_release.py"
    spec = importlib.util.spec_from_file_location("build_hf_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_ids() -> list[str]:
    return [
        line.strip()
        for line in (GENERATOR / "hdris_16k.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _generation_checksums() -> dict[str, str]:
    checksums = {}
    for line in (GENERATOR / "hdris_16k_generation_md5.txt").read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        checksums[filename] = checksum
    return checksums


def test_hdri_selection_and_manifests_are_synchronised():
    asset_ids = _asset_ids()
    assert len(asset_ids) == 167
    assert asset_ids == sorted(set(asset_ids))

    manifest = json.loads((GENERATOR / "hdris_16k_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["resolution"] == "16k"
    assert manifest["format"] == "exr"
    assert manifest["count"] == len(asset_ids)
    assert [asset["asset_id"] for asset in manifest["assets"]] == asset_ids

    generation_checksums = _generation_checksums()
    assert set(generation_checksums) == {f"{asset_id}.exr" for asset_id in asset_ids}
    changed = [
        asset["asset_id"]
        for asset in manifest["assets"]
        if generation_checksums[f"{asset['asset_id']}.exr"] != asset["md5"]
    ]
    assert len(changed) == 36


def test_render_config_contains_the_accepted_scenes_and_profiles():
    config = json.loads((GENERATOR / "scene_render_configs.json").read_text())
    assert set(config["scenes"]) == SCENES
    assert {"train", "train_curated", "eval"} <= set(config["common"])
    assert "--hdri_16k" in config["common"]["train"]
    assert "--hdri_16k" in config["common"]["train_curated"]
    assert "--hdri_16k" in config["common"]["eval"]


def test_scene_provenance_covers_the_accepted_scenes():
    sources = json.loads((GENERATOR / "scene_sources.json").read_text())
    assert sources["schema_version"] == 1
    assert set(sources["scenes"]) == SCENES

    for scene_name, scene in sources["scenes"].items():
        accepted = scene["accepted_scene"]
        assert accepted["filename"] == f"{scene_name}.blend"
        assert len(accepted["sha256"]) == 64
        assert accepted["size_bytes"] > 0
        assert scene["base_model"]["listing_url"].startswith("https://")

    interstellar = sources["scenes"]["interstellar_house"]["base_model"]
    assert interstellar["availability"] == "removed_from_blenderkit_on_2026-07-30"


def test_huggingface_release_uses_an_explicit_allowlist():
    builder = _load_release_builder()
    assert set(builder.SCENES) == SCENES
    assert builder.SPLIT_COUNTS == {"train": 250, "validation": 25, "test": 25}
    assert set(builder.SPLIT_LAYERS["train"]) == {"rgb", "cityscapes_mask"}
    assert {"albedo", "normal", "depth"} <= set(
        builder.SPLIT_LAYERS["validation"]
    )
    assert "_replaced_eval" not in json.dumps(builder.SPLIT_LAYERS)

    card = (GENERATOR / "DATASET_CARD.md").read_text()
    assert "license: cc-by-4.0" in card
    assert "1,500" in card
    assert "@@REPO_ID@@" in card
    assert "archives/interstellar_house.tar.zst" in card
    assert "scenes/<scene>/" in card
