#!/usr/bin/env python3
"""Build the allowlisted Hugging Face release for NeuSky Synthetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

SCENES = (
    "abandoned_buildings",
    "apartment_building",
    "arlanda_uppsala_cathedral",
    "glass_building",
    "interstellar_house",
)
SPLIT_COUNTS = {"train": 250, "validation": 25, "test": 25}
SPLIT_LAYERS = {
    "train": {
        "rgb": ".png",
        "cityscapes_mask": ".png",
    },
    "validation": {
        "rgb": ".png",
        "cityscapes_mask": ".png",
        "albedo": ".exr",
        "normal": ".exr",
        "depth": ".exr",
        "roughness": ".exr",
        "metallic": ".exr",
        "transmission": ".exr",
        "ior": ".exr",
    },
    "test": {
        "rgb": ".png",
        "cityscapes_mask": ".png",
        "albedo": ".exr",
        "normal": ".exr",
        "depth": ".exr",
        "roughness": ".exr",
        "metallic": ".exr",
        "transmission": ".exr",
        "ior": ".exr",
    },
}
PROVENANCE_FILES = (
    "POLYHAVEN_SOURCE_AUDIT.md",
    "hdris_16k.txt",
    "hdris_16k_generation_md5.txt",
    "hdris_16k_manifest.json",
    "scene_render_configs.json",
    "scene_sources.json",
    "synthetic_scenes.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def validate_scene(source: Path, scene: str) -> dict[str, object]:
    transforms_path = source / "transforms.json"
    pointcloud_path = source / "points3d.ply"
    if not transforms_path.is_file() or not pointcloud_path.is_file():
        raise FileNotFoundError(f"{scene}: transforms.json or points3d.ply is missing")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"{scene}: symbolic links are not permitted in the release")

    transforms = json.loads(transforms_path.read_text())
    frames = transforms.get("frames", [])
    split_counts = Counter()
    expected_rgb_stems: dict[str, set[str]] = {
        split: set() for split in SPLIT_COUNTS
    }
    for frame in frames:
        relative = Path(frame["file_path"])
        if len(relative.parts) != 3 or relative.parts[1] != "rgb":
            raise ValueError(f"{scene}: unexpected frame path {relative}")
        split = relative.parts[0]
        if split not in SPLIT_COUNTS:
            raise ValueError(f"{scene}: unexpected split {split}")
        split_counts[split] += 1
        expected_rgb_stems[split].add(relative.stem)

    if dict(split_counts) != SPLIT_COUNTS:
        raise ValueError(
            f"{scene}: split counts {dict(split_counts)} != {SPLIT_COUNTS}"
        )

    file_count = 2
    for split, layers in SPLIT_LAYERS.items():
        for layer, suffix in layers.items():
            layer_dir = source / split / layer
            files = sorted(layer_dir.glob(f"*{suffix}"))
            stems = {path.stem for path in files}
            if stems != expected_rgb_stems[split]:
                missing = sorted(expected_rgb_stems[split] - stems)
                extra = sorted(stems - expected_rgb_stems[split])
                raise ValueError(
                    f"{scene}/{split}/{layer}: missing={missing[:5]} "
                    f"extra={extra[:5]}"
                )
            file_count += len(files)

    return {
        "frames": len(frames),
        "split_counts": dict(split_counts),
        "file_count": file_count,
        "resolution": [transforms["w"], transforms["h"]],
    }


def stage_scene(source: Path, destination: Path, mode: str) -> None:
    for name in ("transforms.json", "points3d.ply"):
        link_or_copy(source / name, destination / name, mode)
    for split, layers in SPLIT_LAYERS.items():
        for layer, suffix in layers.items():
            for path in sorted((source / split / layer).glob(f"*{suffix}")):
                link_or_copy(
                    path,
                    destination / split / layer / path.name,
                    mode,
                )


def archive_scene(
    source: Path,
    output: Path,
    scene: str,
    mode: str,
) -> None:
    """Create a deterministic, independently downloadable scene archive."""
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("building scene archives requires GNU tar and zstd")

    archive_dir = output / "archives"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"{scene}.tar.zst"

    with tempfile.TemporaryDirectory(prefix=f".{scene}-", dir=output) as temporary:
        temporary_root = Path(temporary)
        stage_scene(source, temporary_root / "scenes" / scene, mode)
        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=@0",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "--format=gnu",
                "--use-compress-program=zstd -T0 -3 --no-progress",
                "-cf",
                str(archive_path),
                "-C",
                str(temporary_root),
                f"scenes/{scene}",
            ],
            check=True,
        )


def copy_release_metadata(
    output: Path,
    overview: Path,
    repo_id: str,
    version: str,
    commit: str,
) -> None:
    card = (SCRIPT_DIR / "DATASET_CARD.md").read_text()
    replacements = {
        "@@NEUSKY_COMMIT@@": commit,
        "@@RELEASE_DATE@@": date.today().isoformat(),
        "@@RELEASE_VERSION@@": version,
        "@@REPO_ID@@": repo_id,
    }
    for placeholder, value in replacements.items():
        card = card.replace(placeholder, value)
    if "@@" in card:
        raise ValueError("unresolved placeholder in dataset card")
    (output / "README.md").write_text(card)
    shutil.copy2(SCRIPT_DIR / "DATASET_LICENSE.md", output / "LICENSE.md")

    provenance = output / "provenance"
    provenance.mkdir()
    for filename in PROVENANCE_FILES:
        shutil.copy2(SCRIPT_DIR / filename, provenance / filename)

    benchmark = output / "benchmark"
    benchmark.mkdir()
    shutil.copy2(
        REPO_ROOT / "publication" / "tables_synthetic_full" / "synthetic.csv",
        benchmark / "synthetic_results.csv",
    )

    assets = output / "assets"
    assets.mkdir()
    shutil.copy2(overview, assets / "synthetic_dataset_overview.png")


def write_manifests(
    output: Path,
    version: str,
    commit: str,
    scene_stats: dict[str, dict[str, object]],
) -> None:
    stats = {
        "schema_version": 1,
        "dataset": "NeuSky Synthetic",
        "release_version": version,
        "generator_commit": commit,
        "scenes": scene_stats,
        "totals": {
            "scenes": len(scene_stats),
            "frames": sum(int(stats["frames"]) for stats in scene_stats.values()),
            "train_frames": len(scene_stats) * SPLIT_COUNTS["train"],
            "validation_frames": len(scene_stats) * SPLIT_COUNTS["validation"],
            "test_frames": len(scene_stats) * SPLIT_COUNTS["test"],
        },
    }
    (output / "DATASET_STATS.json").write_text(json.dumps(stats, indent=2) + "\n")

    entries = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "SHA256SUMS"}:
            continue
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "NeuSky Synthetic",
        "release_version": version,
        "generated_on": date.today().isoformat(),
        "generator_repository": "https://github.com/JADGardner/neusky",
        "generator_commit": commit,
        "licence": "CC-BY-4.0",
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(
                f"{sha256(path)}  {path.relative_to(output).as_posix()}"
            )
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


def parse_args() -> argparse.Namespace:
    default_data = Path(
        os.environ.get(
            "NEUSKY_SYN_DATA",
            Path.home() / "data" / "neusky_synthetic_data",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help="NeuSky synthetic working-data root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overview", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="jadgardner/neusky-synthetic",
        help="Hugging Face dataset repository ID written into the card",
    )
    parser.add_argument("--version", default="1.0")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to stage the large scene files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.overview.is_file():
        raise FileNotFoundError(args.overview)

    commit = git_commit()
    scene_stats = {}
    renders = args.data_root.resolve() / "renders"
    for scene in SCENES:
        source = renders / f"{scene}_prepared"
        print(f"[validate] {scene}")
        scene_stats[scene] = validate_scene(source, scene)
        print(f"[archive] {scene}")
        archive_scene(source, output, scene, args.mode)

    copy_release_metadata(
        output=output,
        overview=args.overview.resolve(),
        repo_id=args.repo_id,
        version=args.version,
        commit=commit,
    )
    print("[hash] release files")
    write_manifests(output, args.version, commit, scene_stats)

    forbidden = [
        path
        for path in output.rglob("*")
        if path.name.endswith((".blend", ".blend1"))
        or "_replaced_eval" in path.parts
    ]
    if forbidden:
        raise ValueError(f"forbidden release paths: {forbidden}")

    manifest = json.loads((output / "MANIFEST.json").read_text())
    print(
        f"[done] {manifest['file_count']} files, "
        f"{manifest['total_bytes'] / 2**30:.2f} GiB at {output}"
    )


if __name__ == "__main__":
    main()
