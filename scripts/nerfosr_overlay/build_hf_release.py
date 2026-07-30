#!/usr/bin/env python3
"""Build the allowlisted Hugging Face release of NeuSky's NeRF-OSR additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from datetime import date
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

SCENES = ("lk2", "lwp", "st")
SPLITS = ("train", "validation", "test")
MASK_COUNTS = {
    "lk2": {"train": 160, "validation": 5, "test": 95},
    "lwp": {"train": 258, "validation": 5, "test": 96},
    "st": {"train": 301, "validation": 5, "test": 96},
}
ROTATION_COUNTS = {"lk2": 6, "lwp": 5, "st": 5}
ALLOWED_FINAL_FILES = ("points3d.ply", "envmap_rotations.json")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(16 * 1024)
    marker = b"end_header\n"
    end = header.find(marker)
    if not header.startswith(b"ply\n") or end < 0:
        raise ValueError(f"not a valid PLY: {path}")
    for line in header[:end].decode("ascii").splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    raise ValueError(f"PLY has no vertex count: {path}")


def determinant_3x3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def validate_rotation(matrix: object, context: str) -> None:
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError(f"{context}: rotation must be a 3x3 matrix")

    values = [[float(value) for value in row] for row in matrix]
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError(f"{context}: rotation contains a non-finite value")

    for i in range(3):
        for j in range(3):
            dot = sum(values[k][i] * values[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-5):
                raise ValueError(f"{context}: rotation is not orthonormal")
    if not math.isclose(determinant_3x3(values), 1.0, abs_tol=1e-5):
        raise ValueError(f"{context}: rotation determinant is not one")


def validate_rotations(path: Path, scene: str) -> dict[str, object]:
    rotations = json.loads(path.read_text())
    if not isinstance(rotations, dict) or len(rotations) != ROTATION_COUNTS[scene]:
        raise ValueError(
            f"{scene}: expected {ROTATION_COUNTS[scene]} rotation records, "
            f"found {len(rotations) if isinstance(rotations, dict) else 'invalid'}"
        )

    composed = 0
    direct = 0
    for session, record in rotations.items():
        if not isinstance(record, dict):
            raise ValueError(f"{scene}/{session}: invalid rotation record")
        validate_rotation(record.get("rotation"), f"{scene}/{session}")
        envmap_image = record.get("envmap_image")
        if envmap_image is None:
            if "method" not in record:
                raise ValueError(
                    f"{scene}/{session}: indirect rotation has no provenance"
                )
            composed += 1
            continue
        relative = Path(envmap_image)
        if (
            relative.is_absolute()
            or len(relative.parts) < 4
            or relative.parts[:2] != ("final", "ENV_MAP_CC")
            or ".." in relative.parts
        ):
            raise ValueError(f"{scene}/{session}: invalid envmap path {relative}")
        direct += 1

    return {
        "records": len(rotations),
        "direct_registrations": direct,
        "composed_rotations": composed,
    }


def scene_artifacts(data_root: Path, scene: str) -> list[tuple[Path, Path]]:
    final = data_root / "Data" / scene / "final"
    artifacts = [
        (final / name, Path("Data") / scene / "final" / name)
        for name in ALLOWED_FINAL_FILES
    ]
    for split in SPLITS:
        mask_dir = final / split / "cityscapes_mask"
        for path in sorted(mask_dir.glob("*.png")):
            artifacts.append(
                (
                    path,
                    Path("Data")
                    / scene
                    / "final"
                    / split
                    / "cityscapes_mask"
                    / path.name,
                )
            )
    return artifacts


def validate_scene(data_root: Path, scene: str) -> dict[str, object]:
    final = data_root / "Data" / scene / "final"
    pointcloud = final / "points3d.ply"
    rotations = final / "envmap_rotations.json"
    if not pointcloud.is_file() or not rotations.is_file():
        raise FileNotFoundError(f"{scene}: points3d.ply or rotations JSON is missing")

    split_stats: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        mask_dir = final / split / "cityscapes_mask"
        rgb_dir = final / split / "rgb"
        masks = sorted(mask_dir.glob("*.png"))
        expected_count = MASK_COUNTS[scene][split]
        if len(masks) != expected_count:
            raise ValueError(
                f"{scene}/{split}: expected {expected_count} masks, found {len(masks)}"
            )
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"{scene}/{split}: official RGB directory missing")

        mask_stems = {path.stem for path in masks}
        rgb_stems = {path.stem for path in rgb_dir.iterdir() if path.is_file()}
        if mask_stems != rgb_stems:
            missing = sorted(rgb_stems - mask_stems)
            extra = sorted(mask_stems - rgb_stems)
            raise ValueError(
                f"{scene}/{split}: masks do not match official RGB stems; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        resolutions = sorted({png_size(path) for path in masks})
        split_stats[split] = {
            "masks": len(masks),
            "resolutions": [list(resolution) for resolution in resolutions],
        }

    artifacts = scene_artifacts(data_root, scene)
    symlinks = [source for source, _ in artifacts if source.is_symlink()]
    if symlinks:
        raise ValueError(f"{scene}: symbolic links are not permitted: {symlinks}")

    return {
        "splits": split_stats,
        "masks": sum(MASK_COUNTS[scene].values()),
        "pointcloud_vertices": ply_vertex_count(pointcloud),
        "rotations": validate_rotations(rotations, scene),
    }


def archive_scene(data_root: Path, output: Path, scene: str, mode: str) -> None:
    """Create a deterministic archive that overlays an official download."""
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("building archives requires GNU tar and zstd")

    archive_dir = output / "archives"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"{scene}.tar.zst"

    with tempfile.TemporaryDirectory(prefix=f".{scene}-", dir=output) as temporary:
        temporary_root = Path(temporary)
        for source, relative in scene_artifacts(data_root, scene):
            link_or_copy(source, temporary_root / relative, mode)
        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=@0",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "--format=gnu",
                "--use-compress-program=zstd -T0 -6 --no-progress",
                "-cf",
                str(archive_path),
                "-C",
                str(temporary_root),
                f"Data/{scene}",
            ],
            check=True,
        )


def copy_release_metadata(
    output: Path,
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


def write_manifests(
    output: Path,
    data_root: Path,
    version: str,
    commit: str,
    scene_stats: dict[str, dict[str, object]],
) -> None:
    contents = []
    for scene in SCENES:
        for source, relative in scene_artifacts(data_root, scene):
            contents.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256(source),
                }
            )
    overlay_contents = {
        "schema_version": 1,
        "dataset": "NeuSky additions for NeRF-OSR",
        "release_version": version,
        "generator_commit": commit,
        "scenes": scene_stats,
        "totals": {
            "scenes": len(SCENES),
            "masks": sum(stats["masks"] for stats in scene_stats.values()),
            "pointclouds": len(SCENES),
            "rotation_files": len(SCENES),
            "rotation_records": sum(
                stats["rotations"]["records"] for stats in scene_stats.values()
            ),
        },
        "files": contents,
    }
    (output / "OVERLAY_CONTENTS.json").write_text(
        json.dumps(overlay_contents, indent=2) + "\n"
    )

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
        "dataset": "NeuSky additions for NeRF-OSR",
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
        os.environ.get("NERF_OSR_ROOT", Path.home() / "data" / "NeRF-OSR")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help="Root of the official NeRF-OSR download (containing Data/)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="jadgardner/neusky-nerfosr-overlay",
        help="Hugging Face dataset repository ID written into the card",
    )
    parser.add_argument("--version", default="1.0")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to stage files before archiving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.resolve()
    if not (data_root / "Data").is_dir():
        raise FileNotFoundError(f"NeRF-OSR Data directory not found under {data_root}")

    commit = git_commit()
    scene_stats = {}
    for scene in SCENES:
        print(f"[validate] {scene}")
        scene_stats[scene] = validate_scene(data_root, scene)
        print(f"[archive] {scene}")
        archive_scene(data_root, output, scene, args.mode)

    copy_release_metadata(output, args.repo_id, args.version, commit)
    print("[hash] release files")
    write_manifests(
        output=output,
        data_root=data_root,
        version=args.version,
        commit=commit,
        scene_stats=scene_stats,
    )

    forbidden_suffixes = {
        ".jpg",
        ".jpeg",
        ".exr",
        ".hdr",
        ".bin",
        ".txt",
    }
    forbidden = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise ValueError(f"forbidden release paths: {forbidden}")

    manifest = json.loads((output / "MANIFEST.json").read_text())
    print(
        f"[done] {manifest['file_count']} release files, "
        f"{manifest['total_bytes'] / 2**20:.2f} MiB at {output}"
    )


if __name__ == "__main__":
    main()
