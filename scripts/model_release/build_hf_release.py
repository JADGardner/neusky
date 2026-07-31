#!/usr/bin/env python3
"""Build the allowlisted Hugging Face release for NeuSky checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


RELEASE_VERSION = "1.0.1"


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    collection: str
    scene: str
    source: str
    destination: str
    source_kind: str


RUNS = (
    RunSpec(
        "nerf-osr-lk2",
        "nerf-osr",
        "lk2",
        "code/neusky/outputs/lk2_rngfix_w32cyc_hash21/neusky/2026-07-05_082759",
        "nerf-osr/lk2",
        "phd",
    ),
    RunSpec(
        "nerf-osr-lwp",
        "nerf-osr",
        "lwp",
        "code/neusky/outputs/lwp_rngfix_w32cyc_hash21/neusky/2026-07-05_082758",
        "nerf-osr/lwp",
        "phd",
    ),
    RunSpec(
        "nerf-osr-st",
        "nerf-osr",
        "st",
        "code/neusky/outputs/st_rngfix_w32cyc_hash21/neusky/2026-07-05_082758",
        "nerf-osr/st",
        "phd",
    ),
    RunSpec(
        "synthetic-abandoned-buildings",
        "synthetic",
        "abandoned_buildings",
        "abandoned_buildings",
        "synthetic/abandoned_buildings",
        "synthetic",
    ),
    RunSpec(
        "synthetic-apartment-building",
        "synthetic",
        "apartment_building",
        "apartment_building",
        "synthetic/apartment_building",
        "synthetic",
    ),
    RunSpec(
        "synthetic-arlanda-uppsala-cathedral",
        "synthetic",
        "arlanda_uppsala_cathedral",
        "arlanda_uppsala_cathedral",
        "synthetic/arlanda_uppsala_cathedral",
        "synthetic",
    ),
    RunSpec(
        "synthetic-glass-building",
        "synthetic",
        "glass_building",
        "glass_building",
        "synthetic/glass_building",
        "synthetic",
    ),
    RunSpec(
        "synthetic-interstellar-house",
        "synthetic",
        "interstellar_house",
        "interstellar_house",
        "synthetic/interstellar_house",
        "synthetic",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phd-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--synthetic-source-root",
        type=Path,
        required=True,
        help="Directory containing one staged canonical run directory per scene",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phd_root = args.phd_root.expanduser().resolve()
    synthetic_root = args.synthetic_source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing release: {output}")
    output.mkdir(parents=True)

    models = []
    for spec in RUNS:
        source_run = (
            phd_root / spec.source
            if spec.source_kind == "phd"
            else synthetic_root / spec.source
        )
        destination = output / spec.destination
        files = []
        for name in ("config.yml", "dataparser_transforms.json"):
            files.append(copy_file(source_run / name, destination / name))
        checkpoint = "step-000100000.ckpt"
        files.append(
            copy_file(
                source_run / "nerfstudio_models" / checkpoint,
                destination / "nerfstudio_models" / checkpoint,
            )
        )
        for info in files:
            info["path"] = Path(info["path"]).relative_to(output).as_posix()
        models.append(
            {
                "id": spec.model_id,
                "collection": spec.collection,
                "scene": spec.scene,
                "source_run": spec.source,
                "source_checkpoint": checkpoint,
                "reni_prior": "jadgardner/reni-models:neusky-prior@v1.0",
                "files": files,
            }
        )

    shutil.copy2(Path(__file__).with_name("MODEL_CARD.md"), output / "README.md")
    shutil.copy2(Path(__file__).resolve().parents[2] / "LICENSE", output / "LICENSE")
    manifest = {
        "schema_version": 1,
        "release_version": RELEASE_VERSION,
        "repository": "jadgardner/neusky-models",
        "models": models,
    }
    (output / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = sorted(
        f"{file_info['sha256']}  {file_info['path']}"
        for model in models
        for file_info in model["files"]
    )
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Staged {len(models)} models at {output}")


if __name__ == "__main__":
    main()
