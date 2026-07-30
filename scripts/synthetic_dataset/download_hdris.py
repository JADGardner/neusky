#!/usr/bin/env python3
"""Download the Poly Haven HDRIs used by the NeuSky synthetic dataset.

The selected asset IDs are pinned in ``hdris_16k.txt``. File metadata is
resolved through the public Poly Haven API and written to a JSON manifest
before downloading. Downloads use ``.part`` files and HTTP range requests so
an interrupted 16K download can be resumed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote


FILES_URL = "https://api.polyhaven.com/files/{asset_id}"
USER_AGENT = "neusky-synthetic-dataset/1.0"


@dataclass(frozen=True)
class DownloadItem:
    asset_id: str
    url: str
    size: int
    md5: str

    def filename(self, file_format: str) -> str:
        return f"{self.asset_id}.{file_format}"


def fetch_json(url: str, timeout: float) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def load_asset_ids(path: Path) -> list[str]:
    asset_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not asset_ids:
        raise RuntimeError(f"No asset IDs found in {path}")
    if len(asset_ids) != len(set(asset_ids)):
        raise RuntimeError(f"Duplicate asset IDs found in {path}")
    return asset_ids


def resolve_item(
    asset_id: str,
    resolution: str,
    file_format: str,
    timeout: float,
) -> DownloadItem:
    files = fetch_json(
        FILES_URL.format(asset_id=quote(asset_id)),
        timeout=timeout,
    )
    if not isinstance(files, dict):
        raise RuntimeError(f"{asset_id}: unexpected Poly Haven response")
    entry = files.get("hdri", {}).get(resolution, {}).get(file_format)
    if not isinstance(entry, dict) or "url" not in entry:
        raise RuntimeError(
            f"{asset_id}: no {resolution} {file_format} HDRI is available"
        )
    if entry.get("size") is None or not entry.get("md5"):
        raise RuntimeError(f"{asset_id}: Poly Haven response has no size or MD5")
    return DownloadItem(
        asset_id=asset_id,
        url=str(entry["url"]),
        size=int(entry["size"]),
        md5=str(entry["md5"]),
    )


def resolve_manifest(
    asset_ids: list[str],
    resolution: str,
    file_format: str,
    timeout: float,
    workers: int,
) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resolve_item,
                asset_id,
                resolution,
                file_format,
                timeout,
            ): asset_id
            for asset_id in asset_ids
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            asset_id = futures[future]
            try:
                items.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed to resolve {asset_id}: {exc}") from exc
            if index % 25 == 0 or index == len(asset_ids):
                print(f"[manifest] resolved {index}/{len(asset_ids)}")
    return sorted(items, key=lambda item: item.asset_id)


def write_manifest(
    path: Path,
    items: list[DownloadItem],
    resolution: str,
    file_format: str,
) -> None:
    payload = {
        "schema_version": 1,
        "source": "https://polyhaven.com/hdris",
        "api": "https://api.polyhaven.com/files/{asset_id}",
        "license": "CC0",
        "resolution": resolution,
        "format": file_format,
        "count": len(items),
        "total_bytes": sum(item.size for item in items),
        "assets": [asdict(item) for item in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_manifest(
    path: Path,
    asset_ids: list[str],
    resolution: str,
    file_format: str,
) -> list[DownloadItem]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported manifest schema in {path}")
    if payload.get("resolution") != resolution:
        raise RuntimeError(
            f"{path} is for {payload.get('resolution')}, requested {resolution}"
        )
    if payload.get("format") != file_format:
        raise RuntimeError(
            f"{path} is for {payload.get('format')}, requested {file_format}"
        )
    items = [DownloadItem(**entry) for entry in payload.get("assets", [])]
    manifest_ids = [item.asset_id for item in items]
    if manifest_ids != sorted(asset_ids):
        raise RuntimeError(
            f"{path} does not contain the same assets as the pinned asset list"
        )
    return items


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_generation_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        checksum, filename = line.split(maxsplit=1)
        checksums[filename.strip()] = checksum
    return checksums


def report_upstream_changes(
    items: list[DownloadItem],
    file_format: str,
    generation_checksums_path: Path,
) -> None:
    if not generation_checksums_path.exists():
        return
    generation_checksums = load_generation_checksums(generation_checksums_path)
    changed = [
        item.asset_id
        for item in items
        if generation_checksums.get(item.filename(file_format)) != item.md5
    ]
    if changed:
        print(
            "[manifest] warning: "
            f"{len(changed)} Poly Haven files have minor upstream revisions "
            "relative to the accepted render"
        )
        print(
            "[manifest] the prepared dataset remains exact; a fresh render "
            "may not be bitwise identical"
        )
        print(
            "[manifest] current upstream versions are accepted for source "
            "re-rendering"
        )


def complete_file_is_valid(path: Path, item: DownloadItem, verify_md5: bool) -> bool:
    if not path.exists() or path.stat().st_size != item.size:
        return False
    return not verify_md5 or file_md5(path) == item.md5


def download_once(
    item: DownloadItem,
    destination: Path,
    timeout: float,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item.size:
        partial.unlink()
        offset = 0

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item.url, headers=headers)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            mode = "ab"
        else:
            mode = "wb"
            offset = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    if partial.stat().st_size != item.size:
        raise RuntimeError(
            f"size mismatch: got {partial.stat().st_size}, expected {item.size}"
        )
    partial.replace(destination)


def download_item(
    item: DownloadItem,
    output_dir: Path,
    file_format: str,
    timeout: float,
    retries: int,
    verify_md5: bool,
) -> str:
    destination = output_dir / item.filename(file_format)
    if complete_file_is_valid(destination, item, verify_md5):
        return f"[skip] {item.asset_id}"

    for attempt in range(1, retries + 1):
        try:
            download_once(item, destination, timeout)
            if verify_md5 and file_md5(destination) != item.md5:
                destination.unlink()
                raise RuntimeError("MD5 mismatch")
            return f"[download] {item.asset_id}"
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"{item.asset_id}: failed after {retries} attempts: {exc}"
                ) from exc
            wait_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"[retry] {item.asset_id}: {exc}; "
                f"retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = Path.home() / "data" / "neusky_synthetic_data" / "hdris_16k"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-list",
        type=Path,
        default=script_dir / "hdris_16k.txt",
        help="Pinned Poly Haven asset ID list",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "hdris_16k_manifest.json",
        help="Resolved URL, size and MD5 manifest",
    )
    parser.add_argument(
        "--generation-checksums",
        type=Path,
        default=script_dir / "hdris_16k_generation_md5.txt",
        help="Checksums of the HDRIs used for the accepted dataset render",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--resolution", default="16k")
    parser.add_argument("--format", default="exr")
    parser.add_argument("--metadata-workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-verify-md5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_ids = load_asset_ids(args.asset_list)

    if args.manifest.exists() and not args.refresh_manifest:
        items = read_manifest(
            args.manifest,
            asset_ids,
            args.resolution,
            args.format,
        )
        print(f"[manifest] loaded {len(items)} pinned files from {args.manifest}")
    else:
        items = resolve_manifest(
            asset_ids,
            args.resolution,
            args.format,
            args.timeout,
            args.metadata_workers,
        )
        write_manifest(args.manifest, items, args.resolution, args.format)
        print(f"[manifest] wrote {args.manifest}")

    total_gib = sum(item.size for item in items) / 1024**3
    print(f"[manifest] {len(items)} files, {total_gib:.2f} GiB total")
    report_upstream_changes(
        items,
        args.format,
        args.generation_checksums,
    )
    if args.manifest_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    verify_md5 = not args.no_verify_md5
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as executor:
        futures = [
            executor.submit(
                download_item,
                item,
                args.output,
                args.format,
                args.timeout,
                args.retries,
                verify_md5,
            )
            for item in items
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            print(f"{future.result()} ({index}/{len(items)})")

    print(f"[done] HDRIs are in {args.output}")


if __name__ == "__main__":
    main()
