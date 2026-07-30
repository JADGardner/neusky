#!/usr/bin/env python3
"""Split multipart EXR files from blender_render_scene.py into individual passes.

Each multipart EXR contains parts named: RGB, Depth, Normal, Albedo, Alpha, Roughness, Metallic,
Transmission, IOR.
This script extracts each part into separate EXR files in the directory
structure expected by prepare_synthetic_data.py.

Input structure:
    <input>/
        transforms_train.json
        train/
            multipass/Image0001.exr, Image0002.exr, ...
            hdri/0001.exr, 0002.exr, ...

Output structure:
    <input>/
        transforms_train.json (unchanged)
        train/
            rgb/Image0001.exr, Image0002.exr, ...
            depth/Image0001.exr, ...
            normal/Image0001.exr, ...
            albedo/Image0001.exr, ...
            mask/Image0001.exr, ...
            roughness/Image0001.exr, ...
            metallic/Image0001.exr, ...
            transmission/Image0001.exr, ...
            ior/Image0001.exr, ...
            hdri/ (unchanged)

Usage:
    python split_multipass_exr.py --input /path/to/rendered/dataset
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


def split_multipart_openexr(input_path, output_dirs):
    """Split a multipart EXR using OpenEXR 3.x API."""
    exr = OpenEXR.File(input_path)
    stem = Path(input_path).stem

    for part in exr.parts:
        part_name = part.name()
        channels = list(part.channels.keys())

        if part_name == "RGB":
            out_dir = output_dirs["rgb"]
        elif part_name == "Depth":
            out_dir = output_dirs["depth"]
        elif part_name == "Normal":
            out_dir = output_dirs["normal"]
        elif part_name == "Albedo":
            out_dir = output_dirs["albedo"]
        elif part_name == "Alpha":
            out_dir = output_dirs["mask"]
        else:
            continue

        out_path = os.path.join(out_dir, f"{stem}.exr")

        # Read pixel data from part
        w = part.width
        h = part.height

        # Build header for output EXR
        header = OpenEXR.Header(w, h)
        header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

        chan_data = {}
        for ch_name in channels:
            pixels = part.channel(ch_name)
            # pixels is a numpy array of shape (h, w)
            raw = pixels.astype(np.float32).tobytes()
            short_name = ch_name.split(".")[-1] if "." in ch_name else ch_name
            chan_data[short_name] = raw
            header["channels"][short_name] = Imath.Channel(
                Imath.PixelType(Imath.PixelType.FLOAT)
            )

        # Remove default channels from header
        for default_ch in list(header["channels"].keys()):
            if default_ch not in chan_data:
                del header["channels"][default_ch]

        out_exr = OpenEXR.OutputFile(out_path, header)
        out_exr.writePixels(chan_data)
        out_exr.close()


def split_multipart_simple(input_path, output_dirs):
    """Split multipart EXR using OpenEXR 3.x File API.

    Reads each part and writes it as a standalone EXR.
    """
    exr = OpenEXR.File(input_path)
    stem = Path(input_path).stem

    part_map = {
        "RGB": "rgb",
        "Depth": "depth",
        "Normal": "normal",
        "Albedo": "albedo",
        "Alpha": "mask",
        "Roughness": "roughness",
        "Metallic": "metallic",
        "Transmission": "transmission",
        "IOR": "ior",
    }

    for part in exr.parts:
        part_name = part.name()
        if part_name not in part_map:
            continue

        out_dir = output_dirs[part_map[part_name]]
        out_path = os.path.join(out_dir, f"{stem}.exr")

        w, h = part.width(), part.height()

        # Read channel data via the Channel.pixels attribute
        raw_channels = {}
        for ch_name, ch_obj in part.channels.items():
            pixels = np.array(ch_obj.pixels, dtype=np.float32)
            raw_channels[ch_name] = pixels

        # Write as single-part EXR using the old InputFile/OutputFile API
        header = OpenEXR.Header(w, h)
        header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

        out_channel_data = {}
        dir_key = part_map[part_name]
        for ch_name, pixels in raw_channels.items():
            # Use short channel names (strip "Depth." prefix etc.)
            short = ch_name.split(".")[-1] if "." in ch_name else ch_name

            # Rename single-value channels V -> Y for viewer compatibility
            if short == "V":
                short = "Y"

            if pixels.ndim == 2:
                # Single channel (H, W) -> write as-is
                out_channel_data[short] = pixels.tobytes()
                header["channels"][short] = Imath.Channel(
                    Imath.PixelType(Imath.PixelType.FLOAT)
                )
            elif pixels.ndim == 3:
                # Multi-channel (H, W, C) -> split into R, G, B, A
                ch_names = ["R", "G", "B", "A"][:pixels.shape[2]]
                for ci, cn in enumerate(ch_names):
                    out_channel_data[cn] = pixels[:, :, ci].tobytes()
                    header["channels"][cn] = Imath.Channel(
                        Imath.PixelType(Imath.PixelType.FLOAT)
                    )

        # Remove default channels not in our data
        for default_ch in list(header["channels"].keys()):
            if default_ch not in out_channel_data:
                del header["channels"][default_ch]

        out_exr = OpenEXR.OutputFile(out_path, header)
        out_exr.writePixels(out_channel_data)
        out_exr.close()


def main():
    parser = argparse.ArgumentParser(description="Split multipart EXR files")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to rendered dataset directory")
    args = parser.parse_args()

    if not HAS_OPENEXR:
        print("Error: OpenEXR package required. Install with: pip install OpenEXR")
        return

    input_dir = Path(args.input)
    multipass_dir = input_dir / "train" / "multipass"

    if not multipass_dir.exists():
        print(f"Error: multipass directory not found: {multipass_dir}")
        return

    # Create output directories
    output_dirs = {}
    for pass_name in ["rgb", "depth", "normal", "albedo", "mask", "roughness", "metallic", "transmission", "ior"]:
        d = str(input_dir / "train" / pass_name)
        os.makedirs(d, exist_ok=True)
        output_dirs[pass_name] = d

    # Find all multipart EXR files
    exr_files = sorted(multipass_dir.glob("*.exr"))
    print(f"Found {len(exr_files)} multipart EXR files")

    for exr_path in tqdm(exr_files, desc="Splitting"):
        try:
            split_multipart_simple(str(exr_path), output_dirs)
        except Exception as e:
            print(f"\nError splitting {exr_path.name}: {e}")
            raise

    # Remove multipass directory
    import shutil
    shutil.rmtree(multipass_dir)
    print(f"\nRemoved {multipass_dir}")

    # Summary
    print("\nDone! Split passes:")
    for pass_name, d in output_dirs.items():
        n = len(list(Path(d).glob("*.exr")))
        print(f"  {pass_name}: {n} files")


if __name__ == "__main__":
    main()
