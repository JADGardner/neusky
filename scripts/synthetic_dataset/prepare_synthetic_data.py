#!/usr/bin/env python3
"""Prepare the NeuSky synthetic dataset for training and evaluation.

Converts the raw Blender EXR renders into the format expected by the
CustomNeuskyDataparser.  Train split gets LDR PNGs only by default; val/test
splits additionally include all ground-truth EXR layers for inverse-rendering
metric evaluation (albedo, normal, depth, roughness, metallic, transmission,
ior, mask).  Use --copy-train-gt to include selected GT layers in train too.

    <output>/
        transforms.json
        train/
            rgb/*.png
            cityscapes_mask/*.png
        validation/
            rgb/*.png
            cityscapes_mask/*.png
            albedo/*.exr
            normal/*.exr
            depth/*.exr
            roughness/*.exr
            metallic/*.exr
            transmission/*.exr
            ior/*.exr
        test/
            (same as validation)

Usage (run inside the Docker research container):

    python scripts/synthetic_dataset/prepare_synthetic_data.py \
        --input /data/neusky_synthetic_data/renders/example_eval \
        --output /data/neusky_synthetic_data/renders/example_prepared
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image
from tqdm import tqdm

# Ground-truth EXR layers copied to val/test for inverse-rendering evaluation
GT_LAYERS = ["albedo", "normal", "depth", "roughness", "metallic", "transmission", "ior"]
DEFAULT_TRAIN_GT_LAYERS = ["albedo", "normal", "depth"]


# Cityscapes class colours used by NeuSky
SKY_COLOUR = np.array([70, 130, 180], dtype=np.uint8)
BUILDING_COLOUR = np.array([70, 70, 70], dtype=np.uint8)


def read_exr_rgb(path: str) -> np.ndarray:
    """Read an EXR file and return float32 RGB array of shape (H, W, 3)."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = []
    for ch in ["R", "G", "B"]:
        raw = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        channels.append(np.frombuffer(raw, dtype=np.float32).reshape(h, w))
    return np.stack(channels, axis=-1)


def read_exr_alpha(path: str) -> np.ndarray:
    """Read the alpha channel from an EXR file. Returns float32 (H, W)."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    raw = exr.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(h, w)


def linear_to_srgb(color: np.ndarray) -> np.ndarray:
    """Convert linear RGB float array to sRGB [0,1] with quantile exposure."""
    q = np.quantile(color, 0.98)
    if q > 0:
        color = color / q

    srgb = np.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * np.power(np.clip(np.abs(color), 1e-10, None), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def extract_frame_num(filename: str) -> int:
    """Extract frame number from filenames like Image0027.exr or 0027.exr."""
    matches = re.findall(r"(\d+)", Path(filename).stem)
    if not matches:
        raise ValueError(f"Cannot extract frame number from {filename}")
    return int(matches[-1])


def convert_rgb(exr_path: str, png_path: str):
    """Read a linear EXR and write an sRGB PNG."""
    img = read_exr_rgb(exr_path)
    srgb = linear_to_srgb(img)
    Image.fromarray((srgb * 255).astype(np.uint8)).save(png_path)


def convert_mask_to_cityscapes(exr_path: str, png_path: str):
    """Convert a Blender mask EXR alpha channel to a cityscapes-style RGB PNG.

    Blender 'Film Transparent': alpha=1 → object, alpha=0 → sky.
    """
    alpha = read_exr_alpha(exr_path)

    # alpha > 0.5 → object, else sky
    is_object = alpha > 0.5
    h, w = is_object.shape
    seg = np.zeros((h, w, 3), dtype=np.uint8)
    seg[is_object] = BUILDING_COLOUR
    seg[~is_object] = SKY_COLOUR

    Image.fromarray(seg).save(png_path)


def _evenly_spaced(n_total: int, n_select: int) -> set:
    """Return n_select evenly-spaced indices from range(n_total)."""
    if n_select <= 0 or n_total <= 0:
        return set()
    if n_select >= n_total:
        return set(range(n_total))
    step = n_total / (n_select + 1)
    return {int(round(step * (i + 1))) for i in range(n_select)}


def split_frames(frame_nums: list, n_val: int = None, n_test: int = None,
                 val_every: int = 10, test_every: int = 10):
    """Split sorted frame numbers into train/val/test.

    If n_val and n_test are given, use explicit counts with evenly-spaced
    sampling.  Otherwise fall back to legacy strided mode (val_every/test_every).
    """
    sorted_nums = sorted(frame_nums)
    n = len(sorted_nums)

    if n_val is not None and n_test is not None:
        val_idx = _evenly_spaced(n, n_val)
        remaining = sorted(set(range(n)) - val_idx)
        test_pos = _evenly_spaced(len(remaining), n_test)
        test_idx = {remaining[p] for p in test_pos}

        train = [sorted_nums[i] for i in range(n) if i not in val_idx and i not in test_idx]
        val = [sorted_nums[i] for i in sorted(val_idx)]
        test = [sorted_nums[i] for i in sorted(test_idx)]
    else:
        train, val, test = [], [], []
        for i, fnum in enumerate(sorted_nums):
            if i % val_every == 1:
                val.append(fnum)
            elif i % test_every == 5:
                test.append(fnum)
            else:
                train.append(fnum)

    return {"train": train, "validation": val, "test": test}


# Per-frame metadata keys propagated from render transforms to prepared transforms
FRAME_KEYS = ("fl_x", "fl_y", "cx", "cy", "focal_mm", "exposure_ev",
              "envmap_name", "envmap_url", "envmap_rotation")


def load_render_input(input_dir: Path):
    """Load a raw render directory (post split_multipass_exr).

    Returns (transforms, frame_to_data, available_frame_nums).
    """
    rgb_dir = input_dir / "train" / "rgb"
    transforms_path = input_dir / "transforms_train.json"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

    with open(transforms_path) as f:
        transforms = json.load(f)

    frame_to_data = {}
    for frame in transforms["frames"]:
        frame_to_data[extract_frame_num(frame["file_path"])] = frame

    available = []
    for f in sorted(rgb_dir.glob("*.exr")):
        fnum = extract_frame_num(f.name)
        if fnum in frame_to_data:
            available.append(fnum)

    print(f"Found {len(available)} EXR frames with matching transforms in {input_dir}")
    return transforms, frame_to_data, available


def process_split(split_name, refs, inputs, frame_maps, output_dir, train_gt_layers=()):
    """Convert frames into output_dir/<split_name>/ and return their frame dicts.

    refs is a list of (input_idx, frame_num) tuples so a split can draw frames
    from several render directories (e.g. a varied train render plus a curated
    whole-building render). inputs/frame_maps are parallel lists per render dir.
    """
    (output_dir / split_name / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / split_name / "cityscapes_mask").mkdir(parents=True, exist_ok=True)
    gt_layers_for_split = GT_LAYERS if split_name in ("validation", "test") else list(train_gt_layers)
    for layer in gt_layers_for_split:
        (output_dir / split_name / layer).mkdir(parents=True, exist_ok=True)

    frames = []
    print(f"\nProcessing {split_name} split ({len(refs)} frames)...")
    for idx, (in_idx, fnum) in enumerate(tqdm(refs, desc=split_name)):
        input_dir = inputs[in_idx]
        rgb_dir = input_dir / "train" / "rgb"
        mask_dir = input_dir / "train" / "mask"
        src_exr = f"Image{fnum:04d}.exr"
        dst_stem = f"{idx:04d}"

        # Convert RGB
        convert_rgb(str(rgb_dir / src_exr),
                    str(output_dir / split_name / "rgb" / f"{dst_stem}.png"))

        # Convert mask
        src_mask = mask_dir / src_exr
        if src_mask.exists():
            convert_mask_to_cityscapes(
                str(src_mask),
                str(output_dir / split_name / "cityscapes_mask" / f"{dst_stem}.png"))

        # Copy ground-truth EXR layers for val/test, and optionally for train
        # upper-bound supervision experiments.
        for layer in gt_layers_for_split:
            src_layer = input_dir / "train" / layer / src_exr
            if src_layer.exists():
                shutil.copy2(str(src_layer),
                             str(output_dir / split_name / layer / f"{dst_stem}.exr"))

        # Build frame dict (propagate per-frame intrinsic keys if present)
        src_frame = frame_maps[in_idx][fnum]
        frame_dict = {
            "file_path": f"{split_name}/rgb/{dst_stem}.png",
            "transform_matrix": src_frame["transform_matrix"],
        }
        for key in FRAME_KEYS:
            if key in src_frame:
                frame_dict[key] = src_frame[key]
        frames.append(frame_dict)
    return frames


def print_summary(output_dir: Path, n_frames: int):
    print(f"\nDone! Prepared data written to {output_dir}")
    print(f"  transforms.json: {n_frames} frames")
    for split_name in ["train", "validation", "test"]:
        n_rgb = len(list((output_dir / split_name / "rgb").glob("*.png")))
        n_mask = len(list((output_dir / split_name / "cityscapes_mask").glob("*.png")))
        summary = f"  {split_name}: {n_rgb} rgb, {n_mask} masks"
        gt_counts = []
        for layer in GT_LAYERS:
            n = len(list((output_dir / split_name / layer).glob("*.exr")))
            if n > 0:
                gt_counts.append(f"{n} {layer}")
        if gt_counts:
            summary += f", GT: {', '.join(gt_counts)}"
        print(summary)


def update_eval(args):
    """Replace the validation/test splits of an existing prepared dataset.

    --input is a NEW raw eval render (curated camera views); --output is an
    EXISTING prepared dataset whose train split is kept untouched. The old
    validation/ and test/ directories are deleted and rebuilt from the eval
    render, and transforms.json is rewritten (train frames preserved).
    """
    if len(args.input) != 1:
        raise SystemExit("--update-eval takes exactly one --input render directory")
    input_dir = Path(args.input[0])
    output_dir = Path(args.output)

    existing_path = output_dir / "transforms.json"
    if not existing_path.exists():
        raise FileNotFoundError(
            f"--update-eval requires an existing prepared dataset; "
            f"missing {existing_path}")
    with open(existing_path) as f:
        existing = json.load(f)

    train_frames = [f for f in existing["frames"]
                    if f["file_path"].startswith("train/")]
    if not train_frames:
        raise RuntimeError(f"No train frames found in {existing_path}")

    transforms, frame_to_data, available = load_render_input(input_dir)

    # The eval render must match the prepared dataset's image size
    for key in ("w", "h"):
        if key in existing and key in transforms and \
                int(existing[key]) != int(transforms[key]):
            raise RuntimeError(
                f"Eval render {key}={transforms[key]} does not match "
                f"prepared dataset {key}={existing[key]}")

    n_val = args.n_val if args.n_val is not None else 25
    n_test = args.n_test if args.n_test is not None else 25
    if len(available) < n_val + n_test:
        raise RuntimeError(
            f"Need {n_val + n_test} eval frames (n_val={n_val} + "
            f"n_test={n_test}), found {len(available)}")

    nums = sorted(available)
    val_nums = nums[0::2][:n_val]
    val_set = set(val_nums)
    test_nums = [n for n in nums if n not in val_set][:n_test]

    print(f"Keeping {len(train_frames)} existing train frames; replacing "
          f"validation ({n_val}) and test ({n_test}) from {input_dir}")

    # The raw renders behind the old eval splits may no longer exist, so move
    # them to a backup directory instead of deleting (overwrites any previous
    # backup — we keep one generation).
    backup_dir = output_dir / "_replaced_eval"
    for split in ("validation", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            dst = backup_dir / split
            if dst.exists():
                shutil.rmtree(dst)
            backup_dir.mkdir(exist_ok=True)
            split_dir.rename(dst)
            print(f"  Old {split}/ moved to {dst}")
    if backup_dir.exists():
        shutil.copy2(existing_path, backup_dir / "transforms.json")

    out_frames = list(train_frames)
    out_frames += process_split("validation", [(0, n) for n in val_nums],
                                [input_dir], [frame_to_data], output_dir)
    out_frames += process_split("test", [(0, n) for n in test_nums],
                                [input_dir], [frame_to_data], output_dir)

    out_transforms = {k: v for k, v in existing.items() if k != "frames"}
    out_transforms["frames"] = out_frames
    with open(existing_path, "w") as f:
        json.dump(out_transforms, f, indent=2)

    print_summary(output_dir, len(out_frames))


def main():
    parser = argparse.ArgumentParser(description="Prepare NeuSky synthetic data")
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="One or more render directories. With several, frames "
                             "are concatenated in the given order before splitting "
                             "(e.g. a varied train render + a curated train render)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write prepared data")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Exact number of validation frames (evenly spaced)")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Exact number of test frames (evenly spaced)")
    parser.add_argument("--val-every", type=int, default=10,
                        help="(legacy) Stride for validation split")
    parser.add_argument("--test-every", type=int, default=10,
                        help="(legacy) Stride for test split")
    parser.add_argument("--update-eval", action="store_true",
                        help="Treat --input as a curated eval render and rebuild only "
                             "the validation/test splits of the EXISTING prepared "
                             "dataset at --output (train split kept untouched)")
    parser.add_argument("--copy-train-gt", action="store_true",
                        help="Copy selected GT EXR layers into train/ as well as val/test.")
    parser.add_argument("--train-gt-layers", nargs="+", choices=GT_LAYERS,
                        default=DEFAULT_TRAIN_GT_LAYERS,
                        help="GT layers copied to train/ when --copy-train-gt is set "
                             "(default: albedo normal depth).")
    args = parser.parse_args()

    if args.update_eval:
        update_eval(args)
        return

    inputs = [Path(p) for p in args.input]
    output_dir = Path(args.output)

    loaded = [load_render_input(d) for d in inputs]
    transforms = loaded[0][0]
    frame_maps = [l[1] for l in loaded]

    # Image sizes must agree across inputs
    for d, (t, _, _) in zip(inputs[1:], loaded[1:]):
        for key in ("w", "h"):
            if key in transforms and key in t and int(transforms[key]) != int(t[key]):
                raise RuntimeError(f"{d}: {key}={t[key]} differs from "
                                   f"{inputs[0]}: {key}={transforms[key]}")

    # Concatenate frames across inputs (in the given order) and split by position
    refs = [(i, fnum) for i, (_, _, avail) in enumerate(loaded)
            for fnum in sorted(avail)]
    pos_splits = split_frames(list(range(len(refs))), n_val=args.n_val,
                              n_test=args.n_test, val_every=args.val_every,
                              test_every=args.test_every)
    splits = {name: [refs[p] for p in pos] for name, pos in pos_splits.items()}
    for split_name, split_refs in splits.items():
        print(f"  {split_name}: {len(split_refs)} frames")

    # Build the output transforms.json
    out_transforms = {
        "fl_x": transforms["fl_x"],
        "fl_y": transforms["fl_y"],
        "cx": transforms["cx"],
        "cy": transforms["cy"],
        "w": transforms.get("w", transforms["cx"] * 2),
        "h": transforms.get("h", transforms["cy"] * 2),
        "camera_angle_x": transforms.get("camera_angle_x", 0),
        "camera_angle_y": transforms.get("camera_angle_y", 0),
        "frames": [],
    }

    # Process each split - renumber sequentially starting from 0
    train_gt_layers = args.train_gt_layers if args.copy_train_gt else []
    for split_name, split_refs in splits.items():
        out_transforms["frames"] += process_split(
            split_name, split_refs, inputs, frame_maps, output_dir, train_gt_layers=train_gt_layers)

    # Write transforms.json
    out_transforms_path = output_dir / "transforms.json"
    with open(out_transforms_path, "w") as f:
        json.dump(out_transforms, f, indent=2)

    print_summary(output_dir, len(out_transforms["frames"]))


if __name__ == "__main__":
    main()
