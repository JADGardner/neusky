#!/usr/bin/env python3
"""Add coastal cliff instances to an existing Blender scene.

Opens a scene .blend, imports the coastal_cliff_04 asset multiple times,
places them in a rough arc in the background, and saves. The user can
then open the .blend in Blender and reposition/resize as desired.

Usage:
    blender --background scene.blend --python add_cliffs_to_scene.py -- \
        --cliff_asset /path/to/coastal_cliff_04_8k.blend/coastal_cliff_04_8k.blend \
        --output scene.blend
"""

import argparse
import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Add cliffs to scene")
    parser.add_argument("--cliff_asset", type=str, required=True,
                        help="Path to coastal cliff .blend file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output .blend file")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of cliff instances (default: 5)")
    parser.add_argument("--distance", type=float, default=80.0,
                        help="Distance from origin to place cliffs (default: 80)")
    parser.add_argument("--scale", type=float, default=3.0,
                        help="Scale factor for cliffs (default: 3.0)")
    parser.add_argument("--arc_start", type=float, default=-120.0,
                        help="Arc start angle in degrees (default: -120)")
    parser.add_argument("--arc_end", type=float, default=120.0,
                        help="Arc end angle in degrees (default: 120)")
    return parser.parse_args(argv)


def import_cliff(blend_path, lod="coastal_cliff_04_LOD0"):
    """Import a single cliff mesh (LOD0) from the asset blend file."""
    blend_path = os.path.abspath(blend_path)

    before = set(bpy.data.objects)

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        # Only import LOD0 (highest quality)
        if lod in data_from.objects:
            data_to.objects = [lod]
        else:
            # Fallback: import first object
            data_to.objects = [data_from.objects[0]] if data_from.objects else []
        # Import all materials (needed for textures)
        data_to.materials = data_from.materials

    imported = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported.append(obj)

    return imported


def place_cliffs(args):
    cliff_path = os.path.abspath(args.cliff_asset)
    if not os.path.exists(cliff_path):
        print(f"Error: cliff asset not found at {cliff_path}")
        sys.exit(1)

    arc_start_rad = math.radians(args.arc_start)
    arc_end_rad = math.radians(args.arc_end)

    placed = []
    for i in range(args.count):
        # Spread evenly across the arc
        if args.count > 1:
            t = i / (args.count - 1)
        else:
            t = 0.5
        angle = arc_start_rad + t * (arc_end_rad - arc_start_rad)

        # Import cliff
        objs = import_cliff(cliff_path)
        if not objs:
            print(f"Warning: failed to import cliff instance {i}")
            continue

        cliff = objs[0]

        # Position on the arc
        x = args.distance * math.cos(angle)
        y = args.distance * math.sin(angle)

        cliff.location = mathutils.Vector((x, y, 0.0))
        cliff.scale = mathutils.Vector((args.scale, args.scale, args.scale))

        # Rotate to face the center (cliff's long axis is X, so rotate Z)
        cliff.rotation_euler.z = angle + math.pi / 2

        # Rename for clarity
        cliff.name = f"CoastalCliff_{i:02d}"

        placed.append(cliff)
        print(f"Placed {cliff.name} at ({x:.1f}, {y:.1f}, 0.0) angle={math.degrees(angle):.0f}deg scale={args.scale}")

    bpy.context.view_layer.update()

    # Report
    print(f"\nPlaced {len(placed)} cliff instances in arc from {args.arc_start}° to {args.arc_end}° at r={args.distance}")
    print(f"Each cliff scaled {args.scale}x → approx {87*args.scale:.0f} x {24*args.scale:.0f} x {11*args.scale:.0f} units")
    print(f"\nOpen in Blender to reposition/resize as needed.")

    # Save
    output_abs = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_abs) if os.path.dirname(output_abs) else ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_abs)
    print(f"Saved to: {output_abs}")


if __name__ == "__main__":
    args = parse_args()
    place_cliffs(args)
