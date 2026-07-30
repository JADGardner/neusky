#!/usr/bin/env python3
"""Regenerate vegetation (trees, rocks, shrubs) in a NeuSky scene.

Removes existing vegetation objects and replaces them with new instances
using improved randomisation and collision avoidance against buildings
and cliffs.

Usage:
    blender --background scene.blend --python regenerate_vegetation.py -- \
        --bg_assets_dir /path/to/background_assets \
        --output scene.blend \
        --seed 12345

Cliffs (CoastalCliff_*) and the ground plane are preserved.
"""

import argparse
import math
import os
import random
import sys

import bpy
import mathutils


# Prefixes used by setup_scene.py for vegetation objects
VEGETATION_PREFIXES = ("Tree_", "Boulder_", "Shrub_", "BigTree_", "BoulderOuter_")


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Regenerate vegetation in scene")
    parser.add_argument(
        "--bg_assets_dir",
        type=str,
        required=True,
        help="Directory with background .blend assets (trees, rocks, shrubs)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output .blend file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=5.0,
        help="Minimum XY clearance from buildings/cliffs (default: 5.0)",
    )
    parser.add_argument(
        "--min_spacing",
        type=float,
        default=8.0,
        help="Minimum spacing between vegetation objects (default: 8.0)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Geometry helpers — convex hull polygon collision
# ---------------------------------------------------------------------------

# Maximum XY span (in any axis) for an object to be treated as an obstacle.
# Objects larger than this are likely terrain/environment meshes and are skipped.
MAX_OBSTACLE_SPAN = 300.0


def _cross_2d(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull_2d(points):
    """Andrew's monotone chain convex hull. Returns CCW-ordered vertices."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    # Lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def expand_polygon(polygon, margin):
    """Expand a convex polygon outward from its centroid by *margin* units."""
    if not polygon or margin <= 0:
        return polygon
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    expanded = []
    for px, py in polygon:
        dx, dy = px - cx, py - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            expanded.append((px + margin * dx / dist, py + margin * dy / dist))
        else:
            expanded.append((px + margin, py))
    return expanded


def point_in_convex_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test (works for any simple polygon)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def get_object_xy_hull(obj, margin=0.0):
    """Get the 2D convex hull of an object's world-space bbox projected to XY.

    Returns a list of (x, y) vertices or None.  Much tighter than an AABB for
    rotated objects like cliffs.
    """
    if obj.type != "MESH":
        return None
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

    # Size filter: skip enormous objects (terrain, environment shells)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    if (max(xs) - min(xs)) > MAX_OBSTACLE_SPAN or (max(ys) - min(ys)) > MAX_OBSTACLE_SPAN:
        return None

    pts_2d = list({(round(c.x, 4), round(c.y, 4)) for c in corners})
    hull = convex_hull_2d(pts_2d)
    if margin > 0:
        hull = expand_polygon(hull, margin)
    return hull


def collect_obstacle_hulls(margin=5.0):
    """Collect convex-hull polygons of all obstacle objects.

    Obstacles are everything that is NOT vegetation, NOT the ground plane,
    and not oversized environment meshes.
    """
    obstacles = []
    skipped_large = 0
    for obj in bpy.data.objects:
        name = obj.name
        # Skip vegetation (about to be deleted)
        if any(name.startswith(p) for p in VEGETATION_PREFIXES):
            continue
        # Skip ground plane (may be named differently in manual scenes)
        if "GroundPlane" in name or "Ground" == name:
            continue
        hull = get_object_xy_hull(obj, margin)
        if hull is None:
            if obj.type == "MESH":
                skipped_large += 1
            continue
        obstacles.append(hull)
    if skipped_large:
        print(f"  (Skipped {skipped_large} non-mesh or oversized objects)")
    return obstacles


def point_in_any_obstacle(x, y, obstacles):
    """Check if (x, y) falls inside any obstacle's convex hull polygon."""
    for hull in obstacles:
        if point_in_convex_polygon(x, y, hull):
            return True
    return False


# ---------------------------------------------------------------------------
# Vegetation removal
# ---------------------------------------------------------------------------


def remove_vegetation():
    """Remove all existing vegetation objects and purge orphan data."""
    to_remove = []
    for obj in bpy.data.objects:
        if any(obj.name.startswith(p) for p in VEGETATION_PREFIXES):
            to_remove.append(obj)

    # Remove children before parents to avoid dangling references
    to_remove.sort(key=lambda o: 0 if o.parent is None else 1, reverse=True)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Purge orphan data blocks (meshes, materials, textures) left by old copies
    purged = bpy.data.orphans_purge(do_recursive=True)

    print(f"Removed {len(to_remove)} vegetation objects, purged {purged} orphan data blocks")
    return len(to_remove)


# ---------------------------------------------------------------------------
# Asset import & placement
# ---------------------------------------------------------------------------


# Template cache: {abs_path: [template_objects]}
_template_cache = {}


def _import_template(blend_path):
    """Import LOD0 objects once as hidden templates, cached for reuse."""
    blend_path = os.path.abspath(blend_path)
    if blend_path in _template_cache:
        return _template_cache[blend_path]

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = [
            name
            for name in data_from.objects
            if not any(
                name.endswith(f"_LOD{i}") or f"_LOD{i}." in name for i in range(1, 4)
            )
        ]

    imported = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            obj.hide_viewport = True
            obj.hide_render = True
            imported.append(obj)

    _template_cache[blend_path] = imported
    print(f"  Imported template: {os.path.basename(blend_path)} ({len(imported)} objects)")
    return imported


def cleanup_templates():
    """Remove hidden template objects (mesh data persists via linked copies)."""
    for templates in _template_cache.values():
        for obj in templates:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
    _template_cache.clear()


def place_asset_at(asset_path, x, y, scale, rot_z, name_prefix, index):
    """Create linked duplicates of template objects at (x, y, 0).

    Uses obj.copy() WITHOUT copying obj.data, so all instances share the
    same mesh data and materials.  This is Blender's Alt+D (linked duplicate).
    """
    templates = _import_template(asset_path)
    if not templates:
        return []

    template_set = set(templates)

    # Shallow-copy all template objects (shares mesh data)
    copies = []
    old_to_new = {}
    for obj in templates:
        new_obj = obj.copy()  # Linked duplicate — shares obj.data
        new_obj.hide_viewport = False
        new_obj.hide_render = False
        bpy.context.collection.objects.link(new_obj)
        old_to_new[obj] = new_obj
        copies.append(new_obj)

    # Re-parent children to their corresponding copied parents
    for orig, copy in old_to_new.items():
        if orig.parent in template_set:
            copy.parent = old_to_new[orig.parent]

    # Set transforms on root objects
    copy_set = set(copies)
    for copy in copies:
        orig = [o for o, c in old_to_new.items() if c is copy][0]
        is_root = copy.parent is None or copy.parent not in copy_set
        if is_root:
            copy.location = mathutils.Vector((x, y, 0.0))
            copy.scale = orig.scale * scale
            copy.rotation_euler.z = rot_z
        copy.name = f"{name_prefix}_{index:02d}_{orig.name}"

    return copies


# ---------------------------------------------------------------------------
# Random scatter with collision avoidance
# ---------------------------------------------------------------------------


def scatter_in_annulus(
    rng, r_min, r_max, count, obstacles, placed_positions, min_spacing=8.0, max_attempts=100
):
    """Generate random positions in an annular region.

    Avoids obstacle convex hulls and enforces minimum spacing between
    all previously placed vegetation.  Returns list of (x, y) tuples.
    """
    positions = []
    for _ in range(count):
        for _attempt in range(max_attempts):
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(r_min, r_max)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # Reject if inside an obstacle footprint
            if point_in_any_obstacle(x, y, obstacles):
                continue

            # Reject if too close to already-placed vegetation
            too_close = False
            for px, py in placed_positions:
                if (x - px) ** 2 + (y - py) ** 2 < min_spacing ** 2:
                    too_close = True
                    break
            if too_close:
                continue

            positions.append((x, y))
            placed_positions.append((x, y))
            break
        else:
            print(f"  Warning: skipped 1 instance (no valid position after {max_attempts} attempts)")

    return positions


# ---------------------------------------------------------------------------
# Main vegetation placement
# ---------------------------------------------------------------------------


def regenerate_vegetation(bg_assets_dir, seed=42, margin=5.0, min_spacing=8.0):
    """Remove old vegetation and scatter new instances with collision avoidance."""
    rng = random.Random(seed)

    obstacles = collect_obstacle_hulls(margin)
    print(f"Found {len(obstacles)} obstacle hulls for collision avoidance")

    # Track all placed positions for inter-vegetation spacing
    placed_positions = []
    total_placed = 0

    # --- Small trees: scattered across mid-range ---
    tree_path = os.path.join(bg_assets_dir, "tree_small_02_1k.blend")
    if os.path.exists(tree_path):
        positions = scatter_in_annulus(
            rng, 55, 95, count=14, obstacles=obstacles,
            placed_positions=placed_positions, min_spacing=min_spacing,
        )
        for i, (x, y) in enumerate(positions):
            s = rng.uniform(2.0, 4.0)
            rot = rng.uniform(0, 2 * math.pi)
            objs = place_asset_at(tree_path, x, y, s, rot, "Tree", i)
            total_placed += len(objs)
        print(f"  Placed {len(positions)} small trees (r=55-95)")

    # --- Boulders: mid-ground scatter ---
    boulder_path = os.path.join(bg_assets_dir, "boulder_01_1k.blend")
    if os.path.exists(boulder_path):
        positions = scatter_in_annulus(
            rng, 40, 80, count=10, obstacles=obstacles,
            placed_positions=placed_positions, min_spacing=min_spacing * 0.6,
        )
        for i, (x, y) in enumerate(positions):
            s = rng.uniform(2.5, 7.0)
            rot = rng.uniform(0, 2 * math.pi)
            objs = place_asset_at(boulder_path, x, y, s, rot, "Boulder", i)
            total_placed += len(objs)
        print(f"  Placed {len(positions)} boulders (r=40-80)")

    # --- Shrubs: fill gaps ---
    shrub_path = os.path.join(bg_assets_dir, "shrub_04_1k.blend")
    if os.path.exists(shrub_path):
        positions = scatter_in_annulus(
            rng, 45, 88, count=20, obstacles=obstacles,
            placed_positions=placed_positions, min_spacing=min_spacing * 0.5,
        )
        for i, (x, y) in enumerate(positions):
            s = rng.uniform(2.5, 5.5)
            rot = rng.uniform(0, 2 * math.pi)
            objs = place_asset_at(shrub_path, x, y, s, rot, "Shrub", i)
            total_placed += len(objs)
        print(f"  Placed {len(positions)} shrubs (r=45-88)")

    # --- Large trees: outer ring ---
    big_tree_path = os.path.join(bg_assets_dir, "island_tree_01_1k.blend")
    if os.path.exists(big_tree_path):
        positions = scatter_in_annulus(
            rng, 85, 115, count=10, obstacles=obstacles,
            placed_positions=placed_positions, min_spacing=min_spacing,
        )
        for i, (x, y) in enumerate(positions):
            s = rng.uniform(1.5, 3.0)
            rot = rng.uniform(0, 2 * math.pi)
            objs = place_asset_at(big_tree_path, x, y, s, rot, "BigTree", i)
            total_placed += len(objs)
        print(f"  Placed {len(positions)} large trees (r=85-115)")

    # --- Outer boulders: at terrain rise ---
    if os.path.exists(boulder_path):
        positions = scatter_in_annulus(
            rng, 78, 110, count=14, obstacles=obstacles,
            placed_positions=placed_positions, min_spacing=min_spacing * 0.6,
        )
        for i, (x, y) in enumerate(positions):
            s = rng.uniform(2.0, 5.0)
            rot = rng.uniform(0, 2 * math.pi)
            objs = place_asset_at(boulder_path, x, y, s, rot, "BoulderOuter", i)
            total_placed += len(objs)
        print(f"  Placed {len(positions)} outer boulders (r=78-110)")

    return total_placed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'=' * 60}")
    print("Regenerate Vegetation")
    print(f"{'=' * 60}")
    print(f"Seed:        {args.seed}")
    print(f"Margin:      {args.margin}")
    print(f"Min spacing: {args.min_spacing}")
    print(f"Assets:      {args.bg_assets_dir}")
    print(f"Output:      {args.output}")
    print(f"{'=' * 60}\n")

    # Step 1: Remove old vegetation
    remove_vegetation()

    # Step 2: Place new vegetation with collision avoidance
    bpy.context.view_layer.update()
    n = regenerate_vegetation(args.bg_assets_dir, args.seed, args.margin, args.min_spacing)
    print(f"\nTotal vegetation objects placed: {n}")

    # Step 3: Remove hidden template objects (mesh data persists via copies)
    cleanup_templates()

    # Step 4: Clean up stale library references from template imports
    for lib in list(bpy.data.libraries):
        bpy.data.libraries.remove(lib)

    # Step 5: Save (no make_paths_relative — it breaks library links in bg mode)
    output_abs = os.path.abspath(args.output)
    os.makedirs(
        os.path.dirname(output_abs) if os.path.dirname(output_abs) else ".", exist_ok=True
    )
    bpy.ops.wm.save_as_mainfile(filepath=output_abs)
    print(f"Saved to: {output_abs}")
