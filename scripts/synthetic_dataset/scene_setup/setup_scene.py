#!/usr/bin/env python3
"""Set up a Blender scene for NeuSky rendering from an imported asset.

Imports a 3D model (from .blend, .fbx, .glb, .gltf, .obj), places it in the
scene with a ground plane and optional background walls, then saves the scene
as a .blend file ready for blender_render_scene.py.

Usage (headless):

    blender --background --python setup_scene.py -- \
        --asset /path/to/building.blend \
        --output /path/to/scene.blend \
        --ground_material cobblestone \
        --ground_size 200 \
        --walls \
        --wall_distance 80 \
        --wall_height 30

Or open an existing blend file and import an asset into it:

    blender --background existing.blend --python setup_scene.py -- \
        --asset /path/to/building.fbx \
        --output /path/to/scene.blend
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Set up NeuSky rendering scene")
    parser.add_argument("--asset", type=str, required=True,
                        help="Path to 3D model file (.blend, .fbx, .glb, .gltf, .obj)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output .blend file")
    parser.add_argument("--ground_size", type=float, default=200.0,
                        help="Ground plane size in Blender units (default: 200)")
    parser.add_argument("--ground_texture", type=str, default=None,
                        help="Prefix of ground PBR texture (e.g., 'concrete_floor_02')")
    parser.add_argument("--texture_dir", type=str, default=None,
                        help="Directory containing PBR texture files")
    parser.add_argument("--bg_assets_dir", type=str, default=None,
                        help="Directory with background .blend assets (trees, rocks)")
    parser.add_argument("--walls", action="store_true",
                        help="Add background walls to block horizon")
    parser.add_argument("--wall_distance", type=float, default=80.0,
                        help="Distance of background walls from origin (default: 80)")
    parser.add_argument("--wall_height", type=float, default=30.0,
                        help="Height of background walls (default: 30)")
    parser.add_argument("--center_asset", action="store_true", default=True,
                        help="Center the imported asset at origin (default: True)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for imported asset (auto-computed if not set)")
    parser.add_argument("--target_height", type=float, default=15.0,
                        help="Target height for auto-scaling (default: 15 Blender units)")
    parser.add_argument("--clean_scene", action="store_true", default=True,
                        help="Remove default cube/light/camera before importing (default: True)")
    return parser.parse_args(argv)


def clean_default_scene():
    """Remove default objects (Cube, Light, Camera) from a fresh scene."""
    for name in ["Cube", "Light", "Camera"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def import_asset(asset_path):
    """Import a 3D model from various formats. Returns list of imported objects."""
    asset_path = os.path.abspath(asset_path)
    ext = Path(asset_path).suffix.lower()
    before = set(bpy.data.objects)

    if ext == ".blend":
        # Append objects from the blend file, skipping lower LODs (LOD1-3)
        with bpy.data.libraries.load(asset_path, link=False) as (data_from, data_to):
            data_to.objects = [
                name for name in data_from.objects
                if not any(name.endswith(f"_LOD{i}") or f"_LOD{i}." in name
                           for i in range(1, 4))
            ]
            data_to.materials = data_from.materials

        # Link imported objects to the scene
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)

    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=asset_path)

    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=asset_path)

    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=asset_path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    after = set(bpy.data.objects)
    imported = list(after - before)
    print(f"Imported {len(imported)} objects from {asset_path}")
    return imported


def get_scene_bounds(objects):
    """Get the bounding box of a collection of objects in world space."""
    min_corner = mathutils.Vector((float('inf'),) * 3)
    max_corner = mathutils.Vector((float('-inf'),) * 3)

    for obj in objects:
        if obj.type != 'MESH':
            continue
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        for corner in bbox_corners:
            for i in range(3):
                min_corner[i] = min(min_corner[i], corner[i])
                max_corner[i] = max(max_corner[i], corner[i])

    return min_corner, max_corner


def center_and_scale_objects(objects, target_height=15.0, scale=None):
    """Center imported objects at origin and scale to target height.

    Directly modifies root-level object transforms (no parenting) for
    reliable operation in headless/background mode.
    """
    if not objects:
        return None

    # Ensure world matrices are up-to-date after import
    bpy.context.view_layer.update()

    min_corner, max_corner = get_scene_bounds(objects)
    center = (min_corner + max_corner) / 2.0
    dimensions = max_corner - min_corner
    current_height = dimensions.z
    # Compute scale if not provided
    if scale is None and current_height > 0:
        scale = target_height / current_height
        print(f"Auto-scale: {scale:.4f} (current height {current_height:.1f} -> target {target_height:.1f})")

    # Find root-level objects (those with no parent or parent outside imported set)
    imported_set = set(objects)
    roots = [obj for obj in objects if obj.parent is None or obj.parent not in imported_set]

    # Step 1: Translate to center XY at origin, bottom at Z=0
    offset = mathutils.Vector((-center.x, -center.y, -min_corner.z))
    for obj in roots:
        obj.location += offset

    bpy.context.view_layer.update()

    # Step 2: Scale around origin
    if scale is not None and scale != 1.0:
        for obj in roots:
            obj.location *= scale
            obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    return None


def create_pbr_material(name, texture_dir, texture_prefix):
    """Create a PBR material from Poly Haven texture files.

    Expects files: {prefix}_diff_2k.jpg, {prefix}_nor_gl_2k.jpg, {prefix}_rough_2k.jpg
    """
    texture_dir = os.path.abspath(texture_dir)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    # Diffuse
    diff_path = os.path.join(texture_dir, f"{texture_prefix}_diff_2k.jpg")
    if os.path.exists(diff_path):
        diff_tex = nodes.new("ShaderNodeTexImage")
        diff_tex.image = bpy.data.images.load(diff_path)
        links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])
        links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness
    rough_path = os.path.join(texture_dir, f"{texture_prefix}_rough_2k.jpg")
    if os.path.exists(rough_path):
        rough_tex = nodes.new("ShaderNodeTexImage")
        rough_tex.image = bpy.data.images.load(rough_path)
        rough_tex.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

    # Normal
    nor_path = os.path.join(texture_dir, f"{texture_prefix}_nor_gl_2k.jpg")
    if os.path.exists(nor_path):
        nor_tex = nodes.new("ShaderNodeTexImage")
        nor_tex.image = bpy.data.images.load(nor_path)
        nor_tex.image.colorspace_settings.name = "Non-Color"
        nor_map = nodes.new("ShaderNodeNormalMap")
        links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
        links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def deform_ground_edges(ground_obj, inner_radius=80.0, max_height=15.0,
                        grid_res=400, noise_scale=0.05, seed=42):
    """Deform ground plane edges upward to create hills that hide the horizon.

    Replaces the ground plane mesh with a high-res grid, generates proper UVs,
    enables smooth shading, then raises vertices beyond inner_radius.
    """
    import bmesh

    mesh = ground_obj.data

    # Get current plane extent from existing verts
    half_size = max(abs(v.co.x) for v in mesh.vertices)

    bm = bmesh.new()

    # Build a uniform grid (grid_res x grid_res quads)
    step = 2.0 * half_size / grid_res
    verts_grid = []
    for iy in range(grid_res + 1):
        row = []
        for ix in range(grid_res + 1):
            x = -half_size + ix * step
            y = -half_size + iy * step
            row.append(bm.verts.new((x, y, 0.0)))
        verts_grid.append(row)

    # Create faces
    for iy in range(grid_res):
        for ix in range(grid_res):
            bm.faces.new([
                verts_grid[iy][ix],
                verts_grid[iy][ix + 1],
                verts_grid[iy + 1][ix + 1],
                verts_grid[iy + 1][ix],
            ])

    # Generate UV coordinates (planar projection, normalised 0-1)
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for loop in face.loops:
            u = (loop.vert.co.x + half_size) / (2.0 * half_size)
            v_coord = (loop.vert.co.y + half_size) / (2.0 * half_size)
            loop[uv_layer].uv = (u, v_coord)

    # Deform: raise vertices beyond inner_radius
    max_dist = half_size * math.sqrt(2)  # corner distance
    rng = random.Random(seed)

    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        dist = math.sqrt(v.co.x**2 + v.co.y**2)
        if dist > inner_radius:
            t = (dist - inner_radius) / (max_dist - inner_radius)
            t = min(t, 1.0)
            # Smoothstep for gradual rise
            t = t * t * (3 - 2 * t)
            noise = rng.uniform(-noise_scale, noise_scale) * t
            v.co.z = max_height * t + noise * max_height

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Enable smooth shading on all faces
    for poly in mesh.polygons:
        poly.use_smooth = True


def create_ground_plane(size=200.0, texture_dir=None, texture_prefix=None):
    """Create a ground plane at Z=0 with optional PBR texture.

    Edges are deformed upward to create gentle hills that hide the horizon.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "GroundPlane"

    if texture_dir and texture_prefix:
        mat = create_pbr_material("GroundMaterial", texture_dir, texture_prefix)
        ground.data.materials.append(mat)

        # UV scale: tile the texture across the large ground plane
        # Scale UVs so texture tiles every ~5 Blender units
        tile_factor = size / 5.0
        for node in mat.node_tree.nodes:
            if node.type == "MAPPING":
                node.inputs["Scale"].default_value = (tile_factor, tile_factor, 1.0)
                break
    else:
        # Fallback: simple grey material
        mat = bpy.data.materials.new(name="GroundMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.8
        ground.data.materials.append(mat)

    # Deform edges upward to hide the horizon
    deform_ground_edges(ground, inner_radius=80.0, max_height=15.0)

    return ground


def import_background_asset(blend_path):
    """Import LOD0 objects from a .blend file for use as background geometry.

    Poly Haven assets include multiple LODs (LOD0-LOD3) at the same position.
    Only LOD0 (highest quality) is imported; lower LODs are skipped.

    Uses absolute path for the blend file so Blender correctly resolves
    relative texture paths (//textures/...) from the asset's directory.
    """
    blend_path = os.path.abspath(blend_path)

    imported = []
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        # Filter out LOD1/2/3 — keep LOD0 and objects without LOD suffixes
        data_to.objects = [
            name for name in data_from.objects
            if not any(name.endswith(f"_LOD{i}") or f"_LOD{i}." in name
                       for i in range(1, 4))
        ]
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported.append(obj)

    return imported


def place_background_ring(asset_path, distance, count, scale_range=(1.0, 1.5),
                          z_offset=0.0, name_prefix="BG"):
    """Place instances of a background asset in a ring around the scene."""
    import random as _rng
    _rng.seed(hash((name_prefix, distance, count)) % (2**31))

    placed = []
    for i in range(count):
        angle = 2 * math.pi * i / count + _rng.uniform(-0.2, 0.2)
        r = distance + _rng.uniform(-distance * 0.1, distance * 0.1)
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        objs = import_background_asset(asset_path)
        if not objs:
            continue

        # Find root objects
        obj_set = set(objs)
        roots = [o for o in objs if o.parent is None or o.parent not in obj_set]

        s = _rng.uniform(scale_range[0], scale_range[1])
        rot_z = _rng.uniform(0, 2 * math.pi)

        for obj in roots:
            obj.location = mathutils.Vector((x, y, z_offset))
            obj.scale = obj.scale * s
            obj.rotation_euler.z = rot_z
            obj.name = f"{name_prefix}_{i:02d}_{obj.name}"

        for obj in objs:
            if obj not in roots:
                obj.name = f"{name_prefix}_{i:02d}_{obj.name}"

        placed.extend(objs)

    return placed


def create_background_environment(bg_assets_dir, distance=60.0):
    """Create a natural-looking background environment using trees, rocks, and shrubs.

    Places assets from bg_assets_dir in concentric rings around the scene.
    """
    placed_count = 0

    # Ring of trees (main horizon blocker)
    tree_path = os.path.join(bg_assets_dir, "tree_small_02_1k.blend")
    if os.path.exists(tree_path):
        objs = place_background_ring(tree_path, distance, count=12,
                                     scale_range=(2.0, 3.5), name_prefix="Tree")
        placed_count += len(objs)
        print(f"  Placed {len(objs)} tree objects in ring at r={distance}")

    # Ring of boulders (mid-ground)
    boulder_path = os.path.join(bg_assets_dir, "boulder_01_1k.blend")
    if os.path.exists(boulder_path):
        objs = place_background_ring(boulder_path, distance * 0.7, count=8,
                                     scale_range=(3.0, 6.0), name_prefix="Boulder")
        placed_count += len(objs)
        print(f"  Placed {len(objs)} boulder objects at r={distance*0.7:.0f}")

    # Shrubs to fill gaps between trees and boulders
    shrub_path = os.path.join(bg_assets_dir, "shrub_04_1k.blend")
    if os.path.exists(shrub_path):
        objs = place_background_ring(shrub_path, distance * 0.85, count=16,
                                     scale_range=(3.0, 5.0), name_prefix="Shrub")
        placed_count += len(objs)
        print(f"  Placed {len(objs)} shrub objects at r={distance*0.85:.0f}")

    # Larger trees further back
    big_tree_path = os.path.join(bg_assets_dir, "island_tree_01_1k.blend")
    if os.path.exists(big_tree_path):
        objs = place_background_ring(big_tree_path, distance * 1.3, count=8,
                                     scale_range=(1.5, 2.5), name_prefix="BigTree")
        placed_count += len(objs)
        print(f"  Placed {len(objs)} large tree objects at r={distance*1.3:.0f}")

    # Extra boulders at the terrain rise to mask the transition
    if os.path.exists(boulder_path):
        objs = place_background_ring(boulder_path, distance * 1.0, count=12,
                                     scale_range=(2.0, 4.0), name_prefix="BoulderOuter")
        placed_count += len(objs)
        print(f"  Placed {len(objs)} outer boulder objects at r={distance*1.0:.0f}")

    return placed_count


def setup_scene(args):
    """Main scene setup function."""
    print(f"\n{'='*60}")
    print("NeuSky Scene Setup")
    print(f"{'='*60}")
    print(f"Asset:        {args.asset}")
    print(f"Output:       {args.output}")
    print(f"Ground size:  {args.ground_size}")
    print(f"Walls:        {args.walls}")
    if args.walls:
        print(f"Wall dist:    {args.wall_distance}")
        print(f"Wall height:  {args.wall_height}")
    print(f"Target height: {args.target_height}")
    print(f"{'='*60}\n")

    # Clean default scene
    if args.clean_scene:
        clean_default_scene()

    # Import asset
    imported = import_asset(args.asset)
    if not imported:
        print("Warning: No objects imported!")
        return

    # Center and scale
    if args.center_asset:
        parent = center_and_scale_objects(
            imported,
            target_height=args.target_height,
            scale=args.scale,
        )

    # Recompute bounds after centering
    bpy.context.view_layer.update()
    min_c, max_c = get_scene_bounds(imported)
    dims = max_c - min_c
    print(f"Scene bounds: ({min_c.x:.1f}, {min_c.y:.1f}, {min_c.z:.1f}) to "
          f"({max_c.x:.1f}, {max_c.y:.1f}, {max_c.z:.1f})")
    print(f"Scene dimensions: {dims.x:.1f} x {dims.y:.1f} x {dims.z:.1f}")

    # Create ground plane with PBR texture
    ground = create_ground_plane(args.ground_size, args.texture_dir, args.ground_texture)
    if args.ground_texture:
        print(f"Created ground plane: {args.ground_size}x{args.ground_size} ({args.ground_texture})")
    else:
        print(f"Created ground plane: {args.ground_size}x{args.ground_size} (default grey)")

    # Create background environment
    if args.bg_assets_dir and os.path.isdir(args.bg_assets_dir):
        print("Placing background assets...")
        n = create_background_environment(args.bg_assets_dir, args.wall_distance)
        print(f"Total background objects: {n}")
    elif args.walls:
        # Fallback: simple flat walls if no bg assets
        print("No background assets directory, using simple walls")
        from functools import partial
        walls = []
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = args.wall_distance * math.cos(angle)
            y = args.wall_distance * math.sin(angle)
            bpy.ops.mesh.primitive_plane_add(size=1, location=(x, y, args.wall_height / 2))
            wall = bpy.context.active_object
            wall.name = f"BackgroundWall_{i:02d}"
            wall_width = 2.0 * args.wall_distance * math.tan(math.pi / 8) * 1.1
            wall.scale = (wall_width / 2, 1, args.wall_height / 2)
            wall.rotation_euler.z = angle + math.pi / 2
            mat = bpy.data.materials.new(name=f"WallMat_{i}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.4, 0.38, 0.35, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.9
            wall.data.materials.append(mat)
            walls.append(wall)
        print(f"Created {len(walls)} background walls")

    # Ensure world exists for HDRI shader
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    bpy.context.scene.world.use_nodes = True

    # Fix image paths: after appending from external blend files, image paths
    # are relative to the *original* blend file but Blender can't properly
    # re-relativise them when saving to a new location (no file path is set
    # yet in --background mode). Fix by making all image paths absolute first.
    for img in bpy.data.images:
        if img.filepath and img.filepath.startswith("//"):
            abs_path = bpy.path.abspath(img.filepath)
            if os.path.exists(abs_path):
                img.filepath = abs_path
            else:
                # The path couldn't be resolved from the current (empty) blend path.
                # This happens when the image was appended from another blend file
                # and its relative path doesn't resolve from CWD. Leave as-is;
                # will be fixed by make_paths_relative after saving.
                pass

    # Save
    output_abs = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_abs)

    # Now re-save with paths relative to the output file
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
    print(f"\nSaved scene to: {args.output}")

    # Print recommended render command
    scene_height = dims.z
    sphere_radius = max(dims.x, dims.y) * 1.5
    sphere_center_z = scene_height / 2
    print(f"\nRecommended render command:")
    print(
        f"  blender --background {args.output} "
        "--python scripts/synthetic_dataset/blender_render_scene.py -- \\"
    )
    print(f"    --output /path/to/output \\")
    print(f"    --hdri_dir /path/to/HDRIs \\")
    print(f"    --num_frames 200 \\")
    print(f"    --resolution 1920 1080 \\")
    print(f"    --sphere_radius {sphere_radius:.0f} \\")
    print(f"    --sphere_center 0 0 {sphere_center_z:.0f} \\")
    print(f"    --sphere_elevation_min 10 \\")
    print(f"    --sphere_elevation_max 50 \\")
    print(f"    --samples 128")


if __name__ == "__main__":
    args = parse_args()
    setup_scene(args)
