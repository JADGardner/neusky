#!/usr/bin/env python3
"""Automated Blender rendering script for NeuSky synthetic datasets.

Sets up multi-pass rendering (RGB, depth, normal, albedo, mask, roughness, metallic,
transmission, IOR) with HDRI
cycling for multi-illumination, generates camera positions on a sphere,
renders all passes, and writes transforms.json.

Supports Blender 5.0+ (uses compositing_node_group API).

Usage (headless):

    blender --background scene.blend --python blender_render_scene.py -- \
        --output /path/to/output \
        --hdri_dir /path/to/HDRIs \
        --num_frames 200 \
        --resolution 1920 1080 \
        --focal_mm 50 \
        --sphere_radius 15 \
        --sphere_center 0 0 5 \
        --upper_only \
        --seed 42

The script expects the .blend file to already contain the scene geometry.
It will set up the compositor, camera, world shader, and render settings
automatically. Each frame produces a multipart EXR containing all render
passes (rgb, depth, normal, albedo, alpha) plus a copy of the HDRI used.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import mathutils

# Background vegetation/scenery prefixes — excluded from auto_exclude bbox
# checks because their inflated bounding boxes form a near-continuous shell
# that rejects cameras at many azimuths, limiting coverage to ~180°.
BACKGROUND_PREFIXES = (
    "Tree_", "Boulder_", "Shrub_", "BigTree_", "BoulderOuter_", "CoastalCliff_",
)


# ---------------------------------------------------------------------------
# Argument parsing (everything after '--' on the command line)
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render NeuSky synthetic dataset")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for rendered dataset")
    parser.add_argument("--hdri_dir", type=str, required=True,
                        help="Directory containing HDRI .exr files")
    parser.add_argument("--num_frames", type=int, default=200,
                        help="Number of camera viewpoints to render (default: 200)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                        metavar=("W", "H"),
                        help="Render resolution (default: 1920 1080)")
    parser.add_argument("--focal_mm", type=float, default=50.0,
                        help="Camera focal length in mm (default: 50)")
    parser.add_argument("--focal_mm_range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Per-frame focal length range in mm (e.g. 35 70). "
                             "Overrides --focal_mm. Sampled uniformly per frame.")
    parser.add_argument("--exposure_range", type=float, nargs=2, default=None,
                        metavar=("MIN_EV", "MAX_EV"),
                        help="Per-frame exposure variation in EV stops (e.g. -1.5 1.5). "
                             "Applied as 2^EV multiplier to Cycles film_exposure. "
                             "Only affects RGB, not depth/normal/roughness/metallic.")
    parser.add_argument("--sphere_radius", type=float, default=15.0,
                        help="Camera sampling sphere radius, or max radius if --sphere_radius_min is set (default: 15)")
    parser.add_argument("--sphere_radius_min", type=float, default=None,
                        help="Min camera radius (enables variable distance sampling)")
    parser.add_argument("--sphere_center", type=float, nargs=3, default=[0, 0, 5],
                        metavar=("X", "Y", "Z"),
                        help="Camera look-at / sphere center (default: 0 0 5)")
    parser.add_argument("--sphere_elevation_min", type=float, default=10.0,
                        help="Minimum camera elevation angle in degrees (default: 10)")
    parser.add_argument("--sphere_elevation_max", type=float, default=70.0,
                        help="Maximum camera elevation angle in degrees (default: 70)")
    parser.add_argument("--camera_height_min", type=float, default=None,
                        help="Min camera height above ground (overrides elevation)")
    parser.add_argument("--camera_height_max", type=float, default=None,
                        help="Max camera height above ground (overrides elevation)")
    parser.add_argument("--azimuth_min", type=float, default=0.0,
                        help="Minimum azimuth angle in degrees (default: 0)")
    parser.add_argument("--azimuth_max", type=float, default=360.0,
                        help="Maximum azimuth angle in degrees (default: 360)")
    parser.add_argument("--upper_only", action="store_true",
                        help="Only sample cameras above the ground plane")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for camera sampling (default: 42)")
    parser.add_argument("--samples", type=int, default=128,
                        help="Cycles render samples (default: 128)")
    parser.add_argument("--aabb", type=int, default=4,
                        help="AABB scale for transforms.json (default: 4)")
    parser.add_argument("--look_at", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Camera look-at point (default: same as sphere_center)")
    parser.add_argument("--look_at_z_range", type=float, nargs=2, default=None,
                        metavar=("ZMIN", "ZMAX"),
                        help="Random Z range for look-at point (vary between ground and tower)")
    parser.add_argument("--look_at_bias", type=float, default=0.0,
                        help="Bias look-at XY toward camera position (0=center, 0.3=30%% toward camera). "
                             "Makes cameras look at the nearest part of the building.")
    parser.add_argument("--look_at_building", action="store_true",
                        help="Auto-target look-at on building AABB faces (geometry-aware). "
                             "Overrides --look_at, --look_at_bias, --look_at_z_range.")
    parser.add_argument("--building_clearance", type=float, default=None,
                        help="Min XY distance from building walls (requires --look_at_building). "
                             "Replaces --sphere_radius_min with geometry-aware minimum distance.")
    parser.add_argument("--min_clearance", type=float, default=None,
                        help="Min distance to geometry above camera — rejects cameras "
                             "inside buildings/cliffs via ray casting (e.g. 5.0)")
    parser.add_argument("--exclude_box", type=float, nargs=4, action="append",
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="XY bounding box to exclude cameras from (can specify multiple times)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "exr"],
                        help="Output format: png (smaller, for iteration) or exr (full passes)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Max CPU threads for rendering (0 = auto/all cores, default: 0)")
    parser.add_argument("--hdri_16k", action="store_true",
                        help="Use 16K HDRIs from hdris_16k/ sibling directory instead of hdri_dir")
    parser.add_argument("--auto_exclude", action="store_true",
                        help="Auto-reject cameras inside any mesh bounding box (fast, no ray-cast)")
    parser.add_argument("--min_building_fraction", type=float, default=None,
                        help="Minimum fraction of ray-cast hits through camera that must "
                             "hit scene geometry (0-1, e.g. 0.1 for 10%%). "
                             "Frames below threshold are skipped before rendering.")
    parser.add_argument("--hdri_offset", type=int, default=0,
                        help="Start index offset into the sorted HDRI list when cycling. "
                             "Lets a separate eval render use a different HDRI subsequence "
                             "than the training render (default: 0)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Auto-exclude: collect mesh bounding boxes for camera rejection
# ---------------------------------------------------------------------------

def collect_mesh_bboxes(exclude_names=None):
    """Collect world-space axis-aligned bounding boxes for all mesh objects.

    Returns a list of (min_xyz, max_xyz) tuples. Skips ground planes,
    background vegetation/scenery, and any objects in exclude_names.
    """
    exclude_names = set(exclude_names or [])
    bboxes = []
    skipped_bg = 0
    total_meshes = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        total_meshes += 1
        if obj.name in exclude_names:
            continue
        # Skip ground planes (large flat objects aren't useful for exclusion)
        if "ground" in obj.name.lower():
            continue
        # Skip background vegetation/scenery (inflated bboxes block cameras)
        if any(obj.name.startswith(p) for p in BACKGROUND_PREFIXES):
            skipped_bg += 1
            continue
        bb_world = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        ws_min = tuple(min(v[i] for v in bb_world) for i in range(3))
        ws_max = tuple(max(v[i] for v in bb_world) for i in range(3))
        bboxes.append((ws_min, ws_max))
    if skipped_bg:
        print(f"  Auto-exclude: {len(bboxes)} building bboxes from {total_meshes} meshes "
              f"(skipped {skipped_bg} background objects)")
    return bboxes


def point_inside_any_bbox(pos, bboxes):
    """Check if a point is inside any of the given bounding boxes."""
    x, y, z = pos.x, pos.y, pos.z
    for (mn, mx) in bboxes:
        if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]:
            return True
    return False


def merge_bboxes(bboxes):
    """Merge list of (min_xyz, max_xyz) into a single enclosing AABB."""
    all_min = tuple(min(b[0][i] for b in bboxes) for i in range(3))
    all_max = tuple(max(b[1][i] for b in bboxes) for i in range(3))
    return (all_min, all_max)


def xy_distance_to_bbox(pos, bbox):
    """Compute XY distance from a point to the nearest edge of an AABB."""
    mn, mx = bbox
    dx = max(mn[0] - pos.x, 0.0, pos.x - mx[0])
    dy = max(mn[1] - pos.y, 0.0, pos.y - mx[1])
    return math.sqrt(dx * dx + dy * dy)


def sample_lookat_on_bbox(pos, bbox, rng):
    """Sample a point on the AABB face most visible from camera position.

    Picks the face whose outward normal has the largest dot product with
    the direction from AABB center to camera, then samples a random point
    on that face.
    """
    mn, mx = bbox
    center = [(mn[i] + mx[i]) / 2 for i in range(3)]
    dir_vec = [pos.x - center[0], pos.y - center[1], pos.z - center[2]]

    # 6 faces: (axis_index, sign, fixed_coordinate_value)
    faces = [
        (0, +1, mx[0]), (0, -1, mn[0]),  # +X, -X
        (1, +1, mx[1]), (1, -1, mn[1]),  # +Y, -Y
        (2, +1, mx[2]), (2, -1, mn[2]),  # +Z, -Z
    ]
    # Score: dot product of face normal with camera direction
    best_face = max(faces, key=lambda f: f[1] * dir_vec[f[0]])
    axis, sign, fixed = best_face

    pt = [0.0, 0.0, 0.0]
    pt[axis] = fixed
    for a in range(3):
        if a != axis:
            pt[a] = rng.uniform(mn[a], mx[a])
    return mathutils.Vector(pt)


# ---------------------------------------------------------------------------
# Building coverage check (pre-render ray-cast filter)
# ---------------------------------------------------------------------------

def check_building_coverage(scene, cam_matrix, depsgraph, fl_x, fl_y, cx, cy,
                            res_x, res_y, grid_n=16):
    """Cast rays through a coarse pixel grid and return fraction hitting foreground geometry.

    Only counts hits on building/foreground objects. Ignores ground planes and
    background objects (trees, boulders, etc.) using BACKGROUND_PREFIXES.

    Args:
        cam_matrix: 4x4 camera-to-world matrix (mathutils.Matrix).
    """
    origin = cam_matrix.translation

    hits = 0
    total = grid_n * grid_n
    for gy in range(grid_n):
        for gx in range(grid_n):
            # Pixel position (evenly spaced across frame)
            u = (gx + 0.5) / grid_n * res_x
            v = (gy + 0.5) / grid_n * res_y
            # Camera-space ray direction (Blender: -Z forward, Y up)
            dx = (u - cx) / fl_x
            dy = -(v - cy) / fl_y
            dz = -1.0
            dir_cam = mathutils.Vector((dx, dy, dz)).normalized()
            # Transform to world space (rotation only)
            dir_world = cam_matrix.to_3x3() @ dir_cam
            # Cast ray — check hit object is foreground (not ground/background)
            result, location, normal, index, hit_object, matrix = scene.ray_cast(
                depsgraph, origin, dir_world)
            if result and hit_object is not None:
                name = hit_object.name
                is_background = (
                    "ground" in name.lower()
                    or any(name.startswith(p) for p in BACKGROUND_PREFIXES)
                )
                if not is_background:
                    hits += 1
    return hits / total


# ---------------------------------------------------------------------------
# Camera sampling on a sphere with elevation constraints
# ---------------------------------------------------------------------------

def sample_camera_positions(num_frames, radius, center, elev_min_deg, elev_max_deg,
                            upper_only, seed, height_min=None, height_max=None,
                            azimuth_min_deg=0.0, azimuth_max_deg=360.0,
                            radius_min=None, exclude_boxes=None,
                            look_at=None, look_at_z_range=None,
                            look_at_bias=0.0, building_bbox=None,
                            building_clearance=None, validate_fn=None,
                            focal_mms=None):
    """Sample camera positions on a sphere with constrained elevation.

    If height_min/height_max are provided, cameras are placed at a fixed
    height range with the full radius as XY distance from center, ignoring
    elevation angles. Otherwise uses elevation-based sphere sampling.

    If radius_min is set, radius is sampled uniformly between radius_min and
    radius for each frame, giving a mix of close-up and wide shots.

    azimuth_min_deg/azimuth_max_deg restrict the azimuth range (default: full
    360 degrees). Useful for scenes with background geometry that blocks views
    from certain directions.

    exclude_boxes is a list of (xmin, ymin, xmax, ymax) tuples defining XY
    regions where cameras must not be placed (e.g. building footprints).

    look_at overrides where cameras point (default: orbit center).
    look_at_z_range randomly varies the look-at Z per frame.

    validate_fn is an optional callback(pos, target, focal_mm) -> bool used
    for geometry checks (bbox, ray-cast, coverage). Positions where it
    returns False are re-sampled.

    focal_mms is an optional list of per-frame focal lengths passed to
    validate_fn for coverage checks.

    Returns list of (position, look_at) tuples.
    """
    rng = random.Random(seed)
    center = mathutils.Vector(center)
    look_at_base = mathutils.Vector(look_at) if look_at is not None else center.copy()

    az_min = math.radians(azimuth_min_deg)
    az_max = math.radians(azimuth_max_deg)

    max_retries = 200

    positions = []
    resampled = 0
    for frame_i in range(num_frames):
        for attempt in range(max_retries):
            azimuth = rng.uniform(az_min, az_max)

            # Sample radius (fixed or variable)
            r = rng.uniform(radius_min, radius) if radius_min is not None else radius

            if height_min is not None and height_max is not None:
                x = center.x + r * math.cos(azimuth)
                y = center.y + r * math.sin(azimuth)
                z = rng.uniform(height_min, height_max)
                pos = mathutils.Vector((x, y, z))
            else:
                elevation = rng.uniform(math.radians(elev_min_deg),
                                        math.radians(elev_max_deg))
                x = r * math.cos(elevation) * math.cos(azimuth)
                y = r * math.cos(elevation) * math.sin(azimuth)
                z = r * math.sin(elevation)
                if upper_only:
                    z = abs(z)
                pos = center + mathutils.Vector((x, y, z))

            # Reject positions inside any exclusion box
            if exclude_boxes:
                inside = False
                for xmin, ymin, xmax, ymax in exclude_boxes:
                    if xmin <= pos.x <= xmax and ymin <= pos.y <= ymax:
                        inside = True
                        break
                if inside:
                    continue

            # Reject positions too close to building walls
            if building_clearance is not None and building_bbox is not None:
                if xy_distance_to_bbox(pos, building_bbox) < building_clearance:
                    continue

            # Determine look-at target for this frame (before validate_fn,
            # which may need the target for coverage ray-casting)
            if building_bbox is not None:
                target = sample_lookat_on_bbox(pos, building_bbox, rng)
            elif look_at_z_range is not None:
                look_z = rng.uniform(look_at_z_range[0], look_at_z_range[1])
                target = mathutils.Vector((look_at_base.x, look_at_base.y, look_z))
            else:
                target = look_at_base.copy()

            # Bias look-at XY toward camera position (only when not using building bbox)
            if building_bbox is None and look_at_bias > 0:
                target.x += look_at_bias * (pos.x - target.x)
                target.y += look_at_bias * (pos.y - target.y)

            # Reject positions failing geometry/coverage checks
            if validate_fn:
                fm = focal_mms[frame_i] if focal_mms else None
                if not validate_fn(pos, target, fm):
                    continue

            if attempt > 0:
                resampled += 1
            positions.append((pos, target))
            break
        else:
            print(f"WARNING: Camera {frame_i+1}: no valid position after "
                  f"{max_retries} attempts, using last sample")
            target = look_at_base.copy()
            positions.append((pos, target))

    if resampled:
        print(f"  Resampled {resampled}/{num_frames} camera(s) that were inside "
              f"geometry or exclusion zones")

    return positions


# ---------------------------------------------------------------------------
# Scene and render setup
# ---------------------------------------------------------------------------

def setup_render_settings(scene, args):
    """Configure Cycles render settings."""
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True

    # Enable GPU rendering if available (prefer OptiX > CUDA)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    gpu_enabled = False
    for compute_type in ["OPTIX", "CUDA", "HIP"]:
        try:
            prefs.compute_device_type = compute_type
            prefs.get_devices()
            gpu_devs = [d for d in prefs.devices if d.type != "CPU"]
            if gpu_devs:
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                scene.cycles.device = "GPU"
                gpu_enabled = True
                print(f"GPU:        {gpu_devs[0].name} ({compute_type})")
                break
        except Exception:
            continue
    if not gpu_enabled:
        scene.cycles.device = "CPU"
        print("GPU:        none found, using CPU")

    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.resolution_percentage = 100

    # CPU thread limiting (0 = auto)
    if args.threads > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    else:
        scene.render.threads_mode = "AUTO"

    # Only use transparent film in EXR mode (needed for clean alpha mask).
    # In PNG mode, render the HDRI sky background visibly.
    scene.render.film_transparent = (args.format == "exr")

    if args.format == "exr":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "32"
        # Enable render passes for EXR mode
        view_layer = scene.view_layers[0]
        view_layer.use_pass_normal = True
        view_layer.use_pass_z = True
        view_layer.use_pass_diffuse_color = True
        view_layer.use_pass_environment = True
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"


def setup_world_shader(scene, hdri_path=None):
    """Set up world shader with Environment Texture node and rotation control.

    Node chain: TexCoord -> Mapping (rotation) -> EnvTex -> Background -> Output.
    Returns (env_node, mapping_node) so the caller can update the HDRI image
    and rotation per frame.
    """
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create nodes
    node_texcoord = nodes.new(type="ShaderNodeTexCoord")
    node_mapping = nodes.new(type="ShaderNodeMapping")
    node_mapping.name = "EnvMapRotation"
    node_env = nodes.new(type="ShaderNodeTexEnvironment")
    node_env.name = "Environment Texture"
    node_bg = nodes.new(type="ShaderNodeBackground")
    node_output = nodes.new(type="ShaderNodeOutputWorld")

    # Link: TexCoord -> Mapping -> EnvTex -> Background -> Output
    links.new(node_texcoord.outputs["Generated"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_output.inputs["Surface"])

    if hdri_path:
        node_env.image = bpy.data.images.load(hdri_path)

    return node_env, node_mapping


def normalize_glass_materials():
    """Normalize non-standard glass materials to Principled BSDF with Transmission Weight.

    Handles three cases:
    1. Glass BSDF nodes → replaced with Principled BSDF (Transmission Weight=1.0)
    2. Principled BSDF with low constant alpha (<0.5) → set Transmission Weight = 1-alpha, alpha=1.0
    3. Light Path glass trick (Transparent BSDF + Glossy mix, no active Principled) →
       reconnect existing transmissive Principled BSDF to Material Output

    This ensures all glass/transparent materials use Principled BSDF Transmission Weight,
    which can then be captured as an AOV for ground-truth inverse rendering evaluation.
    """
    glass_converted = 0
    alpha_converted = 0
    trick_converted = 0

    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Case 1: Glass BSDF → Principled BSDF with Transmission Weight
        glass_nodes = [n for n in nodes if n.type == 'BSDF_GLASS']
        for glass_node in glass_nodes:
            principled = nodes.new('ShaderNodeBsdfPrincipled')
            principled.location = glass_node.location

            # Copy Color → Base Color
            color_input = glass_node.inputs["Color"]
            if color_input.is_linked:
                links.new(color_input.links[0].from_socket, principled.inputs["Base Color"])
            else:
                principled.inputs["Base Color"].default_value = color_input.default_value

            # Copy Roughness
            rough_input = glass_node.inputs["Roughness"]
            if rough_input.is_linked:
                links.new(rough_input.links[0].from_socket, principled.inputs["Roughness"])
            else:
                principled.inputs["Roughness"].default_value = rough_input.default_value

            # Copy IOR
            ior_input = glass_node.inputs["IOR"]
            if ior_input.is_linked:
                links.new(ior_input.links[0].from_socket, principled.inputs["IOR"])
            else:
                principled.inputs["IOR"].default_value = ior_input.default_value

            # Copy Normal if connected
            normal_input = glass_node.inputs["Normal"]
            if normal_input.is_linked:
                links.new(normal_input.links[0].from_socket, principled.inputs["Normal"])

            # Set transmission properties
            principled.inputs["Transmission Weight"].default_value = 1.0
            principled.inputs["Metallic"].default_value = 0.0

            # Relink outputs: whoever was connected to Glass BSDF output → Principled BSDF output
            for link in list(glass_node.outputs[0].links):
                links.new(principled.outputs[0], link.to_socket)

            nodes.remove(glass_node)
            glass_converted += 1
            print(f"  Glass→Principled: {mat.name}")

        # Case 2: Principled BSDF with low constant alpha → Transmission Weight
        for node in list(nodes):
            if node.type != 'BSDF_PRINCIPLED':
                continue
            alpha_input = node.inputs["Alpha"]
            if alpha_input.is_linked:
                continue
            alpha_val = alpha_input.default_value
            if alpha_val < 0.5:
                transmission_weight = 1.0 - alpha_val
                node.inputs["Transmission Weight"].default_value = transmission_weight
                alpha_input.default_value = 1.0
                alpha_converted += 1
                print(f"  Alpha({alpha_val:.2f})→Transmission({transmission_weight:.2f}): {mat.name}")

        # Case 3: Light Path glass trick — Transparent BSDF + Glossy mix without
        # an active Principled BSDF.  If the material has a disconnected Principled
        # BSDF with Transmission > 0, reconnect it to the Material Output.
        active_principled = _find_active_principled(nodes)
        if active_principled is None:
            # Check for Transparent BSDF (hallmark of the glass trick)
            has_transparent = any(n.type == 'BSDF_TRANSPARENT' for n in nodes)
            if not has_transparent:
                continue

            # Find a Principled BSDF with Transmission > 0 (the intended glass shader)
            candidate = None
            for node in nodes:
                if node.type != 'BSDF_PRINCIPLED':
                    continue
                trans_input = node.inputs["Transmission Weight"]
                if not trans_input.is_linked and trans_input.default_value > 0:
                    candidate = node
                    break
            if candidate is None:
                continue

            # Find the active Material Output and rewire
            output_node = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
                    output_node = node
                    break
            if output_node is None:
                continue

            surface_input = output_node.inputs.get('Surface')
            if surface_input is None:
                continue

            # Remove existing connection to Surface
            for link in list(surface_input.links):
                links.remove(link)

            # Connect the transmissive Principled BSDF to Material Output
            links.new(candidate.outputs[0], surface_input)
            trick_converted += 1
            trans_val = candidate.inputs["Transmission Weight"].default_value
            ior_val = candidate.inputs["IOR"].default_value
            print(f"  LightPath trick→Principled (Trans={trans_val:.2f}, IOR={ior_val:.2f}): {mat.name}")

    total = glass_converted + alpha_converted + trick_converted
    if total > 0:
        print(f"Normalized {total} glass material(s): "
              f"{glass_converted} Glass BSDF, {alpha_converted} alpha-based, "
              f"{trick_converted} Light Path trick")
    else:
        print("No glass materials found to normalize")
    return total


def _find_active_principled(nodes):
    """Find the Principled BSDF that feeds into the active Material Output.

    Traces backwards from the Material Output's Surface input through
    intermediate nodes (Mix Shader, Add Shader, etc.) to find the first
    Principled BSDF in the chain. Returns None if not found.
    """
    # Find active Material Output
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            output_node = node
            break
    if output_node is None:
        return None

    surface_input = output_node.inputs.get('Surface')
    if surface_input is None or not surface_input.is_linked:
        return None

    # BFS backwards through the node graph
    to_visit = [surface_input.links[0].from_node]
    visited = set()
    while to_visit:
        node = to_visit.pop(0)
        if node.name in visited:
            continue
        visited.add(node.name)
        if node.type == 'BSDF_PRINCIPLED':
            return node
        # Follow all input links backwards
        for inp in node.inputs:
            for link in inp.links:
                to_visit.append(link.from_node)
    return None


def setup_aov_passes(scene):
    """Add Roughness, Metallic, Transmission, and IOR AOVs and wire Principled BSDF inputs to them.

    For each material with a Principled BSDF, creates AOV Output nodes that
    capture the roughness, metallic, transmission weight, and IOR values
    (whether driven by textures or set as default values). Modifies node trees
    in memory only.
    """
    view_layer = scene.view_layers[0]
    for aov_name in ["Roughness", "Metallic", "Transmission", "IOR"]:
        aov = view_layer.aovs.add()
        aov.name = aov_name
        aov.type = 'VALUE'

    # Map AOV names to Principled BSDF input names
    aov_to_input = {
        "Roughness": "Roughness",
        "Metallic": "Metallic",
        "Transmission": "Transmission Weight",
        "IOR": "IOR",
    }

    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find the Principled BSDF that actually drives the Material Output
        # by tracing backwards from the active output node
        principled = _find_active_principled(nodes)
        if principled is None:
            continue

        for aov_name, input_name in aov_to_input.items():
            aov_node = nodes.new('ShaderNodeOutputAOV')
            aov_node.aov_name = aov_name

            prop_input = principled.inputs[input_name]
            if prop_input.is_linked:
                source_socket = prop_input.links[0].from_socket
            else:
                val_node = nodes.new('ShaderNodeValue')
                val_node.outputs[0].default_value = prop_input.default_value
                source_socket = val_node.outputs[0]

            if aov_name == "IOR":
                # Mask IOR to only output where Transmission Weight > 0,
                # so opaque surfaces get IOR=0 instead of the default 1.5
                trans_input = principled.inputs["Transmission Weight"]
                if trans_input.is_linked:
                    trans_socket = trans_input.links[0].from_socket
                else:
                    trans_val = nodes.new('ShaderNodeValue')
                    trans_val.outputs[0].default_value = trans_input.default_value
                    trans_socket = trans_val.outputs[0]

                gt_node = nodes.new('ShaderNodeMath')
                gt_node.operation = 'GREATER_THAN'
                gt_node.inputs[1].default_value = 0.0
                links.new(trans_socket, gt_node.inputs[0])

                mul_node = nodes.new('ShaderNodeMath')
                mul_node.operation = 'MULTIPLY'
                links.new(source_socket, mul_node.inputs[0])
                links.new(gt_node.outputs[0], mul_node.inputs[1])

                links.new(mul_node.outputs[0], aov_node.inputs['Value'])
            else:
                links.new(source_socket, aov_node.inputs['Value'])


def setup_compositor(scene, output_dir):
    """Set up Blender 5.0 compositor for multi-pass EXR output.

    Creates a single File Output node that writes a multipart EXR per frame
    containing all render passes (Image, Depth, Normal, Albedo, Alpha,
    Roughness, Metallic, Transmission, IOR).
    """
    # Create compositor node group (Blender 5.0 API)
    ng = bpy.data.node_groups.new("NeuSkyCompositor", "CompositorNodeTree")
    scene.compositing_node_group = ng

    # Render Layers node
    rl = ng.nodes.new("CompositorNodeRLayers")

    # File Output node (writes multipart EXR in Blender 5.0)
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory = os.path.join(output_dir, "train", "multipass") + "/"
    fo.file_name = "Image0001"

    # Add named items for each render pass
    fo.file_output_items.new(socket_type="RGBA", name="RGB")
    fo.file_output_items.new(socket_type="FLOAT", name="Depth")
    fo.file_output_items.new(socket_type="VECTOR", name="Normal")
    fo.file_output_items.new(socket_type="RGBA", name="Albedo")
    fo.file_output_items.new(socket_type="FLOAT", name="Alpha")
    fo.file_output_items.new(socket_type="FLOAT", name="Roughness")
    fo.file_output_items.new(socket_type="FLOAT", name="Metallic")
    fo.file_output_items.new(socket_type="FLOAT", name="Transmission")
    fo.file_output_items.new(socket_type="FLOAT", name="IOR")

    # Composite render over HDRI background for RGB output.
    # film_transparent=True gives us a clean alpha mask but makes the sky
    # invisible in the Image pass. Alpha Over composites the foreground
    # render over the Environment pass (HDRI as seen by the camera).
    alpha_over = ng.nodes.new("CompositorNodeAlphaOver")
    ng.links.new(rl.outputs["Environment"], alpha_over.inputs["Background"])
    ng.links.new(rl.outputs["Image"], alpha_over.inputs["Foreground"])
    ng.links.new(alpha_over.outputs[0], fo.inputs["RGB"])
    ng.links.new(rl.outputs["Depth"], fo.inputs["Depth"])
    ng.links.new(rl.outputs["Normal"], fo.inputs["Normal"])
    ng.links.new(rl.outputs["Diffuse Color"], fo.inputs["Albedo"])
    ng.links.new(rl.outputs["Alpha"], fo.inputs["Alpha"])
    ng.links.new(rl.outputs["Roughness"], fo.inputs["Roughness"])
    ng.links.new(rl.outputs["Metallic"], fo.inputs["Metallic"])
    ng.links.new(rl.outputs["Transmission"], fo.inputs["Transmission"])
    ng.links.new(rl.outputs["IOR"], fo.inputs["IOR"])

    return fo


def setup_camera(scene, focal_mm):
    """Create or configure the render camera."""
    cam_obj = scene.camera
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj

    # Clear any existing animation so keyframes don't override our positions
    cam_obj.animation_data_clear()

    cam_obj.data.lens = focal_mm
    cam_obj.data.sensor_fit = "HORIZONTAL"
    cam_obj.data.clip_start = 0.1
    cam_obj.data.clip_end = 1000.0

    return cam_obj


def point_camera_at(cam_obj, position, look_at):
    """Position the camera and point it at a target."""
    cam_obj.location = position
    direction = look_at - position
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()


# ---------------------------------------------------------------------------
# Camera intrinsics (matching BlenderNeRF format)
# ---------------------------------------------------------------------------

def get_camera_intrinsics(scene, camera, aabb_scale):
    """Extract camera intrinsics in the same format as BlenderNeRF."""
    cam_data = camera.data
    scale = scene.render.resolution_percentage / 100.0
    w = int(scene.render.resolution_x * scale)
    h = int(scene.render.resolution_y * scale)

    f_mm = cam_data.lens
    sensor_w = cam_data.sensor_width

    if cam_data.sensor_fit == "HORIZONTAL" or \
       (cam_data.sensor_fit == "AUTO" and w >= h):
        fl_x = f_mm / sensor_w * w
        fl_y = fl_x
    else:
        sensor_h = cam_data.sensor_height
        fl_y = f_mm / sensor_h * h
        fl_x = fl_y

    return {
        "camera_angle_x": cam_data.angle_x,
        "camera_angle_y": cam_data.angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": w / 2.0,
        "cy": h / 2.0,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
    }


def matrix_to_list(matrix):
    """Convert a Blender Matrix to a nested list."""
    return [list(row) for row in matrix]


# ---------------------------------------------------------------------------
# HDRI loading
# ---------------------------------------------------------------------------

def load_hdri_files(hdri_dir):
    """Load and sort HDRI file paths from a directory."""
    hdri_dir = Path(hdri_dir)
    hdri_files = sorted([
        str(f) for f in hdri_dir.iterdir()
        if f.suffix.lower() in (".exr", ".hdr")
    ])
    if not hdri_files:
        raise FileNotFoundError(f"No HDRI files found in {hdri_dir}")
    print(f"Found {len(hdri_files)} HDRI files")
    return hdri_files


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

def render_dataset(args):
    """Main function: set up scene and render the full dataset."""
    # Resolve --hdri_16k: swap to hdris_16k/ sibling of the given hdri_dir
    if args.hdri_16k:
        hdri_16k_dir = os.path.join(os.path.dirname(args.hdri_dir.rstrip("/")), "hdris_16k")
        if not os.path.isdir(hdri_16k_dir):
            raise FileNotFoundError(
                f"--hdri_16k specified but directory not found: {hdri_16k_dir}\n"
                f"Copy 16K HDRIs to that directory first.")
        args.hdri_dir = hdri_16k_dir

    scene = bpy.context.scene
    output_dir = args.output
    use_png = (args.format == "png")

    print(f"\n{'='*60}")
    print(f"NeuSky Synthetic Dataset Renderer")
    print(f"{'='*60}")
    print(f"Output:     {output_dir}")
    print(f"HDRI dir:   {args.hdri_dir}")
    if args.hdri_16k:
        print(f"HDRI res:   16K")
    print(f"Frames:     {args.num_frames}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    if args.focal_mm_range:
        print(f"Focal:      {args.focal_mm_range[0]:.0f}-{args.focal_mm_range[1]:.0f}mm (per-frame)")
    else:
        print(f"Focal:      {args.focal_mm}mm")
    if args.exposure_range:
        print(f"Exposure:   {args.exposure_range[0]:+.1f} to {args.exposure_range[1]:+.1f} EV (per-frame)")
    if args.sphere_radius_min is not None:
        print(f"Sphere:     r={args.sphere_radius_min}-{args.sphere_radius}, center={args.sphere_center}")
    else:
        print(f"Sphere:     r={args.sphere_radius}, center={args.sphere_center}")
    if args.camera_height_min is not None:
        print(f"Cam height: {args.camera_height_min}-{args.camera_height_max}m")
    else:
        print(f"Elevation:  {args.sphere_elevation_min}-{args.sphere_elevation_max} deg")
    if args.azimuth_min != 0.0 or args.azimuth_max != 360.0:
        print(f"Azimuth:    {args.azimuth_min}-{args.azimuth_max} deg")
    if args.look_at is not None:
        print(f"Look at:    {args.look_at}")
    if args.look_at_z_range is not None:
        print(f"Look-at Z:  {args.look_at_z_range[0]}-{args.look_at_z_range[1]}")
    if args.look_at_bias > 0:
        print(f"Look bias:  {args.look_at_bias:.0%} toward camera")
    if args.look_at_building:
        print(f"Look-at:    building AABB (geometry-aware)")
    if args.building_clearance is not None:
        print(f"Bldg clear: {args.building_clearance}m from walls")
    if args.min_clearance is not None:
        print(f"Clearance:  {args.min_clearance}m (ray-cast)")
    if args.min_building_fraction is not None:
        print(f"Bldg frac:  {args.min_building_fraction:.0%} minimum (ray-cast, in sampling)")
    if args.auto_exclude:
        print(f"Auto-excl:  bbox check (building meshes only, skipping background)")
    if args.exclude_box:
        for box in args.exclude_box:
            print(f"Exclude:    x=[{box[0]},{box[2]}] y=[{box[1]},{box[3]}]")
    print(f"Samples:    {args.samples}")
    print(f"Seed:       {args.seed}")
    print(f"Format:     {args.format}")
    if args.threads > 0:
        print(f"Threads:    {args.threads}")
    print(f"{'='*60}\n")

    # Load HDRIs
    hdri_files = load_hdri_files(args.hdri_dir)

    # Create output directories
    if use_png:
        rgb_dir = os.path.join(output_dir, "train", "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
    else:
        multipass_dir = os.path.join(output_dir, "train", "multipass")
        os.makedirs(multipass_dir, exist_ok=True)

    # Set up render settings
    setup_render_settings(scene, args)

    # Normalize glass materials before AOV setup (EXR mode only)
    if not use_png:
        normalize_glass_materials()
        setup_aov_passes(scene)

    # Set up world shader (with rotation mapping node)
    env_node, mapping_node = setup_world_shader(scene, hdri_files[0])

    # Set up compositor only for EXR mode
    fo_node = None
    if not use_png:
        fo_node = setup_compositor(scene, output_dir)

    # Set up camera
    cam_obj = setup_camera(scene, args.focal_mm)

    # Pre-sample per-frame intrinsics (before validate_fn so coverage check can use focal_mm)
    intrinsics_rng = random.Random(args.seed + 1)
    sensor_width = cam_obj.data.sensor_width
    res_x, res_y = args.resolution
    cx, cy = res_x / 2.0, res_y / 2.0

    focal_mms = []
    exposure_evs = []
    for _ in range(args.num_frames):
        focal_mms.append(
            intrinsics_rng.uniform(*args.focal_mm_range) if args.focal_mm_range
            else args.focal_mm)
        exposure_evs.append(
            intrinsics_rng.uniform(*args.exposure_range) if args.exposure_range
            else 0.0)

    # Create camera position validator
    validate_fn = None

    # Auto-exclude: fast bbox check against all scene meshes
    if args.auto_exclude:
        mesh_bboxes = collect_mesh_bboxes()
        print(f"Auto-exclude: checking against {len(mesh_bboxes)} mesh bounding boxes")

        def validate_fn(pos, target, focal_mm):
            return not point_inside_any_bbox(pos, mesh_bboxes)

    # Get depsgraph for ray-casting (needed by --min_clearance and --min_building_fraction)
    depsgraph = None
    if args.min_clearance is not None or args.min_building_fraction is not None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    # Ray-cast validator (slower, may hang on complex scenes)
    if args.min_clearance is not None:
        min_cl = args.min_clearance

        # Pre-compute 8 horizontal ray directions + up
        horiz_dirs = [
            mathutils.Vector((math.cos(math.radians(a)), math.sin(math.radians(a)), 0))
            for a in range(0, 360, 45)
        ]
        up = mathutils.Vector((0, 0, 1))

        def validate_fn(pos, target, focal_mm):
            # Check upward clearance (detects roofed buildings, cliffs)
            result, location, *_ = scene.ray_cast(depsgraph, pos, up)
            if result and (location - pos).length < min_cl:
                return False

            # Check horizontal clearance in 8 directions
            # If >=4 of 8 are blocked, camera is enclosed by walls
            blocked = 0
            for d in horiz_dirs:
                result, location, *_ = scene.ray_cast(depsgraph, pos, d)
                if result and (location - pos).length < min_cl:
                    blocked += 1
            return blocked < 4

    # Coverage validate_fn: ray-cast building coverage check during sampling
    if args.min_building_fraction is not None:
        prev_validate_fn = validate_fn
        min_frac = args.min_building_fraction

        def validate_fn(pos, target, focal_mm):
            if prev_validate_fn is not None and not prev_validate_fn(pos, target, focal_mm):
                return False
            direction = target - pos
            rot_quat = direction.to_track_quat("-Z", "Y")
            cam_matrix = mathutils.Matrix.Translation(pos) @ rot_quat.to_matrix().to_4x4()
            fl_x = focal_mm / sensor_width * res_x
            fl_y = fl_x
            coverage = check_building_coverage(
                scene, cam_matrix, depsgraph, fl_x, fl_y, cx, cy, res_x, res_y)
            return coverage >= min_frac

    # Compute building envelope for geometry-aware look-at targeting
    building_bbox = None
    if args.look_at_building:
        building_bboxes = collect_mesh_bboxes()
        if building_bboxes:
            building_bbox = merge_bboxes(building_bboxes)
            mn, mx = building_bbox
            print(f"Building envelope: x=[{mn[0]:.1f},{mx[0]:.1f}] "
                  f"y=[{mn[1]:.1f},{mx[1]:.1f}] z=[{mn[2]:.1f},{mx[2]:.1f}]")
        else:
            print("WARNING: --look_at_building set but no building meshes found, "
                  "falling back to center")

    # Sample camera positions
    positions = sample_camera_positions(
        args.num_frames,
        args.sphere_radius,
        args.sphere_center,
        args.sphere_elevation_min,
        args.sphere_elevation_max,
        args.upper_only,
        args.seed,
        height_min=args.camera_height_min,
        height_max=args.camera_height_max,
        azimuth_min_deg=args.azimuth_min,
        azimuth_max_deg=args.azimuth_max,
        radius_min=args.sphere_radius_min,
        exclude_boxes=args.exclude_box,
        look_at=args.look_at,
        look_at_z_range=args.look_at_z_range,
        look_at_bias=args.look_at_bias,
        building_bbox=building_bbox,
        building_clearance=args.building_clearance,
        validate_fn=validate_fn,
        focal_mms=focal_mms,
    )

    # Get camera intrinsics (global — uses initial focal_mm)
    intrinsics = get_camera_intrinsics(scene, cam_obj, args.aabb)

    # Build transforms data
    transforms = dict(intrinsics)
    transforms["frames"] = []

    ext = "png" if use_png else "exr"
    rng = random.Random(args.seed)
    cam_data = cam_obj.data

    print(f"Rendering {args.num_frames} frames...")

    for i, (pos, look_at) in enumerate(positions):
        frame_num = i + 1
        stem = f"Image{frame_num:04d}"

        # Select HDRI (cycle through available HDRIs, optionally offset)
        hdri_idx = (i + args.hdri_offset) % len(hdri_files)
        hdri_path = hdri_files[hdri_idx]
        hdri_name = Path(hdri_path).stem

        # Update HDRI (remove previous image to avoid memory leak —
        # each 4K EXR is ~134 MB uncompressed, 200 frames = 26 GB without cleanup)
        old_image = env_node.image
        env_node.image = bpy.data.images.load(hdri_path)
        if old_image is not None:
            bpy.data.images.remove(old_image)

        # Apply random Z-axis rotation to the environment map
        envmap_rotation_z = rng.uniform(0, 2 * math.pi)
        mapping_node.inputs["Rotation"].default_value = (0, 0, envmap_rotation_z)

        # Use pre-sampled per-frame intrinsics
        focal_mm = focal_mms[i]
        exposure_ev = exposure_evs[i]
        cam_data.lens = focal_mm
        scene.cycles.film_exposure = 2.0 ** exposure_ev if exposure_ev != 0.0 else 1.0

        # Compute pixel-space intrinsics for this frame
        fl_x = focal_mm / sensor_width * res_x
        fl_y = fl_x

        # Position camera
        point_camera_at(cam_obj, pos, look_at)

        # Update scene (needed for matrix_world to be correct)
        bpy.context.view_layer.update()

        # Set frame number
        scene.frame_set(frame_num)

        if use_png:
            # PNG mode: render directly to file
            scene.render.filepath = os.path.join(rgb_dir, stem)
            bpy.ops.render.render(write_still=True)
        else:
            # EXR mode: compositor writes multipart EXR
            fo_node.file_name = stem
            bpy.ops.render.render(write_still=False)

        # Render log
        print(f"  [{i+1}/{args.num_frames}] frame={frame_num}, "
              f"hdri={hdri_name}, rot_z={math.degrees(envmap_rotation_z):.0f}deg, "
              f"focal={focal_mm:.1f}mm, exposure={exposure_ev:+.2f}EV, "
              f"pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}), "
              f"look=({look_at.x:.1f}, {look_at.y:.1f}, {look_at.z:.1f})")

        # Record transform with envmap metadata and per-frame intrinsics
        transforms["frames"].append({
            "file_path": f"train/rgb/{stem}.{ext}" if use_png else f"train/{stem}.exr",
            "transform_matrix": matrix_to_list(cam_obj.matrix_world),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "focal_mm": focal_mm,
            "exposure_ev": exposure_ev,
            "envmap_name": hdri_name,
            "envmap_url": f"https://polyhaven.com/a/{hdri_name}",
            "envmap_rotation": [0.0, 0.0, envmap_rotation_z],
        })

    # Write transforms.json
    transforms_path = os.path.join(output_dir, "transforms_train.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    # Summary
    rendered = len(transforms['frames'])
    print(f"\nDone! Rendered {rendered} frames to {output_dir}")
    print(f"  transforms_train.json: {rendered} frames")
    if use_png:
        n_rgb = len(os.listdir(rgb_dir))
        print(f"  RGB PNGs: {n_rgb}")
    else:
        print(f"  multipass EXRs: {len(os.listdir(multipass_dir))}")
        print(f"\nNext steps:")
        print(f"  1. Run split_multipass_exr.py to extract individual passes")
        print(f"  2. Run prepare_synthetic_data.py to convert EXR->PNG for NeuSky")


if __name__ == "__main__":
    args = parse_args()
    render_dataset(args)
