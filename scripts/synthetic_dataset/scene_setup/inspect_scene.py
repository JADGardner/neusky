"""Inspect a Blender scene: list objects, bounding boxes, and check camera positions."""
import bpy
import sys
import json
import mathutils
from pathlib import Path

# Parse args after "--"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

# List all mesh objects with world-space bounding boxes
print("\n" + "=" * 80)
print("MESH OBJECTS (sorted by volume, largest first)")
print("=" * 80)

meshes = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    # World-space bounding box
    bbox_world = [obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box]
    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    zs = [v.z for v in bbox_world]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    meshes.append({
        'name': obj.name,
        'loc': obj.matrix_world.translation,
        'bbox_min': (xmin, ymin, zmin),
        'bbox_max': (xmax, ymax, zmax),
        'vol': vol,
    })

meshes.sort(key=lambda m: m['vol'], reverse=True)

for m in meshes[:40]:  # Top 40 by volume
    bmin = m['bbox_min']
    bmax = m['bbox_max']
    loc = m['loc']
    print(f"  {m['name']:<50s}  loc=({loc.x:7.1f}, {loc.y:7.1f}, {loc.z:7.1f})  "
          f"bbox=[({bmin[0]:7.1f}, {bmin[1]:7.1f}, {bmin[2]:7.1f}) → "
          f"({bmax[0]:7.1f}, {bmax[1]:7.1f}, {bmax[2]:7.1f})]  vol={m['vol']:.0f}")

# Check specific camera positions against the church/cathedral objects
print("\n" + "=" * 80)
print("OBJECTS CONTAINING 'church' OR 'cathedral' OR 'fort' OR 'cliff'")
print("=" * 80)

for m in meshes:
    name_lower = m['name'].lower()
    if any(kw in name_lower for kw in ['church', 'cathedral', 'fort', 'cliff', 'rock', 'castle', 'tower']):
        bmin = m['bbox_min']
        bmax = m['bbox_max']
        print(f"  {m['name']:<50s}  "
              f"bbox=[({bmin[0]:7.1f}, {bmin[1]:7.1f}, {bmin[2]:7.1f}) → "
              f"({bmax[0]:7.1f}, {bmax[1]:7.1f}, {bmax[2]:7.1f})]")

# Check camera positions from the render
transforms_path = None
for p in argv:
    if p.endswith('.json'):
        transforms_path = p

if transforms_path and Path(transforms_path).exists():
    print("\n" + "=" * 80)
    print("CAMERA POSITION vs MESH BBOX CHECK")
    print("=" * 80)
    with open(transforms_path) as f:
        transforms = json.load(f)

    for frame in transforms['frames']:
        mat = frame['transform_matrix']
        cx, cy, cz = mat[0][3], mat[1][3], mat[2][3]
        inside = []
        for m in meshes:
            bmin = m['bbox_min']
            bmax = m['bbox_max']
            if (bmin[0] <= cx <= bmax[0] and
                bmin[1] <= cy <= bmax[1] and
                bmin[2] <= cz <= bmax[2]):
                inside.append(m['name'])
        status = f"INSIDE: {', '.join(inside[:5])}" if inside else "OK"
        print(f"  {frame['file_path']:<40s}  pos=({cx:7.1f}, {cy:7.1f}, {cz:7.1f})  {status}")
