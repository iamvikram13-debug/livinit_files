"""
layout/placement_utils.py
-------------------------
Utility functions for asset placement, geometry handling, and layout parsing.

Used by:
  - layout_solver.py (for spatial sanity)
  - grad_solver.py (for geometric checks)
  - usdz_placer.py (for asset positioning)

Simplified for Livinit:
  • No rotated IoU
  • Works fully in 2D for layout reasoning
  • Includes safe fallbacks for random placement
"""

import math
import random
import re
import numpy as np
from shapely.geometry import Polygon, Point, LineString


# -------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------
def triangle_area(A, B, C):
    """Compute area of triangle (A, B, C)."""
    return abs((A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1])) / 2.0)


def random_point_in_triangle(A, B, C):
    """Uniformly sample a random point inside a triangle."""
    r1, r2 = random.random(), random.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    x = (1 - r1 - r2) * A[0] + r1 * B[0] + r2 * C[0]
    y = (1 - r1 - r2) * A[1] + r1 * B[1] + r2 * C[1]
    return (x, y)


def random_point_in_polygon(vertices, add_z=False):
    """
    Sample random point inside a polygon defined by floor vertices.
    """
    polygon = Polygon([(x, y) for x, y, *_ in vertices])
    minx, miny, maxx, maxy = polygon.bounds

    for _ in range(500):
        rx, ry = random.uniform(minx, maxx), random.uniform(miny, maxy)
        if polygon.contains(Point(rx, ry)):
            if add_z:
                return [rx, ry, 0.0]
            return [rx, ry]
    raise RuntimeError("Failed to sample random point inside polygon.")


def get_bbox_corners(position, rotation, bbox_size):
    """
    Compute 3D bounding-box corners after rotation (around Z axis only).
    """
    x, y, z = position
    w, d, h = bbox_size
    theta = math.radians(-rotation[2]) if len(rotation) >= 3 else 0.0

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_w, half_d, half_h = w / 2, d / 2, h / 2

    corners = np.array([
        [-half_w, -half_d, -half_h],
        [ half_w, -half_d, -half_h],
        [ half_w,  half_d, -half_h],
        [-half_w,  half_d, -half_h],
        [-half_w, -half_d,  half_h],
        [ half_w, -half_d,  half_h],
        [ half_w,  half_d,  half_h],
        [-half_w,  half_d,  half_h],
    ])

    rot = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0, 0, 1]
    ])
    rotated = corners @ rot.T
    rotated += np.array([x, y, z])
    return rotated


# -------------------------------------------------------------
# Placement utilities
# -------------------------------------------------------------
def fill_random_unplaced_assets(scene_dict, layout_result, max_retries=10):
    """
    Place assets that failed optimization randomly within the floor boundary.
    """
    unplaced = [aid for aid in scene_dict["assets"] if aid not in layout_result]
    if not unplaced:
        return layout_result

    print(f"🎲 Filling {len(unplaced)} unplaced assets randomly...")
    boundary = Polygon([(x, y) for x, y, _ in scene_dict["boundary"]["floor_vertices"]])

    for aid in unplaced:
        bbox = scene_dict["assets"][aid].get("assetMetadata", {}).get("boundingBox", {})
        sx, sy, sz = bbox.get("x", 1.0), bbox.get("y", 1.0), bbox.get("z", 1.0)
        for _ in range(max_retries):
            rx, ry = random_point_in_polygon(scene_dict["boundary"]["floor_vertices"])
            rz = sz / 2
            rot = [0, 0, random.uniform(0, 360)]

            # ensure inside boundary
            box_poly = Polygon([
                (rx - sx/2, ry - sy/2),
                (rx + sx/2, ry - sy/2),
                (rx + sx/2, ry + sy/2),
                (rx - sx/2, ry + sy/2),
            ])
            if boundary.contains(box_poly):
                layout_result[aid] = {"position": [rx, ry, rz], "rotation": rot}
                break
    return layout_result


# -------------------------------------------------------------
# Extraction utilities (for code-based layouts)
# -------------------------------------------------------------
def extract_numbers(s: str):
    """Extract float numbers from a string (used in debugging)."""
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]


def extract_asset_info(code: str):
    """
    Extract positions and rotations from text-based layout code.
    Example pattern: sofa[0].position = [1,2,3]
    """
    pos_pat = re.compile(r"\w+\[\d\]\.position\s*=\s*\[([\d.,\s-]+)\]")
    rot_pat = re.compile(r"\w+\[\d\]\.rotation\s*=\s*\[([\d.,\s-]+)\]")

    positions, rotations = [], []
    for match in pos_pat.findall(code):
        pos = [float(x) for x in match.split(",")]
        positions.append(pos)
    for match in rot_pat.findall(code):
        rot = [float(x) for x in match.split(",")]
        if len(rot) == 1:
            rotations.append([0, 0, rot[0]])
        else:
            rotations.append(rot)
    return positions, rotations


def extract_initialization_from_string(data_str):
    """
    Parse initialization lines (e.g., sofa[0].position = [x,y,z])
    into structured dict of object positions and rotations.
    """
    pos_pat = re.compile(r"(\w+\[\d+\])\.position = (\[[-\d., ]+\])")
    rot_pat = re.compile(r"(\w+\[\d+\])\.rotation = (\[[-\d., ]+\])")

    positions = pos_pat.findall(data_str)
    rotations = rot_pat.findall(data_str)
    objs = {}

    for obj, pos_str in positions:
        objs[obj] = {"position": eval(pos_str), "rotation": None}
    for obj, rot_str in rotations:
        if obj in objs:
            objs[obj]["rotation"] = eval(rot_str)
    return objs


def replace_z_rot_degree_to_radians(code: str) -> str:
    """
    Replace `.rotation = [degrees]` with `[0, 0, radians]`.
    """
    pat = r'(\w+\[\d+\])\.rotation = \[(-?\d+\.?\d*)\]'

    def repl(m):
        obj = m.group(1)
        deg = float(m.group(2))
        rad = math.radians(deg)
        return f"{obj}.rotation = [0, 0, {rad}]"

    return re.sub(pat, repl, code)
