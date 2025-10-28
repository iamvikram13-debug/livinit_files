"""
layout/constraints.py
---------------------
Defines geometric, spatial, and relational constraint utilities
for the Livinit Layout stage.

Used for:
  • Checking collisions between assets (IoU, bounding boxes)
  • Validating boundary containment
  • Simple geometric helpers for random fallback placement

This version is simplified for Livinit:
  - No rotated IoU (to keep it faster and cleaner)
  - Works in 2D (x, y) plane; assumes z=height
"""

import random
import math
import numpy as np

# -------------------------------------------------------------
# Core geometry utilities
# -------------------------------------------------------------
def iou_polygon(poly1, poly2) -> float:
    """
    Compute Intersection-over-Union (IoU) between two shapely Polygons.
    Returns 0.0 if no overlap.
    """
    # Local import to avoid requiring shapely at package-import time
    from shapely.geometry import Polygon

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return float(inter / union) if union > 0 else 0.0


def is_inside_boundary(point, boundary) -> bool:
    """
    Check if a given (x, y) point is inside the boundary polygon.
    """
    from shapely.geometry import Point
    return boundary.contains(Point(point[0], point[1]))


def get_bbox_polygon(position, bbox_size, rotation_deg=0.0):
    """
    Create a shapely Polygon for an asset's top-down bounding box.
    position: (x, y, z)
    bbox_size: (width_x, depth_y, height_z)
    rotation_deg: rotation around Z axis in degrees
    """
    x, y, _ = position
    w, d, _ = bbox_size
    half_w, half_d = w / 2, d / 2

    # Axis-aligned box before rotation
    corners = np.array([
        [-half_w, -half_d],
        [half_w, -half_d],
        [half_w, half_d],
        [-half_w, half_d]
    ])

    # Apply rotation
    theta = math.radians(rotation_deg)
    rot_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    rotated = corners @ rot_matrix.T

    # Translate to position
    rotated[:, 0] += x
    rotated[:, 1] += y

    from shapely.geometry import Polygon
    return Polygon(rotated)


# -------------------------------------------------------------
# Random placement utilities
# -------------------------------------------------------------
def random_point_in_boundary(boundary_vertices):
    """
    Sample a random (x, y) point inside the floor boundary polygon.
    """
    from shapely.geometry import Polygon, Point
    polygon = Polygon([(x, y) for x, y, _ in boundary_vertices])
    minx, miny, maxx, maxy = polygon.bounds

    for _ in range(1000):  # try up to 1000 times
        rand_x = random.uniform(minx, maxx)
        rand_y = random.uniform(miny, maxy)
        if polygon.contains(Point(rand_x, rand_y)):
            return [rand_x, rand_y]
    raise RuntimeError("Failed to sample random point inside boundary polygon.")


# -------------------------------------------------------------
# Bounding-box corner computation
# -------------------------------------------------------------
def get_bbox_corners(position, rotation_deg, bbox_size):
    """
    Compute all 8 corners of a 3D bounding box after rotation around Z.
    Returns an (8, 3) array of corners.
    """
    x, y, z = position
    w, d, h = bbox_size
    half_w, half_d, half_h = w / 2, d / 2, h / 2

    # Define box corners (local)
    local_corners = np.array([
        [-half_w, -half_d, -half_h],
        [ half_w, -half_d, -half_h],
        [ half_w,  half_d, -half_h],
        [-half_w,  half_d, -half_h],
        [-half_w, -half_d,  half_h],
        [ half_w, -half_d,  half_h],
        [ half_w,  half_d,  half_h],
        [-half_w,  half_d,  half_h],
    ])

    # Apply Z rotation
    theta = math.radians(rotation_deg)
    rot_matrix = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = local_corners @ rot_matrix.T

    # Translate to world position
    rotated[:, 0] += x
    rotated[:, 1] += y
    rotated[:, 2] += z

    return rotated


# -------------------------------------------------------------
# Fallback placement logic
# -------------------------------------------------------------
def fill_random_unplaced_assets(task_dict, layout_result, max_retries=10):
    """
    Fill in missing assets (those without positions) by random placement.
    Ensures assets are inside the boundary.
    """
    unplaced = [aid for aid in task_dict["assets"].keys() if aid not in layout_result]
    if not unplaced:
        return layout_result

    print(f"🌀 Randomly placing {len(unplaced)} unplaced assets...")

    from shapely.geometry import Polygon
    boundary_poly = Polygon([(x, y) for x, y, _ in task_dict["boundary"]["floor_vertices"]])

    for aid in unplaced:
        bbox = task_dict["assets"][aid].get("assetMetadata", {}).get("boundingBox", {})
        size_x, size_y, size_z = bbox.get("x", 1.0), bbox.get("y", 1.0), bbox.get("z", 1.0)

        for _ in range(max_retries):
            rand_x, rand_y = random_point_in_boundary(task_dict["boundary"]["floor_vertices"])
            rand_z = size_z / 2.0
            rotation = [0.0, 0.0, random.uniform(0, 360)]

            bbox_poly = get_bbox_polygon([rand_x, rand_y, rand_z], (size_x, size_y, size_z), rotation[2])
            if boundary_poly.contains(bbox_poly):
                layout_result[aid] = {
                    "position": [rand_x, rand_y, rand_z],
                    "rotation": rotation,
                }
                break

    return layout_result
