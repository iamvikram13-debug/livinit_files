"""
common/transformations.py
--------------------------
Coordinate and rotation utilities shared by layout and USDZ stages.

Livinit coordinate systems:
- Layout system:  X (left–right), Y (forward–back), Z (up)
- USD system:     X (left–right), Y (up), Z (forward–back)

This module standardizes conversions, rotations, and
transform matrix composition.
"""

import math
import numpy as np
from typing import List, Tuple
from pxr import Gf


# -------------------------------------------------------
# Coordinate transforms
# -------------------------------------------------------

def layout_to_usd_coords(position: List[float], rotation: List[float]) -> Tuple[List[float], List[float]]:
    """
    Convert layout-space (X, Y, Z) → USD-space (X, Y, Z).
    Layout:  X (LR), Y (FB), Z (Up)
    USD:     X (LR), Y (Up), Z (FB)
    """
    x, y = position[0], position[1]
    z = position[2] if len(position) > 2 else 0.0

    usd_x = x
    usd_y = z
    usd_z = y

    # Rotation: layout yaw around Z → USD yaw around Y (swap axes)
    rx, ry, rz = rotation if len(rotation) == 3 else (0, 0, 0)
    usd_rot = [float(rx), float(rz), float(ry)]  # reorder to keep correct world orientation

    return [usd_x, usd_y, usd_z], usd_rot


def usd_to_layout_coords(position: List[float]) -> List[float]:
    """
    Convert USD-space → layout-space.
    Useful for debug or reverse-transforming generated layouts.
    """
    x, y, z = position
    layout_x = x
    layout_y = z
    layout_z = y
    return [layout_x, layout_y, layout_z]


# -------------------------------------------------------
# Rotation utilities
# -------------------------------------------------------

def degrees_to_radians(deg: float) -> float:
    return math.radians(deg)


def radians_to_degrees(rad: float) -> float:
    return math.degrees(rad)


def rotation_matrix_yaw(degrees: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for yaw (Z-axis in layout, Y-axis in USD)."""
    theta = math.radians(degrees)
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])


def apply_rotation(position: List[float], rotation_deg: float, pivot: List[float] = [0, 0, 0]) -> List[float]:
    """
    Rotate a 3D point around Y-axis (USD convention) by rotation_deg degrees.
    """
    rot_mat = rotation_matrix_yaw(rotation_deg)
    vec = np.array(position) - np.array(pivot)
    rotated = rot_mat @ vec
    return (rotated + np.array(pivot)).tolist()


# -------------------------------------------------------
# Snapping + normalization
# -------------------------------------------------------

def snap_to_floor(y_val: float, floor_y: float, eps: float = 0.2) -> float:
    """
    Snap object height to detected floor if within epsilon range.
    Ensures furniture sits above floor.
    """
    if abs(y_val - floor_y) < eps:
        return floor_y
    return max(y_val, floor_y + 0.01)


def clamp_rotation(rot: List[float]) -> List[float]:
    """Normalize rotation angles to [0, 360) degrees."""
    return [r % 360 for r in rot]


def clamp_position(pos: List[float], bounds: Tuple[float, float, float, float]) -> List[float]:
    """
    Clamp X,Z position within (xmin, xmax, zmin, zmax).
    """
    xmin, xmax, zmin, zmax = bounds
    x, y, z = pos
    x = max(min(x, xmax), xmin)
    z = max(min(z, zmax), zmin)
    return [x, y, z]


# -------------------------------------------------------
# USD helper wrappers
# -------------------------------------------------------

def vec3f_from_list(lst: List[float]) -> Gf.Vec3f:
    """Convert Python list to USD Vec3f safely."""
    if len(lst) < 3:
        lst = lst + [0.0] * (3 - len(lst))
    return Gf.Vec3f(float(lst[0]), float(lst[1]), float(lst[2]))


def quatf_from_euler(rot_deg: List[float]) -> Gf.Quatf:
    """
    Convert Euler angles (deg) → Quaternion (Gf.Quatf).
    """
    rx, ry, rz = [math.radians(r) for r in rot_deg]
    qx = Gf.Quatf(math.cos(rx/2), math.sin(rx/2), 0, 0)
    qy = Gf.Quatf(math.cos(ry/2), 0, math.sin(ry/2), 0)
    qz = Gf.Quatf(math.cos(rz/2), 0, 0, math.sin(rz/2))
    return qz * qy * qx
