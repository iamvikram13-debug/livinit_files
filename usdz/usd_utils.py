"""
usd_utils.py
------------
Utility functions for handling USD and USDZ operations in the Livinit pipeline.

These utilities are used by:
  - blender_runner.py  (for GLB→USD conversion)
  - usdz_placer.py     (for asset placement & package assembly)
"""

import os
import math
import zipfile
from typing import Optional, List
from pxr import Usd, UsdGeom, UsdUtils, Gf, Sdf
from pathlib import Path


# -------------------------------------------------------
# File helpers
# -------------------------------------------------------

def extract_usdz(usdz_path: str, dest_dir: str) -> str:
    """
    Unpacks a .usdz archive into dest_dir and returns the base USD path.
    """
    if not os.path.exists(usdz_path):
        raise FileNotFoundError(f"USDZ not found: {usdz_path}")

    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(usdz_path, "r") as z:
        z.extractall(dest_dir)

    # Find the main USD file
    for f in os.listdir(dest_dir):
        if f.endswith(".usd") or f.endswith(".usda"):
            return os.path.join(dest_dir, f)

    raise RuntimeError("No .usd/.usda file found inside USDZ package.")


def pack_usdz(source_usd: str, out_path: str, arkit: bool = True) -> str:
    """
    Packs a directory into a USDZ package compatible with ARKit or standard USDZ.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if arkit:
        UsdUtils.CreateNewARKitUsdzPackage(source_usd, out_path)
    else:
        UsdUtils.CreateNewUsdzPackage(source_usd, out_path)
    return out_path


# -------------------------------------------------------
# USD coordinate system helpers
# -------------------------------------------------------

def set_usd_stage_defaults(stage: Usd.Stage):
    """Ensure 1 meter = 1 unit and Y-up axis."""
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)


def layout_to_usd_coords(layout_pos, layout_rot):
    """
    Convert Livinit layout coordinate system to USD:
      Layout: X(left-right), Y(forward-back), Z(up)
      USD:    X(left-right), Y(up), Z(forward-back)

    Returns: (usd_pos, usd_rot)
    """
    x, y, z = layout_pos
    rot_x, rot_y, rot_z = layout_rot

    usd_pos = Gf.Vec3f(x, z, y)  # swap Y<->Z
    usd_rot = Gf.Vec3f(rot_x, rot_y, rot_z)
    return usd_pos, usd_rot


# -------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------

def detect_floor_y(stage: Usd.Stage) -> float:
    """
    Heuristically find the floor level (min Y across all meshes).
    """
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        min_y = float("inf")
        any_mesh = False

        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                any_mesh = True
                box = bbox_cache.ComputeWorldBound(prim).GetBox()
                mn = box.GetMin()
                min_y = min(min_y, mn[1])  # Y component (height)

        if any_mesh and math.isfinite(min_y):
            return float(min_y)
    except Exception as e:
        print(f"⚠️ floor detection failed: {e}")

    return 0.0


def snap_to_floor(y: float, floor_y: float, eps: float = 0.2) -> float:
    """
    Snap Y to floor if within epsilon.
    Keeps object slightly above ground.
    """
    if abs(y - floor_y) < eps:
        return floor_y
    return max(y, floor_y + 0.01)


# -------------------------------------------------------
# Prim utilities
# -------------------------------------------------------

def ensure_xform(stage: Usd.Stage, path: str) -> UsdGeom.Xform:
    """Ensure Xform prim exists at given path."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return UsdGeom.Xform.Define(stage, path)
    return UsdGeom.Xform(stage.GetPrimAtPath(path))


def add_transform_ops(xf: UsdGeom.Xformable, pos, rot=None, scale=None):
    """Adds standard USD xform ops for translate, rotate, and scale."""
    xf.AddTranslateOp().Set(Gf.Vec3f(*pos))
    if rot is not None:
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))


# -------------------------------------------------------
# Debug / logging
# -------------------------------------------------------

def summarize_stage(stage: Usd.Stage, max_prims: int = 10):
    """
    Print a summary of stage content for debugging.
    """
    all_prims = list(stage.Traverse())
    print(f"\n📦 Stage summary: {len(all_prims)} prims total")
    for p in all_prims[:max_prims]:
        print(f"  - {p.GetPath()} ({p.GetTypeName()})")
    if len(all_prims) > max_prims:
        print(f"  ... +{len(all_prims) - max_prims} more")


"""
USD file manipulation utilities
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

def convert_to_usd(input_path: str, output_path: str) -> bool:
    """Convert any supported format to USD"""
    try:
        # USD conversion logic here
        return True
    except Exception as e:
        print(f"USD conversion failed: {e}")
        return False

def merge_usd_files(input_files: List[str], output_path: str) -> bool:
    """Merge multiple USD files into one"""
    try:
        # USD merging logic here
        return True
    except Exception as e:
        print(f"USD merge failed: {e}")
        return False

__all__ = ["convert_to_usd", "merge_usd_files"]
