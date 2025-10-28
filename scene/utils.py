"""
utils.py
---------
Helper utilities for the Scene Generation module in the Livinit pipeline.

This module includes:
  - Category normalization and aliases
  - Supabase asset fetching and parsing
  - Style scoring for asset selection
  - JSON extraction utilities for LLM outputs
  - Floor vertex extraction from USDZ (RoomPlan models)
"""

import os, re, json, random
from typing import Any, Dict, List, Optional
import numpy as np
from supabase import create_client, Client
from pxr import Usd, UsdGeom
from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_ASSETS_TABLE", "assets")

if not (SUPABASE_URL and SUPABASE_KEY):
    raise RuntimeError("Missing env vars: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Category configuration
# ----------------------------
ALLOWED_CATEGORIES = [
    "sofa", "accent_chair", "chairs", "chaise_lounge",
    "coffee_table", "tv_stand", "console", "storage_unit",
    "rug", "floor_lamps", "tv", "dinning"
]

CATEGORY_ALIASES = {
    "sofa": {"sofa", "couch", "sectional"},
    "accent_chair": {"accent_chair", "armchair", "accent_chairs"},
    "chairs": {"chairs", "chair", "dining_chair"},
    "chaise_lounge": {"chaise_lounge", "chaise"},
    "coffee_table": {"coffee_table", "center_table"},
    "tv_stand": {"tv_stand", "media_console"},
    "console": {"console", "console_table"},
    "storage_unit": {"storage_unit", "cabinet", "sideboard"},
    "rug": {"rug", "carpet"},
    "floor_lamps": {"floor_lamps", "floor_lamp"},
    "tv": {"tv", "television"},
    "dinning": {"dinning", "dining", "dining_table"}
}

CATEGORY_PRIORITY = [
    "sofa", "rug", "coffee_table", "tv_stand", "tv",
    "accent_chair", "floor_lamps", "storage_unit"
]

# ----------------------------
# Basic utility functions
# ----------------------------
def _safe_name(name: str) -> str:
    """Normalize filenames or identifiers into safe ASCII tokens."""
    n = name.strip().replace("-", "_").replace(".", "_").replace(" ", "_")
    if n and n[0].isdigit():
        n = "asset_" + n
    return n

def normalize_category(cat: str) -> Optional[str]:
    """Map fuzzy or plural names into a canonical category name."""
    c = (cat or "").strip().lower()
    for canon, variants in CATEGORY_ALIASES.items():
        if c == canon or c in variants:
            return canon
    return c if c in ALLOWED_CATEGORIES else None

# ----------------------------
# Supabase helpers
# ----------------------------
def fetch_assets_from_supabase(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch all asset rows from the Supabase `assets` table."""
    q = supabase.table(SUPABASE_TABLE).select("*")
    if limit:
        q = q.limit(limit)
    res = q.execute()
    return res.data or []

# ----------------------------
# Style scoring
# ----------------------------
def style_score(row: Dict[str, Any], room_type: str, color_palette: str) -> int:
    """
    Compute a lightweight 'style fit' score between the asset and room type.
    """
    hay = " ".join([
        row.get("description", "") or "",
        " ".join(row.get("tags", []) or [])
    ]).lower()

    s = 0
    for tok in room_type.lower().split():
        if tok in hay:
            s += 1
            break
    for tok in color_palette.lower().split():
        if tok in hay:
            s += 1
            break
    return s

# ----------------------------
# LLM JSON cleanup
# ----------------------------
def extract_json_block(text: str) -> str:
    """Extract the first valid JSON block from an LLM output."""
    t = text.strip()
    first, last = t.find("{"), t.rfind("}")
    if first != -1 and last != -1:
        return t[first:last+1]
    return "{}"

# ----------------------------
# Floor extraction from USDZ
# ----------------------------
def extract_floor_vertices_from_usdz(usdz_path: str):
    """
    Extract floor mesh vertices from a USDZ RoomPlan model.
    Returns list of [x, y, z] vertices on the floor plane.
    """
    if not os.path.exists(usdz_path):
        raise FileNotFoundError(f"USDZ not found: {usdz_path}")

    stage = Usd.Stage.Open(usdz_path)
    meshes = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    if not meshes:
        raise ValueError("No Mesh prims found in USDZ file.")

    mesh_vertices = []
    for m in meshes:
        pts = UsdGeom.Mesh(m).GetPointsAttr().Get()
        if pts:
            mesh_vertices.append(np.array(pts))

    floor = max(mesh_vertices, key=lambda v: np.ptp(v[:, 0]) * np.ptp(v[:, 1]))
    top = floor[np.isclose(floor[:, 2], 0.0, atol=0.05)]
    rounded = np.round(top, 3)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    uniq = top[idx]
    uniq = uniq[np.argsort(uniq[:, 0])]
    if len(uniq) > 20:
        uniq = uniq[::len(uniq)//20][:20]
    return uniq.tolist()

"""
Utility functions for scene generation and validation
"""
def validate_scene(scene_data: Dict[str, Any]) -> bool:
    """
    Validate scene data structure and required fields
    
    Args:
        scene_data: Dictionary containing scene information
        
    Returns:
        bool: True if scene is valid, False otherwise
    """
    required_fields = ['room_type', 'size', 'furniture']
    
    if not all(field in scene_data for field in required_fields):
        return False
        
    if not isinstance(scene_data['furniture'], list):
        return False
        
    return True

def load_scene_file(filepath: Path) -> Dict[str, Any]:
    """
    Load scene data from JSON file
    
    Args:
        filepath: Path to scene JSON file
        
    Returns:
        Dict containing scene data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_scene_file(scene_data: Dict[str, Any], filepath: Path) -> None:
    """
    Save scene data to JSON file
    
    Args:
        scene_data: Dictionary containing scene information
        filepath: Output path for JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2)

def validate_furniture_item(item: Dict[str, Any]) -> bool:
    """
    Validate individual furniture item data
    
    Args:
        item: Dictionary containing furniture item data
        
    Returns:
        bool: True if item is valid, False otherwise
    """
    required_fields = ['id', 'type', 'dimensions']
    return all(field in item for field in required_fields)
