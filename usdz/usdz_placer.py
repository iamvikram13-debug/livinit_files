"""
usdz_placer.py
---------------
Assembles a final USDZ scene by:
  1. Unpacking a base room USDZ (from iOS RoomPlan)
  2. Converting and placing assets defined in layout.json
  3. Snapping furniture to the detected floor
  4. Repacking into an ARKit-compatible USDZ file
"""

import os
import json
import math
import tempfile
import shutil
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from pxr import Usd, UsdGeom, Gf

from .usd_utils import (
    extract_usdz,
    pack_usdz,
    detect_floor_y,
    snap_to_floor,
    ensure_xform,
    add_transform_ops,
    layout_to_usd_coords,
    set_usd_stage_defaults,
)
from .blender_runner import convert_glb_to_usd


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _safe_name(name: str) -> str:
    """Normalize asset names for USD prims."""
    n = name.strip().replace("-", "_").replace(".", "_").replace(" ", "_")
    if n and n[0].isdigit():
        n = "asset_" + n
    return n


def _ensure_file_from_url(url: str, dest_dir: str) -> str:
    """Download a file from URL if not cached locally."""
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(urlparse(url).path)
    out = os.path.join(dest_dir, base)
    if not os.path.exists(out):
        print(f"🌐 Downloading: {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
    return out


# -------------------------------------------------------
# Main Composer
# -------------------------------------------------------

def compose_usdz_from_layout(
    usdz_input: str,
    scene_dict: Dict[str, Any],
    layout_dict: Dict[str, Any],
    final_usdz: str,
    tmp_dir: Optional[str] = None,
    prefer_local_assets_dir: Optional[str] = None,
    cleanup: bool = True,
    arkit_package: bool = True,
) -> str:
    """
    Main entrypoint for Livinit USDZ assembly.
    """
    if not os.path.exists(usdz_input):
        raise FileNotFoundError(f"USDZ input not found: {usdz_input}")

    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="livinit_usdz_")
    unpack_dir = os.path.join(tmp_dir, "unpacked")
    os.makedirs(unpack_dir, exist_ok=True)

    print(f"📦 Extracting base USDZ → {unpack_dir}")
    base_usd = extract_usdz(usdz_input, unpack_dir)

    # Open the USD stage
    stage = Usd.Stage.Open(base_usd)
    set_usd_stage_defaults(stage)

    # Detect floor height
    floor_y = detect_floor_y(stage)
    if not math.isfinite(floor_y) or abs(floor_y) > 10.0:
        print(f"⚠️ Floor detection failed, using default 0.0")
        floor_y = 0.0
    print(f"🏠 Detected floor Y = {floor_y:.3f}m")

    # Ensure /Scene exists
    scene_root = ensure_xform(stage, "/Scene")

    pkg_assets_rel = "Assets"
    pkg_assets_abs = os.path.join(unpack_dir, pkg_assets_rel)
    os.makedirs(pkg_assets_abs, exist_ok=True)

    scene_assets = (scene_dict.get("scene") or scene_dict).get("assets", {})
    total = len(layout_dict)
    print(f"\n🎯 Placing {total} assets into scene...")

    for idx, (key, props) in enumerate(layout_dict.items(), start=1):
        print(f"\n[{idx}/{total}] → {key}")

        safe_key = _safe_name(key)
        prim_path = f"/Scene/{safe_key}"

        # 1️⃣ Resolve GLB source
        model_path = None
        if prefer_local_assets_dir:
            candidate = os.path.join(prefer_local_assets_dir, f"{key}.glb")
            if os.path.exists(candidate):
                model_path = candidate
                print(f"📁 Using local model: {candidate}")

        if not model_path:
            asset_info = scene_assets.get(key, {})
            model_url = asset_info.get("model_url")
            if not model_url:
                print(f"⚠️ Missing model_url for {key}, skipping.")
                continue
            model_path = _ensure_file_from_url(model_url, os.path.join(tmp_dir, "downloads"))

        # 2️⃣ Convert GLB → USD
        asset_dir = os.path.join(pkg_assets_abs, safe_key)
        os.makedirs(asset_dir, exist_ok=True)
        usd_payload_abs = convert_glb_to_usd(model_path, asset_dir)

        # 3️⃣ Compute relative path for referencing
        usd_payload_rel = os.path.relpath(usd_payload_abs, start=os.path.dirname(base_usd)).replace("\\", "/")

        # 4️⃣ Compute position + rotation
        pos = props.get("position", [0, 0, 0])
        rot = props.get("rotation", [0, 0, 0])
        scale = props.get("scale", [1, 1, 1])

        usd_pos, usd_rot = layout_to_usd_coords(pos, rot)
        usd_pos[1] = snap_to_floor(usd_pos[1], floor_y, eps=0.3)

        print(f"📍 Position = {usd_pos}, Rotation = {usd_rot}, Scale = {scale}")

        # 5️⃣ Create prim & add transform ops
        asset_xform = ensure_xform(stage, prim_path)
        xf = UsdGeom.Xformable(asset_xform)
        add_transform_ops(xf, usd_pos, usd_rot, scale)

        # 6️⃣ Add reference to converted USD
        prim = asset_xform.GetPrim()
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(usd_payload_rel)

        print(f"✅ Added {safe_key} → {usd_payload_rel}")

    # Save stage
    print("\n💾 Saving composed USD...")
    stage.GetRootLayer().Save()

    # 7️⃣ Repack USDZ
    print(f"📦 Repacking USDZ → {final_usdz}")
    pack_usdz(base_usd, final_usdz, arkit=arkit_package)

    print(f"\n🎉 Final USDZ ready: {final_usdz}")

    # Optional cleanup
    if cleanup:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("🧹 Temporary files cleaned up.")

    return final_usdz


# -------------------------------------------------------
# CLI wrapper
# -------------------------------------------------------

def compose_usdz_from_files(
    usdz_input: str,
    scene_json_path: str,
    layout_json_path: str,
    final_usdz: str,
    prefer_local_assets_dir: Optional[str] = None,
    cleanup: bool = True,
) -> str:
    """Convenience function that reads JSON from files."""
    with open(scene_json_path, "r", encoding="utf-8") as f:
        scene_dict = json.load(f)
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout_dict = json.load(f)

    return compose_usdz_from_layout(
        usdz_input=usdz_input,
        scene_dict=scene_dict,
        layout_dict=layout_dict,
        final_usdz=final_usdz,
        prefer_local_assets_dir=prefer_local_assets_dir,
        cleanup=cleanup,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compose a decorated USDZ scene.")
    parser.add_argument("--input", required=True, help="Base room USDZ file")
    parser.add_argument("--scene", required=True, help="Scene JSON path")
    parser.add_argument("--layout", required=True, help="Layout JSON path")
    parser.add_argument("--output", required=True, help="Final USDZ output path")
    parser.add_argument("--local", help="Optional local GLB folder")
    parser.add_argument("--no-clean", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    compose_usdz_from_files(
        usdz_input=args.input,
        scene_json_path=args.scene,
        layout_json_path=args.layout,
        final_usdz=args.output,
        prefer_local_assets_dir=args.local,
        cleanup=not args.no_clean,
    )
