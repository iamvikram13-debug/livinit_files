"""
blender_runner.py
-----------------
Handles conversion of GLB/GLTF → USD for the Livinit pipeline.

Features:
  • Uses Blender in headless mode if available.
  • Falls back to Pixar's usd_from_gltf if Blender is not installed.
  • Safe cleanup of temp files and graceful error handling.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional


# -------------------------------------------------------
# Blender discovery
# -------------------------------------------------------

def find_blender_executable() -> Optional[str]:
    """Locate Blender executable via BLENDER_HOME or common paths."""
    import glob

    blender_home = os.getenv("BLENDER_HOME")
    if blender_home:
        candidate = os.path.join(blender_home, "blender.exe" if os.name == "nt" else "blender")
        if os.path.exists(candidate):
            return candidate

    exe = shutil.which("blender")
    if exe:
        return exe

    common_roots = [
        r"C:\Program Files\Blender Foundation",
        r"C:\Program Files (x86)\Blender Foundation",
        "/Applications/Blender.app/Contents/MacOS",
        "/usr/local/bin",
    ]
    for root in common_roots:
        if not os.path.exists(root):
            continue
        matches = glob.glob(os.path.join(root, "**", "blender.exe" if os.name == "nt" else "blender"), recursive=True)
        if matches:
            return matches[0]

    return None


# -------------------------------------------------------
# Blender conversion script generator
# -------------------------------------------------------

def _make_blender_script(glb_path: str, out_usd: str) -> str:
    """Create a Python script for Blender to import GLB and export USD."""
    script = f"""
import bpy, sys, os
bpy.ops.wm.read_factory_settings(use_empty=True)
glb_path = r"{glb_path}"
out_usd = r"{out_usd}"

try:
    bpy.ops.import_scene.gltf(filepath=glb_path)
except Exception as e:
    print(f"Import failed: {{e}}")
    sys.exit(1)

os.makedirs(os.path.dirname(out_usd), exist_ok=True)
try:
    bpy.ops.wm.usd_export(
        filepath=out_usd,
        export_materials=True,
        export_textures=True,
        export_normals=True,
        export_animation=False,
        export_armatures=False,
        use_instancing=False,
        export_custom_properties=True
    )
except Exception as e:
    print(f"Export failed: {{e}}")
    sys.exit(1)

print("✓ GLB→USD conversion successful.")
sys.exit(0)
"""
    return script


# -------------------------------------------------------
# Main conversion function
# -------------------------------------------------------

def convert_glb_to_usd(glb_path: str, out_dir: str) -> str:
    """
    Convert a GLB file to USD format using Blender (preferred)
    or fallback usd_from_gltf if Blender unavailable.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    out_usd = os.path.join(out_dir, f"{base_name}.usdc")

    if os.path.exists(out_usd):
        print(f"ℹ Using cached USD: {out_usd}")
        return out_usd

    blender_exe = find_blender_executable()
    if blender_exe:
        print(f"🎨 Using Blender for GLB→USD conversion: {blender_exe}")
        tmp_dir = tempfile.mkdtemp(prefix="blender_convert_")
        script_path = os.path.join(tmp_dir, "convert.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_make_blender_script(glb_path, out_usd))

        try:
            proc = subprocess.run(
                [blender_exe, "--background", "--python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if proc.returncode == 0 and os.path.exists(out_usd):
                print("✅ Blender conversion successful.")
                return out_usd
            else:
                print("⚠ Blender conversion failed, falling back to usd_from_gltf.")
                print(proc.stdout or proc.stderr)
        except Exception as e:
            print(f"⚠ Blender conversion error: {e}, falling back to usd_from_gltf.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Fallback: usd_from_gltf
    candidates = ["usd_from_gltf", "usd_from_gltf.exe"]
    usd_exec = None
    for c in candidates:
        path = shutil.which(c)
        if path:
            usd_exec = path
            break

    if not usd_exec:
        raise RuntimeError(
            "Neither Blender nor usd_from_gltf found. "
            "Install Blender or Pixar USD tools, or set BLENDER_HOME."
        )

    cmd = [usd_exec, glb_path, "-o", out_usd, "--metersPerUnit", "1.0", "--upAxis", "y"]
    print(f"🔧 Running fallback: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"usd_from_gltf failed:\n{proc.stderr}")

    if not os.path.exists(out_usd):
        raise RuntimeError("GLB conversion did not produce a USD file.")

    print("✅ usd_from_gltf conversion successful.")
    return out_usd


# -------------------------------------------------------
# Simple CLI entrypoint (optional)
# -------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert GLB → USD (Livinit)")
    parser.add_argument("glb", help="Input .glb file")
    parser.add_argument("out", help="Output directory for .usdc")
    args = parser.parse_args()

    result = convert_glb_to_usd(args.glb, args.out)
    print(f"✅ Done: {result}")


"""
Blender automation for GLB to USD conversion
"""
import os
import subprocess
from pathlib import Path
import logging
from typing import Optional

from .usd_utils import convert_to_usd, merge_usd_files  # Removed invalid import of '_'

def convert_glb_to_usd(
    input_path: str, 
    output_path: str,
    blender_path: Optional[str] = None
) -> bool:
    """
    Convert GLB to USD using Blender headless mode
    
    Args:
        input_path: Path to input GLB file
        output_path: Path for output USD file
        blender_path: Optional path to Blender executable
    
    Returns:
        bool: True if conversion successful
    """
    blender_path = blender_path or os.getenv("BLENDER_PATH", "blender")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return False

    if not os.path.exists(blender_path):
        logging.error(f"Blender not found at: {blender_path}")
        return False

    cmd = [
        blender_path,
        "--background",
        "--python-expr",
        f"import bpy; bpy.ops.import_scene.gltf(filepath='{input_path}'); bpy.ops.wm.usd_export(filepath='{output_path}')"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"Successfully converted {Path(input_path).name} to USD")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Blender conversion failed: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during conversion: {str(e)}")
        return False
