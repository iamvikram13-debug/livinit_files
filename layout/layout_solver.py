"""
layout/layout_solver.py
-----------------------
High-level solver that orchestrates the Livinit layout stage.

Responsibilities:
  • Take scene.json (task_description, boundary, assets)
  • Generate or import initial layout positions
  • Run gradient optimization (from grad_solver)
  • Validate and fill missing assets (from constraints)
  • Output layout.json in standardized format
"""

import os
import json
import random
import tempfile
import subprocess
from typing import Dict, Any, Optional

from .grad_solver import solve_layout
from .constraints import fill_random_unplaced_assets

# -------------------------------------------------------------
# Helper: run external LayoutVLM if available
# -------------------------------------------------------------
def _run_layoutvlm_subprocess(scene_json_path: str, save_dir: str, model_name: str = "gpt-4") -> Optional[Dict[str, Any]]:
    """
    If LayoutVLM repository is available, run its main script via subprocess.
    Returns parsed layout.json if successful, else None.
    """
    try:
        cmd = [
            "python",
            "main.py",
            "--scene_json_file", scene_json_path,
            "--save_dir", save_dir,
            "--model", model_name,
        ]
        print(f"🚀 Running LayoutVLM subprocess:\n{' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if process.returncode == 0:
            output_path = os.path.join(save_dir, "layout.json")
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    return json.load(f)
        print("⚠️ LayoutVLM subprocess failed or layout.json missing.")
        print("STDERR:", process.stderr[:400])
        return None
    except Exception as e:
        print(f"⚠️ LayoutVLM subprocess error: {e}")
        return None


# -------------------------------------------------------------
# Mock / heuristic layout generator
# -------------------------------------------------------------
def _generate_initial_layout(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a basic initial layout if LayoutVLM isn't available.
    Assets are placed roughly within room bounds with random rotation.
    """
    boundary = scene_dict["boundary"]["floor_vertices"]
    min_x = min(v[0] for v in boundary)
    max_x = max(v[0] for v in boundary)
    min_y = min(v[1] for v in boundary)
    max_y = max(v[1] for v in boundary)

    layout = {}
    for idx, (aid, ainfo) in enumerate(scene_dict["assets"].items()):
        x = random.uniform(min_x + 0.5, max_x - 0.5)
        y = random.uniform(min_y + 0.5, max_y - 0.5)
        z = float(ainfo.get("assetMetadata", {}).get("boundingBox", {}).get("z", 0.8)) / 2
        rot = [0.0, 0.0, random.uniform(0, 360)]
        layout[aid] = {"position": [x, y, z], "rotation": rot}
    print(f"🧠 Generated heuristic initial layout for {len(layout)} assets.")
    return layout


# -------------------------------------------------------------
# Main solver
# -------------------------------------------------------------
def run_layout_solver(scene_dict: Dict[str, Any], use_layoutvlm: bool = True, model_name: str = "gpt-4") -> Dict[str, Any]:
    """
    Main entrypoint for layout solving.
    - Attempts to call LayoutVLM if available.
    - Falls back to heuristic placement + gradient refinement otherwise.
    """
    tmp_dir = tempfile.mkdtemp(prefix="livinit_layout_")
    scene_path = os.path.join(tmp_dir, "scene.json")

    with open(scene_path, "w") as f:
        json.dump(scene_dict, f, indent=2)

    layout = None
    if use_layoutvlm:
        layout = _run_layoutvlm_subprocess(scene_path, tmp_dir, model_name)

    if layout is None:
        layout = _generate_initial_layout(scene_dict)

    # Optimize layout using gradient solver
    print("⚙️ Running gradient-based refinement...")
    layout = solve_layout(layout, scene_dict["boundary"])

    # Fill missing assets if any
    print("🧩 Checking for unplaced assets...")
    layout = fill_random_unplaced_assets(scene_dict, layout)

    print("✅ Layout solving complete.")
    return layout


# -------------------------------------------------------------
# Convenience function (for main pipeline)
# -------------------------------------------------------------
def save_layout_to_file(layout: Dict[str, Any], save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"💾 Layout saved to {save_path}")
    return save_path


# -------------------------------------------------------------
# CLI test
# -------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test
    sample_scene = {
        "task_description": "Arrange a minimalist living room.",
        "layout_criteria": ["Place sofa facing TV", "Keep rug centered"],
        "boundary": {"floor_vertices": [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]]},
        "assets": {
            "sofa-0": {"assetMetadata": {"boundingBox": {"x": 2, "y": 1, "z": 1}}},
            "tv_stand-0": {"assetMetadata": {"boundingBox": {"x": 1.2, "y": 0.5, "z": 0.8}}},
            "rug-0": {"assetMetadata": {"boundingBox": {"x": 3, "y": 2, "z": 0.05}}},
        },
    }

    print("🏗 Running layout solver test...")
    layout_dict = run_layout_solver(sample_scene, use_layoutvlm=False)
    save_layout_to_file(layout_dict, "./layout.json")
