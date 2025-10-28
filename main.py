"""
main.py
--------
Livinit Unified Pipeline Entrypoint

Stages:
1️⃣ Scene generation (scene_generator)
2️⃣ Layout solving (layout.layout_solver)
3️⃣ USDZ composition (usdz.usdz_placer)
"""

import os
import json
import traceback
import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Pipeline modules ---
from scene.scene_generator import generate_scene, BuildSceneIn
from layout.layout_solver import run_layout_solver
from usdz.usdz_placer import compose_usdz_from_layout

# --- Common utilities ---
from common.io_utils import log, write_json
from common.plot_utils import visualize_layout

# -----------------------------------------------------
# Environment setup
# -----------------------------------------------------

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set in environment")

# -----------------------------------------------------
# FastAPI setup
# -----------------------------------------------------

app = FastAPI(title="Livinit 3D Interior API", version="1.0.0")

# -----------------------------------------------------
# Pydantic input model for /infer
# -----------------------------------------------------

class PipelineInput(BaseModel):
    size: str
    room_type: str
    color_palette: str
    budget: str
    user_prompt: str
    usdz_path: Optional[str] = None  # Local RoomPlan USDZ
    local_assets_dir: Optional[str] = None  # Optional cache dir for GLBs

# -----------------------------------------------------
# Core pipeline
# -----------------------------------------------------

def run_pipeline(payload: PipelineInput, output_dir: str) -> dict:
    """
    Executes full Livinit flow:
      scene.json → layout.json → decorated.usdz
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        log("🚀 Starting Livinit pipeline...", "INFO")

        # 1️⃣ Scene generation
        log("🔹 Stage 1: Scene generation", "INFO")
        scene_dict = generate_scene(BuildSceneIn(**payload.dict()))
        scene_json_path = os.path.join(output_dir, "scene.json")
        write_json(scene_dict, scene_json_path)
        log(f"✅ Scene JSON saved → {scene_json_path}")

        # 2️⃣ Layout solving
        log("🔹 Stage 2: Layout solver", "INFO")
        layout_dict = run_layout_solver(scene_dict)
        layout_json_path = os.path.join(output_dir, "layout.json")
        write_json(layout_dict, layout_json_path)
        log(f"✅ Layout JSON saved → {layout_json_path}")

        # Optional layout visualization
        vis_path = os.path.join(output_dir, "layout_preview.png")
        if "boundary" in scene_dict and "floor_vertices" in scene_dict["boundary"]:
            visualize_layout(scene_dict, layout_dict, vis_path)

        # 3️⃣ USDZ placement
        log("🔹 Stage 3: USDZ placement", "INFO")
        base_usdz = payload.usdz_path or "floor.usdz"
        final_usdz = os.path.join(output_dir, "final_decorated.usdz")

        compose_usdz_from_layout(
            usdz_input=base_usdz,
            scene_dict={"scene": scene_dict},
            layout_dict=layout_dict,
            final_usdz=final_usdz,
            prefer_local_assets_dir=payload.local_assets_dir,
        )

        log("🎉 Pipeline completed successfully!", "OK")

        return {
            "scene_json": scene_json_path,
            "layout_json": layout_json_path,
            "final_usdz": final_usdz,
            "preview_image": vis_path
        }

    except Exception as e:
        log(f"❌ Pipeline failed: {e}", "ERROR")
        traceback.print_exc()
        raise


# -----------------------------------------------------
# API endpoint
# -----------------------------------------------------

@app.post("/infer")
async def infer_api(
    size: str = Form(...),
    room_type: str = Form(...),
    color_palette: str = Form(...),
    budget: str = Form(...),
    user_prompt: str = Form(...),
    usdz_file: Optional[UploadFile] = File(None)
):
    """Run full pipeline via REST API."""
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        payload_dict = {
            "size": size,
            "room_type": room_type,
            "color_palette": color_palette,
            "budget": budget,
            "user_prompt": user_prompt
        }

        if usdz_file:
            usdz_path = os.path.join(temp_dir, usdz_file.filename)
            with open(usdz_path, "wb") as f:
                f.write(await usdz_file.read())
            payload_dict["usdz_path"] = usdz_path

        pipeline_input = PipelineInput(**payload_dict)
        output_dir = os.path.join("outputs", "api_run")
        
        results = run_pipeline(pipeline_input, output_dir)
        return FileResponse(results["final_usdz"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# CLI mode
# -----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Livinit Full Pipeline CLI")
    parser.add_argument("--input", required=True, help="Path to input JSON with scene parameters")
    parser.add_argument("--out_dir", default="outputs/run", help="Output directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload_data = json.load(f)

    payload = PipelineInput(**payload_data)
    results = run_pipeline(payload, args.out_dir)

    print("\n=== Livinit Pipeline Complete ===")
    print(json.dumps(results, indent=2))
