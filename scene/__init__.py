"""
Scene generation module for the Livinit pipeline.

This package handles:
  • Asset and style selection from Supabase
  • Scene description (task + layout criteria) generation via LLM
  • Floor boundary extraction from input USDZ files (RoomPlan models)
  • Assembly of the final `scene.json` file for LayoutVLM

Usage:
    from scene.scene_generator import generate_scene, BuildSceneIn

    payload = BuildSceneIn(
        size="large",
        room_type="minimalist",
        color_palette="pastel",
        budget="0 - 5000 dollars",
        user_prompt="a living room with sofa and TV",
        usdz_path="/path/to/floor.usdz"
    )
    scene_dict = generate_scene(payload)
"""

from .scene_generator import (
    BuildSceneIn,
    generate_scene
)
from .utils import validate_scene  # Remove process_scene_data as it doesn't exist
from .prompts import get_scene_prompt, parse_llm_response, generate_scene_from_llm

__all__ = [
    "generate_scene",
    "BuildSceneIn",
    "validate_scene",
    "get_scene_prompt",
    "parse_llm_response",
    "generate_scene_from_llm"
]
