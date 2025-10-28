"""
prompts.py
-----------
Prompt templates for scene generation in the Livinit pipeline.

This module contains reusable LLM system and user prompts used by
the Scene Generation stage (`scene_generator.py`).

Each prompt is designed to produce *strict JSON output* that can be parsed
by the pipeline without ambiguity.
"""

# ============================================================
# CATEGORY SELECTION PROMPT
# ============================================================

CATEGORY_PLANNER_SYSTEM = """
You are an expert interior designer helping to plan the furniture categories
for a living-room scene in 3D.

Your job is to choose which furniture categories should appear in the layout,
and how many of each, based on the user's intent, room size, and style.

Return **only valid JSON** with this schema:
{
  "selected_categories": [
    {"name": "<allowed_category>", "count": <int>},
    ...
  ]
}

Guidelines:
- Use ONLY categories from the given "Allowed categories" list.
- Do NOT invent, rename, or merge categories.
- Choose categories and counts that make sense for the given size and style.
- Prioritize realism — a living room typically includes sofa, rug, coffee_table,
  tv_stand, tv, accent_chair, floor_lamps, and sometimes console, chairs, or storage_unit.
- The total number of items should roughly match the size target range.
- Keep the counts realistic (e.g., 1 sofa, 1 rug, 1 tv_stand, etc.).
"""

# Example usage in scene_generator.py:
# llm(CATEGORY_PLANNER_SYSTEM, user_prompt)


# ============================================================
# TASK + LAYOUT CRITERIA PROMPT
# ============================================================

TASK_CRITERIA_SYSTEM = """
You are an AI interior design assistant who writes precise scene instructions
for a 3D layout generator.

Return **only JSON** with these keys:
{
  "task_description": "<4–6 sentences describing the scene>",
  "layout_criteria": ["<8 concise layout constraints>"]
}

Guidelines:
- Mention ONLY asset categories that are actually in the scene.
- Use realistic, testable distances (in METERS).
- Avoid vague words like "beautiful", "cozy", or "nice".
- Focus on measurable spatial relations, e.g.:
  - "Place the sofa at least 0.5 meters from the wall"
  - "Center the coffee_table in front of the sofa"
  - "Maintain 1 meter of walking space between major objects"
- Keep tone descriptive and professional.
- Distances and dimensions must always be expressed in meters (m).
"""

# ============================================================
# FALLBACK PROMPTS (Optional)
# ============================================================

FALLBACK_TASK_TEMPLATE = (
    "Design a {room_type} living room with {asset_list} "
    "in a {color_palette} color palette. "
    "Focus on functional arrangement and clear spacing."
)

FALLBACK_CRITERIA = [
    "Maintain 0.6–1.0 meters of clear walking paths.",
    "Keep at least 0.1 meters between furniture and walls.",
    "Center the coffee_table in front of the sofa if both are present.",
    "Orient the sofa toward the tv_stand or main focal wall.",
    "Ensure rugs extend at least 0.3 meters beyond sofa edges.",
    "Place accent_chairs within 2 meters of the sofa for conversation.",
    "Position floor_lamps near seating areas but not blocking paths.",
    "Keep tv_stand and storage_unit against walls for stability."
]

"""
Prompt generation and response parsing for scene generation
"""
from typing import Dict, Any, List
import json
import os
import dotenv
from openai import OpenAI
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_scene_prompt(
    room_type: str,
    size: str,
    color_palette: str,
    budget: str,
    user_prompt: str
) -> str:
    """
    Generate a prompt for the LLM to create a scene description
    
    Args:
        room_type: Type of room (bedroom, living room, etc.)
        size: Room size description
        color_palette: Preferred color scheme
        budget: Budget constraint
        user_prompt: Additional user requirements
        
    Returns:
        str: Formatted prompt for scene generation
    """
    return f"""Create a detailed room layout for a {size} {room_type}.
    Color palette: {color_palette}
    Budget: {budget}
    Additional requirements: {user_prompt}
    
    Please provide a structured description including:
    1. Room dimensions and shape
    2. Key furniture pieces with approximate sizes
    3. Color and material suggestions
    4. Placement recommendations
    """

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM response into structured scene data
    
    Args:
        response: Raw text response from LLM
        
    Returns:
        Dict containing parsed scene information
    """
    try:
        # Default structure for scene data
        scene_data = {
            "room": {
                "dimensions": {},
                "features": []
            },
            "furniture": [],
            "colors": [],
            "materials": []
        }
        
        # Here you would implement the actual parsing logic
        # For now, returning a basic structure
        return scene_data
        
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")

def generate_scene_from_llm(
    room_type: str,
    size: str,
    color_palette: str,
    budget: str,
    user_prompt: str
) -> Dict[str, Any]:
    """
    Generate complete scene using LLM
    
    Args:
        room_type: Type of room
        size: Room size description
        color_palette: Preferred color scheme
        budget: Budget constraint
        user_prompt: Additional user requirements
        
    Returns:
        Dict containing complete scene description
    """
    prompt = get_scene_prompt(room_type, size, color_palette, budget, user_prompt)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an interior design expert."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return parse_llm_response(response.choices[0].message.content)
