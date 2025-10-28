"""
scene_generator.py
------------------
Core scene generation pipeline for Livinit.

Steps:
1. Fetch candidate assets from Supabase.
2. Use LLM (GPT-5) to select furniture categories and counts.
3. Match best assets for those categories (by style & palette).
4. Generate scene-level description and layout criteria.
5. Optionally extract the floor boundary from an input USDZ file.
6. Return the complete `scene.json` structure for LayoutVLM.

Output JSON format:
{
  "task_description": "...",
  "layout_criteria": [...],
  "boundary": {...},
  "assets": {...}
}
"""

import os, re, json, random
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import logging

# local imports
from .prompts import (
    CATEGORY_PLANNER_SYSTEM,
    TASK_CRITERIA_SYSTEM,
    FALLBACK_TASK_TEMPLATE,
    FALLBACK_CRITERIA,
)
from .utils import (
    fetch_assets_from_supabase,
    normalize_category,
    style_score,
    extract_floor_vertices_from_usdz,
    extract_json_block,
    ALLOWED_CATEGORIES,
    CATEGORY_PRIORITY,
    validate_scene,
    validate_furniture_item,
)

# ----------------------------
# ENV & CLIENTS
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing env var: OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Input schema
# ----------------------------
class BuildSceneIn(BaseModel):
    size: str
    room_type: str
    color_palette: str
    budget: str
    user_prompt: str
    usdz_path: Optional[str] = Field(None, description="Local USDZ path for floor extraction")

# Default fallback boundary
DEFAULT_BOUNDARY = {
    "floor_vertices": [[-6, -4, 0], [6, -4, 0], [6, 4, 0], [-6, 4, 0]],
    "wall_height": 2.8,
}

# ----------------------------
# LLM helper
# ----------------------------
def llm(system: str, user: str, temperature: float = 0.3) -> str:
    """Unified wrapper for GPT-5 chat completion."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# ----------------------------
# Step 1 – Category planning
# ----------------------------
def plan_categories(user_prompt: str, size: str, room_type: str, color_palette: str) -> List[Dict[str, Any]]:
    """Use GPT-5 to pick which categories (and how many) appear in the scene."""
    size = size.lower()
    target_counts = {"small": (6, 8), "medium": (8, 12), "large": (12, 16)}
    tgt_lo, tgt_hi = target_counts.get(size, (6, 10))

    user_msg = f"""
User prompt: {user_prompt}
Room size: {size} (target count {tgt_lo}-{tgt_hi})
Style: {room_type}
Color palette: {color_palette}
Allowed categories: {", ".join(ALLOWED_CATEGORIES)}

Return JSON exactly in this format:
{{"selected_categories":[{{"name":"<allowed_category>","count":<int>}}, ...]}}
"""
    raw = llm(CATEGORY_PLANNER_SYSTEM, user_msg, 0.2)
    try:
        data = json.loads(extract_json_block(raw))
        plan = data.get("selected_categories", [])
    except Exception:
        plan = []

    # fallback or padding to reach min count
    total = sum(p.get("count", 0) for p in plan)
    if total < tgt_lo:
        for cat in CATEGORY_PRIORITY:
            if total >= tgt_lo:
                break
            plan.append({"name": cat, "count": 1})
            total += 1
    return plan

# ----------------------------
# Step 2 – Asset selection
# ----------------------------
def select_assets_from_supabase(plan: List[Dict[str, Any]], room_type: str, color_palette: str) -> Dict[str, Any]:
    """Match chosen categories to real Supabase assets."""
    rows = fetch_assets_from_supabase()
    if not rows:
        raise RuntimeError("No assets found in Supabase table.")

    # bucket by normalized category
    buckets: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in ALLOWED_CATEGORIES}
    for r in rows:
        canon = normalize_category(r.get("category"))
        if canon in buckets:
            buckets[canon].append(r)

    # sort each bucket by style match and cost
    for cat, items in buckets.items():
        for it in items:
            it["_style_score"] = style_score(it, room_type, color_palette)

        # Clean and normalize cost to float
            cost_val = it.get("cost")
            if isinstance(cost_val, str):
            # Remove $ and non-numeric parts
                cleaned = cost_val.replace("$", "").replace(",", "").split()[0]
                try:
                    it["_cost_float"] = float(cleaned)
                except ValueError:
                    it["_cost_float"] = 0.0
            elif isinstance(cost_val, (int, float)):
                it["_cost_float"] = float(cost_val)
            else:
                it["_cost_float"] = 0.0

    # Sort by style score first, then cost
        items.sort(key=lambda x: (x["_style_score"], -x["_cost_float"]), reverse=True)


    # pick assets
    selected: Dict[str, Any] = {}
    cat_counter: Dict[str, int] = {}

    for req in plan:
        cat, count = req["name"], req["count"]
        if cat not in buckets or not buckets[cat]:
            continue
        for _ in range(count):
            idx = cat_counter.get(cat, 0)
            if idx >= len(buckets[cat]):
                break
            row = buckets[cat][idx]
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", row["name"]).strip("_")
            asset_var = clean_name if clean_name.startswith(cat) else f"{cat}_{clean_name}"
            selected[f"{cat}-{idx}"] = {
                "category": cat,
                "model_url": row.get("model_url"),
                "metadata_url": row.get("metadata_url"),
                "asset_var_name": asset_var,
            }
            cat_counter[cat] = idx + 1
    return selected

# ----------------------------
# Step 3 – Task description & layout criteria
# ----------------------------
def build_task_and_criteria(user_prompt: str, size: str, room_type: str,
                            color_palette: str, assets: Dict[str, Any]) -> Dict[str, Any]:
    """Generate textual scene summary and layout criteria using GPT-5."""
    cats = sorted({v["category"] for v in assets.values()})
    user_msg = f"""
User prompt: {user_prompt}
Room size: {size}
Style: {room_type}
Color palette: {color_palette}
Selected categories: {", ".join(cats)}

Return JSON:
{{"task_description": "...", "layout_criteria": ["...", "..."]}}
"""
    try:
        raw = llm(TASK_CRITERIA_SYSTEM, user_msg, 0.3)
        data = json.loads(extract_json_block(raw))
        return {
            "task_description": data.get("task_description"),
            "layout_criteria": data.get("layout_criteria"),
        }
    except Exception:
        # fallback
        desc = FALLBACK_TASK_TEMPLATE.format(
            room_type=room_type,
            asset_list=", ".join(cats),
            color_palette=color_palette
        )
        return {"task_description": desc, "layout_criteria": FALLBACK_CRITERIA}

# ----------------------------
# Step 4 – Floor boundary extraction
# ----------------------------
def get_boundary(usdz_path: Optional[str]) -> Dict[str, Any]:
    """Extract or default floor boundary."""
    if not usdz_path:
        return DEFAULT_BOUNDARY
    try:
        verts = extract_floor_vertices_from_usdz(usdz_path)
        return {"floor_vertices": verts, "wall_height": 2.8}
    except Exception as e:
        print(f"⚠️ Floor extraction failed ({e}), using default.")
        return DEFAULT_BOUNDARY

# ----------------------------
# MAIN ENTRYPOINT
# ----------------------------
def generate_scene(payload: BuildSceneIn) -> Dict[str, Any]:
    """Main high-level function: generate full scene.json."""
    logging.info(f"🪄 Generating scene for: {payload.room_type}")
    
    plan = plan_categories(payload.user_prompt, payload.size, payload.room_type, payload.color_palette)
    assets = select_assets_from_supabase(plan, payload.room_type, payload.color_palette)
    meta = build_task_and_criteria(payload.user_prompt, payload.size, payload.room_type, payload.color_palette, assets)
    boundary = get_boundary(payload.usdz_path)

    scene = {
        "task_description": meta["task_description"],
        "layout_criteria": meta["layout_criteria"],
        "boundary": boundary,
        "assets": assets,
    }

    # Log the generated scene data
    logging.info(f"Generated scene data: {scene}")

    # Validate the scene data
    if not validate_scene(scene):
        raise ValueError("Generated scene data is invalid")

    print("✅ Scene JSON assembled successfully.")

    return scene


# ----------------------------
# CLI quick test
# ----------------------------
if __name__ == "__main__":
    test_input = BuildSceneIn(
        size="medium",
        room_type="modern minimalist",
        color_palette="warm neutrals",
        budget="0 - 5000 USD",
        user_prompt="a cozy living room with sofa, TV stand, and reading chair",
        usdz_path=None,
    )
    result = generate_scene(test_input)
    print(json.dumps(result, indent=2))
