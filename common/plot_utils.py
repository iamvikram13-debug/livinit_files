"""
common/plot_utils.py
--------------------
Visualization helpers for Livinit:
- Visualize floor boundaries and layout grids
- Annotate asset placements
- Overlay bounding boxes or text
Uses both OpenCV (for image compositing) and Matplotlib (for plotting geometry).
"""

import os
import cv2
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime


from .io_utils import log


# -------------------------------------------------------
# Color + drawing helpers
# -------------------------------------------------------

def get_random_color(seed: Optional[int] = None) -> Tuple[int, int, int]:
    """Generate a random RGB color (OpenCV format: BGR)."""
    if seed is not None:
        np.random.seed(seed)
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int],
              color: Tuple[int, int, int] = (255, 255, 255),
              bg_color: Tuple[int, int, int] = (0, 0, 0),
              font_scale: float = 0.6, thickness: int = 1) -> np.ndarray:
    """Draw readable text with background."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - baseline), (x + w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img


# -------------------------------------------------------
# Geometry visualizations (Matplotlib)
# -------------------------------------------------------

def visualize_layout(
    floor_vertices: List[List[float]],
    layout_dict: Dict[str, Dict[str, List[float]]],
    output_path: str,
    show_ids: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 200,
):
    """
    Plot floor boundary and all placed assets from layout.json.
    Each asset is shown as a point or bounding box (if rotation/scale given).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Floor boundary polygon
    floor_poly = Polygon([(x, y) for x, y, *_ in floor_vertices])
    fx, fy = floor_poly.exterior.xy
    ax.plot(fx, fy, color="black", linewidth=2, label="Floor Boundary")

    # Draw each asset
    for idx, (asset_id, data) in enumerate(layout_dict.items()):
        pos = data.get("position", [0, 0, 0])
        rot = data.get("rotation", [0, 0, 0])
        x, y = pos[0], pos[1]
        color = np.random.rand(3,)
        ax.scatter(x, y, c=[color], s=30)
        if show_ids:
            ax.text(x + 0.1, y + 0.1, asset_id, fontsize=8, color=color)
        # Draw small arrow for facing direction
        angle = math.radians(rot[-1])
        ax.arrow(x, y, 0.3 * math.cos(angle), 0.3 * math.sin(angle),
                 head_width=0.1, fc=color, ec=color)

    ax.set_aspect("equal")
    ax.set_title("Livinit Layout Visualization")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log(f"🖼 Layout visualization saved → {output_path}", "OK")


# -------------------------------------------------------
# Image overlays (OpenCV + PIL)
# -------------------------------------------------------

def overlay_text_on_image(
    image_path: str,
    text: str,
    output_path: str,
    font_size: int = 24,
    color: Tuple[int, int, int] = (255, 255, 255),
):
    """Overlay large title text on top of an image (PIL-based)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    w, h = img.size
    text_w, text_h = draw.textsize(text, font=font)
    draw.text(((w - text_w) / 2, 10), text, font=font, fill=color)

    img.save(output_path)
    log(f"🖋 Text overlay saved → {output_path}", "OK")


def overlay_boxes(
    image_path: str,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
):
    """
    Overlay bounding boxes (x1, y1, x2, y2) on an image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = get_random_color(i)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = labels[i] if labels and i < len(labels) else f"Box {i+1}"
        draw_text(img, label, (x1, max(20, y1 - 10)), color=color)

    out = output_path or os.path.splitext(image_path)[0] + "_boxed.png"
    cv2.imwrite(out, img)
    log(f"📦 Boxes overlaid → {out}", "OK")
    return out


# -------------------------------------------------------
# Debug helper (side-by-side comparison)
# -------------------------------------------------------

def compare_images(img1_path: str, img2_path: str, output_path: str):
    """Horizontally stack two images for visual comparison."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the images could not be read.")

    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2
    cv2.imwrite(output_path, canvas)
    log(f"🔍 Comparison image saved → {output_path}", "OK")


# -------------------------------------------------------
# Demo / CLI
# -------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Livinit visualization utilities")
    parser.add_argument("--layout", help="Path to layout.json")
    parser.add_argument("--floor", help="Path to floor_vertices.json")
    parser.add_argument("--out", default="debug_layout.png")
    args = parser.parse_args()

    if args.layout and args.floor:
        layout = json.load(open(args.layout))
        floor = json.load(open(args.floor))
        visualize_layout(floor["floor_vertices"], layout, args.out)
    else:
        log("No layout/floor input provided — exiting.", "WARN")
