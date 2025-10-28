"""
common/colors.py
----------------
Color utilities for the Livinit pipeline.

Includes:
- Range conversions (0–1 ↔ 0–255)
- Format conversions (RGB, RGBA, BGR, HEX)
- Category color mapping (tab10 / tab20 / viridis / jet)
- Basic color blending and random sampling

Used by visualization (plot_utils) and optional texture colorization.
"""

import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

# -------------------------------------------------------
# Range conversions
# -------------------------------------------------------

def convert_color_range(
    color: Union[List[float], Tuple[float, ...]],
    from_range: str = "0-1",
    to_range: str = "0-255"
) -> List[float]:
    """Convert color values between normalized (0–1) and integer (0–255) ranges."""
    if from_range == "0-1" and to_range == "0-255":
        return [int(c * 255) for c in color]
    elif from_range == "0-255" and to_range == "0-1":
        return [c / 255 for c in color]
    elif from_range == to_range:
        return list(color)
    else:
        raise ValueError(f"Unsupported conversion {from_range} → {to_range}")


# -------------------------------------------------------
# Format conversions
# -------------------------------------------------------

def convert_color_format(
    color: Union[List[float], Tuple[float, ...]],
    from_format: str = "rgba",
    to_format: str = "rgb",
    alpha_value: float = 1.0
) -> List[float]:
    """Convert color representation between RGB, RGBA, and BGR."""
    if from_format == "rgba" and to_format == "rgb":
        return list(color[:3])
    elif from_format == "rgb" and to_format == "rgba":
        return list(color) + [alpha_value]
    elif from_format == "rgb" and to_format == "bgr":
        return list(color[::-1])
    elif from_format == "bgr" and to_format == "rgb":
        return list(color[::-1])
    elif from_format == to_format:
        return list(color)
    else:
        raise ValueError(f"Unsupported conversion {from_format} → {to_format}")


# -------------------------------------------------------
# Palette sampling
# -------------------------------------------------------

def get_categorical_colors(
    num_categories: int,
    colormap_name: str = "tab10",
    color_range: str = "0-255",
    color_format: str = "rgb"
) -> List[List[float]]:
    """Return N visually distinct colors from a matplotlib colormap."""
    if colormap_name == "tab10":
        cmap = plt.cm.tab10
    elif colormap_name == "tab20":
        cmap = plt.cm.tab20
    elif colormap_name == "viridis":
        cmap = plt.cm.viridis
    elif colormap_name == "jet":
        cmap = plt.cm.jet
    else:
        raise ValueError(f"Unsupported colormap: {colormap_name}")

    norm = plt.Normalize(0, num_categories - 1)
    colors = [cmap(norm(i)) for i in range(num_categories)]
    colors = [convert_color_range(c, "0-1", color_range) for c in colors]
    colors = [convert_color_format(c, "rgba", color_format) for c in colors]

    # Optional tab20 shuffle to enhance contrast
    if colormap_name == "tab20" and num_categories == 20:
        colors = colors[::2] + colors[1::2]
    return colors


# -------------------------------------------------------
# Utility operations
# -------------------------------------------------------

def random_color(color_range: str = "0-255", color_format: str = "rgb") -> List[int]:
    """Return a single random RGB color."""
    c = [random.randint(0, 255) for _ in range(3)]
    if color_range == "0-1":
        c = [x / 255 for x in c]
    return convert_color_format(c, "rgb", color_format)


def blend_colors(color1: List[float], color2: List[float], alpha: float = 0.5) -> List[float]:
    """Linearly blend two colors."""
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
    return [c1 * (1 - alpha) + c2 * alpha for c1, c2 in zip(color1, color2)]


def to_hex(color: Union[List[int], List[float]], from_range: str = "0-255") -> str:
    """Convert RGB color to HEX string."""
    if from_range == "0-1":
        color = convert_color_range(color, "0-1", "0-255")
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def from_hex(hex_str: str) -> List[int]:
    """Convert HEX string to RGB (0–255)."""
    hex_str = hex_str.strip("#")
    return [int(hex_str[i:i + 2], 16) for i in (0, 2, 4)]


# -------------------------------------------------------
# Demo (optional)
# -------------------------------------------------------

if __name__ == "__main__":
    colors = get_categorical_colors(6, "tab10", color_range="0-255")
    for i, c in enumerate(colors):
        print(f"Color {i+1}: {c} → HEX {to_hex(c)}")
