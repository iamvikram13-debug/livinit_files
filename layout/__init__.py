"""
Layout package for Livinit pipeline
Handles layout generation and optimization
"""
from .layout_solver import run_layout_solver, save_layout_to_file
from .grad_solver import optimize_layout, solve_layout, compute_boundary_box
from .constraints import (
    iou_polygon,
    is_inside_boundary,
    get_bbox_polygon,
    fill_random_unplaced_assets,
)
from .scene_classes import AssetInstance, AssetGroup, Wall, Scene
from .placement_utils import (
    random_point_in_polygon,
    get_bbox_corners as placement_get_bbox_corners,
)

__all__ = [
    "run_layout_solver",
    "save_layout_to_file",
    "optimize_layout",
    "solve_layout",
    "compute_boundary_box",
    "iou_polygon",
    "is_inside_boundary",
    "get_bbox_polygon",
    "fill_random_unplaced_assets",
    "AssetInstance",
    "AssetGroup",
    "Wall",
    "Scene",
    "random_point_in_polygon",
    "placement_get_bbox_corners",
]
