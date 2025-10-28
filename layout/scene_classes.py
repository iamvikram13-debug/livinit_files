"""
layout/scene_classes.py
-----------------------
Defines lightweight scene object representations for Livinit layout system.

Classes:
  - AssetInstance : Individual placed furniture piece
  - AssetGroup    : Group of asset instances (category)
  - Wall          : Structural wall segment
  - Scene         : Container for all assets, walls, and boundary
"""

import math
import torch
from shapely.geometry import Polygon


# -------------------------------------------------------------
# Core Asset Instance
# -------------------------------------------------------------
class AssetInstance:
    """
    Represents a single furniture instance.
    Attributes:
      id           : unique identifier (e.g., "sofa_0")
      position     : [x, y, z]
      rotation     : [0, 0, yaw_deg]
      size         : [x, y, z]
      onCeiling    : bool (for hanging objects)
      optimize     : int (1=optimize, 0=fixed)
    """

    def __init__(self, id, position, rotation, size=None, onCeiling=False, optimize=1):
        self.id = id
        self.position = torch.tensor(position, dtype=torch.float32)
        self.rotation = torch.tensor(rotation, dtype=torch.float32)
        self.size = size or [1.0, 1.0, 1.0]
        self.onCeiling = onCeiling
        self.optimize = optimize

    def get_theta(self):
        """Return yaw rotation in radians."""
        if len(self.rotation) < 3:
            return 0.0
        return math.radians(self.rotation[2].item() if isinstance(self.rotation, torch.Tensor) else self.rotation[2])

    def get_2dpolygon(self):
        """
        Return 2D bounding polygon (XY plane) for collision checks.
        """
        w, d, _ = self.size
        half_w, half_d = w / 2, d / 2
        theta = self.get_theta()

        # Define local corners (centered on origin)
        corners = torch.tensor([
            [-half_w, -half_d],
            [ half_w, -half_d],
            [ half_w,  half_d],
            [-half_w,  half_d],
        ], dtype=torch.float32)

        # Rotation matrix
        rot = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ], dtype=torch.float32)

        rotated = corners @ rot.T
        rotated[:, 0] += self.position[0]
        rotated[:, 1] += self.position[1]
        return rotated

    def as_dict(self):
        """Return JSON-compatible dictionary."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "size": self.size,
            "onCeiling": self.onCeiling,
            "optimize": self.optimize,
        }


# -------------------------------------------------------------
# AssetGroup (collection of instances)
# -------------------------------------------------------------
class AssetGroup:
    """
    Container for all instances of a single category (e.g., all sofas).
    """

    def __init__(self, category):
        self.category = category
        self.placements = []

    def add_instance(self, asset_instance):
        assert isinstance(asset_instance, AssetInstance)
        self.placements.append(asset_instance)

    def __len__(self):
        return len(self.placements)

    def __iter__(self):
        return iter(self.placements)


# -------------------------------------------------------------
# Wall
# -------------------------------------------------------------
class Wall:
    """
    Represents a wall segment defined by two 3D points.
    """

    def __init__(self, id, vertices):
        self.id = id
        self.corner1 = vertices[0]
        self.corner2 = vertices[1]

    def as_line(self):
        """Return shapely LineString for geometry ops."""
        from shapely.geometry import LineString
        return LineString([self.corner1[:2], self.corner2[:2]])

    def as_dict(self):
        return {"id": self.id, "corner1": self.corner1, "corner2": self.corner2}


# -------------------------------------------------------------
# Scene container
# -------------------------------------------------------------
class Scene:
    """
    The full scene representation, including assets and walls.
    """

    def __init__(self, boundary_vertices):
        self.boundary = boundary_vertices
        self.assets = {}
        self.walls = self._make_walls_from_boundary(boundary_vertices)

    def _make_walls_from_boundary(self, boundary):
        walls = []
        for i in range(len(boundary)):
            c1 = boundary[i]
            c2 = boundary[(i + 1) % len(boundary)]
            walls.append(Wall(f"wall_{i}", [c1, c2]))
        return walls

    def add_asset(self, category, asset_instance):
        if category not in self.assets:
            self.assets[category] = AssetGroup(category)
        self.assets[category].add_instance(asset_instance)

    def as_dict(self):
        return {
            "boundary": self.boundary,
            "assets": {cat: [a.as_dict() for a in grp] for cat, grp in self.assets.items()},
            "walls": [w.as_dict() for w in self.walls],
        }

    def __repr__(self):
        return f"<Scene assets={len(self.assets)} walls={len(self.walls)}>"
