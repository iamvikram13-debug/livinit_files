"""
layout/grad_solver.py
---------------------
Differentiable layout optimization engine for Livinit.

This module takes initial positions and rotations (from LLM or LayoutVLM)
and refines them to satisfy spatial constraints.

Simplified version:
  - No rotated IoU
  - Uses distance, boundary, and clearance penalties
  - Pure PyTorch (no external differentiable geometry libs)
"""

import math
import random

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def pairwise_distance(a, b):
    """Compute pairwise L2 distance between two sets of (x, y) points."""
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    return torch.norm(diff, dim=-1)

def boundary_penalty(points, boundary_min, boundary_max, margin=0.05):
    """
    Penalize points that fall outside the room boundary.
    boundary_min, boundary_max: torch tensors (2,)
    """
    penalty = torch.zeros(1, device=points.device)
    below_min = (boundary_min - points).clamp(min=0)
    above_max = (points - boundary_max).clamp(min=0)
    penalty += below_min.sum() + above_max.sum()
    return penalty * 10.0  # stronger weight

def spacing_penalty(points, min_clearance=0.3):
    """
    Encourage assets to maintain min_clearance distance apart.
    """
    dist = pairwise_distance(points, points)
    mask = torch.triu(torch.ones_like(dist), diagonal=1)
    violation = (min_clearance - dist).clamp(min=0) * mask
    return violation.sum() * 5.0

def orientation_smoothness(rotations):
    """
    Encourage smooth rotation angles (to avoid extreme jitter).
    """
    diffs = torch.diff(rotations)
    return (diffs**2).sum() * 0.1

# -------------------------------------------------------------
# Optimization core
# -------------------------------------------------------------
def optimize_layout(init_positions, init_rotations, boundary, max_steps=200, lr=0.01):
    """
    Run gradient descent to adjust asset positions within constraints.

    init_positions: torch.Tensor (N, 2)
    init_rotations: torch.Tensor (N,)
    boundary: dict with min/max coords
    """
    # Import torch lazily so importing this module doesn't require torch at package-import time.
    global torch
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos = init_positions.clone().detach().to(device).requires_grad_(True)
    rot = init_rotations.clone().detach().to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([pos, rot], lr=lr)

    boundary_min = torch.tensor(boundary["min"], device=device)
    boundary_max = torch.tensor(boundary["max"], device=device)

    for step in range(max_steps):
        optimizer.zero_grad()

        loss_boundary = boundary_penalty(pos, boundary_min, boundary_max)
        loss_spacing = spacing_penalty(pos)
        loss_rot = orientation_smoothness(rot)

        total_loss = loss_boundary + loss_spacing + loss_rot

        total_loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == max_steps - 1:
            print(f"🌀 Step {step:03d} | total={total_loss.item():.4f} "
                  f"(boundary={loss_boundary.item():.3f}, spacing={loss_spacing.item():.3f})")

        # early stopping
        if total_loss.item() < 1e-3:
            break

    return pos.detach().cpu(), rot.detach().cpu()

# -------------------------------------------------------------
# Utility: make boundary box tensor
# -------------------------------------------------------------
def compute_boundary_box(floor_vertices):
    """
    Compute min/max XY bounds from a list of floor vertices.
    """
    # Ensure torch is available at runtime
    global torch
    try:
        import torch
        torch
    except Exception:
        raise RuntimeError("grad_solver: 'torch' is required to compute boundary box at runtime")

    verts = torch.tensor([[x, y] for x, y, _ in floor_vertices])
    min_xy = verts.min(dim=0).values
    max_xy = verts.max(dim=0).values
    return {"min": min_xy.tolist(), "max": max_xy.tolist()}

# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
def solve_layout(init_layout, boundary):
    """
    Wrapper for end-to-end optimization.

    init_layout: dict {asset_id: {"position": [x,y,z], "rotation": [0,0,deg]}}
    boundary: dict with "floor_vertices"
    """
    # Import torch lazily so module import is lightweight
    global torch
    try:
        import torch
        torch
    except Exception:
        # Defer the error until runtime so package import won't fail.
        pass

    N = len(init_layout)
    if N == 0:
        raise ValueError("No assets to optimize")

    # prepare tensors
    init_positions = torch.tensor(
        [[v["position"][0], v["position"][1]] for v in init_layout.values()],
        dtype=torch.float32,
    )
    init_rotations = torch.tensor(
        [v["rotation"][2] for v in init_layout.values()],
        dtype=torch.float32,
    )
    bounds = compute_boundary_box(boundary["floor_vertices"])

    # run optimization
    final_pos, final_rot = optimize_layout(init_positions, init_rotations, bounds)

    # rebuild dictionary
    optimized = {}
    for (key, old), pos, rot in zip(init_layout.items(), final_pos, final_rot):
        optimized[key] = {
            "position": [float(pos[0]), float(pos[1]), old["position"][2]],
            "rotation": [0.0, 0.0, float(rot)],
        }

    return optimized

# -------------------------------------------------------------
# CLI quick test
# -------------------------------------------------------------
if __name__ == "__main__":
    # Fake example
    floor = [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]]
    layout = {
        "sofa-0": {"position": [0.0, 0.0, 0.5], "rotation": [0, 0, 45]},
        "table-0": {"position": [1.0, 1.0, 0.5], "rotation": [0, 0, 90]},
        "lamp-0": {"position": [-1.0, 1.0, 0.5], "rotation": [0, 0, 30]},
    }

    print("🔧 Running gradient layout solver...")
    result = solve_layout(layout, {"floor_vertices": floor})
    print("\n✅ Optimized layout:")
    for k, v in result.items():
        print(k, v)
