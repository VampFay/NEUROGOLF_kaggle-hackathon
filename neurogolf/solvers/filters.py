"""
neurogolf/solvers/filters.py — Try various local 3x3 conv filters.

Useful for tasks that involve neighbor-based rules (e.g., cellular automata,
edge detection, contour filling, "fill between markers").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx

from .base import Solver
from .. import dsl
from ..arc_data import get_pairs


def _conv_onehot(weight: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Apply a 3x3 conv to a one-hot-encoded grid and return argmax.

    weight: (out_C=10, in_C=10, kH=3, kW=3)
    grid:   (H, W) int64
    Returns: (H, W) int64 (argmax over output channels)
    """
    from ..arc_data import grid_to_onehot
    from ..constants import MAX_GRID, NUM_COLORS
    H, W = grid.shape
    onehot = grid_to_onehot(grid.tolist())  # (1, 10, 30, 30)
    # Pad weight to operate on (1, 10, 30, 30)
    # Use a simple conv via numpy
    out = np.zeros((NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
    half = 1
    for c_out in range(NUM_COLORS):
        for c_in in range(NUM_COLORS):
            for dh in range(-half, half + 1):
                for dw in range(-half, half + 1):
                    w = weight[c_out, c_in, dh + half, dw + half]
                    if w == 0:
                        continue
                    # Shift onehot[c_in] by (-dh, -dw) and accumulate
                    shifted = np.zeros_like(onehot[0, c_in])
                    # Source: (i, j) in shifted comes from (i + dh, j + dw) in onehot
                    # Equivalent to: out[c_out, i, j] += w * onehot[c_in, i+dh, j+dw]
                    # Implement via slicing
                    for i in range(MAX_GRID):
                        si = i + dh
                        if si < 0 or si >= MAX_GRID:
                            continue
                        for j in range(MAX_GRID):
                            sj = j + dw
                            if sj < 0 or sj >= MAX_GRID:
                                continue
                            out[c_out, i, j] += w * onehot[0, c_in, si, sj]
    return out.argmax(axis=0)[:H, :W]


def _try_filter(weight: np.ndarray, pairs) -> bool:
    """Check if a conv filter (weight) correctly maps all (input, output) pairs."""
    for inp, out in pairs:
        pred = _conv_onehot(weight, inp)
        if pred.shape != out.shape:
            return False
        if not np.array_equal(pred, out):
            return False
    return True


# ----- Filter templates -----


def _identity_filter() -> np.ndarray:
    """3x3 identity: only center, diagonal weights."""
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for c in range(10):
        W[c, c, 1, 1] = 1.0
    return W


def _color_substitution_filter(mapping: dict[int, int]) -> np.ndarray:
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for frm, to in mapping.items():
        W[to, frm, 1, 1] = 1.0
    return W


def _mask_filter(mask_color: int) -> np.ndarray:
    """Pass through cell only if it equals mask_color."""
    # Actually this needs Mul not Conv — skip for now
    pass


def _neighbor_count_filter(target_color: int, threshold: int, output_color: int) -> np.ndarray:
    """If a cell has >= threshold neighbors of target_color, output output_color;
    else keep input color.

    Implemented as: out[output_color] = sum of neighbor onehots of target_color
                    out[input_color]   = 1 (current color)
    Then argmax picks output_color if sum >= 1 (assuming output_color != input_color).
    For threshold > 1, we need bias.
    """
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    # Current color: identity at center
    for c in range(10):
        W[c, c, 1, 1] = 1.0
    # Neighbor count: out[output_color] += in[target_color] for all 8 neighbors
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            if dh == 0 and dw == 0:
                continue
            W[output_color, target_color, dh + 1, dw + 1] = 1.0
    return W


class ConvFilterSolver(Solver):
    """Try a library of pre-defined 3x3 conv filters."""
    name = "conv_filter"

    CANDIDATE_FILTERS = [
        ("identity", _identity_filter),
    ]

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # 1. Try identity (already covered by IdentitySolver, but cheap)
        # 2. Try every (target_color, output_color) neighbor count filter
        for target_c in range(10):
            for output_c in range(10):
                if target_c == output_c:
                    continue
                W = _neighbor_count_filter(target_c, 1, output_c)
                if _try_filter(W, pairs):
                    return dsl.single_layer_conv2d(W)
        # 3. Try every 2-color substitution + neighbor count
        # (more candidates can be added)
        return None


class ColorSubstitutionSolver(Solver):
    """Try every single-color and 2-color substitution (exhaustive over colors).

    Already covered by ColorMapSolver but kept as a slightly different fallback
    that doesn't require same shapes everywhere.
    """
    name = "color_substitution"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        # Try every (from, to) single-color substitution
        for frm in range(10):
            for to in range(10):
                if frm == to:
                    continue
                W = _color_substitution_filter({frm: to})
                if _try_filter(W, pairs):
                    return dsl.single_layer_conv2d(W)
        return None
