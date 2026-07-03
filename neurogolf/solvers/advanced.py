"""
neurogolf/solvers/advanced.py — More advanced solvers.

Covers:
  - ScaleUpSolver: output = input scaled by integer factor k
  - CropSolver: output = top-left HxW of input (identity works due to validator cropping)
  - ShiftSolver: output = input shifted by (dh, dw)
  - TileSolver: output = input repeated (Tile op)
  - KroneckerSolver: out[i*k+a, j*k+b] = in[a,b] if in[i,j] != 0 else 0
  - ConcatRepeatSolver: out = input + first K rows repeated (extend pattern)
  - ConditionalSliceColorMapSolver: slice a sub-region based on a separator color,
    then apply a color map
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto

from .base import Solver
from .. import dsl
from ..arc_data import get_pairs
from ..constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_io_value_info(name: str, shape=list(IO_SHAPE)) -> onnx.ValueInfoProto:
    return h.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _make_simple_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto] = None,
                       name: str = "neurogolf") -> onnx.ModelProto:
    initializers = initializers or []
    graph = h.make_graph(
        nodes, name,
        inputs=[_make_io_value_info(INPUT_NAME)],
        outputs=[_make_io_value_info(OUTPUT_NAME)],
        initializer=initializers,
    )
    model = h.make_model(graph, producer_name="neurogolf",
                          opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# ScaleUpSolver
# ---------------------------------------------------------------------------


class ScaleUpSolver(Solver):
    """Output is input scaled by an integer factor k (k=2,3,4)."""
    name = "scale_up"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        k = None
        for inp, out in pairs:
            if inp.shape[0] == 0 or inp.shape[1] == 0:
                return None
            kh = out.shape[0] / inp.shape[0]
            kw = out.shape[1] / inp.shape[1]
            if not kh.is_integer() or not kw.is_integer() or kh != kw:
                return None
            kk = int(kh)
            if k is None:
                k = kk
            elif k != kk:
                return None
        if k is None or k <= 1:
            return None
        for inp, out in pairs:
            scaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
            if scaled.shape != out.shape or not np.array_equal(scaled, out):
                return None
        nodes = [
            h.make_node("Constant", [], ["roi"],
                         value=h.make_tensor("roi_v", TensorProto.FLOAT, [0], [])),
            h.make_node("Constant", [], ["scales"],
                         value=h.make_tensor("scales_v", TensorProto.FLOAT, [4],
                                              [1.0, 1.0, float(k), float(k)])),
            h.make_node("Resize", [INPUT_NAME, "roi", "scales"], [OUTPUT_NAME],
                         mode="nearest",
                         nearest_mode="floor",
                         coordinate_transformation_mode="asymmetric"),
        ]
        return _make_simple_model(nodes, name="scale_up")


# ---------------------------------------------------------------------------
# CropSolver
# ---------------------------------------------------------------------------


class CropSolver(Solver):
    """Output is top-left HxW of input (validator crops automatically)."""
    name = "crop_top_left"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            H, W = out.shape
            if inp.shape[0] < H or inp.shape[1] < W:
                return None
            sub = inp[:H, :W]
            if not np.array_equal(sub, out):
                return None
        return dsl.identity()


# ---------------------------------------------------------------------------
# ShiftSolver
# ---------------------------------------------------------------------------


class ShiftSolver(Solver):
    """Output = input shifted by (dh, dw). Pads with 0 (color 0).

    Implemented using Pad + Slice (or just Slice with appropriate offsets).
    Output dimensions = input dimensions (validator crops).
    """
    name = "shift"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine the shift (dh, dw) — must be consistent across all pairs
        shift = None
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
            # Find the shift by trying all possibilities
            H, W = inp.shape
            found = None
            for dh in range(-H + 1, H):
                for dw in range(-W + 1, W):
                    # Apply shift: out[i, j] = inp[i - dh, j - dw] if in bounds else 0
                    candidate = np.zeros_like(inp)
                    for i in range(H):
                        for j in range(W):
                            si = i - dh
                            sj = j - dw
                            if 0 <= si < H and 0 <= sj < W:
                                candidate[i, j] = inp[si, sj]
                    if np.array_equal(candidate, out):
                        found = (dh, dw)
                        break
                if found:
                    break
            if found is None:
                return None
            if shift is None:
                shift = found
            elif shift != found:
                return None
        if shift is None:
            return None
        dh, dw = shift
        if dh == 0 and dw == 0:
            return dsl.identity()
        return _shift_model(dh, dw)


def _shift_model(dh: int, dw: int) -> onnx.ModelProto:
    """Build a network that shifts the input by (dh, dw) — out[i,j] = inp[i-dh, j-dw].

    Uses Slice with negative-start trickery inside the 30x30 frame, then
    places the result back at the top-left.
    """
    # Strategy: Slice the input from (dh, dw) to (dh+30, dw+30) within the 30x30 frame.
    # If dh > 0: source starts at row dh, output starts at row 0.
    # If dh < 0: source starts at row 0, output starts at row -dh (so we Pad top by -dh).
    # Simpler: Pad input with zeros on the side opposite to the shift direction,
    # then slice the top-left 30x30.
    #
    # Equivalent: use Slice with starts=[max(0,dh), max(0,dw)] ends=[30+max(0,dh), 30+max(0,dw)]
    # But this loses cells that fall off the bottom.
    # Better: Pad then Slice.
    #
    # Pad: top = max(0, -dh), bottom = max(0, dh), left = max(0, -dw), right = max(0, dw)
    # Then slice top-left 30x30.

    top = max(0, -dh)
    bottom = max(0, dh)
    left = max(0, -dw)
    right = max(0, dw)
    pads = [0, 0, top, left, 0, 0, bottom, right]
    nodes = [
        h.make_node("Constant", [], ["pads_c"],
                     value=h.make_tensor("pads_v", TensorProto.INT64, [8], pads)),
        h.make_node("Constant", [], ["val_c"],
                     value=h.make_tensor("val_v", TensorProto.FLOAT, [], [0.0])),
        h.make_node("Pad", [INPUT_NAME, "pads_c", "val_c"], ["padded"],
                     mode="constant"),
        # Slice padded (30+|dh| x 30+|dw|) to top-left 30x30
        h.make_node("Constant", [], ["starts"],
                     value=h.make_tensor("starts_v", TensorProto.INT64, [4], [0, 0, 0, 0])),
        h.make_node("Constant", [], ["ends"],
                     value=h.make_tensor("ends_v", TensorProto.INT64, [4],
                                          [1, NUM_COLORS, MAX_GRID, MAX_GRID])),
        h.make_node("Constant", [], ["axes"],
                     value=h.make_tensor("axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
        h.make_node("Slice", ["padded", "starts", "ends", "axes"], [OUTPUT_NAME]),
    ]
    return _make_simple_model(nodes, name=f"shift_{dh}_{dw}")


# ---------------------------------------------------------------------------
# TileSolver
# ---------------------------------------------------------------------------


class TileSolver(Solver):
    """Output = input pattern repeated (tile_factor_x, tile_factor_y) times.

    E.g., 3x3 input tiled 3x3 -> 9x9 output where the 3x3 pattern repeats.
    Implemented using ONNX Tile op.
    """
    name = "tile"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine tile factor
        tf = None
        for inp, out in pairs:
            if inp.shape[0] == 0 or inp.shape[1] == 0:
                return None
            th = out.shape[0] / inp.shape[0]
            tw = out.shape[1] / inp.shape[1]
            if not th.is_integer() or not tw.is_integer():
                return None
            t = (int(th), int(tw))
            if tf is None:
                tf = t
            elif tf != t:
                return None
        if tf is None or tf == (1, 1):
            return None
        # Verify
        for inp, out in pairs:
            tiled = np.tile(inp, tf)
            if tiled.shape != out.shape or not np.array_equal(tiled, out):
                return None
        th, tw = tf
        # Build Tile model: repeats = [1, 1, th, tw]
        nodes = [
            h.make_node("Constant", [], ["repeats"],
                         value=h.make_tensor("repeats_v", TensorProto.INT64, [4],
                                              [1, 1, th, tw])),
            h.make_node("Tile", [INPUT_NAME, "repeats"], [OUTPUT_NAME]),
        ]
        return _make_simple_model(nodes, name=f"tile_{th}x{tw}")


# ---------------------------------------------------------------------------
# KroneckerSolver — out[i*k+a, j*k+b] = in[a,b] if in[i,j]!=0 else 0
# ---------------------------------------------------------------------------


class KroneckerSolver(Solver):
    """Kronecker-like product: each non-zero input cell becomes a copy of the
    input pattern in the output; zero input cells become zero blocks.

    Matches tasks like 007bbfb7 (3x3 -> 9x9 conditional tiling).

    Implementation: out = Tile(input, kxk) * Resize(input, kxk) where
    Resize gives the "block value" and Tile gives the "within-block value".
    But we need to handle the "is block non-zero" check correctly in one-hot.
    """
    name = "kronecker"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine k
        k = None
        for inp, out in pairs:
            if inp.shape[0] == 0 or inp.shape[1] == 0:
                return None
            if inp.shape[0] * inp.shape[1] == 0:
                return None
            # Kronecker requires input to be kxk (square)
            if inp.shape[0] != inp.shape[1]:
                return None
            kh = out.shape[0] / inp.shape[0]
            kw = out.shape[1] / inp.shape[1]
            if not kh.is_integer() or not kw.is_integer() or kh != kw:
                return None
            kk = int(kh)
            # Input size must equal k (Kronecker pattern: input is the "tile" itself)
            if inp.shape[0] != kk:
                return None
            if k is None:
                k = kk
            elif k != kk:
                return None
        if k is None or k <= 1:
            return None
        # Verify: out[i*k+a, j*k+b] = in[a, b] if in[i, j] != 0 else 0
        for inp, out in pairs:
            H, W = inp.shape
            for i in range(H):
                for j in range(W):
                    block = out[i*k:(i+1)*k, j*k:(j+1)*k]
                    if inp[i, j] == 0:
                        if not np.array_equal(block, np.zeros((k, k))):
                            return None
                            break
                    else:
                        if not np.array_equal(block, inp):
                            return None
                            break
        return _kronecker_model(k)


def _kronecker_model(k: int) -> onnx.ModelProto:
    """Build a Kronecker-product network with factor k.

    Requires input to be kxk (placed in top-left of 30x30 frame, rest zero).

    For one-hot input I (1, 10, 30, 30):
      S = Slice(I, top-left kxk)              → (1, 10, k, k)
      T1 = Resize(S, scale=k)                 → (1, 10, k*k, k*k)  block-constant tiling
      T2 = Tile(S, repeats=[1,1,k,k])         → (1, 10, k*k, k*k)  pattern-tiling
      gate = 1 - T1[0:1]                      → 1 where block non-zero, else 0
      O_small = T2 * gate                     → (1, 10, k*k, k*k)
      O = Pad(O_small, to 30x30)              → (1, 10, 30, 30)
    """
    kk = k * k
    pad_h = MAX_GRID - kk
    pad_w = MAX_GRID - kk
    nodes = [
        # Slice input to (1, 10, k, k) — top-left kxk
        h.make_node("Constant", [], ["s_starts"],
                     value=h.make_tensor("s_starts_v", TensorProto.INT64, [4], [0, 0, 0, 0])),
        h.make_node("Constant", [], ["s_ends"],
                     value=h.make_tensor("s_ends_v", TensorProto.INT64, [4], [1, NUM_COLORS, k, k])),
        h.make_node("Constant", [], ["s_axes"],
                     value=h.make_tensor("s_axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
        h.make_node("Slice", [INPUT_NAME, "s_starts", "s_ends", "s_axes"], ["S"]),
        # T1 = Resize(S, scale=k) → (1, 10, k*k, k*k)
        h.make_node("Constant", [], ["roi1"],
                     value=h.make_tensor("roi1_v", TensorProto.FLOAT, [0], [])),
        h.make_node("Constant", [], ["scales1"],
                     value=h.make_tensor("scales1_v", TensorProto.FLOAT, [4],
                                          [1.0, 1.0, float(k), float(k)])),
        h.make_node("Resize", ["S", "roi1", "scales1"], ["T1"],
                     mode="nearest", nearest_mode="floor",
                     coordinate_transformation_mode="asymmetric"),
        # T2 = Tile(S, repeats=[1,1,k,k]) → (1, 10, k*k, k*k)
        h.make_node("Constant", [], ["repeats2"],
                     value=h.make_tensor("repeats2_v", TensorProto.INT64, [4],
                                          [1, 1, k, k])),
        h.make_node("Tile", ["S", "repeats2"], ["T2"]),
        # gate = 1 - T1[0:1] (channel 0 of T1) — shape (1, 1, k*k, k*k)
        h.make_node("Constant", [], ["g_starts"],
                     value=h.make_tensor("g_starts_v", TensorProto.INT64, [4], [0, 0, 0, 0])),
        h.make_node("Constant", [], ["g_ends"],
                     value=h.make_tensor("g_ends_v", TensorProto.INT64, [4], [1, 1, kk, kk])),
        h.make_node("Constant", [], ["g_axes"],
                     value=h.make_tensor("g_axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
        h.make_node("Slice", ["T1", "g_starts", "g_ends", "g_axes"], ["T1_c0"]),
        # gate = 1 - T1_c0 (broadcasts: (1,1,1,1) - (1,1,k*k,k*k) → (1,1,k*k,k*k))
        h.make_node("Constant", [], ["one"],
                     value=h.make_tensor("one_v", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])),
        h.make_node("Sub", ["one", "T1_c0"], ["gate"]),
        # O_small = T2 * gate (broadcasts channel)
        h.make_node("Mul", ["T2", "gate"], ["O_small"]),
        # Pad O_small to (1, 10, 30, 30)
        h.make_node("Constant", [], ["pad_c"],
                     value=h.make_tensor("pad_v", TensorProto.INT64, [8],
                                          [0, 0, 0, 0, 0, 0, pad_h, pad_w])),
        h.make_node("Constant", [], ["pad_val"],
                     value=h.make_tensor("pad_val_v", TensorProto.FLOAT, [], [0.0])),
        h.make_node("Pad", ["O_small", "pad_c", "pad_val"], [OUTPUT_NAME], mode="constant"),
    ]
    return _make_simple_model(nodes, name=f"kronecker_{k}x{k}")


# ---------------------------------------------------------------------------
# ConcatRepeatSolver — extend input by repeating first K rows (or cols)
# ---------------------------------------------------------------------------


class ConcatRepeatSolver(Solver):
    """Output = input + first K rows of input appended at the bottom (or similar).

    Matches task 017c7c7b (6x3 -> 9x3, output = input + first 3 rows).
    """
    name = "concat_repeat"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine direction (rows or cols) and K
        direction = None
        K = None
        for inp, out in pairs:
            H_in, W_in = inp.shape
            H_out, W_out = out.shape
            # Check rows extension
            if W_in == W_out and H_out > H_in:
                # Check if out[:H_in] == inp and out[H_in:] == inp[:K] for some K
                if not np.array_equal(out[:H_in], inp):
                    return None
                K_candidate = H_out - H_in
                if not np.array_equal(out[H_in:], inp[:K_candidate]):
                    return None
                if direction is None:
                    direction = "rows"
                    K = K_candidate
                elif direction != "rows" or K != K_candidate:
                    return None
            elif H_in == H_out and W_out > W_in:
                if not np.array_equal(out[:, :W_in], inp):
                    return None
                K_candidate = W_out - W_in
                if not np.array_equal(out[:, W_in:], inp[:, :K_candidate]):
                    return None
                if direction is None:
                    direction = "cols"
                    K = K_candidate
                elif direction != "cols" or K != K_candidate:
                    return None
            else:
                return None
        if direction is None:
            return None
        return _concat_repeat_model(direction, K)


def _concat_repeat_model(direction: str, K: int) -> onnx.ModelProto:
    """Build a network that outputs [input, input[:K]] (rows) or [input | input[:,:K]] (cols)
    in the top-left of the 30x30 frame.
    """
    if direction == "rows":
        # Slice input[:K] (top K rows) — slice on axis 2
        nodes = [
            h.make_node("Constant", [], ["starts"],
                         value=h.make_tensor("starts_v", TensorProto.INT64, [4], [0, 0, 0, 0])),
            h.make_node("Constant", [], ["ends"],
                         value=h.make_tensor("ends_v", TensorProto.INT64, [4], [1, NUM_COLORS, K, MAX_GRID])),
            h.make_node("Constant", [], ["axes"],
                         value=h.make_tensor("axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
            h.make_node("Slice", [INPUT_NAME, "starts", "ends", "axes"], ["top_K"]),
            # Concat along axis 2 (rows): input + top_K
            h.make_node("Concat", [INPUT_NAME, "top_K"], [OUTPUT_NAME], axis=2),
        ]
    else:  # cols
        nodes = [
            h.make_node("Constant", [], ["starts"],
                         value=h.make_tensor("starts_v", TensorProto.INT64, [4], [0, 0, 0, 0])),
            h.make_node("Constant", [], ["ends"],
                         value=h.make_tensor("ends_v", TensorProto.INT64, [4], [1, NUM_COLORS, MAX_GRID, K])),
            h.make_node("Constant", [], ["axes"],
                         value=h.make_tensor("axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
            h.make_node("Slice", [INPUT_NAME, "starts", "ends", "axes"], ["left_K"]),
            h.make_node("Concat", [INPUT_NAME, "left_K"], [OUTPUT_NAME], axis=3),
        ]
    return _make_simple_model(nodes, name=f"concat_repeat_{direction}_{K}")


# ---------------------------------------------------------------------------
# ConditionalSliceColorMapSolver — slice sub-region based on a separator,
# then apply a color map
# ---------------------------------------------------------------------------


class ConditionalSliceColorMapSolver(Solver):
    """Find a separator column (a column of a single non-zero color), slice the
    region to one side of it, and apply a color map.

    Matches task 0520fde7: input 3x7 has a column of 5s at column 3; output is
    the right half (cols 4-6) with color 1 -> 2.
    """
    name = "slice_separators_color_map"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Find the separator structure (must be consistent across pairs)
        sep_info = None  # (axis, index, side, color_map)
        for inp, out in pairs:
            H_in, W_in = inp.shape
            H_out, W_out = out.shape
            # Try horizontal separator (a column of same color)
            found = None
            for sep_idx in range(W_in):
                col = inp[:, sep_idx]
                if (col == col[0]).all() and col[0] != 0:
                    sep_color = int(col[0])
                    # Try left half
                    left = inp[:, :sep_idx]
                    if left.shape == out.shape:
                        # Find color map
                        m = {}
                        valid = True
                        for c in range(10):
                            in_cells = (left == c)
                            if in_cells.any():
                                out_colors = np.unique(out[in_cells])
                                if len(out_colors) != 1:
                                    valid = False
                                    break
                                m[c] = int(out_colors[0])
                        if valid and m:
                            found = ("col", sep_idx, "left", m, sep_color)
                            break
                    # Try right half
                    right = inp[:, sep_idx + 1:]
                    if right.shape == out.shape:
                        m = {}
                        valid = True
                        for c in range(10):
                            in_cells = (right == c)
                            if in_cells.any():
                                out_colors = np.unique(out[in_cells])
                                if len(out_colors) != 1:
                                    valid = False
                                    break
                                m[c] = int(out_colors[0])
                        if valid and m:
                            found = ("col", sep_idx, "right", m, sep_color)
                            break
            if found is None:
                # Try row separator
                for sep_idx in range(H_in):
                    row = inp[sep_idx, :]
                    if (row == row[0]).all() and row[0] != 0:
                        sep_color = int(row[0])
                        top = inp[:sep_idx, :]
                        if top.shape == out.shape:
                            m = {}
                            valid = True
                            for c in range(10):
                                in_cells = (top == c)
                                if in_cells.any():
                                    out_colors = np.unique(out[in_cells])
                                    if len(out_colors) != 1:
                                        valid = False
                                        break
                                    m[c] = int(out_colors[0])
                            if valid and m:
                                found = ("row", sep_idx, "top", m, sep_color)
                                break
                        bot = inp[sep_idx + 1:, :]
                        if bot.shape == out.shape:
                            m = {}
                            valid = True
                            for c in range(10):
                                in_cells = (bot == c)
                                if in_cells.any():
                                    out_colors = np.unique(out[in_cells])
                                    if len(out_colors) != 1:
                                        valid = False
                                        break
                                    m[c] = int(out_colors[0])
                            if valid and m:
                                found = ("row", sep_idx, "bottom", m, sep_color)
                                break
            if found is None:
                return None
            if sep_info is None:
                sep_info = found
            elif sep_info != found:
                return None
        if sep_info is None:
            return None
        axis, idx, side, mapping, sep_color = sep_info
        return _slice_colormap_model(axis, idx, side, mapping)


def _slice_colormap_model(axis: str, idx: int, side: str, mapping: dict[int, int]) -> onnx.ModelProto:
    """Build a network that slices the input on one side of a separator line
    and applies a color map to the result.

    axis = "col" or "row"; idx = separator index; side = "left"/"right"/"top"/"bottom"
    """
    # Step 1: slice
    if axis == "col":
        if side == "left":
            starts = [0, 0, 0, 0]
            ends = [1, NUM_COLORS, MAX_GRID, idx]
        else:  # right
            starts = [0, 0, 0, idx + 1]
            ends = [1, NUM_COLORS, MAX_GRID, MAX_GRID]
    else:  # row
        if side == "top":
            starts = [0, 0, 0, 0]
            ends = [1, NUM_COLORS, idx, MAX_GRID]
        else:  # bottom
            starts = [0, 0, idx + 1, 0]
            ends = [1, NUM_COLORS, MAX_GRID, MAX_GRID]
    # Build color map conv weight (1x1)
    W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
    full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
    for frm, to in full_map.items():
        W[to, frm, 0, 0] = 1.0

    nodes = [
        h.make_node("Constant", [], ["starts"],
                     value=h.make_tensor("starts_v", TensorProto.INT64, [4], starts)),
        h.make_node("Constant", [], ["ends"],
                     value=h.make_tensor("ends_v", TensorProto.INT64, [4], ends)),
        h.make_node("Constant", [], ["axes"],
                     value=h.make_tensor("axes_v", TensorProto.INT64, [4], [0, 1, 2, 3])),
        h.make_node("Slice", [INPUT_NAME, "starts", "ends", "axes"], ["sliced"]),
        # Color map: 1x1 conv
        h.make_node("Conv", ["sliced", "cmW"], ["cm_out"],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        h.make_node("Identity", ["cm_out"], [OUTPUT_NAME]),
    ]
    inits = [h.make_tensor("cmW", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist())]
    return _make_simple_model(nodes, initializers=inits, name="slice_colormap")
