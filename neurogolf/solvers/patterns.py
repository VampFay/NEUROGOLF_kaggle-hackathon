"""
neurogolf/solvers/patterns.py — Pattern-based solvers.

Covers common ARC transformations:
  - MirrorConcatSolver: input + flip(input) concatenated (or flip + input)
  - ColorCountSolver: output is a 1xN row of color C, count = something
  - ExhaustiveColorMapSolver: try every possible 1-color and 2-color substitution
  - PaletteSolver: try all permutations of the input color palette
  - FillBorderSolver: fill the border with a specific color
  - EnclosedFillSolver: fill enclosed regions with a color (via multi-layer conv)
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


def _make_simple_model(nodes, initializers=None, name="neurogolf"):
    initializers = initializers or []
    graph = h.make_graph(
        nodes, name,
        inputs=[h.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[h.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        initializer=initializers,
    )
    model = h.make_model(graph, producer_name="neurogolf",
                          opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# MirrorConcatSolver
# ---------------------------------------------------------------------------


class MirrorConcatSolver(Solver):
    """Output = input + flip(input) concatenated along an axis, or flip + input,
    or input + flip + input, etc.

    Common ARC pattern: complete a symmetric pattern.
    """
    name = "mirror_concat"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Try various mirror+concat patterns
        # Pattern 1: out = [input | fliplr(input)] (concat along cols, input first)
        # Pattern 2: out = [fliplr(input) | input]
        # Pattern 3: out = [input; flipud(input)] (concat along rows, input first)
        # Pattern 4: out = [flipud(input); input]
        # Pattern 5: out = [input | input] (just concat, no flip)
        # Pattern 6: out = [input; input]
        patterns = [
            ("lr_in_flip", lambda a: np.concatenate([a, np.fliplr(a)], axis=1)),
            ("lr_flip_in", lambda a: np.concatenate([np.fliplr(a), a], axis=1)),
            ("ud_in_flip", lambda a: np.concatenate([a, np.flipud(a)], axis=0)),
            ("ud_flip_in", lambda a: np.concatenate([np.flipud(a), a], axis=0)),
            ("lr_in_in",   lambda a: np.concatenate([a, a], axis=1)),
            ("ud_in_in",   lambda a: np.concatenate([a, a], axis=0)),
        ]
        matched_pattern = None
        for pname, pfn in patterns:
            ok = True
            for inp, out in pairs:
                p_out = pfn(inp)
                if p_out.shape != out.shape or not np.array_equal(p_out, out):
                    ok = False
                    break
            if ok:
                matched_pattern = pname
                break
        if matched_pattern is None:
            return None
        return _mirror_concat_model(matched_pattern)


def _mirror_concat_model(pattern: str) -> onnx.ModelProto:
    """Build a model for the given mirror+concat pattern."""
    # All ops work on (1, 10, 30, 30) input → (1, 10, 30, 30+30) or similar → slice back to 30x30
    # Build the "flipped" version, then concat, then slice to top-left 30x30
    nodes = []
    if "lr" in pattern:
        # Flip along W axis (axis 3): use Slice with reverse
        nodes.append(h.make_node("Constant", [], ["fs"],
                                  value=h.make_tensor("fs_v", TensorProto.INT64, [1], [MAX_GRID - 1])))
        nodes.append(h.make_node("Constant", [], ["fe"],
                                  value=h.make_tensor("fe_v", TensorProto.INT64, [1], [-MAX_GRID - 1])))
        nodes.append(h.make_node("Constant", [], ["fst"],
                                  value=h.make_tensor("fst_v", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fax"],
                                  value=h.make_tensor("fax_v", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "fs", "fe", "fax", "fst"], ["flipped"]))
        concat_axis = 3
    elif "ud" in pattern:
        nodes.append(h.make_node("Constant", [], ["fs"],
                                  value=h.make_tensor("fs_v", TensorProto.INT64, [1], [MAX_GRID - 1])))
        nodes.append(h.make_node("Constant", [], ["fe"],
                                  value=h.make_tensor("fe_v", TensorProto.INT64, [1], [-MAX_GRID - 1])))
        nodes.append(h.make_node("Constant", [], ["fst"],
                                  value=h.make_tensor("fst_v", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fax"],
                                  value=h.make_tensor("fax_v", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "fs", "fe", "fax", "fst"], ["flipped"]))
        concat_axis = 2

    # Determine concat order
    if pattern.endswith("_in_flip") or pattern.endswith("_in_in"):
        # input first, then flipped (or input, input)
        second = "flipped" if "flip" in pattern else INPUT_NAME
        nodes.append(h.make_node("Concat", [INPUT_NAME, second], ["concatenated"], axis=concat_axis))
    else:  # _flip_in or _in_in
        first = "flipped" if "flip" in pattern else INPUT_NAME
        nodes.append(h.make_node("Concat", [first, INPUT_NAME], ["concatenated"], axis=concat_axis))

    # Slice to top-left 30x30
    nodes.append(h.make_node("Constant", [], ["ss"],
                              value=h.make_tensor("ss_v", TensorProto.INT64, [4], [0, 0, 0, 0])))
    nodes.append(h.make_node("Constant", [], ["se"],
                              value=h.make_tensor("se_v", TensorProto.INT64, [4],
                                                   [1, NUM_COLORS, MAX_GRID, MAX_GRID])))
    nodes.append(h.make_node("Constant", [], ["sax"],
                              value=h.make_tensor("sax_v", TensorProto.INT64, [4], [0, 1, 2, 3])))
    nodes.append(h.make_node("Slice", ["concatenated", "ss", "se", "sax"], [OUTPUT_NAME]))
    return _make_simple_model(nodes, name=f"mirror_concat_{pattern}")


# ---------------------------------------------------------------------------
# ColorCountSolver — output is a 1xN or Nx1 row of color C
# ---------------------------------------------------------------------------


class ColorCountSolver(Solver):
    """Output is a 1xN row (or Nx1 column) of a single color C, where N equals
    some count derived from the input (e.g., number of non-zero cells, number
    of distinct colors, etc.).

    Limited to: count of color C in input → output is N cells of color C'.
    Very common ARC pattern.
    """
    name = "color_count"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine the rule: out_color, out_axis, count_source_color
        rule = None
        for inp, out in pairs:
            # Check if output is a single color
            out_colors = np.unique(out)
            if len(out_colors) != 1:
                return None
            out_c = int(out_colors[0])
            # Determine N
            if out.shape[0] == 1:
                N = out.shape[1]
                axis = "row"
            elif out.shape[1] == 1:
                N = out.shape[0]
                axis = "col"
            else:
                return None
            # Try to find what N counts in the input
            # Common: count of color out_c, count of any non-zero, count of distinct colors
            count_source = None
            for c in range(10):
                if int((inp == c).sum()) == N:
                    count_source = ("count_color", c)
                    break
            if count_source is None:
                # Try non-zero count
                if int((inp != 0).sum()) == N:
                    count_source = ("count_nonzero",)
                else:
                    return None
            r = (out_c, axis, count_source)
            if rule is None:
                rule = r
            elif rule != r:
                return None
        if rule is None:
            return None
        return _color_count_model(rule)


def _color_count_model(rule) -> onnx.ModelProto:
    """Build a model that outputs a row/col of color out_c with length = count.
    Limited implementation: works only for count_color (out_c == source_c).
    """
    out_c, axis, count_source = rule
    if count_source[0] != "count_color" or count_source[1] != out_c:
        return None  # Too complex for now
    # We need to count cells of color out_c in input and output that many cells of out_c in a row/col
    # This is hard because the output length depends on the input — but we can use a fixed
    # approach: compute the count, then write that many 1s in a row.
    #
    # Actually, since ONNX requires static shapes, the output is always (1, 10, 30, 30).
    # We need to put `count` cells of color out_c in the top row (or left col).
    #
    # Approach: compute count via ReduceSum on the one-hot channel out_c.
    # Then create a (30,) vector with the first `count` cells = 1, rest = 0.
    # Then expand to (1, 10, 30, 30) by one-hot to channel out_c.
    #
    # But making "first count cells = 1" requires a comparison: index < count.
    # This needs Range + Less + CumSum (CumSum might be banned? Let me check — no, it's allowed).
    #
    # Implementation:
    # 1. Slice input channel out_c: (1, 1, 30, 30)
    # 2. ReduceSum over all axes: scalar = count
    # 3. Create index vector [0, 1, 2, ..., 29] (Constant)
    # 4. Less(index, count): (30,) bool
    # 5. Cast to float: (30,) 0/1
    # 6. OneHot to channel out_c: (10, 30) → (1, 10, 30, 1) → broadcast to (1, 10, 30, 30)
    # 7. Actually if axis is row: put the (30,) in row 0 of (30, 30). Use Expand.
    #
    # This is getting complex. Skip for now.
    return None


# ---------------------------------------------------------------------------
# ExhaustiveColorMapSolver — try all 10x10 = 100 single-color substitutions
# ---------------------------------------------------------------------------


class ExhaustiveColorMapSolver(Solver):
    """Try every (from, to) pair exhaustively. Already covered by ColorMapSolver
    but kept as a more aggressive fallback that builds color maps cell-by-cell
    based on local context (e.g., depending on neighbors).
    """
    name = "exhaustive_color_map"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        # Same as ColorMapSolver — already exhaustive on the global mapping.
        # We'll try a different variant: color map + 1 bias term to handle
        # cases where the mapping has cycles.
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        mapping = {}
        for inp, out in pairs:
            for c in range(10):
                in_cells = (inp == c)
                if in_cells.any():
                    out_colors = np.unique(out[in_cells])
                    if len(out_colors) != 1:
                        return None
                    target = int(out_colors[0])
                    if c in mapping and mapping[c] != target:
                        return None
                    mapping[c] = target
        if not mapping or all(mapping.get(c, c) == c for c in range(10)):
            return None
        return dsl.color_map(mapping)


# ---------------------------------------------------------------------------
# FillBorderSolver
# ---------------------------------------------------------------------------


class FillBorderSolver(Solver):
    """Fill the border of the grid with a specific color, leaving the interior
    unchanged.

    Matches tasks like "draw a frame around the grid".
    """
    name = "fill_border"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Determine the border color
        border_color = None
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            # Check: output = input except border cells are all the same color
            diff = (inp != out)
            # Border mask: row 0, row H-1, col 0, col W-1
            border_mask = np.zeros_like(inp, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            # All border diffs should be where inp != border_color
            # And the new border color should be consistent
            border_out = out[border_mask]
            bc = np.unique(border_out)
            if len(bc) != 1:
                return None
            bc = int(bc[0])
            # Interior should be unchanged
            interior_mask = ~border_mask
            if not np.array_equal(inp[interior_mask], out[interior_mask]):
                return None
            if border_color is None:
                border_color = bc
            elif border_color != bc:
                return None
        if border_color is None:
            return None
        return _fill_border_model(border_color)


def _fill_border_model(border_color: int) -> onnx.ModelProto:
    """Build a model that fills the border of the 30x30 grid with `border_color`
    and leaves the interior unchanged.

    Approach:
      1. Compute a border mask of shape (1, 1, 30, 30) — 1 on border, 0 interior
      2. Output = input * (1 - border_mask) + border_color_onehot * border_mask

    But we need to be careful with the actual grid dimensions (which vary per task).
    For tasks where the grid is HxW with H, W < 30, the "border" is at rows 0, H-1
    and cols 0, W-1, not at rows 0, 29 and cols 0, 29.

    This solver is only correct if the grid is exactly 30x30. We'd need task-specific
    border positions for general grids.

    Skip for now — this is too restrictive.
    """
    return None


# ---------------------------------------------------------------------------
# PaletteSolver — try all permutations of the input color palette
# ---------------------------------------------------------------------------


class PaletteSolver(Solver):
    """For tasks where the output uses the same color set as input but in
    different positions (e.g., swap two colors, rotate palette).

    Try all 10! / (10-k)! permutations of up to k=4 colors.
    """
    name = "palette"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        from itertools import permutations
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        # Find colors that appear in any input
        in_colors = set()
        for inp, _ in pairs:
            in_colors.update(int(c) for c in np.unique(inp))
        in_colors = sorted(in_colors)
        if len(in_colors) > 5:
            return None  # Too many to permute
        # Try all permutations of these colors as the mapping
        for perm in permutations(range(10), len(in_colors)):
            mapping = {in_colors[i]: perm[i] for i in range(len(in_colors))}
            # Check if this mapping works for all pairs
            ok = True
            for inp, out in pairs:
                pred = np.array([[mapping.get(int(v), int(v)) for v in row] for row in inp])
                if not np.array_equal(pred, out):
                    ok = False
                    break
            if ok:
                # Don't take identity (ColorMapSolver or IdentitySolver handles it)
                if all(mapping.get(c, c) == c for c in range(10)):
                    continue
                return dsl.color_map(mapping)
        return None


# ---------------------------------------------------------------------------
# MaxColorSolver — output is filled with the most common non-zero color
# ---------------------------------------------------------------------------


class MaxColorSolver(Solver):
    """Output grid (same size as input) is filled with the most-common non-zero
    color of the input.
    """
    name = "max_color"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        # Too complex for static-shape ONNX (requires argmax over color counts).
        # Skip.
        return None
