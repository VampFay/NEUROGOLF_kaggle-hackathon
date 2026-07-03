"""
neurogolf/solvers/simple.py — Simplest, highest-ROI solvers.

These cover the most common ARC patterns with very small networks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx

from .base import Solver
from .. import dsl
from ..arc_data import get_pairs, get_train_pairs


class IdentitySolver(Solver):
    """Solves tasks where output == input."""
    name = "identity"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Need same shapes everywhere AND same content
        for inp, out in pairs:
            if inp.shape != out.shape or not np.array_equal(inp, out):
                return None
        return dsl.identity()


class ColorMapSolver(Solver):
    """Solves tasks where each input color maps to a fixed output color
    (per-pair, all pairs share the same mapping), and shapes match.

    Cost: 100 params (1x1 conv). Score ~ 13.8.
    """
    name = "color_map"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # Shapes must match
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        # Build the color map: input color c -> output color c'
        # All pairs must agree.
        mapping: dict[int, int] = {}
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
        # Identity check (already covered by IdentitySolver but cheap to repeat)
        if all(mapping.get(c, c) == c for c in range(10)):
            return None  # let IdentitySolver take it (smaller)
        return dsl.color_map(mapping)


class ConstantSolver(Solver):
    """Solves tasks where output is a constant grid (same for all inputs).

    Cost: 10*H*W floats. Usually only worthwhile for very small outputs.
    Score is poor but eligibility gives at least 1 point.
    """
    name = "constant"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        outs = [p[1] for p in pairs]
        # All outputs must be identical
        first = outs[0]
        for o in outs[1:]:
            if o.shape != first.shape or not np.array_equal(o, first):
                return None
        return dsl.constant_grid(first.tolist())


class BiasColorMapSolver(Solver):
    """Color map but with a learned bias to make argmax robust.

    Sometimes a plain color map ties (e.g., mapping 5->0 makes 0 tie with the
    original 0).  We add a small bias to break ties.  Cost = 100 + 10.
    """
    name = "bias_color_map"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        mapping: dict[int, int] = {}
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
        if all(mapping.get(c, c) == c for c in range(10)):
            return None
        # Build weight + bias: W[to, from, 0, 0] = 1 if mapping[from]==to else 0
        # Bias: -0.5 for channels that don't receive any color, 0 for those that do.
        # Actually we need to be careful: a channel that is the target for some
        # input color should get value 1, all others 0 — already works with no bias.
        # But if mapping[0] = 5, then channel 0 gets 0 (no input maps to it) and
        # channel 5 gets 1 (when input is 0). Argmax picks 5. OK.
        # The only ambiguous case: multiple input colors mapping to the same target.
        # That's fine, argmax still works (target channel is 1, others 0).
        # So this solver is actually equivalent to ColorMapSolver in most cases.
        # We add bias only if the mapping has a cycle (e.g., 0<->5 swap), where
        # we need a slight asymmetry. But color_map already handles swaps correctly.
        return None  # Disabled — color_map handles all cases


class ReplaceColorSolver(Solver):
    """If only ONE color changes between input and output, and it changes
    consistently, use a single color replacement.

    Cost: same as ColorMapSolver (100 params).  Useful as a sanity fallback.
    """
    name = "replace_color"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None
        # Find colors that change
        changes: dict[int, int] = {}
        for inp, out in pairs:
            diff = inp != out
            if not diff.any():
                continue
            for c in range(10):
                in_cells = (inp == c) & diff
                if in_cells.any():
                    out_colors = np.unique(out[in_cells])
                    if len(out_colors) != 1:
                        return None
                    target = int(out_colors[0])
                    if c in changes and changes[c] != target:
                        return None
                    changes[c] = target
        if not changes:
            return None
        return dsl.color_map(changes)
