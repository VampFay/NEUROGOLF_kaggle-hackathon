"""
neurogolf/solvers/cellular.py — Cellular-automaton-style solvers.

SingleRuleCASolver: tries (X, Y, Z, threshold) rules — cell of color X with
K neighbors of color Y becomes Z.

MultiRuleCASolver: tries to find ALL (Y → Z) rules for empty cells
(cell of color 0 with K neighbors of color Y becomes Z) and combines them
into a single conv-based network.
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


NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbor_count(grid: np.ndarray, color: int, neighbors=NEIGHBORS_8) -> np.ndarray:
    H, W = grid.shape
    count = np.zeros((H, W), dtype=np.int32)
    for dh, dw in neighbors:
        shifted = np.full_like(grid, 0)
        src_i_start = max(0, dh)
        src_i_end = min(H, H + dh)
        src_j_start = max(0, dw)
        src_j_end = min(W, W + dw)
        dst_i_start = max(0, -dh)
        dst_i_end = dst_i_start + (src_i_end - src_i_start)
        dst_j_start = max(0, -dw)
        dst_j_end = dst_j_start + (src_j_end - src_j_start)
        if src_i_end > src_i_start and src_j_end > src_j_start:
            shifted[dst_i_start:dst_i_end, dst_j_start:dst_j_end] = \
                grid[src_i_start:src_i_end, src_j_start:src_j_end]
        count += (shifted == color).astype(np.int32)
    return count


class CellularAutomatonSolver(Solver):
    """Single-rule CA: cell of color X with K neighbors of color Y → Z."""
    name = "cellular_automaton"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None

        in_colors = set()
        out_colors = set()
        for inp, out in pairs:
            in_colors.update(int(c) for c in np.unique(inp))
            out_colors.update(int(c) for c in np.unique(out))

        for X in sorted(in_colors):
            for Y in sorted(in_colors | {0}):
                for Z in sorted(out_colors):
                    if Z == X:
                        continue
                    for threshold in range(1, 9):
                        ok = True
                        for inp, out in pairs:
                            count_Y = _neighbor_count(inp, Y)
                            rule_mask = (inp == X) & (count_Y >= threshold)
                            if not (out[rule_mask] == Z).all():
                                ok = False
                                break
                            non_rule_mask = ~rule_mask
                            if not (out[non_rule_mask] == inp[non_rule_mask]).all():
                                ok = False
                                break
                        if ok:
                            return _ca_model(X, Y, Z, threshold)
        return None


def _ca_model(X: int, Y: int, Z: int, threshold: int) -> onnx.ModelProto:
    W_count = np.zeros((1, NUM_COLORS, 3, 3), dtype=np.float32)
    for dh, dw in NEIGHBORS_8:
        W_count[0, Y, dh + 1, dw + 1] = 1.0

    W_X = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    W_X[0, X, 0, 0] = 1.0

    one_hot_z = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    one_hot_z[0, Z, 0, 0] = 1.0

    nodes = [
        h.make_node("Conv", [INPUT_NAME, "W_count"], ["count_Y"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        h.make_node("Conv", [INPUT_NAME, "W_X"], ["ch_X"],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        h.make_node("Constant", [], ["thr"],
                     value=h.make_tensor("thr_v", TensorProto.FLOAT, [1], [float(threshold)])),
        h.make_node("GreaterOrEqual", ["count_Y", "thr"], ["count_ge_b"]),
        h.make_node("Cast", ["count_ge_b"], ["count_ge_f"], to=TensorProto.FLOAT),
        h.make_node("Constant", [], ["zero"],
                     value=h.make_tensor("zero_v", TensorProto.FLOAT, [1], [0.0])),
        h.make_node("Greater", ["ch_X", "zero"], ["ch_X_pos_b"]),
        h.make_node("Cast", ["ch_X_pos_b"], ["ch_X_pos_f"], to=TensorProto.FLOAT),
        h.make_node("Mul", ["ch_X_pos_f", "count_ge_f"], ["rule_mask_f"]),
        h.make_node("Constant", [], ["one"],
                     value=h.make_tensor("one_v", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])),
        h.make_node("Sub", ["one", "rule_mask_f"], ["not_rule"]),
        h.make_node("Mul", [INPUT_NAME, "not_rule"], ["passthrough"]),
        h.make_node("Mul", ["one_hot_z", "rule_mask_f"], ["rule_z_broadcast"]),
        h.make_node("Add", ["passthrough", "rule_z_broadcast"], [OUTPUT_NAME]),
    ]
    inits = [
        h.make_tensor("W_count", TensorProto.FLOAT,
                       list(W_count.shape), W_count.flatten().tolist()),
        h.make_tensor("W_X", TensorProto.FLOAT,
                       list(W_X.shape), W_X.flatten().tolist()),
        h.make_tensor("one_hot_z", TensorProto.FLOAT,
                       list(one_hot_z.shape), one_hot_z.flatten().tolist()),
    ]
    graph = h.make_graph(
        nodes, f"ca_{X}_{Y}_{Z}_{threshold}",
        inputs=[h.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[h.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        initializer=inits,
    )
    model = h.make_model(graph, producer_name="neurogolf",
                          opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# MultiRuleCASolver — find all (Y → Z) rules for empty-cell fills
# ---------------------------------------------------------------------------


class MultiRuleCASolver(Solver):
    """Find all rules of form: cell of color 0 with >= 1 neighbor of color Y → color Z.

    Tries both 4-neighbor (orthogonal) and 8-neighbor (orthogonal+diagonal) patterns.
    Also tries diagonal-only patterns.

    Builds a single conv that:
      - Preserves non-zero cells (W[c, c, 0, 0] = 2 for c > 0)
      - Preserves zero cells with no non-zero neighbors (W[0, 0, 0, 0] = 1)
      - Fills empty cells based on neighbor colors (W[Z, Y, dh, dw] = 1 for selected neighbors)
      - Uses bias to break ties: bias[0] = -0.5
    """
    name = "multi_rule_ca"

    NEIGHBOR_SETS = {
        "4": NEIGHBORS_4,
        "8": NEIGHBORS_8,
        "diag": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    }

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape:
                return None

        # Try each neighbor set
        for nset_name, neighbors in self.NEIGHBOR_SETS.items():
            mapping = self._find_mapping(pairs, neighbors)
            if mapping is None:
                continue
            # Verify the mapping
            if self._verify(pairs, mapping, neighbors):
                return _multi_rule_ca_model(mapping, neighbors)
        return None

    def _find_mapping(self, pairs, neighbors):
        """Find the (Y → Z) mapping for empty cells. Returns None if inconsistent."""
        mapping = {}
        for inp, out in pairs:
            diff = (inp != out)
            if not diff.any():
                continue
            # All changed cells must be empty (color 0)
            if not (inp[diff] == 0).all():
                return None
            for i, j in zip(*np.where(diff)):
                target_Z = int(out[i, j])
                neighbor_colors = set()
                for dh, dw in neighbors:
                    ni, nj = i + dh, j + dw
                    if 0 <= ni < inp.shape[0] and 0 <= nj < inp.shape[1]:
                        c = int(inp[ni, nj])
                        if c != 0:
                            neighbor_colors.add(c)
                if len(neighbor_colors) != 1:
                    return None
                Y = next(iter(neighbor_colors))
                if Y in mapping and mapping[Y] != target_Z:
                    return None
                mapping[Y] = target_Z
        return mapping if mapping else None

    def _verify(self, pairs, mapping, neighbors) -> bool:
        """Verify the mapping against all pairs."""
        for inp, out in pairs:
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] != 0:
                        if out[i, j] != inp[i, j]:
                            return False
                        continue
                    neighbor_colors = set()
                    for dh, dw in neighbors:
                        ni, nj = i + dh, j + dw
                        if 0 <= ni < inp.shape[0] and 0 <= nj < inp.shape[1]:
                            c = int(inp[ni, nj])
                            if c != 0:
                                neighbor_colors.add(c)
                    expected_Z = 0
                    triggers = [c for c in neighbor_colors if c in mapping]
                    if triggers:
                        zs = set(mapping[c] for c in triggers)
                        if len(zs) > 1:
                            return False
                        expected_Z = mapping[triggers[0]]
                    if int(out[i, j]) != expected_Z:
                        return False
        return True


def _multi_rule_ca_model(mapping: dict[int, int], neighbors) -> onnx.ModelProto:
    """Build a conv-based model implementing multiple (Y → Z) rules with the
    given neighbor set.
    """
    W = np.zeros((NUM_COLORS, NUM_COLORS, 3, 3), dtype=np.float32)
    W[0, 0, 1, 1] = 1.0
    for c in range(1, NUM_COLORS):
        W[c, c, 1, 1] = 2.0
    for Y, Z in mapping.items():
        for dh, dw in neighbors:
            W[Z, Y, dh + 1, dw + 1] = 1.0

    bias = np.zeros(NUM_COLORS, dtype=np.float32)
    bias[0] = -0.5

    inits = [
        h.make_tensor("W", TensorProto.FLOAT, list(W.shape), W.flatten().tolist()),
        h.make_tensor("B", TensorProto.FLOAT, list(bias.shape), bias.flatten().tolist()),
    ]
    nodes = [
        h.make_node("Conv", [INPUT_NAME, "W", "B"], [OUTPUT_NAME],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
    ]
    graph = h.make_graph(
        nodes, f"multi_ca_{'_'.join(f'{y}{z}' for y, z in mapping.items())}",
        inputs=[h.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[h.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        initializer=inits,
    )
    model = h.make_model(graph, producer_name="neurogolf",
                          opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return model
