"""
neurogolf/solvers/transforms.py — Geometric transforms (flip, rotate, transpose).

These need careful handling because the output may have swapped H/W (rotate 90,
transpose). Since our network must output (1, 10, 30, 30) regardless, the
transformed grid still fits in 30x30 (just possibly transposed), and the
validator's crop uses the expected output dims — so it works.
"""
from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto

from .base import Solver
from .. import dsl
from ..arc_data import get_pairs
from ..constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS


def _slice_op(input_name: str, output_name: str, axis: int, reverse: bool = False) -> onnx.NodeProto:
    """Create a Slice node that reverses along an axis (if reverse) or is identity."""
    if not reverse:
        return onnx.helper.make_node("Identity", [input_name], [output_name])
    # Slice with negative step
    # starts = [MAX_GRID-1], ends = [-MAX_GRID-1], steps = [-1] for that axis
    # Build per-axis slice; for axes != axis, take all
    axes = [axis]
    starts = onnx.helper.make_tensor("starts_" + output_name, TensorProto.INT64, [1], [MAX_GRID - 1])
    ends = onnx.helper.make_tensor("ends_" + output_name, TensorProto.INT64, [1], [-MAX_GRID - 1])
    steps = onnx.helper.make_tensor("steps_" + output_name, TensorProto.INT64, [1], [-1])
    n_starts = onnx.helper.make_node("Constant", [], ["starts_" + output_name], value=starts)
    n_ends = onnx.helper.make_node("Constant", [], ["ends_" + output_name], value=ends)
    n_steps = onnx.helper.make_node("Constant", [], ["steps_" + output_name], value=steps)
    n_axes = onnx.helper.make_node("Constant", [], ["axes_" + output_name],
                                    value=onnx.helper.make_tensor("axes_v_" + output_name, TensorProto.INT64, [1], [axis]))
    n_slice = onnx.helper.make_node("Slice",
                                     [input_name, "starts_" + output_name, "ends_" + output_name,
                                      "axes_" + output_name, "steps_" + output_name],
                                     [output_name])
    return [n_starts, n_ends, n_steps, n_axes, n_slice]


def flip_horizontal_model() -> onnx.ModelProto:
    """Flip the grid horizontally (left-right reverse along W axis = axis 3)."""
    nodes = _slice_op(INPUT_NAME, OUTPUT_NAME, axis=3, reverse=True)
    if isinstance(nodes, onnx.NodeProto):
        nodes = [nodes]
    graph = onnx.helper.make_graph(
        nodes, "flip_h",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def flip_vertical_model() -> onnx.ModelProto:
    """Flip the grid vertically (top-bottom reverse along H axis = axis 2)."""
    nodes = _slice_op(INPUT_NAME, OUTPUT_NAME, axis=2, reverse=True)
    if isinstance(nodes, onnx.NodeProto):
        nodes = [nodes]
    graph = onnx.helper.make_graph(
        nodes, "flip_v",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def transpose_model() -> onnx.ModelProto:
    """Transpose H and W (axes 2 and 3)."""
    node = onnx.helper.make_node("Transpose", [INPUT_NAME], [OUTPUT_NAME], perm=[0, 1, 3, 2])
    graph = onnx.helper.make_graph(
        [node], "transpose",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def rotate_180_model() -> onnx.ModelProto:
    """Rotate the grid 180 degrees = flip H then flip V."""
    n1 = _slice_op(INPUT_NAME, "mid_v", axis=2, reverse=True)
    if isinstance(n1, onnx.NodeProto):
        n1 = [n1]
    n2 = _slice_op("mid_v", OUTPUT_NAME, axis=3, reverse=True)
    if isinstance(n2, onnx.NodeProto):
        n2 = [n2]
    graph = onnx.helper.make_graph(
        n1 + n2, "rotate_180",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def rotate_90_cw_model() -> onnx.ModelProto:
    """Rotate 90 degrees clockwise = transpose then flip H (along W axis)."""
    n_t = onnx.helper.make_node("Transpose", [INPUT_NAME], ["t"], perm=[0, 1, 3, 2])
    n2 = _slice_op("t", OUTPUT_NAME, axis=3, reverse=True)
    if isinstance(n2, onnx.NodeProto):
        n2 = [n2]
    graph = onnx.helper.make_graph(
        [n_t] + n2, "rotate_90_cw",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def rotate_270_cw_model() -> onnx.ModelProto:
    """Rotate 270 degrees clockwise = transpose then flip V (along H axis)."""
    n_t = onnx.helper.make_node("Transpose", [INPUT_NAME], ["t"], perm=[0, 1, 3, 2])
    n2 = _slice_op("t", OUTPUT_NAME, axis=2, reverse=True)
    if isinstance(n2, onnx.NodeProto):
        n2 = [n2]
    graph = onnx.helper.make_graph(
        [n_t] + n2, "rotate_270_cw",
        inputs=[onnx.helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
        outputs=[onnx.helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(IO_SHAPE))],
    )
    model = onnx.helper.make_model(graph, producer_name="neurogolf",
                                    opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _all_transform_models() -> list[tuple[str, onnx.ModelProto, Callable]]:
    """Return list of (name, model, transform_fn) for all transforms we support."""
    return [
        ("flip_h", flip_horizontal_model(), lambda a: np.fliplr(a)),
        ("flip_v", flip_vertical_model(), lambda a: np.flipud(a)),
        ("transpose", transpose_model(), lambda a: a.T),
        ("rotate_180", rotate_180_model(), lambda a: np.rot90(a, 2)),
        ("rotate_90_cw", rotate_90_cw_model(), lambda a: np.rot90(a, -1)),
        ("rotate_270_cw", rotate_270_cw_model(), lambda a: np.rot90(a, 1)),
    ]


class GeometricTransformSolver(Solver):
    """Try each geometric transform; pick the one that matches all pairs."""
    name = "geom_transform"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        for tname, model, tfn in _all_transform_models():
            ok = True
            for inp, out in pairs:
                # Apply transform to input, place in top-left of 30x30, compare to output
                transformed = tfn(inp)
                if transformed.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(transformed, out):
                    ok = False
                    break
            if ok:
                return model
        return None


class ColorMapThenTransformSolver(Solver):
    """Try color_map then a geometric transform.

    Useful when both colors and positions change.
    """
    name = "color_map_then_transform"

    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        pairs = get_pairs(task)
        # First, find a color map that aligns colors
        # Then check if a transform aligns positions
        mapping: dict[int, int] = {}
        for inp, out in pairs:
            # Try each transform; if any works, store the color-mapped version
            pass

        # Naive approach: try all combinations of (color map) x (transform).
        # The color map can be derived after applying the transform.
        for tname, _, tfn in _all_transform_models():
            # Compute (transformed_input, output) pairs and try to find a color map
            t_pairs = []
            ok_shape = True
            for inp, out in pairs:
                t_inp = tfn(inp)
                if t_inp.shape != out.shape:
                    ok_shape = False
                    break
                t_pairs.append((t_inp, out))
            if not ok_shape:
                continue
            # Find color map
            mapping: dict[int, int] = {}
            valid = True
            for t_inp, out in t_pairs:
                for c in range(10):
                    in_cells = (t_inp == c)
                    if in_cells.any():
                        out_colors = np.unique(out[in_cells])
                        if len(out_colors) != 1:
                            valid = False
                            break
                        target = int(out_colors[0])
                        if c in mapping and mapping[c] != target:
                            valid = False
                            break
                        mapping[c] = target
                if not valid:
                    break
            if not valid or not mapping:
                continue
            # Build the chained model
            cm_model = dsl.color_map(mapping)
            # We need to apply the transform AFTER the color map — but actually
            # transform on the input first, then color map. Order matters!
            # Since color map is a per-cell operation, it commutes with transforms.
            # So we can apply color map first then transform, OR transform first
            # then color map. Either way works. We'll chain as color_map then transform.
            from ..dsl import chain
            # Need to build transform model again (we discarded it above)
            for tname2, t_model, _ in _all_transform_models():
                if tname2 == tname:
                    return chain([cm_model, t_model])
        return None
