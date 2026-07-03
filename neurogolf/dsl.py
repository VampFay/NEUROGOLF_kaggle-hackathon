"""
neurogolf/dsl.py — A small DSL of ARC primitives that emit tiny ONNX networks.

Every builder returns an `onnx.ModelProto` ready to be saved and validated.

Convention (see constants.py):
  Input  "input":  (1, 10, 30, 30) float32 — one-hot encoded grid
  Output "output": (1, 10, 30, 30) float32 — argmax over channel gives the grid

Strategy: each DSL primitive builds a tiny ONNX graph by hand using
`onnx.helper` so we have full control over parameter count and file size.

The primitives compose via `chain()` which glues two networks by renaming
the intermediate tensor.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto, numpy_helper

from .constants import IO_SHAPE, INPUT_NAME, OUTPUT_NAME, MAX_GRID, NUM_COLORS, BANNED_OPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor_value_info(name: str, shape=IO_SHAPE, dtype=TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(name, dtype, list(shape))


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return numpy_helper.from_array(arr, name=name)


def _empty_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto],
                 intermediates: list[str] | None = None) -> onnx.ModelProto:
    """Build a minimal ONNX model with the standard I/O signature."""
    intermediates = intermediates or []
    graph = onnx.helper.make_graph(
        nodes,
        name="neurogolf",
        inputs=[_make_tensor_value_info(INPUT_NAME)],
        outputs=[_make_tensor_value_info(OUTPUT_NAME)],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="neurogolf-dsl",
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    # Statically-defined shapes — no symbolic dims
    model.ir_version = 8
    return model


def count_params(model: onnx.ModelProto) -> int:
    """Total number of parameters (elements in all initializers)."""
    n = 0
    for init in model.graph.initializer:
        n += int(np.prod(init.dims)) if init.dims else 1
    return n


def model_size_bytes(model: onnx.ModelProto) -> int:
    """Serialized file size in bytes."""
    return len(model.SerializeToString())


def model_cost(model: onnx.ModelProto) -> int:
    """Total cost = #params + #bytes."""
    return count_params(model) + model_size_bytes(model)


def model_score(model: onnx.ModelProto) -> float:
    import math
    cost = model_cost(model)
    return max(1.0, 25.0 - math.log(cost))


def uses_banned_ops(model: onnx.ModelProto) -> set[str]:
    used = {n.op_type for n in model.graph.node}
    return used & BANNED_OPS


def has_dynamic_shapes(model: onnx.ModelProto) -> bool:
    """Check whether any value info has a symbolic (non-integer) dim."""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:  # symbolic
                return True
    return False


def validate_model_structure(model: onnx.ModelProto) -> tuple[bool, str]:
    """Static structural checks: shape, banned ops, size."""
    if has_dynamic_shapes(model):
        return False, "Dynamic shapes detected"
    banned = uses_banned_ops(model)
    if banned:
        return False, f"Banned ops: {banned}"
    sz = model_size_bytes(model)
    if sz > 1_440_000:
        return False, f"File too large: {sz} bytes > 1.44MB"
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        return False, f"onnx.checker: {e}"
    return True, "OK"


# ---------------------------------------------------------------------------
# Primitive: Identity (zero-parameter pass-through)
# ---------------------------------------------------------------------------


def identity() -> onnx.ModelProto:
    """Output = Input. Cost ~ 0 params, ~150 bytes."""
    nodes = [onnx.helper.make_node("Identity", [INPUT_NAME], [OUTPUT_NAME])]
    return _empty_model(nodes, [])


# ---------------------------------------------------------------------------
# Primitive: Single conv2d layer (matches competition example)
# ---------------------------------------------------------------------------


def single_layer_conv2d(weight: np.ndarray, bias: np.ndarray | None = None) -> onnx.ModelProto:
    """Single 2D convolution layer.

    weight: (out_C, in_C, kH, kW) float32
    bias:   (out_C,) float32 or None

    The network applies Conv to the one-hot input and writes the result
    directly to output (no activation, no argmax — the validator argmaxes).
    """
    assert weight.ndim == 4
    out_C, in_C, kH, kW = weight.shape
    assert in_C == NUM_COLORS, f"in_channels must be {NUM_COLORS}, got {in_C}"
    assert out_C == NUM_COLORS, f"out_channels must be {NUM_COLORS}, got {out_C}"

    pad_h = kH // 2
    pad_w = kW // 2

    init_w = _make_initializer("conv_w", weight.astype(np.float32))
    inits = [init_w]
    node_inputs = [INPUT_NAME, "conv_w"]

    if bias is not None:
        init_b = _make_initializer("conv_b", bias.astype(np.float32))
        inits.append(init_b)
        node_inputs.append("conv_b")

    nodes = [
        onnx.helper.make_node(
            "Conv",
            inputs=node_inputs,
            outputs=[OUTPUT_NAME],
            kernel_shape=[kH, kW],
            pads=[pad_h, pad_w, pad_h, pad_w],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
    ]
    return _empty_model(nodes, inits)


def conv_weight_from_fn(weight_fn: Callable[[int, int, tuple[int, int]], float],
                        kernel_size: int = 3) -> np.ndarray:
    """Build a conv weight tensor from a Python weight function (as in the
    competition example).

    weight_fn(channel_out, channel_in, (dh, dw)) -> float
    where (dh, dw) ranges over (-(k//2)..k//2, -(k//2)..k//2)
    """
    k = kernel_size
    half = k // 2
    W = np.zeros((NUM_COLORS, NUM_COLORS, k, k), dtype=np.float32)
    for co in range(NUM_COLORS):
        for ci in range(NUM_COLORS):
            for dh in range(-half, half + 1):
                for dw in range(-half, half + 1):
                    W[co, ci, dh + half, dw + half] = float(weight_fn(co, ci, (dh, dw)))
    return W


# ---------------------------------------------------------------------------
# Primitive: Color map (a single conv2d that implements a lookup table)
# ---------------------------------------------------------------------------


def color_map(mapping: dict[int, int]) -> onnx.ModelProto:
    """Map each color c to mapping[c] (default: identity if not in mapping).

    Implemented as a 1x1 conv where W[to, from, 0, 0] = 1 if mapping[from]==to
    else 0, plus a bias that ensures the target color wins argmax.

    To make argmax work robustly: for each input color `from`, we want the
    output channel `mapping[from]` to have a large positive value and all
    others to have 0.  So:
      W[to, from, 0, 0] = 1  if mapping[from] == to else 0
      bias[to] = 0
    Then output[to, h, w] = 1 if mapping[grid[h,w]] == to else 0.  Argmax
    picks the channel that's 1, i.e. mapping[grid[h,w]].  Cost = 100 params.
    """
    full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
    W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
    for frm, to in full_map.items():
        W[to, frm, 0, 0] = 1.0
    return single_layer_conv2d(W, bias=None)


# ---------------------------------------------------------------------------
# Primitive: Chain two networks
# ---------------------------------------------------------------------------


def chain(models: list[onnx.ModelProto]) -> onnx.ModelProto:
    """Compose models sequentially. The output of model[i] feeds into model[i+1]."""
    if len(models) == 1:
        return models[0]
    # Merge all nodes/initializers into one graph, renaming intermediates.
    all_nodes: list[onnx.NodeProto] = []
    all_inits: list[onnx.TensorProto] = []
    seen_init_names: set[str] = set()
    current_input = INPUT_NAME
    for i, m in enumerate(models):
        # Rename initializers to avoid clashes
        prefix = f"m{i}_"
        for init in m.graph.initializer:
            new_name = prefix + init.name
            new_init = onnx.TensorProto()
            new_init.CopyFrom(init)
            new_init.name = new_name
            all_inits.append(new_init)
            seen_init_names.add(new_name)

        # Rename nodes
        for node in m.graph.node:
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            # Rename inputs
            new_inputs = []
            for inp in list(new_node.input):
                if inp == INPUT_NAME:
                    new_inputs.append(current_input)
                elif inp in seen_init_names:
                    # already prefixed
                    new_inputs.append(inp)
                elif any(init.name == inp for init in m.graph.initializer):
                    # this is an initializer in this model — needs prefix
                    new_inputs.append(prefix + inp)
                else:
                    new_inputs.append(inp)
            new_inputs = [prefix + inp if any(init.name == inp for init in m.graph.initializer) else
                          (current_input if inp == INPUT_NAME else inp)
                          for inp in new_node.input]
            new_node.input[:] = new_inputs

            # Rename outputs
            new_outputs = []
            for out in list(new_node.output):
                if out == OUTPUT_NAME:
                    new_outputs.append(f"mid_{i}" if i < len(models) - 1 else OUTPUT_NAME)
                else:
                    new_outputs.append(prefix + out)
            new_node.output[:] = new_outputs

            all_nodes.append(new_node)

        # Update current_input for next model
        if i < len(models) - 1:
            current_input = f"mid_{i}"

    return _empty_model(all_nodes, all_inits)


# ---------------------------------------------------------------------------
# Primitive: Bias-only "constant" network (output a fixed grid)
# ---------------------------------------------------------------------------


def constant_grid(grid: list[list[int]]) -> onnx.ModelProto:
    """Output a constant grid regardless of input. Uses a 1x1 conv with bias
    equal to the one-hot encoding of `grid` and weight=0.

    Cost: 10 params (bias vector per spatial location is not feasible with conv
    alone — we need a tensor constant of shape (1, 10, 30, 30) which is 9000
    floats = 36000 bytes, scoring ~6.6).  A cheaper approach: use a Constant
    op directly.  Cost = 9000 floats = 36000 bytes, score ~ 6.6.

    This is only worth it for very small constant answers; usually a
    transformation is cheaper.
    """
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    const = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
    for c in range(NUM_COLORS):
        const[0, c, :H, :W] = (arr == c).astype(np.float32)
    init = _make_initializer("const", const)
    nodes = [onnx.helper.make_node("Identity", ["const"], [OUTPUT_NAME])]
    return _empty_model(nodes, [init])


# ---------------------------------------------------------------------------
# Primitive: Threshold / argmax layer (for tasks needing discretization)
# ---------------------------------------------------------------------------


def argmax_over_channels() -> onnx.ModelProto:
    """Argmax over channel dim and re-one-hot. Useful as a final cleanup stage
    after a logits-producing conv.

    ArgMax(axis=1, keepdims=False) -> (1, 30, 30) int64 with values 0..9
    OneHot(axis=1, depth=10)       -> (1, 10, 30, 30) float32
    """
    nodes = [
        onnx.helper.make_node(
            "ArgMax", [INPUT_NAME], ["am"],
            axis=1, keepdims=False, select_last_index=0,
        ),
        onnx.helper.make_node("Cast", ["am"], ["am_i64"], to=TensorProto.INT64),
        onnx.helper.make_node("Constant", [], ["depth"],
                              value=onnx.helper.make_tensor("depth_v", TensorProto.INT64, [], [NUM_COLORS])),
        # values tensor of shape (2,): [off_value, on_value]
        onnx.helper.make_node("Constant", [], ["values"],
                              value=onnx.helper.make_tensor("values_v", TensorProto.FLOAT, [2], [0.0, 1.0])),
        onnx.helper.make_node(
            "OneHot",
            ["am_i64", "depth", "values"],
            ["oh"],
            axis=1,
        ),
        onnx.helper.make_node("Identity", ["oh"], [OUTPUT_NAME]),
    ]
    return _empty_model(nodes, [])


# ---------------------------------------------------------------------------
# Primitive: Masked / gated composition
# ---------------------------------------------------------------------------


def mask_apply(mask_color: int) -> onnx.ModelProto:
    """Pass through the input only at cells where input has color `mask_color`;
    zero elsewhere. Implemented as element-wise multiplication with a 1x1 conv
    that produces a per-cell mask channel.

    Actually simpler: use Mul with a constant mask computed from input.
    A 1x1 conv producing a single channel = sum over input channels with weight
    [1 if c==mask_color else 0 for c in range(10)] gives a (1, 1, 30, 30) mask.
    We then expand the mask to 10 channels and multiply.

    But Mul of (1,10,30,30) * (1,1,30,30) broadcasts.  That works.
    Cost: 10 params for the conv, plus one Mul node.
    """
    W = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    W[0, mask_color, 0, 0] = 1.0
    inits = [_make_initializer("mask_w", W)]
    nodes = [
        onnx.helper.make_node(
            "Conv", [INPUT_NAME, "mask_w"], ["mask"],
            kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
        ),
        onnx.helper.make_node("Mul", [INPUT_NAME, "mask"], [OUTPUT_NAME]),
    ]
    return _empty_model(nodes, inits)


# ---------------------------------------------------------------------------
# Primitive: Replace color c1 with c2 only where the cell currently is c1
# (Same as color_map but more explicit; useful when the change is sparse.)
# ---------------------------------------------------------------------------


def replace_color(from_c: int, to_c: int) -> onnx.ModelProto:
    m = {from_c: to_c}
    return color_map(m)


# ---------------------------------------------------------------------------
# Primitive: Multi-layer conv stack (for tasks needing local reasoning)
# ---------------------------------------------------------------------------


def conv_stack(layers: list[tuple[np.ndarray, np.ndarray | None, str | None]]) -> onnx.ModelProto:
    """Build a multi-layer conv stack. Each layer is (weight, bias, activation).

    activation: None, "relu", "sigmoid", or "tanh"

    The final layer's output goes to OUTPUT_NAME.
    """
    all_nodes = []
    all_inits = []
    current = INPUT_NAME
    for i, (W, B, act) in enumerate(layers):
        w_name = f"W{i}"
        all_inits.append(_make_initializer(w_name, W.astype(np.float32)))
        node_inputs = [current, w_name]
        if B is not None:
            b_name = f"B{i}"
            all_inits.append(_make_initializer(b_name, B.astype(np.float32)))
            node_inputs.append(b_name)

        out_C, in_C, kH, kW = W.shape
        pad_h = kH // 2
        pad_w = kW // 2
        conv_out = f"conv{i}"
        nodes = [onnx.helper.make_node(
            "Conv", node_inputs, [conv_out],
            kernel_shape=[kH, kW], pads=[pad_h, pad_w, pad_h, pad_w],
            strides=[1, 1], dilations=[1, 1], group=1,
        )]
        current = conv_out
        if act == "relu":
            act_out = f"act{i}"
            nodes.append(onnx.helper.make_node("Relu", [conv_out], [act_out]))
            current = act_out
        elif act == "sigmoid":
            act_out = f"act{i}"
            nodes.append(onnx.helper.make_node("Sigmoid", [conv_out], [act_out]))
            current = act_out
        elif act == "tanh":
            act_out = f"act{i}"
            nodes.append(onnx.helper.make_node("Tanh", [conv_out], [act_out]))
            current = act_out
        all_nodes.extend(nodes)

    # Final identity to OUTPUT_NAME if needed
    if current != OUTPUT_NAME:
        all_nodes.append(onnx.helper.make_node("Identity", [current], [OUTPUT_NAME]))

    return _empty_model(all_nodes, all_inits)


# ---------------------------------------------------------------------------
# Utility: save / load
# ---------------------------------------------------------------------------


def save_model(model: onnx.ModelProto, path: str) -> None:
    onnx.save(model, path)


def load_model(path: str) -> onnx.ModelProto:
    return onnx.load(path)
