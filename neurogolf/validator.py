"""
neurogolf/validator.py — Local validator mirroring the competition's checks.

The validator does two things:
  1. Structural validation: shape constraints, banned ops, file size, onnx.checker
  2. Functional validation: run the network on all task pairs and check the
     output matches the expected ARC output (after argmax + crop).

The competition mentions a "small private benchmark suite" beyond the public
tasks. We can't replicate that here, but functional correctness on the public
training pairs (especially the test pair, which is held out in the JSON) is
a strong proxy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort

from . import dsl
from .arc_data import grid_to_onehot, onehot_to_grid, get_pairs, load_task
from .constants import IO_SHAPE, MAX_GRID, NUM_COLORS


def structural_check(model: onnx.ModelProto) -> tuple[bool, str]:
    """Check structural constraints (no dynamic shapes, no banned ops, size limit)."""
    return dsl.validate_model_structure(model)


def run_model(model: onnx.ModelProto, input_grid: list[list[int]]) -> np.ndarray:
    """Run model on a single input grid, return the (1, 10, 30, 30) output."""
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    inp = grid_to_onehot(input_grid)
    out = sess.run(None, {"input": inp})[0]
    return out


def functional_check(model: onnx.ModelProto, task: dict, verbose: bool = False) -> tuple[bool, list[str]]:
    """Check if model correctly transforms every (input, output) pair in the task.

    Returns (all_correct, list_of_failure_descriptions).
    """
    failures = []
    pairs = get_pairs(task)
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as e:
        return False, [f"Failed to load ONNX session: {e}"]

    for i, (inp_arr, exp_arr) in enumerate(pairs):
        inp = grid_to_onehot(inp_arr.tolist())
        try:
            out = sess.run(None, {"input": inp})[0]
        except Exception as e:
            failures.append(f"Pair {i}: inference failed: {e}")
            continue
        H, W = exp_arr.shape
        pred_grid = np.array(onehot_to_grid(out, H, W))
        if not np.array_equal(pred_grid, exp_arr):
            if verbose:
                print(f"Pair {i} MISMATCH:")
                print(f"  Input ({inp_arr.shape}):\n{inp_arr}")
                print(f"  Expected ({exp_arr.shape}):\n{exp_arr}")
                print(f"  Predicted:\n{pred_grid}")
            failures.append(f"Pair {i}: mismatch")
    return len(failures) == 0, failures


def evaluate_model(model: onnx.ModelProto, task: dict) -> dict:
    """Full evaluation: structural + functional + cost + score."""
    s_ok, s_msg = structural_check(model)
    f_ok, f_msgs = functional_check(model, task)
    cost = dsl.model_cost(model)
    return {
        "structural_ok": s_ok,
        "structural_msg": s_msg,
        "functional_ok": f_ok,
        "functional_failures": f_msgs,
        "params": dsl.count_params(model),
        "size_bytes": dsl.model_size_bytes(model),
        "cost": cost,
        "score": dsl.model_score(model),
        "eligible_for_points": s_ok and f_ok,
    }


def print_eval(model: onnx.ModelProto, task: dict) -> None:
    e = evaluate_model(model, task)
    print(f"Structural: {'OK' if e['structural_ok'] else 'FAIL: ' + e['structural_msg']}")
    print(f"Functional: {'OK' if e['functional_ok'] else 'FAIL: ' + str(e['functional_failures'][:3])}")
    print(f"Params:     {e['params']}")
    print(f"Size:       {e['size_bytes']} bytes")
    print(f"Cost:       {e['cost']}")
    print(f"Score:      {e['score']:.3f}")
    print(f"Eligible:   {e['eligible_for_points']}")
