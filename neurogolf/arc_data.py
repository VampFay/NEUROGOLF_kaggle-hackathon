"""
neurogolf/arc_data.py — Load and manipulate ARC-AGI task data.
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MAX_GRID, NUM_COLORS


TRAINING_DIR = Path("/home/z/my-project/data/arc_agi/data/training")


def list_task_files() -> list[Path]:
    """Return alphabetically sorted list of all 400 ARC-AGI training task files."""
    return sorted(TRAINING_DIR.glob("*.json"))


def task_id_to_filename(task_id: int) -> str:
    """Map a 1-indexed task ID (1..400) to its ARC task JSON filename (without extension)."""
    files = list_task_files()
    if not (1 <= task_id <= len(files)):
        raise IndexError(f"task_id {task_id} out of range (1..{len(files)})")
    return files[task_id - 1].stem


def filename_to_task_id(filename: str) -> int:
    """Inverse of task_id_to_filename."""
    stem = Path(filename).stem
    files = list_task_files()
    for i, f in enumerate(files, start=1):
        if f.stem == stem:
            return i
    raise KeyError(stem)


def load_task(task_id_or_filename: int | str) -> dict[str, Any]:
    """Load a task by 1-indexed ID or by filename stem (e.g. '007bbfb7')."""
    if isinstance(task_id_or_filename, int):
        path = list_task_files()[task_id_or_filename - 1]
    else:
        path = TRAINING_DIR / f"{task_id_or_filename}.json"
    with open(path) as f:
        return json.load(f)


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    """Convert an ARC grid (list of lists of ints) to a 2D numpy array (H, W)."""
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def array_to_grid(arr: np.ndarray) -> list[list[int]]:
    """Convert a 2D numpy array back to ARC grid format."""
    return arr.astype(int).tolist()


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    """One-hot encode an ARC grid to shape (1, NUM_COLORS, MAX_GRID, MAX_GRID) float32.

    The grid is placed in the top-left corner; remaining cells are zero-padded.
    Color 0 in the grid maps to channel 0 being 1 (this is intentional — color 0
    is a real ARC color, distinct from padding which also happens to be channel 0).

    Note: This means padding (channel-0 zero cells) is indistinguishable from
    actual color-0 cells. This matches the competition's apparent convention
    from the single_layer_conv2d_network example.
    """
    arr = grid_to_array(grid)
    H, W = arr.shape
    out = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
    for c in range(NUM_COLORS):
        out[0, c, :H, :W] = (arr == c).astype(np.float32)
    return out


def onehot_to_grid(onehot: np.ndarray, out_h: int, out_w: int) -> list[list[int]]:
    """Convert a (1, NUM_COLORS, MAX_GRID, MAX_GRID) one-hot/logits tensor to a grid
    of shape (out_h, out_w) by argmax over the channel dim and cropping to top-left.

    If `onehot` is already one-hot (values 0/1) this returns the encoded grid.
    If `onehot` is logits (any float), argmax picks the most-likely color.
    """
    if onehot.ndim == 4:
        onehot = onehot[0]  # (C, H, W)
    grid = onehot.argmax(axis=0)  # (H, W)
    grid = grid[:out_h, :out_w]
    return grid.astype(int).tolist()


def get_pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all (input, output) pairs from a task as (in_array, out_array)."""
    pairs = []
    for p in task.get("train", []) + task.get("test", []):
        pairs.append((grid_to_array(p["input"]), grid_to_array(p["output"])))
    return pairs


def get_train_pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(grid_to_array(p["input"]), grid_to_array(p["output"])) for p in task.get("train", [])]


def get_test_pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(grid_to_array(p["input"]), grid_to_array(p["output"])) for p in task.get("test", [])]


def task_signature(task: dict[str, Any]) -> dict:
    """Compute a summary signature for classification."""
    pairs = get_pairs(task)
    train_pairs = get_train_pairs(task)
    test_pairs = get_test_pairs(task)
    in_sizes = [p[0].shape for p in pairs]
    out_sizes = [p[1].shape for p in pairs]
    sig = {
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "in_sizes": in_sizes,
        "out_sizes": out_sizes,
        "all_same_size": all(s == in_sizes[0] for s in in_sizes + out_sizes),
        "in_eq_out_all": all(p[0].shape == p[1].shape for p in pairs),
        "in_colors": set(),
        "out_colors": set(),
        "all_in_colors": set(),
        "all_out_colors": set(),
    }
    for inp, out in pairs:
        sig["all_in_colors"].update(int(c) for c in np.unique(inp))
        sig["all_out_colors"].update(int(c) for c in np.unique(out))
        sig["in_colors"].add(frozenset(int(c) for c in np.unique(inp)))
        sig["out_colors"].add(frozenset(int(c) for c in np.unique(out)))
    sig["in_colors"] = [sorted(s) for s in sig["in_colors"]]
    sig["out_colors"] = [sorted(s) for s in sig["out_colors"]]
    sig["all_in_colors"] = sorted(sig["all_in_colors"])
    sig["all_out_colors"] = sorted(sig["all_out_colors"])
    return sig
