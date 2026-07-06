"""Solver v3 — try more patterns including:
- Move object to bottom/right of grid
- Color map + tile with variable output
- Non-bijective color map + scale
- Multiple color swaps (3+ colors)
"""
import sys, os, json, time, zipfile, numpy as np
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import onnx
import onnx.helper as h
from onnx import TensorProto
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from dsl_transpiler import (
    Transpiler, py_color_map, py_crop, py_pad_to,
    py_flip_lr, py_flip_ud, py_rot180, py_transpose, py_rot90, py_rot270,
    py_scale_up, py_scale_down, py_tile, py_repeat_rows, py_repeat_cols
)


def try_multi_color_swap(pairs):
    """Try swapping 3+ colors simultaneously."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    # Derive multi-color mapping
    mapping = {}
    ok = True
    for inp, out in pairs:
        for c in range(NUM_COLORS):
            in_cells = (inp == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok or len(mapping) < 3:
        return None
    if not any(k != v for k, v in mapping.items()):
        return None
    # Verify
    for inp, out in pairs:
        expected = inp.copy()
        for k, v in mapping.items():
            expected[inp == k] = v
        if not np.array_equal(expected, out):
            return None
    t = Transpiler()
    t.color_map(mapping)
    return t.build()


def try_color_map_then_crop_static(pairs):
    """Color map then crop to fixed size. Output is always same shape."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    if not all(out.shape == (out_h, out_w) for _, out in pairs):
        return None
    
    # Derive color map from cropped input
    mapping = {}
    ok = True
    for inp, out in pairs:
        cropped = inp[:out_h, :out_w]
        for c in range(NUM_COLORS):
            in_cells = (cropped == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok or not any(k != v for k, v in mapping.items()):
        return None
    # Verify
    for inp, out in pairs:
        cropped = inp[:out_h, :out_w]
        mapped = cropped.copy()
        for k, v in mapping.items():
            mapped[cropped == k] = v
        if not np.array_equal(mapped, out):
            return None
    t = Transpiler()
    t.crop_top_left(out_h, out_w)
    t.color_map(mapping)
    return t.build()


def try_crop_then_color_map(pairs):
    """Crop to fixed size then color map."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    if not all(out.shape == (out_h, out_w) for _, out in pairs):
        return None
    
    # First check if it's just a crop (no color map)
    just_crop = True
    for inp, out in pairs:
        if not np.array_equal(inp[:out_h, :out_w], out):
            just_crop = False; break
    if just_crop:
        t = Transpiler()
        t.crop_top_left(out_h, out_w)
        return t.build()
    
    # Try crop + color map
    mapping = {}
    ok = True
    for inp, out in pairs:
        cropped = inp[:out_h, :out_w]
        for c in range(NUM_COLORS):
            in_cells = (cropped == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok or not any(k != v for k, v in mapping.items()):
        return None
    for inp, out in pairs:
        cropped = inp[:out_h, :out_w]
        mapped = cropped.copy()
        for k, v in mapping.items():
            mapped[cropped == k] = v
        if not np.array_equal(mapped, out):
            return None
    t = Transpiler()
    t.crop_top_left(out_h, out_w)
    t.color_map(mapping)
    return t.build()


def try_color_map_then_scale_down(pairs):
    """Color map then scale down by integer factor."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    if not all(out.shape == (out_h, out_w) for _, out in pairs): return None
    if in_h % out_h != 0 or in_w % out_w != 0: return None
    k_h = in_h // out_h
    k_w = in_w // out_w
    if k_h != k_w or k_h < 2 or k_h > 5: return None
    k = k_h
    
    # Derive color map: scale down output, compare to input
    mapping = {}
    ok = True
    for inp, out in pairs:
        scaled_down = out[::k, ::k]  # Wait, this is wrong direction
        # Actually: output = color_map(input[::k, ::k])
        cropped = inp[::k, ::k]
        if cropped.shape != out.shape:
            ok = False; break
        for c in range(NUM_COLORS):
            in_cells = (cropped == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok or not any(k2 != v for k2, v in mapping.items()):
        return None
    for inp, out in pairs:
        cropped = inp[::k, ::k]
        mapped = cropped.copy()
        for k2, v in mapping.items():
            mapped[cropped == k2] = v
        if not np.array_equal(mapped, out):
            return None
    t = Transpiler()
    t.crop_top_left(in_h, in_w)
    t.scale_down(k)
    t.color_map(mapping)
    t.pad_to(MAX_GRID, MAX_GRID)
    return t.build()


def try_scale_down_then_color_map(pairs):
    """Scale down then color map."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    if not all(out.shape == (out_h, out_w) for _, out in pairs): return None
    if in_h % out_h != 0 or in_w % out_w != 0: return None
    k_h = in_h // out_h
    k_w = in_w // out_w
    if k_h != k_w or k_h < 2 or k_h > 5: return None
    k = k_h
    
    mapping = {}
    ok = True
    for inp, out in pairs:
        scaled = inp[::k, ::k]
        if scaled.shape != out.shape:
            ok = False; break
        for c in range(NUM_COLORS):
            in_cells = (scaled == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok or not any(k2 != v for k2, v in mapping.items()):
        return None
    for inp, out in pairs:
        scaled = inp[::k, ::k]
        mapped = scaled.copy()
        for k2, v in mapping.items():
            mapped[scaled == k2] = v
        if not np.array_equal(mapped, out):
            return None
    t = Transpiler()
    t.crop_top_left(in_h, in_w)
    t.scale_down(k)
    t.color_map(mapping)
    t.pad_to(MAX_GRID, MAX_GRID)
    return t.build()


def try_dihedral_then_crop(pairs):
    """Dihedral transform then crop to fixed size."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    if not all(out.shape == (out_h, out_w) for _, out in pairs): return None
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    
    dihedrals = [
        ("identity", lambda g: g),
        ("flip_lr", py_flip_lr),
        ("flip_ud", py_flip_ud),
        ("rot180", py_rot180),
        ("rot90", py_rot90),
        ("rot270", py_rot270),
        ("transpose", py_transpose),
    ]
    
    for dname, dfn in dihedrals:
        ok = True
        for inp, out in pairs:
            transformed = dfn(inp)
            if transformed.shape[0] < out_h or transformed.shape[1] < out_w:
                ok = False; break
            if not np.array_equal(transformed[:out_h, :out_w], out):
                ok = False; break
        if ok:
            t = Transpiler()
            t.crop_top_left(in_h, in_w)
            if dname != "identity":
                if dname == "flip_lr": t.flip_lr()
                elif dname == "flip_ud": t.flip_ud()
                elif dname == "rot90": t.rot90()
                elif dname == "rot180": t.rot180()
                elif dname == "rot270": t.rot270()
                elif dname == "transpose": t.transpose()
            t.crop_top_left(out_h, out_w)
            return t.build()
    return None


def try_dihedral_then_crop_then_colormap(pairs):
    """Dihedral → crop → color map."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    if not all(out.shape == (out_h, out_w) for _, out in pairs): return None
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    
    dihedrals = [
        ("identity", lambda g: g),
        ("flip_lr", py_flip_lr),
        ("flip_ud", py_flip_ud),
        ("rot180", py_rot180),
        ("rot90", py_rot90),
        ("rot270", py_rot270),
        ("transpose", py_transpose),
    ]
    
    for dname, dfn in dihedrals:
        mapping = {}
        ok = True
        for inp, out in pairs:
            transformed = dfn(inp)
            if transformed.shape[0] < out_h or transformed.shape[1] < out_w:
                ok = False; break
            cropped = transformed[:out_h, :out_w]
            for c in range(NUM_COLORS):
                in_cells = (cropped == c)
                if not in_cells.any(): continue
                out_at = out[in_cells]
                out_colors = np.unique(out_at)
                if len(out_colors) != 1:
                    ok = False; break
                t = int(out_colors[0])
                if c in mapping and mapping[c] != t:
                    ok = False; break
                mapping[c] = t
            if not ok: break
        if not ok: continue
        if not any(k != v for k, v in mapping.items()): continue
        # Verify
        valid = True
        for inp, out in pairs:
            transformed = dfn(inp)
            cropped = transformed[:out_h, :out_w]
            mapped = cropped.copy()
            for k, v in mapping.items():
                mapped[cropped == k] = v
            if not np.array_equal(mapped, out):
                valid = False; break
        if not valid: continue
        t = Transpiler()
        t.crop_top_left(in_h, in_w)
        if dname != "identity":
            if dname == "flip_lr": t.flip_lr()
            elif dname == "flip_ud": t.flip_ud()
            elif dname == "rot90": t.rot90()
            elif dname == "rot180": t.rot180()
            elif dname == "rot270": t.rot270()
            elif dname == "transpose": t.transpose()
        t.crop_top_left(out_h, out_w)
        t.color_map(mapping)
        return t.build()
    return None


SOLVER_V3_DETECTORS = [
    ("multi_color_swap", try_multi_color_swap),
    ("color_map_then_crop_static", try_color_map_then_crop_static),
    ("crop_then_color_map", try_crop_then_color_map),
    ("color_map_then_scale_down", try_color_map_then_scale_down),
    ("scale_down_then_color_map", try_scale_down_then_color_map),
    ("dihedral_then_crop", try_dihedral_then_crop),
    ("dihedral_then_crop_then_colormap", try_dihedral_then_crop_then_colormap),
]


def try_solver_v3(task):
    pairs = arc_data.get_pairs(task)
    for name, detector in SOLVER_V3_DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        e = validator.evaluate_model(model, task)
        if e["eligible_for_points"]:
            model.ClearField("producer_name")
            model.ClearField("producer_version")
            model.ClearField("doc_string")
            model.ClearField("domain")
            model.ClearField("model_version")
            model.graph.ClearField("doc_string")
            if len(model.graph.name) > 1:
                model.graph.name = "g"
            e2 = validator.evaluate_model(model, task)
            if e2["eligible_for_points"]:
                return model, name, e2["score"]
    return None, None, 0


def main():
    with open("/home/z/my-project/data/final_comprehensive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    solved = 0
    score = 0.0
    breakdown = {}
    
    output_path = "/home/z/my-project/download/submission.zip"
    t0 = time.time()
    
    with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved:
            try:
                task = arc_data.load_task(tid)
                model, method, sc = try_solver_v3(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Solver V3 Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
