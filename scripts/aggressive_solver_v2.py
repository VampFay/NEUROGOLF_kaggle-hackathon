"""Aggressive solver v2 — try exhaustive color maps + geometric transforms.
For each task, try:
1. All 10!/(10-n)! color permutations (limited to colors present in input)
2. All 8 dihedral transforms
3. Combinations: color_map then dihedral, dihedral then color_map
4. Color map with multiple simultaneous changes
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
from itertools import permutations


def try_all_color_permutations(pairs):
    """Try all color permutations (bijective maps) of the colors present in input."""
    # Get colors present across all pairs
    all_in_colors = set()
    all_out_colors = set()
    for inp, out in pairs:
        all_in_colors.update(int(c) for c in np.unique(inp))
        all_out_colors.update(int(c) for c in np.unique(out))
    
    # Must be same set for bijective
    if all_in_colors != all_out_colors:
        return None
    
    # Limit: if more than 7 colors, too many permutations
    colors_list = sorted(all_in_colors)
    if len(colors_list) > 7:
        return None
    
    # Try all permutations of the non-zero colors
    zero = 0 if 0 in colors_list else None
    nonzero_colors = [c for c in colors_list if c != 0]
    if len(nonzero_colors) > 6:
        return None
    
    for perm in permutations(nonzero_colors):
        # Build mapping: nonzero_colors[i] → perm[i]
        mapping = {nonzero_colors[i]: perm[i] for i in range(len(nonzero_colors))}
        if 0 is not None:
            mapping[0] = 0
        
        # Quick check: is this identity?
        if all(k == v for k, v in mapping.items()):
            continue
        
        # Verify
        ok = True
        for inp, out in pairs:
            expected = inp.copy()
            for k, v in mapping.items():
                expected[inp == k] = v
            if not np.array_equal(expected, out):
                ok = False; break
        if ok:
            # Build ONNX
            t = Transpiler()
            t.color_map(mapping)
            return t.build()
    return None


def try_color_permutation_then_dihedral(pairs):
    """Try all color permutations × all dihedral transforms."""
    all_in_colors = set()
    all_out_colors = set()
    for inp, out in pairs:
        all_in_colors.update(int(c) for c in np.unique(inp))
        all_out_colors.update(int(c) for c in np.unique(out))
    if all_in_colors != all_out_colors:
        return None
    
    nonzero_colors = sorted([c for c in all_in_colors if c != 0])
    if len(nonzero_colors) > 6:
        return None
    
    dihedrals = [
        ("identity", lambda g: g),
        ("flip_lr", py_flip_lr),
        ("flip_ud", py_flip_ud),
        ("rot180", py_rot180),
        ("rot90", py_rot90),
        ("rot270", py_rot270),
        ("transpose", py_transpose),
    ]
    
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs):
        return None
    
    for dname, dfn in dihedrals:
        for perm in permutations(nonzero_colors):
            mapping = {nonzero_colors[i]: perm[i] for i in range(len(nonzero_colors))}
            if 0 in all_in_colors:
                mapping[0] = 0
            if all(k == v for k, v in mapping.items()) and dname == "identity":
                continue
            
            ok = True
            for inp, out in pairs:
                mapped = inp.copy()
                for k, v in mapping.items():
                    mapped[inp == k] = v
                transformed = dfn(mapped)
                if not np.array_equal(transformed, out):
                    ok = False; break
            if ok:
                t = Transpiler()
                t.crop_top_left(in_h, in_w)
                t.color_map(mapping)
                if dname != "identity":
                    if dname == "flip_lr": t.flip_lr()
                    elif dname == "flip_ud": t.flip_ud()
                    elif dname == "rot90": t.rot90()
                    elif dname == "rot180": t.rot180()
                    elif dname == "rot270": t.rot270()
                    elif dname == "transpose": t.transpose()
                t.pad_to(MAX_GRID, MAX_GRID)
                return t.build()
    return None


def try_dihedral_then_color_permutation(pairs):
    """Try all dihedral × all color permutations."""
    all_in_colors = set()
    all_out_colors = set()
    for inp, out in pairs:
        all_in_colors.update(int(c) for c in np.unique(inp))
        all_out_colors.update(int(c) for c in np.unique(out))
    if all_in_colors != all_out_colors:
        return None
    
    nonzero_colors = sorted([c for c in all_in_colors if c != 0])
    if len(nonzero_colors) > 6:
        return None
    
    dihedrals = [
        ("identity", lambda g: g),
        ("flip_lr", py_flip_lr),
        ("flip_ud", py_flip_ud),
        ("rot180", py_rot180),
        ("rot90", py_rot90),
        ("rot270", py_rot270),
        ("transpose", py_transpose),
    ]
    
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs):
        return None
    
    for dname, dfn in dihedrals:
        for perm in permutations(nonzero_colors):
            mapping = {nonzero_colors[i]: perm[i] for i in range(len(nonzero_colors))}
            if 0 in all_in_colors:
                mapping[0] = 0
            if all(k == v for k, v in mapping.items()) and dname == "identity":
                continue
            
            ok = True
            for inp, out in pairs:
                transformed = dfn(inp)
                mapped = transformed.copy()
                for k, v in mapping.items():
                    mapped[transformed == k] = v
                if not np.array_equal(mapped, out):
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
                t.color_map(mapping)
                t.pad_to(MAX_GRID, MAX_GRID)
                return t.build()
    return None


def try_non_bijective_color_map(pairs):
    """Try non-bijective color maps (many-to-one mappings)."""
    # For each color in input, try mapping it to each possible output color
    # This is 10^10 combinations — too many. Use a smarter approach:
    # Derive the mapping from pairs (if consistent)
    mapping = {}
    ok = True
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
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
    if not ok or not mapping:
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
    # Build
    t = Transpiler()
    t.color_map(mapping)
    return t.build()


def try_non_bijective_color_map_then_dihedral(pairs):
    """Non-bijective color map then dihedral."""
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs):
        return None
    if not all(inp.shape == out.shape for inp, out in pairs):
        return None
    
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
        # Derive color map: transform input, then map to output
        mapping = {}
        ok = True
        for inp, out in pairs:
            transformed = dfn(inp)
            for c in range(NUM_COLORS):
                in_cells = (transformed == c)
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
            mapped = transformed.copy()
            for k, v in mapping.items():
                mapped[transformed == k] = v
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
        t.color_map(mapping)
        t.pad_to(MAX_GRID, MAX_GRID)
        return t.build()
    return None


def try_dihedral_then_non_bijective_color_map(pairs):
    """Dihedral then non-bijective color map."""
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs):
        return None
    if not all(inp.shape == out.shape for inp, out in pairs):
        return None
    
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
        # Derive color map: map input to dfn^{-1}(output)
        rev_dfn = {
            "flip_lr": py_flip_lr, "flip_ud": py_flip_ud, "rot180": py_rot180,
            "rot90": py_rot270, "rot270": py_rot90, "transpose": py_transpose,
            "identity": lambda g: g,
        }.get(dname, lambda g: g)
        
        mapping = {}
        ok = True
        for inp, out in pairs:
            rev_out = rev_dfn(out)
            for c in range(NUM_COLORS):
                in_cells = (inp == c)
                if not in_cells.any(): continue
                out_at = rev_out[in_cells]
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
            mapped = inp.copy()
            for k, v in mapping.items():
                mapped[inp == k] = v
            transformed = dfn(mapped)
            if not np.array_equal(transformed, out):
                valid = False; break
        if not valid: continue
        
        t = Transpiler()
        t.crop_top_left(in_h, in_w)
        t.color_map(mapping)
        if dname != "identity":
            if dname == "flip_lr": t.flip_lr()
            elif dname == "flip_ud": t.flip_ud()
            elif dname == "rot90": t.rot90()
            elif dname == "rot180": t.rot180()
            elif dname == "rot270": t.rot270()
            elif dname == "transpose": t.transpose()
        t.pad_to(MAX_GRID, MAX_GRID)
        return t.build()
    return None


AGGRESSIVE_DETECTORS = [
    ("non_bijective_color_map", try_non_bijective_color_map),
    ("non_bijective_cm_then_dihedral", try_non_bijective_color_map_then_dihedral),
    ("dihedral_then_non_bijective_cm", try_dihedral_then_non_bijective_color_map),
    ("all_color_permutations", try_all_color_permutations),
    ("color_perm_then_dihedral", try_color_permutation_then_dihedral),
    ("dihedral_then_color_perm", try_dihedral_then_color_permutation),
]


def try_aggressive_detectors(task):
    pairs = arc_data.get_pairs(task)
    for name, detector in AGGRESSIVE_DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        e = validator.evaluate_model(model, task)
        if e["eligible_for_points"]:
            # Strip metadata
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
                model, method, sc = try_aggressive_detectors(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Aggressive Solver Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
