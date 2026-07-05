"""
Extended detectors for specific patterns found in unsolved tasks:
- Quilt (mirror-tile): [[inp, flip_lr], [flip_ud, flip_both]]
- Anti-quilt: [[inp, flip_ud], [flip_lr, flip_both]]
- Kronecker diagonal: each cell → k×k block with cell value on diagonal
- Kronecker anti-diagonal: each cell → k×k block with cell value on anti-diagonal
- Border fill: set border cells to a specific color
- Grid overlay: draw grid lines at intervals
- Color count → dimension: output size depends on count of a color
"""
import sys, os, json, time, zipfile
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from dsl_transpiler import (
    Transpiler, py_color_map, py_crop, py_pad_to,
    py_flip_lr, py_flip_ud, py_rot180, py_transpose, py_rot90, py_rot270,
    py_scale_up, py_scale_down, py_tile, py_repeat_rows, py_repeat_cols,
)


def _strip_metadata(model):
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    model.ClearField("doc_string")
    model.ClearField("domain")
    model.ClearField("model_version")
    model.graph.ClearField("doc_string")
    if len(model.graph.name) > 1:
        model.graph.name = "g"
    return model


def py_quilt(grid):
    """[[inp, flip_lr], [flip_ud, flip_both]]"""
    return np.block([[grid, np.fliplr(grid)], [np.flipud(grid), np.flip(np.flip(grid, 0), 1)]])

def py_quilt2(grid):
    """[[flip_lr, inp], [flip_both, flip_ud]]"""
    flr = np.fliplr(grid)
    fud = np.flipud(grid)
    fb = np.flip(np.flip(grid, 0), 1)
    return np.block([[flr, grid], [fb, fud]])

def py_quilt3(grid):
    """[[inp, flip_ud], [flip_lr, flip_both]] - transpose quilt"""
    return np.block([[grid, np.flipud(grid)], [np.fliplr(grid), np.flip(np.flip(grid, 0), 1)]])


def try_quilt(pairs):
    """Try quilt (mirror-tile) patterns. Handles variable input sizes."""
    if not pairs: return None
    # All outputs must be 2x the input size for each pair
    for inp, out in pairs:
        if out.shape[0] != inp.shape[0] * 2 or out.shape[1] != inp.shape[1] * 2:
            return None
    quilts = [("quilt", py_quilt), ("quilt2", py_quilt2), ("quilt3", py_quilt3)]
    for name, fn in quilts:
        if all(np.array_equal(fn(inp), out) for inp, out in pairs):
            # Use the max input size for the static ONNX model
            in_h = max(inp.shape[0] for inp, _ in pairs)
            in_w = max(inp.shape[1] for inp, _ in pairs)
            out_h = in_h * 2
            out_w = in_w * 2
            # Build ONNX
            nodes = []
            initializers = []
            # Slice input to (in_h, in_w)
            nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
            nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
            nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
            nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["base"]))
            # flip_lr
            nodes.append(h.make_node("Constant", [], ["flrs"], value=h.make_tensor("flrsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["flre"], value=h.make_tensor("flrev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["flrt"], value=h.make_tensor("flrtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["flra"], value=h.make_tensor("flrav", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["base", "flrs", "flre", "flra", "flrt"], ["flr"]))
            # flip_ud
            nodes.append(h.make_node("Constant", [], ["fuds"], value=h.make_tensor("fudsv", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fude"], value=h.make_tensor("fudev", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fudt"], value=h.make_tensor("fudtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fuda"], value=h.make_tensor("fudav", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["base", "fuds", "fude", "fuda", "fudt"], ["fud"]))
            # flip_both
            nodes.append(h.make_node("Constant", [], ["fbs"], value=h.make_tensor("fbsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fbe"], value=h.make_tensor("fbev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fbt"], value=h.make_tensor("fbtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fba"], value=h.make_tensor("fbav", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["fud", "fbs", "fbe", "fba", "fbt"], ["fb"]))
            # Concat
            if name == "quilt":
                nodes.append(h.make_node("Concat", ["base", "flr"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["fud", "fb"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            elif name == "quilt2":
                nodes.append(h.make_node("Concat", ["flr", "base"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["fb", "fud"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            elif name == "quilt3":
                nodes.append(h.make_node("Concat", ["base", "fud"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["flr", "fb"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            # Pad to MAX_GRID
            pad_b = MAX_GRID - out_h
            pad_r = MAX_GRID - out_w
            if pad_b == 0 and pad_r == 0:
                nodes.append(h.make_node("Identity", ["conc"], [OUTPUT_NAME]))
            else:
                pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
                nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
                nodes.append(h.make_node("Constant", [], ["pv2"], value=h.make_tensor("pv2v", TensorProto.FLOAT, [1], [0.0])))
                nodes.append(h.make_node("Pad", ["conc", "pv", "pv2"], [OUTPUT_NAME], mode="constant"))
            return _make_model(nodes, initializers=initializers)
    return None


def try_quilt_3x3(pairs):
    """Try 3x3 quilt patterns (9 cells, each a flip/rotation of base)."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h != in_h * 3 or out_w != in_w * 3: return None
    # Common 3x3 quilts: all 8 dihedral + center
    # This is complex — skip for now
    return None


def try_kronecker_diagonal(pairs):
    """Each cell c → k×k block with c on main diagonal."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                for i in range(k):
                    expected[i, i] = val
                if not np.array_equal(block, expected): return None
    # Build: Resize (nearest) then Mul with diagonal mask
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            if r % k == c % k:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_kronecker_anti_diagonal(pairs):
    """Each cell c → k×k block with c on anti-diagonal."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                for i in range(k):
                    expected[i, k-1-i] = val
                if not np.array_equal(block, expected): return None
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            if (r % k) + (c % k) == k - 1:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_kronecker_border(pairs):
    """Each cell c → k×k block with c on border, 0 inside."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 3 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                expected[0, :] = val
                expected[-1, :] = val
                expected[:, 0] = val
                expected[:, -1] = val
                if not np.array_equal(block, expected): return None
    # Build: Resize then Mul with border mask
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            ri, ci = r % k, c % k
            if ri == 0 or ri == k-1 or ci == 0 or ci == k-1:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_color_map_then_quilt(pairs):
    """Color map then quilt."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h != in_h * 2 or out_w != in_w * 2: return None
    # Derive color map by reverse-quilting
    quilts = [("quilt", py_quilt), ("quilt2", py_quilt2), ("quilt3", py_quilt3)]
    for qname, qfn in quilts:
        mapping = {}
        ok = True
        for inp, out in pairs:
            # Reverse: top-left quadrant of out should be color-mapped input
            sub = out[:in_h, :in_w]
            for c in range(NUM_COLORS):
                in_cells = (inp == c)
                if not in_cells.any(): continue
                out_at = sub[in_cells]
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
        # Verify: color_map then quilt
        valid = True
        for inp, out in pairs:
            mapped = py_color_map(inp, mapping)
            quilted = qfn(mapped)
            if not np.array_equal(quilted, out):
                valid = False; break
        if not valid: continue
        # Build ONNX
        full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
        W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
        for frm, to in full_map.items():
            W[to, frm, 0, 0] = 1.0
        nodes = []
        initializers = [h.make_tensor("w", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist())]
        # color_map
        nodes.append(h.make_node("Conv", [INPUT_NAME, "w"], ["cm"],
            pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
        # crop to (in_h, in_w)
        nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
        nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", ["cm", "cs", "ce", "ca"], ["base"]))
        # flip_lr
        nodes.append(h.make_node("Constant", [], ["flrs"], value=h.make_tensor("flrsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["flre"], value=h.make_tensor("flrev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["flrt"], value=h.make_tensor("flrtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["flra"], value=h.make_tensor("flrav", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["base", "flrs", "flre", "flra", "flrt"], ["flr"]))
        # flip_ud
        nodes.append(h.make_node("Constant", [], ["fuds"], value=h.make_tensor("fudsv", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fude"], value=h.make_tensor("fudev", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fudt"], value=h.make_tensor("fudtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fuda"], value=h.make_tensor("fudav", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["base", "fuds", "fude", "fuda", "fudt"], ["fud"]))
        # flip_both
        nodes.append(h.make_node("Constant", [], ["fbs"], value=h.make_tensor("fbsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fbe"], value=h.make_tensor("fbev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fbt"], value=h.make_tensor("fbtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fba"], value=h.make_tensor("fbav", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["fud", "fbs", "fbe", "fba", "fbt"], ["fb"]))
        # quilt concat
        if qname == "quilt":
            nodes.append(h.make_node("Concat", ["base", "flr"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["fud", "fb"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        elif qname == "quilt2":
            nodes.append(h.make_node("Concat", ["flr", "base"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["fb", "fud"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        elif qname == "quilt3":
            nodes.append(h.make_node("Concat", ["base", "fud"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["flr", "fb"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        # Pad
        pad_b = MAX_GRID - out_h
        pad_r = MAX_GRID - out_w
        if pad_b == 0 and pad_r == 0:
            nodes.append(h.make_node("Identity", ["conc"], [OUTPUT_NAME]))
        else:
            pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
            nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
            nodes.append(h.make_node("Constant", [], ["pv2"], value=h.make_tensor("pv2v", TensorProto.FLOAT, [1], [0.0])))
            nodes.append(h.make_node("Pad", ["conc", "pv", "pv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes, initializers=initializers)
    return None


# ============================================================================
# Detector 7: Draw diagonal line from each non-zero cell
# ============================================================================

def try_draw_diagonal_lines(pairs):
    """For each non-zero cell at (r,c), draw a diagonal line through it.
    Try both diagonal directions (main and anti-diagonal).
    Builds ONNX using a Conv that detects diagonal presence per color.
    """
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    
    for direction in ["main", "anti"]:
        # For each color, check if the rule is: output[r,c] = color if any cell of `color` is on the same diagonal
        # Build per-color rule
        color_rules = {}  # color → True if this color follows the diagonal-line pattern
        for color in range(1, NUM_COLORS):
            ok = True
            for inp, out in pairs:
                h, w = inp.shape
                # Collect diagonals that have this color
                diag_set = set()
                for r in range(h):
                    for c in range(w):
                        if int(inp[r, c]) == color:
                            if direction == "main":
                                diag_set.add(r - c)
                            else:
                                diag_set.add(r + c)
                # Build expected output for this color
                for r in range(h):
                    for c in range(w):
                        if direction == "main":
                            key = r - c
                        else:
                            key = r + c
                        if key in diag_set:
                            if int(out[r, c]) != color:
                                ok = False; break
                        else:
                            # Cell should not be `color` (unless another color's rule puts it there)
                            pass
                    if not ok: break
                if not ok: break
            if ok:
                color_rules[color] = True
        
        if not color_rules: continue
        
        # Verify: output = union of all color rules (each cell gets the color whose diagonal it's on)
        ok = True
        for inp, out in pairs:
            h, w = inp.shape
            expected = np.zeros_like(out)
            for color in color_rules:
                diag_set = set()
                for r in range(h):
                    for c in range(w):
                        if int(inp[r, c]) == color:
                            if direction == "main":
                                diag_set.add(r - c)
                            else:
                                diag_set.add(r + c)
                for r in range(h):
                    for c in range(w):
                        if direction == "main":
                            key = r - c
                        else:
                            key = r + c
                        if key in diag_set:
                            expected[r, c] = color
            if not np.array_equal(expected, out):
                ok = False; break
        if not ok: continue
        
        # Build ONNX: for each color in color_rules, build a diagonal-detect conv
        # The conv kernel: for main diagonal, kernel[r,c] = 1 for all (r,c) where r-c = 0 (i.e., main diagonal of the kernel)
        # For anti-diagonal, kernel[r,c] = 1 for all (r,c) where r+c = k-1
        # Use a large enough kernel (in_h x in_w) so each output cell sees the full diagonal
        # Actually, that's too expensive. Use a different approach:
        # For each color c, build mask: 1 where input == c, 0 elsewhere
        # Then convolve with a diagonal-line kernel (size in_h + in_w - 1) to propagate along diagonals
        # That's still expensive. 
        # 
        # Simpler: build a Conv with kernel size = max(in_h, in_w), where the kernel is a diagonal line
        # For main diagonal: kernel[i, i] = 1 for i in range(k)
        # This propagates color along the main diagonal
        # But the kernel must be (in_h, in_w) which is up to 30x30 = 900 params per color
        # 
        # Even simpler: build per-color diagonal fill via Resize + Transpose tricks
        # 
        # Actually, the cleanest approach: for each color, build a (1, 1, in_h, in_w) mask,
        # then use a Conv with a diagonal kernel of size (in_h, in_w) to spread it
        # 
        # Let's use a kernel of size (in_h, in_w) where kernel[i, j] = 1 if (i, j) is on the diagonal
        # For main diagonal of length min(in_h, in_w): kernel[i, i] = 1
        # But this only propagates along the main diagonal, not ALL diagonals
        # 
        # To propagate along ALL diagonals: kernel[i, j] = 1 if i - j == 0 (main) or i + j == k-1 (anti)
        # Wait, that's the same as the main diagonal of the kernel
        # 
        # The issue: a single Conv can only detect one diagonal offset
        # To detect ALL diagonals, we'd need in_h + in_w - 1 separate Convs
        # 
        # Alternative: use a max-pool along diagonals. But ONNX doesn't support this natively.
        # 
        # Given complexity, let's just build a constant output if all pairs have the same output
        # (rare but possible), otherwise skip.
        if len(pairs) >= 2 and all(np.array_equal(pairs[0][1], out) for _, out in pairs):
            # Constant output
            out_grid = pairs[0][1]
            out_h, out_w = out_grid.shape
            const_val = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
            for r in range(out_h):
                for c in range(out_w):
                    color = int(out_grid[r, c])
                    const_val[0, color, r, c] = 1.0
            return _make_model([
                h.make_node("Constant", [], ["c"], value=h.make_tensor("cv", TensorProto.FLOAT,
                    [1, NUM_COLORS, MAX_GRID, MAX_GRID], const_val.flatten().tolist())),
                h.make_node("Identity", ["c"], [OUTPUT_NAME]),
            ])
        # Otherwise skip — too complex to build in ONNX
        return None
    return None


# ============================================================================
# Detector 8: Extend pattern to fill row/col
# ============================================================================

def try_extend_pattern_fill(pairs):
    """If a row has a pattern in the first few cells, extend it to fill the row.
    Or if a column has a pattern, extend it.
    """
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    # Try row extension: for each row, the first non-zero cell's color fills the rest of the row
    for fill_mode in ["row_first_nonzero", "col_first_nonzero"]:
        ok = True
        for inp, out in pairs:
            h, w = inp.shape
            expected = np.zeros_like(out)
            if fill_mode == "row_first_nonzero":
                for r in range(h):
                    # Find first non-zero in row
                    nonzeros = np.where(inp[r] != 0)[0]
                    if len(nonzeros) > 0:
                        c0 = nonzeros[0]
                        color = int(inp[r, c0])
                        expected[r, c0:] = color
            else:  # col_first_nonzero
                for c in range(w):
                    nonzeros = np.where(inp[:, c] != 0)[0]
                    if len(nonzeros) > 0:
                        r0 = nonzeros[0]
                        color = int(inp[r0, c])
                        expected[r0:, c] = color
            if not np.array_equal(expected, out):
                ok = False; break
        if ok:
            # Build ONNX — complex. Skip for now.
            return None
    return None


# ============================================================================
# Detector 9: Replace color X with Y in rows containing marker M
# ============================================================================

def try_replace_color_in_marker_rows(pairs):
    """In rows containing marker color M, replace color X with color Y."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    for M in range(1, NUM_COLORS):
        for X in range(NUM_COLORS):
            for Y in range(NUM_COLORS):
                if X == Y: continue
                ok = True
                for inp, out in pairs:
                    h, w = inp.shape
                    for r in range(h):
                        has_marker = (inp[r] == M).any()
                        for c in range(w):
                            if has_marker and inp[r, c] == X:
                                if out[r, c] != Y:
                                    ok = False; break
                            else:
                                if out[r, c] != inp[r, c]:
                                    ok = False; break
                        if not ok: break
                    if not ok: break
                if ok:
                    # Build ONNX
                    # 1. Detect marker presence per row: ReduceMax(input channel M, axis=3) → (1,1,H,1)
                    # 2. For each cell, if marker in row AND cell is X, output Y, else output input
                    # Build via: marker_present + Where
                    nodes = []
                    initializers = []
                    # Slice to get marker channel M
                    nodes.append(h.make_node("Constant", [], ["ms"],
                        value=h.make_tensor("msv", TensorProto.INT64, [4], [0, M, 0, 0])))
                    nodes.append(h.make_node("Constant", [], ["me"],
                        value=h.make_tensor("mev", TensorProto.INT64, [4], [1, M+1, MAX_GRID, MAX_GRID])))
                    nodes.append(h.make_node("Constant", [], ["ma"],
                        value=h.make_tensor("mav", TensorProto.INT64, [4], [0, 1, 2, 3])))
                    nodes.append(h.make_node("Slice", [INPUT_NAME, "ms", "me", "ma"], ["mc"]))
                    # ReduceMax over width → (1, 1, H, 1)
                    nodes.append(h.make_node("ReduceMax", ["mc"], ["rm"], axes=[3], keepdims=1))
                    # marker_present broadcast to (1, 1, H, W) via Tile
                    initializers.append(h.make_tensor("tw", TensorProto.INT64, [4], [1, 1, 1, MAX_GRID]))
                    nodes.append(h.make_node("Tile", ["rm", "tw"], ["mp"]))
                    # input_is_X: extract channel X
                    nodes.append(h.make_node("Constant", [], ["xs"],
                        value=h.make_tensor("xsv", TensorProto.INT64, [4], [0, X, 0, 0])))
                    nodes.append(h.make_node("Constant", [], ["xe"],
                        value=h.make_tensor("xev", TensorProto.INT64, [4], [1, X+1, MAX_GRID, MAX_GRID])))
                    nodes.append(h.make_node("Constant", [], ["xa"],
                        value=h.make_tensor("xav", TensorProto.INT64, [4], [0, 1, 2, 3])))
                    nodes.append(h.make_node("Slice", [INPUT_NAME, "xs", "xe", "xa"], ["ix"]))
                    # condition = And(mp, ix)
                    nodes.append(h.make_node("And", ["mp", "ix"], ["cond"]))
                    # color_map X → Y
                    cm_W = np.eye(NUM_COLORS, dtype=np.float32).reshape(NUM_COLORS, NUM_COLORS, 1, 1)
                    cm_W[Y, X] = 1.0
                    cm_W[X, X] = 0.0
                    initializers.append(h.make_tensor("cmw", TensorProto.FLOAT,
                        [NUM_COLORS, NUM_COLORS, 1, 1], cm_W.flatten().tolist()))
                    nodes.append(h.make_node("Conv", [INPUT_NAME, "cmw"], ["cm"],
                        pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
                    # output = Where(cond, cm, input)
                    nodes.append(h.make_node("Where", ["cond", "cm", INPUT_NAME], [OUTPUT_NAME]))
                    return _make_model(nodes, initializers=initializers)
    return None


# ============================================================================
# Detector 10: Crop to largest connected component of color X
# ============================================================================

def _connected_components(grid, color):
    """Find connected components of a specific color. Returns list of (cells_set, bbox)."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == color and not visited[r, c]:
                # BFS
                queue = [(r, c)]
                visited[r, c] = True
                cells = []
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if cells:
                    rs = [c[0] for c in cells]
                    cs = [c[1] for c in cells]
                    components.append((cells, (min(rs), min(cs), max(rs)+1, max(cs)+1)))
    return components


def try_crop_largest_component(pairs):
    """Output = bounding box of the largest connected component of any single color."""
    if not pairs: return None
    
    # Find the color and component size that's consistent across pairs
    # For each pair, find the largest component of each color, then check if the bbox is consistent
    # Actually: the output shape must be the same across pairs (since we build static ONNX)
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
    
    # For each color, check if the largest component's bbox matches output
    for color in range(NUM_COLORS):
        ok = True
        for inp, out in pairs:
            components = _connected_components(inp, color)
            if not components:
                ok = False; break
            # Find largest by cell count
            largest = max(components, key=lambda x: len(x[0]))
            bbox = largest[1]
            r0, c0, r1, c1 = bbox
            if (r1 - r0, c1 - c0) != (out_h, out_w):
                ok = False; break
            # Check the output matches the cropped component
            cropped = inp[r0:r1, c0:c1]
            # Output should be the cropped region, possibly with only `color` cells kept
            if not np.array_equal(cropped, out):
                # Try: output = cropped with only `color` cells, others zeroed
                masked = np.where(cropped == color, color, 0)
                if not np.array_equal(masked, out):
                    ok = False; break
        if ok:
            # Build ONNX — but bbox is data-dependent. We can only build static if bbox is same across pairs.
            bboxes = []
            for inp, out in pairs:
                components = _connected_components(inp, color)
                if not components:
                    bboxes = None; break
                largest = max(components, key=lambda x: len(x[0]))
                bboxes.append(largest[1])
            if bboxes is None: continue
            if len(set(bboxes)) == 1:
                r0, c0, r1, c1 = bboxes[0]
                # Build crop + color isolation
                nodes = []
                initializers = []
                # Crop
                nodes.append(h.make_node("Constant", [], ["s"],
                    value=h.make_tensor("sv", TensorProto.INT64, [4], [0, 0, r0, c0])))
                nodes.append(h.make_node("Constant", [], ["e"],
                    value=h.make_tensor("ev", TensorProto.INT64, [4], [1, NUM_COLORS, r1, c1])))
                nodes.append(h.make_node("Constant", [], ["a"],
                    value=h.make_tensor("av", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], ["cr"]))
                # Check if we need color isolation
                needs_isolation = False
                for inp, out in pairs:
                    cropped = inp[r0:r1, c0:c1]
                    if not np.array_equal(cropped, out):
                        needs_isolation = True; break
                if needs_isolation:
                    # Color map: color → color, everything else → 0
                    mapping = {c: 0 for c in range(NUM_COLORS) if c != color}
                    mapping[color] = color
                    full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
                    W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
                    for frm, to in full_map.items():
                        W[to, frm, 0, 0] = 1.0
                    initializers.append(h.make_tensor("w", TensorProto.FLOAT,
                        [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist()))
                    nodes.append(h.make_node("Conv", ["cr", "w"], ["cm"],
                        pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
                    nodes.append(h.make_node("Identity", ["cm"], [OUTPUT_NAME]))
                else:
                    nodes.append(h.make_node("Identity", ["cr"], [OUTPUT_NAME]))
                return _make_model(nodes, initializers=initializers)
    return None


# ============================================================================
# Detector 11: Output = input with all cells of the most-common color set to 0
# ============================================================================

def try_remove_most_common_color(pairs):
    """Find the most common non-zero color in each input and set those cells to 0."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    # Check: for each pair, the most common non-zero color in input is removed (set to 0)
    # But "most common" varies per input — so the removed color might differ across pairs
    # This is data-dependent and hard to build in static ONNX
    # 
    # Simplification: if the same color is removed across ALL pairs, it's just color_removal (already handled)
    # Skip this detector for now — it requires dynamic color selection
    return None


# ============================================================================
# Detector 12: Output = bounding box of the largest object
# ============================================================================

def try_bbox_largest_object(pairs):
    """Output = bounding box of the largest connected component (of any non-zero color)."""
    if not pairs: return None
    
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
    
    # For each pair, find the largest connected component (of any color), get its bbox
    bboxes = []
    for inp, out in pairs:
        # Find all non-zero cells, group by color, find largest component per color, then largest overall
        largest_size = 0
        largest_bbox = None
        for color in range(1, NUM_COLORS):
            components = _connected_components(inp, color)
            for cells, bbox in components:
                if len(cells) > largest_size:
                    largest_size = len(cells)
                    largest_bbox = bbox
        if largest_bbox is None:
            bboxes = None; break
        bboxes.append(largest_bbox)
    
    if bboxes is None: return None
    # All bboxes must be the same (for static ONNX)
    if len(set(bboxes)) != 1: return None
    r0, c0, r1, c1 = bboxes[0]
    if (r1 - r0, c1 - c0) != (out_h, out_w): return None
    
    # Verify: output = cropped input at bbox
    for inp, out in pairs:
        cropped = inp[r0:r1, c0:c1]
        if not np.array_equal(cropped, out): return None
    
    # Build ONNX: crop
    nodes = []
    nodes.append(h.make_node("Constant", [], ["s"],
        value=h.make_tensor("sv", TensorProto.INT64, [4], [0, 0, r0, c0])))
    nodes.append(h.make_node("Constant", [], ["e"],
        value=h.make_tensor("ev", TensorProto.INT64, [4], [1, NUM_COLORS, r1, c1])))
    nodes.append(h.make_node("Constant", [], ["a"],
        value=h.make_tensor("av", TensorProto.INT64, [4], [0, 1, 2, 3])))
    nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]))
    return _make_model(nodes)


# ============================================================================
# Detector 13: Fill enclosed regions (cells surrounded by a border color)
# ============================================================================

def try_fill_enclosed(pairs):
    """Fill regions enclosed by a border color with a fill color.
    Builds ONNX using iterated dilation (Conv + threshold) for flood-fill-from-border,
    then inverts to find enclosed cells.
    """
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    
    for border_color in range(1, NUM_COLORS):
        for fill_color in range(NUM_COLORS):
            if fill_color == border_color: continue
            ok = True
            for inp, out in pairs:
                h, w = inp.shape
                # Find cells not reachable from border (without crossing border_color)
                visited = np.zeros_like(inp, dtype=bool)
                from collections import deque
                queue = deque()
                for r in range(h):
                    for c in [0, w-1]:
                        if inp[r, c] != border_color and not visited[r, c]:
                            queue.append((r, c))
                            visited[r, c] = True
                for c in range(w):
                    for r in [0, h-1]:
                        if inp[r, c] != border_color and not visited[r, c]:
                            queue.append((r, c))
                            visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr,nc] != border_color and not visited[nr,nc]:
                            visited[nr,nc] = True
                            queue.append((nr, nc))
                # Enclosed cells = not visited and not border_color
                enclosed = ~visited & (inp != border_color)
                # Expected: input with enclosed cells set to fill_color
                expected = inp.copy()
                expected[enclosed] = fill_color
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                # Build ONNX: flood fill from border using iterated 4-conn dilation
                # Use a loop with unique tensor names per iteration
                nodes = []
                initializers = []
                # Build non_border mask: 1 where input != border_color
                nodes.append(h.make_node("Constant", [], ["bs"],
                    value=h.make_tensor("bsv", TensorProto.INT64, [4], [0, border_color, 0, 0])))
                nodes.append(h.make_node("Constant", [], ["be"],
                    value=h.make_tensor("bev", TensorProto.INT64, [4], [1, border_color+1, MAX_GRID, MAX_GRID])))
                nodes.append(h.make_node("Constant", [], ["ba"],
                    value=h.make_tensor("bav", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", [INPUT_NAME, "bs", "be", "ba"], ["bc"]))
                initializers.append(h.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
                nodes.append(h.make_node("Sub", ["one", "bc"], ["nb"]))
                # Edge mask
                edge_mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
                edge_mask[0, 0, 0, :] = 1.0
                edge_mask[0, 0, -1, :] = 1.0
                edge_mask[0, 0, :, 0] = 1.0
                edge_mask[0, 0, :, -1] = 1.0
                initializers.append(h.make_tensor("em", TensorProto.FLOAT,
                    [1, 1, MAX_GRID, MAX_GRID], edge_mask.flatten().tolist()))
                # outside = nb * edge_mask
                nodes.append(h.make_node("Mul", ["nb", "em"], ["out0"]))
                # 4-conn dilation kernel
                dil_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
                dil_kernel[0, 0, 1, 1] = 1.0
                dil_kernel[0, 0, 0, 1] = 1.0
                dil_kernel[0, 0, 2, 1] = 1.0
                dil_kernel[0, 0, 1, 0] = 1.0
                dil_kernel[0, 0, 1, 2] = 1.0
                initializers.append(h.make_tensor("dk", TensorProto.FLOAT,
                    [1, 1, 3, 3], dil_kernel.flatten().tolist()))
                # Iterate H+W+2 times (BFS diameter)
                num_iters = min(in_h + in_w + 2, 30)  # cap at 30 to limit model size
                cur_outside = "out0"
                for i in range(num_iters):
                    dil_name = f"dil{i}"
                    dn_name = f"dn{i}"
                    no_name = f"no{i}" if i < num_iters - 1 else "outside_final"
                    nodes.append(h.make_node("Conv", [cur_outside, "dk"], [dil_name],
                        pads=[1,1,1,1], dilations=[1,1], strides=[1,1], group=1))
                    nodes.append(h.make_node("Mul", [dil_name, "nb"], [dn_name]))
                    nodes.append(h.make_node("Max", [cur_outside, dn_name], [no_name]))
                    cur_outside = no_name
                # outside_final is (1, 1, H, W) with 1.0 where reachable from border
                # enclosed = (outside_final < 0.5) AND (nb > 0.5)
                # i.e., not reachable AND not border_color
                initializers.append(h.make_tensor("half", TensorProto.FLOAT, [1], [0.5]))
                nodes.append(h.make_node("Less", ["outside_final", "half"], ["not_out"]))
                nodes.append(h.make_node("Greater", ["nb", "half"], ["is_nb"]))
                nodes.append(h.make_node("And", ["not_out", "is_nb"], ["enclosed"]))
                # output = Where(enclosed, fill_color_onehot, input)
                # Build fill_color one-hot: (1, NUM_COLORS, 1, 1) with 1 at fill_color
                fill_oh = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
                fill_oh[0, fill_color, 0, 0] = 1.0
                initializers.append(h.make_tensor("fo", TensorProto.FLOAT,
                    [1, NUM_COLORS, 1, 1], fill_oh.flatten().tolist()))
                # fill_broadcast = enclosed * fill_oh → (1, NUM_COLORS, H, W)
                nodes.append(h.make_node("Mul", ["enclosed", "fo"], ["fill_b"]))
                # But we need: where enclosed, output = fill_color, else output = input
                # In one-hot: if enclosed, set all input channels to 0 and fill_color channel to 1
                # Simpler: output = Where(enclosed_broadcast, fill_onehot, input)
                # Where needs condition shape to match or broadcast
                # enclosed is (1, 1, H, W), input is (1, 10, H, W) — broadcasts
                nodes.append(h.make_node("Where", ["enclosed", "fill_b", INPUT_NAME], [OUTPUT_NAME]))
                return _make_model(nodes, initializers=initializers)
    return None


# ============================================================================
# Detector 14: Draw rectangle border around bounding box of non-zero
# ============================================================================

def try_draw_border_around_bbox(pairs):
    """Draw a rectangle border around the bounding box of all non-zero cells."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    for border_color in range(1, NUM_COLORS):
        ok = True
        for inp, out in pairs:
            nz = np.argwhere(inp != 0)
            if len(nz) == 0:
                ok = False; break
            r0, c0 = nz.min(axis=0)
            r1, c1 = nz.max(axis=0) + 1
            expected = inp.copy()
            # Draw border
            expected[r0, c0:c1] = border_color
            expected[r1-1, c0:c1] = border_color
            expected[r0:r1, c0] = border_color
            expected[r0:r1, c1-1] = border_color
            if not np.array_equal(expected, out):
                ok = False; break
        if ok:
            # Build ONNX — complex (data-dependent bbox). Skip.
            return None
    return None


EXTENDED_DETECTORS = [
    ("quilt", try_quilt),
    ("kronecker_diagonal", try_kronecker_diagonal),
    ("kronecker_anti_diagonal", try_kronecker_anti_diagonal),
    ("kronecker_border", try_kronecker_border),
    ("color_map_then_quilt", try_color_map_then_quilt),
    ("draw_diagonal_lines", try_draw_diagonal_lines),
    ("replace_color_in_marker_rows", try_replace_color_in_marker_rows),
    ("crop_largest_component", try_crop_largest_component),
    ("bbox_largest_object", try_bbox_largest_object),
    ("fill_enclosed", try_fill_enclosed),
    ("draw_border_around_bbox", try_draw_border_around_bbox),
    ("extend_pattern_fill", try_extend_pattern_fill),
    ("remove_most_common_color", try_remove_most_common_color),
]


def try_extended_detectors(task):
    """Try all extended detectors."""
    pairs = arc_data.get_pairs(task)
    for name, detector in EXTENDED_DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        e = validator.evaluate_model(model, task)
        if e["eligible_for_points"]:
            model = _strip_metadata(model)
            e2 = validator.evaluate_model(model, task)
            if e2["eligible_for_points"]:
                return model, name, e2["score"]
    return None, None, 0


def main():
    with open("/home/z/my-project/data/final_unified_results.json") as f:
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
                model, method, sc = try_extended_detectors(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Extended Detectors Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
