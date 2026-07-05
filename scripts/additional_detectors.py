"""
Additional detectors for common ARC patterns found in unsolved tasks:
- Shift/translate: move all non-zero cells by (dr, dc)
- Shift with wraparound
- Extend row to the right with object color
- Crop right half / left half / specific columns
- Count non-zero → output size
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


def try_shift_translate(pairs):
    """Shift all non-zero cells by (dr, dc). Cells that go off-grid are dropped."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    in_h, in_w = pairs[0][0].shape
    if not all(inp.shape == (in_h, in_w) for inp, _ in pairs): return None
    
    # Try all shifts (dr, dc) in range [-5, 5]
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if dr == 0 and dc == 0: continue
            ok = True
            for inp, out in pairs:
                h, w = inp.shape
                expected = np.zeros_like(out)
                for r in range(h):
                    for c in range(w):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            expected[nr, nc] = inp[r, c]
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                # Build ONNX: Pad then Slice (or Slice then Pad)
                # To shift by (dr, dc): pad top/bottom by dr, left/right by dc, then crop
                # Pad: if dr > 0, pad top by dr; if dr < 0, pad bottom by -dr
                # Similarly for dc
                pad_top = max(0, dr)
                pad_bottom = max(0, -dr)
                pad_left = max(0, dc)
                pad_right = max(0, -dc)
                nodes = []
                initializers = []
                # Crop input to (in_h, in_w) first
                nodes.append(h.make_node("Constant", [], ["cs"],
                    value=h.make_tensor("csv", TensorProto.INT64, [4], [0, 0, 0, 0])))
                nodes.append(h.make_node("Constant", [], ["ce"],
                    value=h.make_tensor("cev", TensorProto.INT64, [4], [1, NUM_COLORS, in_h, in_w])))
                nodes.append(h.make_node("Constant", [], ["ca"],
                    value=h.make_tensor("cav", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["cr"]))
                # Pad
                if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
                    nodes.append(h.make_node("Identity", ["cr"], ["pd"]))
                else:
                    pads = [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]
                    initializers.append(h.make_tensor("pv", TensorProto.INT64, [8], pads))
                    initializers.append(h.make_tensor("pval", TensorProto.FLOAT, [1], [0.0]))
                    nodes.append(h.make_node("Pad", ["cr", "pv", "pval"], ["pd"], mode="constant"))
                # Now pd has shape (1, 10, in_h + |dr|, in_w + |dc|)
                # We need to crop to (in_h, in_w) starting at offset (pad_top + ... )
                # Actually after padding, the shifted content is at:
                # If dr > 0: content moved down by dr, so crop starting at row dr
                # If dr < 0: content moved up, so crop starting at row 0 (but bottom dr rows are gone)
                # The crop start should be: pad_top if dr > 0, else 0... wait
                # After padding: new grid has the content shifted.
                # Original cell (r, c) is now at (r + pad_top, c + pad_left) in padded grid.
                # We want output (nr, nc) = input (nr - dr, nc - dc).
                # In padded grid: output (nr, nc) = padded (nr + pad_top - dr, nc + pad_left - dc)
                #   = padded (nr + pad_top - dr, ...)
                # We want to crop starting at (pad_top - dr, pad_left - dc)... 
                # Wait, let me think again.
                # shift (dr, dc): output[r+dr, c+dc] = input[r, c]
                # So output[r, c] = input[r-dr, c-dc] (if in bounds)
                # After padding input with pad_top, pad_left:
                #   padded[r + pad_top, c + pad_left] = input[r, c]
                # So padded[r, c] = input[r - pad_top, c - pad_left]
                # We want output[r, c] = input[r - dr, c - dc] = padded[r - dr + pad_top, c - dc + pad_left]
                # So crop starts at (pad_top - dr, pad_left - dc)... but pad_top = max(0, dr)
                # If dr > 0: pad_top = dr, so crop starts at (dr - dr, ...) = (0, ...)
                # If dr < 0: pad_top = 0, so crop starts at (0 - dr, ...) = (-dr, ...) = (|dr|, ...)
                # That makes sense!
                crop_r_start = pad_top - dr  # = 0 if dr>=0, = -dr if dr<0
                crop_c_start = pad_left - dc
                # But wait — if dr > 0, crop_r_start = dr - dr = 0... that's wrong
                # Let me re-derive. shift (dr, dc) means: move content DOWN by dr, RIGHT by dc
                # output[r, c] = input[r - dr, c - dc]
                # If dr > 0 (move down): we pad top by dr, then the content starts at row dr in padded
                # We want output[r, c] = padded[r + dr - dr, ...] = padded[r, ...]... no
                # Actually: padded[dr, 0] = input[0, 0] (because we padded top by dr)
                # We want output[0, 0] = input[-dr, 0] which is out of bounds if dr > 0 → 0
                # And output[dr, 0] = input[0, 0] = padded[dr, 0]
                # So output = padded[dr:, ...] cropped to in_h rows
                # If dr > 0: crop starts at row dr (which is pad_top)
                # If dr < 0: pad_bottom = -dr, padded has extra rows at bottom. output = padded[0:in_h, ...]
                #   But wait, we padded bottom, so content is at top. output[0,0] = padded[0,0] = input[0,0]
                #   But we want output[0,0] = input[-dr, 0]... which is input[|dr|, 0]
                #   That's wrong. Let me reconsider.
                # 
                # OK I think the issue is: "shift by (dr, dc)" is ambiguous.
                # Let me define: shift(dr, dc) means output[r, c] = input[r - dr, c - dc]
                # So positive dr shifts content DOWN (content at row 0 moves to row dr)
                # 
                # To implement: pad input at TOP by max(0, dr) and at LEFT by max(0, dc)
                # and pad at BOTTOM by max(0, -dr) and RIGHT by max(0, -dc)
                # Then crop the region [pad_top : pad_top + in_h, pad_left : pad_left + in_w]... 
                # No wait. Let me just test with a simple case.
                # dr=1, dc=0: output[r, c] = input[r-1, c]. Row 0 of output = input[-1, c] = 0.
                # Pad top by 1: padded[1, c] = input[0, c]. Crop [1:1+in_h, :] → output[0, c] = padded[1, c] = input[0, c]. WRONG.
                # 
                # I think the issue is my shift direction. Let me re-check the Python:
                # expected[nr, nc] = inp[r, c] where nr = r + dr, nc = c + dc
                # So output[r + dr, c + dc] = input[r, c]
                # → output[r, c] = input[r - dr, c - dc]
                # For dr=1: output[1, c] = input[0, c], output[0, c] = input[-1, c] = 0
                # 
                # To build: pad input at top by dr (if dr > 0), then crop from row 0
                # padded[0, c] = 0 (padding), padded[1, c] = input[0, c]
                # output[0, c] = padded[0, c] = 0 ✓
                # output[1, c] = padded[1, c] = input[0, c] ✓
                # So: pad top by dr (if dr > 0), pad bottom by -dr (if dr < 0)
                # Then crop starting at row 0, but we need in_h rows from the padded grid
                # The padded grid has in_h + |dr| rows. We want the FIRST in_h rows (if dr > 0)
                # or the LAST in_h rows (if dr < 0)
                # If dr > 0: crop [0 : in_h, ...] from padded (which has in_h + dr rows)
                # If dr < 0: crop [-dr : -dr + in_h, ...] from padded (which has in_h + |dr| rows)
                # = crop [|dr| : |dr| + in_h, ...]
                # 
                # So crop_r_start = max(0, -dr) = pad_bottom
                # crop_c_start = max(0, -dc) = pad_right
                crop_r_start = pad_bottom  # = max(0, -dr)
                crop_c_start = pad_right   # = max(0, -dc)
                # Crop to (in_h, in_w)
                nodes.append(h.make_node("Constant", [], ["s2"],
                    value=h.make_tensor("s2v", TensorProto.INT64, [4], [0, 0, crop_r_start, crop_c_start])))
                nodes.append(h.make_node("Constant", [], ["e2"],
                    value=h.make_tensor("e2v", TensorProto.INT64, [4], [1, NUM_COLORS, crop_r_start + in_h, crop_c_start + in_w])))
                nodes.append(h.make_node("Constant", [], ["a2"],
                    value=h.make_tensor("a2v", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", ["pd", "s2", "e2", "a2"], ["sh"]))
                # Pad to MAX_GRID
                pad_b = MAX_GRID - in_h
                pad_r = MAX_GRID - in_w
                if pad_b == 0 and pad_r == 0:
                    nodes.append(h.make_node("Identity", ["sh"], [OUTPUT_NAME]))
                else:
                    pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
                    initializers.append(h.make_tensor("pv2", TensorProto.INT64, [8], pads))
                    initializers.append(h.make_tensor("pval2", TensorProto.FLOAT, [1], [0.0]))
                    nodes.append(h.make_node("Pad", ["sh", "pv2", "pval2"], [OUTPUT_NAME], mode="constant"))
                return _make_model(nodes, initializers=initializers)
    return None


def try_crop_columns(pairs):
    """Crop specific columns: right half, left half, or specific column range."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
        if inp.shape[0] != out_h: return None
    
    # Try: output = input[:, c0:c1] for some fixed c0, c1
    # Check if all pairs use the same column range
    for c0 in range(MAX_GRID):
        for c1 in range(c0 + 1, MAX_GRID + 1):
            if c1 - c0 != out_w: continue
            ok = True
            for inp, out in pairs:
                if inp.shape[1] < c1: 
                    ok = False; break
                if not np.array_equal(inp[:, c0:c1], out):
                    ok = False; break
            if ok:
                # Build ONNX: crop columns c0:c1
                nodes = []
                nodes.append(h.make_node("Constant", [], ["s"],
                    value=h.make_tensor("sv", TensorProto.INT64, [4], [0, 0, 0, c0])))
                nodes.append(h.make_node("Constant", [], ["e"],
                    value=h.make_tensor("ev", TensorProto.INT64, [4], [1, NUM_COLORS, out_h, c1])))
                nodes.append(h.make_node("Constant", [], ["a"],
                    value=h.make_tensor("av", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]))
                return _make_model(nodes)
    return None


def try_crop_rows(pairs):
    """Crop specific rows: top half, bottom half, or specific row range."""
    if not pairs: return None
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
        if inp.shape[1] != out_w: return None
    
    for r0 in range(MAX_GRID):
        for r1 in range(r0 + 1, MAX_GRID + 1):
            if r1 - r0 != out_h: continue
            ok = True
            for inp, out in pairs:
                if inp.shape[0] < r1:
                    ok = False; break
                if not np.array_equal(inp[r0:r1, :], out):
                    ok = False; break
            if ok:
                nodes = []
                nodes.append(h.make_node("Constant", [], ["s"],
                    value=h.make_tensor("sv", TensorProto.INT64, [4], [0, 0, r0, 0])))
                nodes.append(h.make_node("Constant", [], ["e"],
                    value=h.make_tensor("ev", TensorProto.INT64, [4], [1, NUM_COLORS, r1, out_w])))
                nodes.append(h.make_node("Constant", [], ["a"],
                    value=h.make_tensor("av", TensorProto.INT64, [4], [0, 1, 2, 3])))
                nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]))
                return _make_model(nodes)
    return None


def try_extend_row_right(pairs):
    """For each row, extend the rightmost non-zero cell's color to fill the rest of the row."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    ok = True
    for inp, out in pairs:
        h, w = inp.shape
        for r in range(h):
            # Find rightmost non-zero in input row
            nonzeros = np.where(inp[r] != 0)[0]
            if len(nonzeros) > 0:
                c_last = nonzeros[-1]
                color = int(inp[r, c_last])
                # Output: cells c_last..w-1 should be `color`
                for c in range(c_last, w):
                    if int(out[r, c]) != color:
                        ok = False; break
                # Cells before c_last should match input
                for c in range(c_last):
                    if int(out[r, c]) != int(inp[r, c]):
                        ok = False; break
            else:
                # No non-zero in row — output should be all zero
                if not np.all(out[r] == 0):
                    ok = False; break
        if not ok: break
    if ok:
        # Build ONNX — complex (data-dependent). Skip.
        return None
    return None


def try_partial_repeat_with_colormap(pairs):
    """Partial repeat: output = color_map(input) repeated 1.5x (6→9 rows: rows 0-5 then 0-2)."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_w != in_w: return None
    if out_h <= in_h: return None
    # Check if out_h = in_h + in_h//2 (1.5x)
    if out_h != in_h + in_h // 2: return None
    n_extra = out_h - in_h  # = in_h // 2
    
    # Derive color map from first in_h rows of output vs input
    mapping = {}
    ok = True
    for inp, out in pairs:
        for c in range(NUM_COLORS):
            in_cells = (inp == c)
            if not in_cells.any(): continue
            # Check corresponding output cells in first in_h rows
            out_sub = out[:in_h]
            out_at = out_sub[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1:
                ok = False; break
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t:
                ok = False; break
            mapping[c] = t
        if not ok: break
    if not ok: return None
    if not any(k != v for k, v in mapping.items()): return None
    
    # Verify: output = color_map(input) then repeat first n_extra rows
    for inp, out in pairs:
        mapped = inp.copy()
        for k, v in mapping.items():
            mapped[inp == k] = v
        expected = np.vstack([mapped, mapped[:n_extra]])
        if not np.array_equal(expected, out):
            return None
    
    # Build ONNX: color_map + concat (rows 0..in_h-1 + rows 0..n_extra-1)
    nodes = []
    initializers = []
    full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
    W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
    for frm, to in full_map.items():
        W[to, frm, 0, 0] = 1.0
    initializers.append(h.make_tensor("w", TensorProto.FLOAT,
        [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist()))
    nodes.append(h.make_node("Conv", [INPUT_NAME, "w"], ["cm"],
        pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
    # Crop to (in_h, in_w)
    nodes.append(h.make_node("Constant", [], ["cs"],
        value=h.make_tensor("csv", TensorProto.INT64, [4], [0, 0, 0, 0])))
    nodes.append(h.make_node("Constant", [], ["ce"],
        value=h.make_tensor("cev", TensorProto.INT64, [4], [1, NUM_COLORS, in_h, in_w])))
    nodes.append(h.make_node("Constant", [], ["ca"],
        value=h.make_tensor("cav", TensorProto.INT64, [4], [0, 1, 2, 3])))
    nodes.append(h.make_node("Slice", ["cm", "cs", "ce", "ca"], ["base"]))
    # Slice first n_extra rows
    nodes.append(h.make_node("Constant", [], ["es"],
        value=h.make_tensor("esv", TensorProto.INT64, [4], [0, 0, 0, 0])))
    nodes.append(h.make_node("Constant", [], ["ee"],
        value=h.make_tensor("eev", TensorProto.INT64, [4], [1, NUM_COLORS, n_extra, in_w])))
    nodes.append(h.make_node("Constant", [], ["ea"],
        value=h.make_tensor("eav", TensorProto.INT64, [4], [0, 1, 2, 3])))
    nodes.append(h.make_node("Slice", ["base", "es", "ee", "ea"], ["extra"]))
    # Concat base + extra along axis 2 (rows)
    nodes.append(h.make_node("Concat", ["base", "extra"], ["conc"], axis=2))
    # Pad to MAX_GRID
    pad_b = MAX_GRID - out_h
    pad_r = MAX_GRID - out_w
    if pad_b == 0 and pad_r == 0:
        nodes.append(h.make_node("Identity", ["conc"], [OUTPUT_NAME]))
    else:
        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
        initializers.append(h.make_tensor("pv", TensorProto.INT64, [8], pads))
        initializers.append(h.make_tensor("pval", TensorProto.FLOAT, [1], [0.0]))
        nodes.append(h.make_node("Pad", ["conc", "pv", "pval"], [OUTPUT_NAME], mode="constant"))
    return _make_model(nodes, initializers=initializers)


ADDITIONAL_DETECTORS = [
    ("shift_translate", try_shift_translate),
    ("crop_columns", try_crop_columns),
    ("crop_rows", try_crop_rows),
    ("partial_repeat_with_colormap", try_partial_repeat_with_colormap),
    ("extend_row_right", try_extend_row_right),
]


def try_additional_detectors(task):
    """Try all additional detectors."""
    pairs = arc_data.get_pairs(task)
    for name, detector in ADDITIONAL_DETECTORS:
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
                model, method, sc = try_additional_detectors(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Additional Detectors Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
