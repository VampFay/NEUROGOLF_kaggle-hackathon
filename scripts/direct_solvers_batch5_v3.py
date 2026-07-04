"""Batch 5 v3 — final fixes."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 6: Extract right half (after separator), map 1→2, ALL else→0
def solve_task6(grid):
    H, W = grid.shape
    sep_col = None
    for j in range(W):
        col = grid[:, j]
        nonzero = col[col != 0]
        if len(nonzero) > 0 and (col == nonzero[0]).all():
            sep_col = j; break
    if sep_col is None: return grid
    right = grid[:, sep_col+1:]
    # Only keep 1s (mapped to 2), everything else → 0
    result = np.where(right == 1, 2, 0)
    # Trim trailing zero columns
    nonzero_cols = [j for j in range(result.shape[1]) if (result[:, j] != 0).any()]
    if nonzero_cols:
        result = result[:, :max(nonzero_cols)+1]
    return result

# Task 24: Fill rows first (for non-2 colors), then fill columns (for color 2)
def solve_task24(grid):
    result = grid.copy()
    H, W = grid.shape
    # First pass: fill rows for non-2 colors
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c == 2: continue
            result[i, :] = c
    # Second pass: fill columns for color 2 (overwrites row fills)
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 2:
                result[:, j] = c
    return result

# Task 26: Extract 3 cols around "1" col, map 1→8, keep 9s, drop 0s? No.
# From data: input has 9s and 1s. Output has 8s and 0s.
# 1→8, 9→0, 0→0. Simple color map.
def solve_task26(grid):
    H, W = grid.shape
    col_1 = None
    for j in range(W):
        if (grid[:, j] == 1).any():
            col_1 = j; break
    if col_1 is None: return grid
    start = max(0, col_1 - 1)
    end = min(W, col_1 + 2)
    result = grid[:, start:end].copy()
    # Map: 1→8, 9→0
    result = np.where(result == 1, 8, result)
    result = np.where(result == 9, 0, result)
    return result

# Task 27: Fill inside of L-shape
# Fix: fill cells that have 1 ABOVE and 1 LEFT (inside corner)
# But ALSO must have 1 BELOW and 1 RIGHT (truly enclosed)
# The issue was the bounding box was too wide. Need to check per-row and per-col.
def solve_task27(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0: continue
            # Check if this cell is "inside" the L
            # Has 1 above in same column
            has_above = any(grid[k, j] == 1 for k in range(0, i))
            # Has 1 below in same column
            has_below = any(grid[k, j] == 1 for k in range(i+1, H))
            # Has 1 left in same row
            has_left = any(grid[i, k] == 1 for k in range(0, j))
            # Has 1 right in same row
            has_right = any(grid[i, k] == 1 for k in range(j+1, W))
            if has_above and has_below and has_left and has_right:
                result[i, j] = 2
    return result

# === Also include ALL previously verified solvers ===
def solve_task3(grid):
    result = np.where(grid == 1, 2, grid)
    H, W = grid.shape
    for P in range(1, H):
        if all(np.array_equal(result[i], result[i + P]) for i in range(H - P)):
            break
    else:
        P = H
    out_h = H + H // 2
    return np.array([result[i % P] for i in range(out_h)])

def solve_task10(grid):
    H, W = grid.shape
    cols_with_5 = sorted(set(j for j in range(W) if (grid[:, j] == 5).any()))
    if not cols_with_5: return grid
    groups = []; current = [cols_with_5[0]]
    for c in cols_with_5[1:]:
        if c == current[-1] + 1: current.append(c)
        else: groups.append(current); current = [c]
    groups.append(current)
    heights = [(sum(1 for i in range(H) if grid[i, g[0]] == 5), idx) for idx, g in enumerate(groups)]
    heights.sort(reverse=True)
    color_for_group = {}
    for rank, (_, idx) in enumerate(heights):
        color_for_group[idx] = rank + 1
    result = grid.copy()
    for idx, group in enumerate(groups):
        c = color_for_group[idx]
        for col in group:
            result[grid[:, col] == 5, col] = c
    return result

def solve_task15(grid):
    result = grid.copy()
    H, W = grid.shape
    for color, fc in [(2, 4), (1, 7)]:
        positions = list(zip(*np.where(grid == color)))
        for r, c in positions:
            if color == 2:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
            elif color == 1:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
    return result

def solve_task40(grid):
    result = grid.copy()
    H, W = grid.shape
    left_vals = set(grid[i, 0] for i in range(H) if grid[i, 0] != 0)
    right_vals = set(grid[i, -1] for i in range(H) if grid[i, -1] != 0)
    top_vals = set(grid[0, j] for j in range(W) if grid[0, j] != 0)
    bot_vals = set(grid[-1, j] for j in range(W) if grid[-1, j] != 0)
    left_color = left_vals.pop() if len(left_vals) == 1 else None
    right_color = right_vals.pop() if len(right_vals) == 1 else None
    top_color = top_vals.pop() if len(top_vals) == 1 else None
    bot_color = bot_vals.pop() if len(bot_vals) == 1 else None
    boundary_colors = {c for c in [left_color, right_color, top_color, bot_color] if c is not None}
    if not boundary_colors: return grid
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c in boundary_colors: continue
            if left_color is not None and right_color is not None and left_color != right_color:
                result[i, j] = left_color if j < W/2 else right_color
            elif top_color is not None and bot_color is not None and top_color != bot_color:
                result[i, j] = top_color if i < H/2 else bot_color
    return result

def solve_task45(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        if grid[i, 0] != 0 and grid[i, -1] != 0 and grid[i, 0] == grid[i, -1]:
            result[i, :] = grid[i, 0]
    return result

def solve_task47(grid):
    H, W = grid.shape
    crosses = []
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            cross = np.zeros((H, W), dtype=int)
            cross[i, :] = c; cross[:, j] = c
            crosses.append((c, cross))
    result = grid.copy()
    fill_count = np.zeros((H, W), dtype=int)
    for c, cross in crosses:
        mask = cross != 0
        result[mask] = np.where(fill_count[mask] == 0, c, result[mask])
        fill_count[mask] += 1
    if len(crosses) >= 2:
        overlap = fill_count >= 2
        result[overlap] = 2
    return result

def solve_task49(grid):
    best_color = 0; best_area = 999; best_shape = None
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions: continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        area = len(positions)
        if area < best_area:
            best_area = area; best_color = color
            best_shape = grid[min(rows):max(rows)+1, min(cols):max(cols)+1]
    if best_color == 0 or best_shape is None: return grid
    return best_shape.copy()

# === VERIFY ALL ===
def verify(name, fn, tid):
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    for i, (inp, out) in enumerate(pairs):
        try:
            result = np.array(fn(inp.copy()))
            if result.shape != out.shape or not np.array_equal(result, out):
                diffs = int((result != out).sum()) if result.shape == out.shape else -1
                print(f"  {name} task {tid}: pair {i} FAIL ({diffs} diffs)")
                return False
        except Exception as e:
            print(f"  {name} task {tid}: pair {i} ERROR: {str(e)[:80]}")
            return False
    print(f"  {name} task {tid}: ALL {len(pairs)} PAIRS OK ✓")
    return True

if __name__ == "__main__":
    solved = 0; total = 0
    all_solvers = [
        ("task3", solve_task3, 3), ("task6", solve_task6, 6),
        ("task10", solve_task10, 10), ("task15", solve_task15, 15),
        ("task24", solve_task24, 24), ("task26", solve_task26, 26),
        ("task27", solve_task27, 27), ("task40", solve_task40, 40),
        ("task45", solve_task45, 45), ("task47", solve_task47, 47),
        ("task49", solve_task49, 49),
    ]
    for name, fn, tid in all_solvers:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
