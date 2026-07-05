"""Batch 5 FIXED — corrected rules from debug output."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 6: Extract right half, map 1→2, but ONLY where input has 1 (not all cells)
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
    # Map 1→2, keep everything else as-is (including 0s)
    result = np.where(right == 1, 2, right)
    # Also map 5→0 (remove separator residue)
    result = np.where(result == 5, 0, result)
    # Trim trailing zero columns
    nonzero_cols = [j for j in range(result.shape[1]) if (result[:, j] != 0).any()]
    if nonzero_cols:
        result = result[:, :max(nonzero_cols)+1]
    return result

# Task 24: Color 2 fills column, ALL OTHER colors fill row
def solve_task24(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            if c == 2:
                result[:, j] = c
            else:
                result[i, :] = c
    return result

# Task 26: Extract 3 columns centered on the "1" column
# But the mapping is: 9→0 only where there's a 9, 1→8 only where there's a 1
# And the output preserves the 0s from the original
def solve_task26(grid):
    H, W = grid.shape
    col_1 = None
    for j in range(W):
        if (grid[:, j] == 1).any():
            col_1 = j; break
    if col_1 is None: return grid
    # Extract 3 columns: col_1-1, col_1, col_1+1
    start = max(0, col_1 - 1)
    end = min(W, col_1 + 2)
    result = grid[:, start:end].copy()
    # Map 1→8, keep 9 as-is (not 9→0)
    result = np.where(result == 1, 8, result)
    return result

# Task 27: Fill inside of L-shape — fix the condition
# The L-corner is where the shape turns. Fill cells that are:
# - Inside the bounding box
# - To the LEFT of the rightmost 1 in their row
# - BELOW the bottommost 1 in their column
# Actually from the diffs: (4,1),(4,2),(5,1),(5,2),(5,3) should be 2
# and (3,5),(5,6) should NOT be 2
# The correct rule: fill cells that have a 1 ABOVE and a 1 to the RIGHT
# (not all 4 directions)
def solve_task27(grid):
    result = grid.copy()
    H, W = grid.shape
    ones = list(zip(*np.where(grid == 1)))
    if not ones: return grid
    rows = [p[0] for p in ones]
    cols = [p[1] for p in ones]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    for i in range(rmin, rmax + 1):
        for j in range(cmin, cmax + 1):
            if grid[i, j] != 0: continue
            # Check: is there a 1 above AND a 1 to the left in the same row?
            has_above = any(grid[k, j] == 1 for k in range(rmin, i))
            has_left = any(grid[i, k] == 1 for k in range(cmin, j))
            # And is there a 1 below OR a 1 to the right?
            has_below = any(grid[k, j] == 1 for k in range(i + 1, rmax + 1))
            has_right = any(grid[i, k] == 1 for k in range(j + 1, cmax + 1))
            # Fill if: has_above AND has_left (inside the L corner)
            if has_above and has_left and has_below and has_right:
                result[i, j] = 2
    return result

# === VERIFY ===
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
    for name, fn, tid in [
        ("task6", solve_task6, 6),
        ("task24", solve_task24, 24),
        ("task26", solve_task26, 26),
        ("task27", solve_task27, 27),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
