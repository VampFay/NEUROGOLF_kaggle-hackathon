"""Batch 7 v3 — more fixes."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 52: Keep only the row where ALL cells are the SAME color.
# Then replace that color with 5. Other rows become 0.
def solve_task52(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    for i in range(H):
        row = grid[i]
        nonzero = row[row != 0]
        if len(nonzero) > 0 and (nonzero == nonzero[0]).all() and len(nonzero) == W:
            result[i] = 5
    return result

# Task 56: Output = 2 if corners are filled (diagonal), 1 otherwise
# But pair 1 has corners filled AND center filled → still output 2
# Fix: check if ALL 4 corners are filled
def solve_task56(grid):
    H, W = grid.shape
    colors = grid[grid != 0]
    if len(colors) == 0: return np.array([[0]])
    majority = int(np.bincount(colors).argmax())
    positions = set(zip(*np.where(grid == majority)))
    corners = {(0,0), (0,W-1), (H-1,0), (H-1,W-1)}
    if corners.issubset(positions):
        return np.array([[2]])
    else:
        return np.array([[1]])

# Task 60: verified ✓ (keep as is)
def solve_task60(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        left = grid[i, 0] if grid[i, 0] != 0 else None
        right = grid[i, -1] if grid[i, -1] != 0 else None
        if left is not None and right is not None and left != right:
            mid = W // 2
            result[i, :mid] = left
            result[i, mid] = 5
            result[i, mid+1:] = right
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
        ("task52", solve_task52, 52),
        ("task56", solve_task56, 56),
        ("task60", solve_task60, 60),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
