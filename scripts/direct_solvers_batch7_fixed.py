"""Batch 7 FIXED — corrected rules from debug."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 42: Draw diagonal lines — also extend beyond markers to grid edge
def solve_task42(grid):
    result = grid.copy()
    H, W = grid.shape
    markers = {}
    for i in range(H):
        for j in range(W):
            c = int(grid[i, j])
            if c != 0:
                markers.setdefault(c, []).append((i, j))
    for c, positions in markers.items():
        if len(positions) != 2: continue
        (r1, c1), (r2, c2) = positions
        dr = r2 - r1; dc = c2 - c1
        if abs(dr) != abs(dc): continue
        sr = 1 if dr > 0 else -1; sc = 1 if dc > 0 else -1
        # Fill between markers
        steps = abs(dr)
        for k in range(1, steps):
            r = r1 + k * sr; cc = c1 + k * sc
            if 0 <= r < H and 0 <= cc < W and result[r, cc] == 0:
                result[r, cc] = 8
        # Extend beyond both markers to grid edges
        for k in range(-max(H, W), max(H, W)):
            r = r1 + k * sr; cc = c1 + k * sc
            if 0 <= r < H and 0 <= cc < W and result[r, cc] == 0:
                # Only fill if on the same diagonal line
                if (r - r1) * sc == (cc - c1) * sr:
                    result[r, cc] = 8
    return result

# Task 51: Extend only if there's a "seed" pattern (3+ connected cells)
def solve_task51(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        # Find connected blocks of same color
        row = grid[i]
        j = 0
        while j < W:
            if row[j] != 0:
                color = row[j]; start = j
                while j < W and row[j] == color:
                    j += 1
                block_len = j - start
                # Only extend if block is >= 3 cells AND there's a gap after
                if block_len >= 3 and j < W and row[j] == 0:
                    # Check if there's another block after the gap
                    has_more = any(row[k] != 0 for k in range(j, W))
                    if not has_more:
                        # Extend to right edge
                        for k in range(j, W):
                            result[i, k] = color
            else:
                j += 1
    return result

# Task 56: Output = 1 if shape is plus/cross, 2 if shape is X/diagonal
def solve_task56(grid):
    H, W = grid.shape
    colors = grid[grid != 0]
    if len(colors) == 0: return np.array([[0]])
    majority = int(np.bincount(colors).argmax())
    # Check if the shape is diagonal (corners filled, center empty)
    positions = list(zip(*np.where(grid == majority)))
    corners = [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]
    center = (H//2, W//2)
    corner_count = sum(1 for p in positions if p in corners)
    center_filled = center in positions
    if corner_count >= 2 and not center_filled:
        return np.array([[2]])  # diagonal/X shape
    else:
        return np.array([[1]])  # plus/connected shape

# Task 60: Fill row — left color up to midpoint, then 5, then right color
# From data: 1.........2 → 11111522222
# The "5" is at the exact midpoint
def solve_task60(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        left = grid[i, 0] if grid[i, 0] != 0 else None
        right = grid[i, -1] if grid[i, -1] != 0 else None
        if left is not None and right is not None and left != right:
            mid = W // 2
            result[i, :mid] = left
            result[i, mid] = 5  # midpoint gets 5
            result[i, mid+1:] = right
    return result

# Task 52: Keep only rows where all non-zero cells are the same color
def solve_task52(grid):
    H, W = grid.shape
    keep_rows = []
    for i in range(H):
        row = grid[i]
        nonzero = row[row != 0]
        if len(nonzero) > 0 and (nonzero == nonzero[0]).all():
            keep_rows.append(row)
    if not keep_rows: return grid
    return np.array(keep_rows)

# Task 59: Replace non-5 markers with 5, and fill adjacent cells
def solve_task59(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find rows/cols with 5-barriers
    h_barriers = [i for i in range(H) if (grid[i] == 5).all()]
    if not h_barriers: return grid
    # For each section between barriers, replace stray colors with 5
    for i in range(H):
        if i in h_barriers: continue
        for j in range(W):
            if grid[i, j] != 0 and grid[i, j] != 5:
                result[i, j] = 5
    # Also: fill the "corner" where sections meet
    # From data: cells adjacent to barriers get filled
    for i in range(H):
        for j in range(W):
            if result[i, j] == 0:
                # Check if adjacent to a 5-barrier
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and result[ni, nj] == 5:
                        # Check if this is a corner cell
                        if (i in h_barriers or ni in h_barriers):
                            result[i, j] = 5
                        break
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
        ("task42", solve_task42, 42),
        ("task51", solve_task51, 51),
        ("task52", solve_task52, 52),
        ("task56", solve_task56, 56),
        ("task60", solve_task60, 60),
        ("task59", solve_task59, 59),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
