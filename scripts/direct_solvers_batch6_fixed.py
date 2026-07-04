"""Batch 6 FIXED — corrected rules from debug."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 28: Fill marker row and GRID EDGE only (not section boundary)
def solve_task28(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    markers = []
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                markers.append((i, j, int(grid[i, j])))
    if not markers: return grid
    markers.sort()
    # Split at midpoint between consecutive markers
    sections = []
    prev_end = -1
    for idx, (r, c, color) in enumerate(markers):
        if idx < len(markers) - 1:
            end = (r + markers[idx + 1][0]) // 2
        else:
            end = H - 1
        sections.append((prev_end + 1, end, r, color))
        prev_end = end
    for start, end, marker_row, color in sections:
        for i in range(start, end + 1):
            if i == marker_row:
                result[i, :] = color
            elif i == 0 or i == H - 1:
                result[i, :] = color
            else:
                result[i, 0] = color
                result[i, -1] = color
    return result

# Task 30: Shift blocks toward center until they overlap
def solve_task30(grid):
    H, W = grid.shape
    color_ranges = {}
    for c in range(1, 10):
        rows = [i for i in range(H) if (grid[i] == c).any()]
        if rows:
            color_ranges[c] = (min(rows), max(rows))
    if len(color_ranges) < 2: return grid
    # Find center
    all_starts = [r[0] for r in color_ranges.values()]
    all_ends = [r[1] for r in color_ranges.values()]
    center = (max(all_starts) + min(all_ends)) // 2
    result = np.zeros_like(grid)
    for c, (rmin, rmax) in color_ranges.items():
        block_height = rmax - rmin + 1
        # Shift so block is centered on `center`
        new_rmin = center - block_height // 2
        new_rmax = new_rmin + block_height - 1
        shift = new_rmin - rmin
        for i in range(H):
            for j in range(W):
                if grid[i, j] == c:
                    ni = i + shift
                    if 0 <= ni < H:
                        result[ni, j] = c
    return result

# Task 34: 3-wide diagonal from marker, centered on the "2" position
def solve_task34(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 4 and j + 1 < W and grid[i, j + 1] == 2:
                # Diagonal center starts at col j+1, going up-right
                center_col = j + 1
                for dr in range(H):
                    r = i - dr
                    c = center_col + dr
                    if r < 0 or c >= W: break
                    for dc in range(-1, 2):
                        cc = c + dc
                        if 0 <= cc < W and result[r, cc] == 0:
                            result[r, cc] = 4
                return result
            elif grid[i, j] == 4 and j > 0 and grid[i, j - 1] == 2:
                # Diagonal going up-left
                center_col = j - 1
                for dr in range(H):
                    r = i - dr
                    c = center_col - dr
                    if r < 0 or c < 0: break
                    for dc in range(-1, 2):
                        cc = c + dc
                        if 0 <= cc < W and result[r, cc] == 0:
                            result[r, cc] = 4
                return result
    return result

# Task 41: Fill between same-color markers only
def solve_task41(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        # Group non-zero positions by color
        color_positions = {}
        for j in range(W):
            c = int(grid[i, j])
            if c != 0:
                color_positions.setdefault(c, []).append(j)
        # Fill between leftmost and rightmost of each color
        for c, positions in color_positions.items():
            if len(positions) >= 2:
                left, right = positions[0], positions[-1]
                result[i, left:right + 1] = c
    return result

# Task 32: Stack non-zero to bottom (already working)
def solve_task32(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    for j in range(W):
        col = grid[:, j]
        nonzero = col[col != 0]
        result[H - len(nonzero):, j] = nonzero
    return result

# Task 35: Marker propagates to nearest 8-cell (already working)
def solve_task35(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c == 8: continue
            for k in range(j + 1, W):
                if grid[i, k] == 8: result[i, k] = c; break
                elif grid[i, k] != 0 and grid[i, k] != 8: break
            for k in range(j - 1, -1, -1):
                if grid[i, k] == 8: result[i, k] = c; break
                elif grid[i, k] != 0 and grid[i, k] != 8: break
            for k in range(i + 1, H):
                if grid[k, j] == 8: result[k, j] = c; break
                elif grid[k, j] != 0 and grid[k, j] != 8: break
            for k in range(i - 1, -1, -1):
                if grid[k, j] == 8: result[k, j] = c; break
                elif grid[k, j] != 0 and grid[k, j] != 8: break
    return result

# Task 43: Copy row 0 pattern to marker rows (already working)
def solve_task43(grid):
    result = grid.copy()
    H, W = grid.shape
    if H < 2: return grid
    pattern = grid[0]
    marker_color = None
    for j in range(W):
        if pattern[j] != 0:
            marker_color = pattern[j]; break
    if marker_color is None: return grid
    for i in range(1, H):
        if grid[i, -1] != 0:
            for j in range(W):
                if pattern[j] != 0 and pattern[j] == marker_color:
                    result[i, j] = 2
    return result

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
    for name, fn, tid in [
        ("task28", solve_task28, 28),
        ("task30", solve_task30, 30),
        ("task32", solve_task32, 32),
        ("task34", solve_task34, 34),
        ("task35", solve_task35, 35),
        ("task41", solve_task41, 41),
        ("task43", solve_task43, 43),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
