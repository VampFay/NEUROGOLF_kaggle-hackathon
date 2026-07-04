"""Batch 6 — analyzed from task data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 28: Single marker creates rectangular frame sections
# Each marker at (r,c) fills its row and the nearest grid edge row.
# Side borders (col 0 and W-1) for all rows in the section.
def solve_task28(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    # Find all single-cell markers
    markers = []
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                markers.append((i, j, int(grid[i, j])))
    if not markers: return grid
    markers.sort()
    # Split grid into sections at midpoint between markers
    sections = []
    prev_end = -1
    for idx, (r, c, color) in enumerate(markers):
        if idx < len(markers) - 1:
            next_r = markers[idx + 1][0]
            end = (r + next_r) // 2
        else:
            end = H - 1
        sections.append((prev_end + 1, end, r, color))
        prev_end = end
    # Fill each section
    for start, end, marker_row, color in sections:
        for i in range(start, end + 1):
            if i == marker_row:
                result[i, :] = color  # fill row
            elif i == start or i == end:
                result[i, :] = color  # fill edge row
            else:
                result[i, 0] = color
                result[i, -1] = color  # side borders
    return result

# Task 30: Keep only the overlapping rows of multi-colored blocks
def solve_task30(grid):
    H, W = grid.shape
    # Find row ranges for each color
    color_ranges = {}
    for c in range(1, 10):
        rows = [i for i in range(H) if (grid[i] == c).any()]
        if rows:
            color_ranges[c] = (min(rows), max(rows))
    if len(color_ranges) < 2: return grid
    # Find overlap: max of mins, min of maxes
    max_start = max(r[0] for r in color_ranges.values())
    min_end = min(r[1] for r in color_ranges.values())
    if max_start > min_end: return grid
    # Keep only cells in the overlapping range
    result = np.zeros_like(grid)
    for c, (rmin, rmax) in color_ranges.items():
        for i in range(max_start, min_end + 1):
            for j in range(W):
                if grid[i, j] == c:
                    result[i, j] = c
    return result

# Task 32: Stack all non-zero values to bottom of each column
def solve_task32(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    for j in range(W):
        # Get non-zero values in this column (top to bottom order)
        col = grid[:, j]
        nonzero = col[col != 0]
        # Place at bottom
        result[H - len(nonzero):, j] = nonzero
    return result

# Task 34: Draw 3-wide diagonal stripe going up-right from marker
def solve_task34(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find marker (color 4 adjacent to color 2)
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 4:
                # Draw 3-wide diagonal going up-right
                for dr in range(H):
                    r = i - dr
                    c = j + dr
                    if r < 0 or c >= W: break
                    for dc in range(3):
                        cc = c - 1 + dc
                        if 0 <= cc < W and result[r, cc] == 0:
                            result[r, cc] = 4
                return result
    return result

# Task 41: Fill triangle between pairs of same-color markers
def solve_task41(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        # Find leftmost and rightmost non-zero in this row
        nonzero = [j for j in range(W) if grid[i, j] != 0]
        if len(nonzero) >= 2:
            c = grid[i, nonzero[0]]  # color of the markers
            left, right = nonzero[0], nonzero[-1]
            result[i, left:right+1] = c
    return result

# Task 43: Copy row 0 pattern to rows with marker at last column
# Replace marker color with 2 in the copy
def solve_task43(grid):
    result = grid.copy()
    H, W = grid.shape
    if H < 2: return grid
    pattern = grid[0]
    marker_color = None
    for j in range(W):
        if pattern[j] != 0:
            marker_color = pattern[j]
            break
    if marker_color is None: return grid
    for i in range(1, H):
        if grid[i, -1] != 0:
            # Copy pattern, replace marker_color with 2
            for j in range(W):
                if pattern[j] != 0 and pattern[j] == marker_color:
                    result[i, j] = 2
    return result

# Task 35: Marker color propagates to nearest 8-cell in same row/col
def solve_task35(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c == 8: continue
            # Propagate right in same row
            for k in range(j + 1, W):
                if grid[i, k] == 8:
                    result[i, k] = c
                    break
                elif grid[i, k] != 0 and grid[i, k] != 8:
                    break
            # Propagate left
            for k in range(j - 1, -1, -1):
                if grid[i, k] == 8:
                    result[i, k] = c
                    break
                elif grid[i, k] != 0 and grid[i, k] != 8:
                    break
            # Propagate down
            for k in range(i + 1, H):
                if grid[k, j] == 8:
                    result[k, j] = c
                    break
                elif grid[k, j] != 0 and grid[k, j] != 8:
                    break
            # Propagate up
            for k in range(i - 1, -1, -1):
                if grid[k, j] == 8:
                    result[k, j] = c
                    break
                elif grid[k, j] != 0 and grid[k, j] != 8:
                    break
    return result

# Task 33: Copy pattern between sections defined by vertical 8-lines
def solve_task33(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find vertical 8-lines (columns where all rows have 8)
    v_lines = [j for j in range(W) if (grid[:, j] == 8).all()]
    if len(v_lines) < 1: return grid
    # Add grid boundaries
    boundaries = [-1] + v_lines + [W]
    # For each section between boundaries, find the pattern and copy to other sections
    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        if start <= end:
            sections.append((start, end))
    if len(sections) < 2: return result
    # Find the section with a pattern (non-8, non-0 content)
    pattern_section = None
    for idx, (s, e) in enumerate(sections):
        for i in range(H):
            for j in range(s, e + 1):
                if grid[i, j] != 0 and grid[i, j] != 8:
                    pattern_section = idx
                    break
            if pattern_section is not None: break
    if pattern_section is None: return result
    ps, pe = sections[pattern_section]
    pw = pe - ps + 1
    # Copy pattern to all other sections
    for idx, (s, e) in enumerate(sections):
        if idx == pattern_section: continue
        sw = e - s + 1
        for i in range(H):
            for j in range(sw):
                src_j = ps + j
                dst_j = s + j
                if dst_j <= e and grid[i, src_j] != 8:
                    if result[i, dst_j] == 0 or result[i, dst_j] == 8:
                        result[i, dst_j] = grid[i, src_j]
    return result

# Task 36: Count objects, output as small grid
# From data: input 30x30, output 5x3. 
# The output encodes the count of 1s and 5s as a pattern of 3s.
# Actually: output is a visual representation of the count.
# Pair 0: many 1s and 5s → output is all 3s (5x3 = 15 cells of 3)
# Pair 2: fewer 2s → output is 3x3 with 4s
# This is complex — skip for now
def solve_task36(grid):
    return grid  # placeholder

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
        ("task28", solve_task28, 28),
        ("task30", solve_task30, 30),
        ("task32", solve_task32, 32),
        ("task34", solve_task34, 34),
        ("task35", solve_task35, 35),
        ("task41", solve_task41, 41),
        ("task43", solve_task43, 43),
        ("task33", solve_task33, 33),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
