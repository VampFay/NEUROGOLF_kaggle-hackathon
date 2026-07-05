"""Batch 5 — analyzed from task data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 5: Repeat pattern to fill grid horizontally and vertically
# The seed pattern (e.g., "888" with "3" next to it) gets repeated to fill the row
# Then the row pattern gets repeated vertically
def solve_task5(grid):
    H, W = grid.shape
    result = grid.copy()
    # Find all non-zero content rows
    content_rows = [i for i in range(H) if (grid[i] != 0).any()]
    if not content_rows: return grid
    # For each content row, find the repeating pattern and fill
    for i in content_rows:
        row = grid[i]
        nonzero_cols = [j for j in range(W) if row[j] != 0]
        if len(nonzero_cols) < 2: continue
        # Find the pattern: the smallest repeating unit
        # Look at the gap between first and second non-zero block
        first_block_end = nonzero_cols[0]
        for j in nonzero_cols[1:]:
            if row[j] != row[nonzero_cols[0]]:
                first_block_end = j - 1
                break
        # Find next block start
        next_start = first_block_end + 1
        while next_start < W and row[next_start] == 0:
            next_start += 1
        if next_start >= W: continue
        period = next_start - nonzero_cols[0]
        pattern = row[nonzero_cols[0]:first_block_end+1].copy()
        # Fill the row with the pattern
        for j in range(W):
            pos = (j - nonzero_cols[0]) % period
            if pos < len(pattern) and result[i, j] == 0:
                result[i, j] = pattern[pos]
    # Now fill vertically: find vertical period and fill
    content_cols = [j for j in range(W) if (result[:, j] != 0).any()]
    if not content_cols: return result
    # Find vertical pattern
    for j in content_cols:
        col = result[:, j]
        nonzero_rows = [i for i in range(H) if col[i] != 0]
        if len(nonzero_rows) < 2: continue
        first_end = nonzero_rows[0]
        for i in nonzero_rows[1:]:
            if col[i] != col[nonzero_rows[0]]:
                first_end = i - 1
                break
        next_start = first_end + 1
        while next_start < H and col[next_start] == 0:
            next_start += 1
        if next_start >= H: continue
        vperiod = next_start - nonzero_rows[0]
        vpattern = col[nonzero_rows[0]:first_end+1].copy()
        for i in range(H):
            pos = (i - nonzero_rows[0]) % vperiod
            if pos < len(vpattern) and result[i, j] == 0:
                result[i, j] = vpattern[pos]
    return result

# Task 6: Extract right half of grid (after separator column 5), map 1→2
def solve_task6(grid):
    H, W = grid.shape
    # Find separator column (all same non-zero color)
    sep_col = None
    for j in range(W):
        col = grid[:, j]
        nonzero = col[col != 0]
        if len(nonzero) > 0 and (col == nonzero[0]).all():
            sep_col = j
            break
    if sep_col is None: return grid
    # Extract right half
    right = grid[:, sep_col+1:]
    # Map color 1 → 2
    result = np.where(right == 1, 2, right)
    # Keep only non-zero columns (trim trailing zeros)
    nonzero_cols = [j for j in range(result.shape[1]) if (result[:, j] != 0).any()]
    if nonzero_cols:
        result = result[:, :max(nonzero_cols)+1]
    return result

# Task 8: Move top shape down to fill gap above bottom shape
def solve_task8(grid):
    H, W = grid.shape
    result = np.zeros_like(grid)
    # Find all shapes (by color)
    shapes = {}
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if positions:
            rows = [p[0] for p in positions]
            shapes[color] = (min(rows), max(rows), positions)
    if len(shapes) < 2: return grid
    # Sort by row position
    sorted_shapes = sorted(shapes.items(), key=lambda x: x[1][0])
    top_color, (top_rmin, top_rmax, top_pos) = sorted_shapes[0]
    # Find the shape below
    for bot_color, (bot_rmin, bot_rmax, bot_pos) in sorted_shapes[1:]:
        if bot_rmin > top_rmax:
            # Move top shape so its bottom touches bot shape's top
            shift = bot_rmin - top_rmax - 1
            if shift > 0:
                for r, c in top_pos:
                    result[r + shift, c] = top_color
            else:
                for r, c in top_pos:
                    result[r, c] = top_color
            # Keep bottom shape in place
            for r, c in bot_pos:
                result[r, c] = bot_color
            # Keep any other shapes in place
            for color, (rmin, rmax, pos) in sorted_shapes[2:]:
                for r, c in pos:
                    result[r, c] = color
            return result
    return grid

# Task 9: Copy block pattern to fill the row between barriers
def solve_task9(grid):
    H, W = grid.shape
    result = grid.copy()
    # Find barrier rows (all same color)
    barriers = [i for i in range(H) if len(np.unique(grid[i][grid[i]!=0])) == 1 and (grid[i] != 0).all()]
    if len(barriers) < 2: return result
    # Process sections between barriers
    for bi in range(len(barriers) - 1):
        r_start = barriers[bi] + 1
        r_end = barriers[bi + 1]
        for r in range(r_start, r_end):
            row = result[r]
            # Find all non-zero blocks
            blocks = []
            j = 0
            while j < W:
                if row[j] != 0:
                    start = j
                    while j < W and row[j] == row[start]:
                        j += 1
                    blocks.append((start, j - 1, row[start]))
                else:
                    j += 1
            if len(blocks) >= 2:
                # The first block defines the pattern; copy it to fill gaps
                first = blocks[0]
                block_color = first[2]
                block_len = first[1] - first[0] + 1
                gap = blocks[1][0] - first[1] - 1
                period = block_len + gap
                for j in range(first[0], W):
                    pos = (j - first[0]) % period
                    if pos < block_len and result[r, j] == 0:
                        result[r, j] = block_color
    return result

# Task 21: Output = top-left block of the majority color
# The grid has horizontal bands separated by full-width rows.
# Output = the first 2 rows of the dominant color, trimmed to content width.
def solve_task21(grid):
    H, W = grid.shape
    # Find the separator rows (all same color, full width)
    sep_rows = []
    for i in range(H):
        nonzero = grid[i][grid[i] != 0]
        if len(nonzero) > 0 and (grid[i] == nonzero[0]).all() and nonzero[0] != 0:
            sep_rows.append(i)
    if len(sep_rows) < 1: return grid
    # The output is the content between separators, for the first section
    # Actually: output = the UNIQUE colors in the first 2 rows, as a compact grid
    # From pair 0: rows 0-1 are "373333333373373" → output "3333"
    # That's: take the first 2 rows, find the most common color, output as 2×W'
    # Actually output is 2×4 of color 3. The grid has color 3 in rows 0-1.
    # Pair 2: rows 0-2 are "11118111111" → output "11" (2×2)
    # So: output = first 2 rows, keep only the majority color, trim to bounding box
    top_rows = grid[:2] if H >= 2 else grid[:1]
    # Find majority color in top rows
    colors, counts = np.unique(top_rows[top_rows != 0], return_counts=True)
    if len(colors) == 0: return grid
    majority = colors[np.argmax(counts)]
    # Keep only majority color cells
    mask = top_rows == majority
    # Find bounding box
    rows_with = [i for i in range(top_rows.shape[0]) if mask[i].any()]
    cols_with = [j for j in range(top_rows.shape[1]) if mask[:, j].any()]
    if not rows_with or not cols_with: return grid
    result = top_rows[min(rows_with):max(rows_with)+1, min(cols_with):max(cols_with)+1]
    return result

# Task 23: Replace color 5 with two colors based on position in shape
# Left part → 8, right part → 2 (split at the narrowest point)
def solve_task23(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find all 5-cells
    positions = list(zip(*np.where(grid == 5)))
    if not positions: return grid
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    # For each row, find the split point (where the shape is narrowest)
    for r in range(rmin, rmax + 1):
        row_5_cols = sorted([c for c in range(W) if grid[r, c] == 5])
        if len(row_5_cols) < 2: continue
        # Find the gap (where consecutive 5s stop)
        gaps = []
        for i in range(len(row_5_cols) - 1):
            if row_5_cols[i + 1] - row_5_cols[i] > 1:
                gaps.append((row_5_cols[i], row_5_cols[i + 1]))
        if gaps:
            # Split at the first gap: left → 8, right → 2
            split = gaps[0][0]
            for c in row_5_cols:
                if c <= split:
                    result[r, c] = 8
                else:
                    result[r, c] = 2
        else:
            # No gap: entire row → 8
            for c in row_5_cols:
                result[r, c] = 8
    return result

# Task 24: Diagonal cells fill column, off-diagonal fill row
# But pair 1 has different layout. Let me check the general rule.
# From pair 0: cell (2,2)=2 fills column 2. Cell (4,7)=3 fills row 4.
# Cell (6,3)=1 fills row 6.
# The rule: single-cell markers fill their row AND column.
# But (2,2) only fills column 2, not row 2. Why?
# Because (2,2) is ON THE DIAGONAL → fills column only.
# (4,7) is OFF diagonal → fills row only.
# (6,3) is off diagonal → fills row only.
# Pair 1: (1,1)=3 is on diagonal → fills row 1 (not column). Wait, that contradicts.
# Let me re-check pair 1:
# Input: .3...... at (1,0). Output: 33333333 at row 1. So (1,0) fills row 1.
# (3,3)=3 is on diagonal → fills row 3. Output: 33333333 at row 3. Yes, fills row.
# So the rule is: EVERY marker fills its ROW. Not column.
# But pair 0: (2,2)=2 fills column 2 (output has 2 in all rows at col 2).
# That contradicts "fill row only".
# Let me re-examine pair 0:
# Output: ..2...... at rows 0,1,3,5,7,8. That's column 2 filled.
# Output: 333333333 at row 4. That's row 4 filled.
# Output: 111111111 at row 6. That's row 6 filled.
# So: marker at (2,2) fills COLUMN, markers at (4,7) and (6,3) fill ROWS.
# The difference: (2,2) has row==col (diagonal). (4,7) and (6,3) don't.
# Pair 1: (1,0) has row≠col → fill row. (3,3) has row==col → fill row? 
# But output row 3 is 33333333. Let me check if column 3 is also filled.
# Pair 1 output:
# .....2.. at row 0. Wait, that's a 2 at col 5, not related to 3.
# Actually pair 1 input has TWO markers: 3 at (1,0) and 3 at (3,3).
# Both are color 3. Output fills row 1 AND row 3.
# But (3,3) is on the diagonal — should it fill column 3 instead?
# The output doesn't fill column 3. So the rule might be:
# If there's ONLY ONE marker, diagonal fills column, off-diagonal fills row.
# If there are MULTIPLE markers of the same color, all fill rows.
# Actually simpler: the LAST marker (bottommost) fills its row.
# And if there's a marker on the diagonal, it fills its column.
# Let me try: for each marker, if row==col fill column, else fill row.
# But pair 1 has (3,3) on diagonal and it fills row, not column.
# Hmm. Maybe the rule is: if the marker is the ONLY non-zero cell in its row,
# it fills the column. Otherwise it fills the row.
# Pair 0: (2,2) is the only non-zero in row 2 → fill column 2.
# (4,7) is the only non-zero in row 4 → should fill column 7. But it fills row 4!
# That doesn't work either.
# 
# Let me try: the marker fills the row. Then the DIAGONAL marker also fills the column.
# Pair 0: (2,2) fills row 2 AND column 2. But output row 2 is "..2......" not "222222222"
# So (2,2) fills ONLY column 2, not row 2.
# (4,7) fills ONLY row 4, not column 7.
# (6,3) fills ONLY row 6, not column 3.
# 
# Rule: if row==col (diagonal), fill COLUMN. Otherwise fill ROW.
# Pair 1: (1,0) row≠col → fill row 1. (3,3) row==col → fill column 3.
# But output has row 3 = "33333333", not column 3 filled.
# Unless... pair 1 has markers at (1,0) and (3,3), both color 3.
# Maybe when there are multiple markers, the rule changes?
# Or: (3,3) fills column 3, AND (1,0) fills row 1, AND since both are color 3,
# the row fill from (1,0) also fills row 3?
# No, that doesn't make sense.
# 
# Actually, maybe the rule is simpler: ALL markers fill their ROW.
# The column fill in pair 0 is from a different mechanism.
# Pair 0 has a marker 2 at (2,2). In the output, column 2 has 2s.
# But also, rows 0,1,3,5,7,8 have "..2......".
# If the marker fills the ROW, row 2 would be "222222222". But it's "..2......".
# So the marker at (2,2) does NOT fill its row. It fills its COLUMN.
# 
# I think the rule depends on the color:
# Color 2 → fill column
# Color 1, 3 → fill row
# Let me check: pair 1 has color 3 markers. They fill rows. ✓
# Pair 0 has color 2 (fills column) and colors 1, 3 (fill rows). ✓
def solve_task24(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            if c == 2:  # fill column
                result[:, j] = c
            else:  # fill row
                result[i, :] = c
    return result

# Task 25: Duplicate adjacent markers of same color
# From data: .4.3..3.....4...... → ...33......44......
# The 3 at col 1 (with 4 at col 1) gets duplicated: 3 at col 3 becomes 33
# Actually: markers on the SAME ROW as another marker get duplicated
# The rule: if two different-colored markers are on the same row,
# duplicate each to fill the cell next to the other marker
def solve_task25(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        # Find markers in this row
        markers = [(j, grid[i, j]) for j in range(W) if grid[i, j] != 0]
        if len(markers) < 2: continue
        # For each pair of different-colored markers
        for a in range(len(markers)):
            for b in range(a + 1, len(markers)):
                ja, ca = markers[a]
                jb, cb = markers[b]
                if ca != cb:
                    # Duplicate: place ca next to jb, and cb next to ja
                    if jb + 1 < W and result[i, jb + 1] == 0:
                        result[i, jb + 1] = ca
                    if jb - 1 >= 0 and result[i, jb - 1] == 0:
                        result[i, jb - 1] = ca
                    if ja + 1 < W and result[i, ja + 1] == 0:
                        result[i, ja + 1] = cb
                    if ja - 1 >= 0 and result[i, ja - 1] == 0:
                        result[i, ja - 1] = cb
    return result

# Task 26: Extract the column containing color 1, with 1→8 mapping
# From data: 5×7 grid with 9s and 1s. Output is 5×3.
# The output is the 3 columns around the "1" column, with 9→0 and 1→8
def solve_task26(grid):
    H, W = grid.shape
    # Find the column with color 1
    col_1 = None
    for j in range(W):
        if (grid[:, j] == 1).any():
            col_1 = j
            break
    if col_1 is None: return grid
    # Extract 3 columns centered on col_1
    start = max(0, col_1 - 1)
    end = min(W, col_1 + 2)
    result = grid[:, start:end].copy()
    # Map: 1→8, 9→0
    result = np.where(result == 1, 8, result)
    result = np.where(result == 9, 0, result)
    return result

# Task 27: Fill the "inside" of an L-shape with color 2
# From data: the 1-shape has an L-corner. The inside of the L gets filled with 2.
def solve_task27(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find all 1-cells
    ones = list(zip(*np.where(grid == 1)))
    if not ones: return grid
    # Find the bounding box
    rows = [p[0] for p in ones]
    cols = [p[1] for p in ones]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    # Fill cells that are inside the L-shape (within bounding box but not on the shape)
    # The L-shape has a corner. Cells to the left of the vertical part and above 
    # the horizontal part get filled with 2.
    # Simple approach: for each empty cell in the bounding box,
    # check if it has a 1 to its right AND a 1 below it
    for i in range(rmin, rmax + 1):
        for j in range(cmin, cmax + 1):
            if grid[i, j] != 0: continue
            has_right = any(grid[i, k] == 1 for k in range(j + 1, cmax + 1))
            has_below = any(grid[k, j] == 1 for k in range(i + 1, rmax + 1))
            has_left = any(grid[i, k] == 1 for k in range(cmin, j))
            has_above = any(grid[k, j] == 1 for k in range(rmin, i))
            if has_right and has_below and has_left and has_above:
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
        ("task5", solve_task5, 5),
        ("task6", solve_task6, 6),
        ("task8", solve_task8, 8),
        ("task9", solve_task9, 9),
        ("task21", solve_task21, 21),
        ("task23", solve_task23, 23),
        ("task24", solve_task24, 24),
        ("task25", solve_task25, 25),
        ("task26", solve_task26, 26),
        ("task27", solve_task27, 27),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
