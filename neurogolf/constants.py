"""
neurogolf/constants.py — Shared constants for the NeuroGolf pipeline.

I/O Convention (reverse-engineered from competition example)
============================================================
Each ONNX network takes a single input tensor of shape (1, 10, 30, 30) float32,
which is a one-hot encoding of the ARC grid (10 colors 0-9, max grid 30x30,
padded with zeros).  The network must produce a single output tensor of the
same shape (1, 10, 30, 30) float32 whose argmax-over-channels yields the
output grid (also padded to 30x30 with zeros).

The competition's validator is responsible for:
  1. Reading the raw ARC grid (list of lists of ints 0-9)
  2. Padding to 30x30 with 0
  3. One-hot encoding to (1, 10, 30, 30)
  4. Running the ONNX network
  5. Taking argmax over channel dim -> (1, 30, 30) grid
  6. Cropping to the expected output dimensions (which the validator knows
     from the test pair)

So our job per task: design a small ONNX network mapping
(1, 10, 30, 30) -> (1, 10, 30, 30) that yields the correct argmax for every
input/output pair in that task.
"""

# Grid dimensions
MAX_GRID = 30
NUM_COLORS = 10

# Standard I/O tensor shape: (batch, channels, height, width)
IO_SHAPE = (1, NUM_COLORS, MAX_GRID, MAX_GRID)

# Input/output tensor names used by the validator
INPUT_NAME = "input"
OUTPUT_NAME = "output"

# Hard constraints from the competition rules
MAX_FILE_BYTES = 1_440_000  # 1.44 MB
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}

# Scoring: max(1, 25 - ln(cost)) where cost = #params + #bytes
# So smaller is better. Reference points:
#   cost=10   -> score=22.70
#   cost=100  -> score=20.40
#   cost=1000 -> score=18.10
#   cost=10000-> score=15.81
def score(cost: int) -> float:
    import math
    return max(1.0, 25.0 - math.log(cost))
