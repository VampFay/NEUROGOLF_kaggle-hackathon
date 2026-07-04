"""Inspect specific ARC tasks to understand what transformations they need."""
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data
import numpy as np

def show(tid):
    fname = arc_data.task_id_to_filename(tid)
    task = arc_data.load_task(tid)
    print(f"\n=== Task {tid} ({fname}) ===")
    pairs = arc_data.get_pairs(task)
    print(f"  {len(pairs)} pairs")
    for i, (inp, out) in enumerate(pairs[:3]):
        print(f"  Pair {i}: in={inp.shape}, out={out.shape}")
        print(f"    IN:  {inp.tolist()}")
        print(f"    OUT: {out.tolist()}")
    print(f"  In colors: {sorted(set(int(c) for p in pairs for c in np.unique(p[0])))}")
    print(f"  Out colors: {sorted(set(int(c) for p in pairs for c in np.unique(p[1])))}")
    sig = arc_data.task_signature(task)
    print(f"  in_eq_out_all: {sig['in_eq_out_all']}")
    print(f"  all_same_size: {sig['all_same_size']}")

# Inspect the first 10 tasks
for tid in range(1, 11):
    show(tid)
