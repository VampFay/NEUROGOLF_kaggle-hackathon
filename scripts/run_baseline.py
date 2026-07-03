"""Run all solvers on all 400 ARC tasks and report the baseline."""
import sys
sys.path.insert(0, "/home/z/my-project")
import json
import time
from neurogolf import dsl, validator, arc_data
from neurogolf.solvers import simple, transforms, filters, advanced, run_solvers

SOLVERS = [
    simple.IdentitySolver(),
    simple.ColorMapSolver(),
    simple.ReplaceColorSolver(),
    transforms.GeometricTransformSolver(),
    transforms.ColorMapThenTransformSolver(),
    # Skip ConvFilterSolver for now (too slow)
    # filters.ConvFilterSolver(),
    filters.ColorSubstitutionSolver(),
    advanced.ScaleUpSolver(),
    advanced.CropSolver(),
    advanced.ShiftSolver(),
    advanced.TileSolver(),
    advanced.KroneckerSolver(),
    advanced.ConcatRepeatSolver(),
    advanced.ConditionalSliceColorMapSolver(),
    simple.ConstantSolver(),
]

results = []
solved = 0
total_score = 0.0
breakdown = {}
fails = []
t0 = time.time()
for tid in range(1, 401):
    task = arc_data.load_task(tid)
    fname = arc_data.task_id_to_filename(tid)
    result = run_solvers(task, SOLVERS, verbose=False)
    if result and result.eligible:
        solved += 1
        total_score += result.score
        breakdown[result.solver_name] = breakdown.get(result.solver_name, 0) + 1
        results.append({
            "task_id": tid,
            "filename": fname,
            "solver": result.solver_name,
            "cost": result.cost,
            "score": result.score,
            "eligible": True,
        })
    else:
        best = result.solver_name if result else "none"
        fails.append((tid, fname, best))
        results.append({
            "task_id": tid,
            "filename": fname,
            "solver": best,
            "cost": result.cost if result else 0,
            "score": result.score if result else 0,
            "eligible": False,
        })

elapsed = time.time() - t0
print(f"\n=== Baseline Summary ===")
print(f"Solved: {solved}/400 ({100*solved/400:.1f}%)")
print(f"Total score: {total_score:.2f}")
print(f"Elapsed: {elapsed:.1f}s")
print(f"\nSolver breakdown:")
for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
    print(f"  {s:35s}: {c}")

# Save full results
with open("/home/z/my-project/data/baseline_results.json", "w") as f:
    json.dump({"summary": {"solved": solved, "total": 400, "score": total_score,
                            "elapsed": elapsed, "breakdown": breakdown},
               "results": results}, f, indent=2)
print(f"\nDetailed results saved to /home/z/my-project/data/baseline_results.json")
print(f"\nFirst 30 solved tasks:")
for r in results:
    if r["eligible"]:
        print(f"  task {r['task_id']:3d} ({r['filename']}): {r['solver']:30s} cost={r['cost']:5d} score={r['score']:.2f}")
