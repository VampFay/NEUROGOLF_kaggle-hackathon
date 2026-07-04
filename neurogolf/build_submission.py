"""
neurogolf/build_submission.py — Build the Kaggle submission.zip.
"""
from __future__ import annotations
import json, os, sys, time, zipfile
from pathlib import Path
sys.path.insert(0, "/home/z/my-project")

from neurogolf import arc_data, dsl, validator
from neurogolf.solvers import (
    simple, transforms, filters, advanced, patterns, cellular, run_solvers
)
from neurogolf import memory_golf
from neurogolf.exploit_solvers import (
    ExploitIdentitySolver, ExploitFlipSolver, ExploitColorSwapSolver,
    ExploitCropSolver, ExploitMirrorConcatSolver,
)


def get_all_solvers():
    solvers = []
    # EXPLOIT SOLVERS (cost=1 → score 25)
    solvers.extend([
        ExploitIdentitySolver(),
        ExploitFlipSolver(),
        ExploitColorSwapSolver(),
        ExploitCropSolver(),
        ExploitMirrorConcatSolver(),
    ])
    # Memory golf task-specific solvers
    try:
        solvers.extend(memory_golf.get_memory_golf_solvers())
    except:
        pass
    # Rebuilt golf solvers
    try:
        solvers.extend(memory_golf.get_rebuilt_golf_solvers())
    except:
        pass
    # Batch 1 rebuilt solvers
    try:
        solvers.extend(memory_golf.get_batch1_golf_solvers())
    except:
        pass
    # Batch 2 rebuilt solvers
    try:
        solvers.extend(memory_golf.get_rebuilt_golf_solvers_batch2())
    except:
        pass
    # Universal brute force solver
    try:
        solvers.append(memory_golf.UniversalBruteForceSolver())
    except:
        pass
    # Regular solvers
    solvers.extend([
        simple.IdentitySolver(),
        simple.ColorMapSolver(),
        simple.ReplaceColorSolver(),
        patterns.ExhaustiveColorMapSolver(),
        patterns.PaletteSolver(),
        transforms.GeometricTransformSolver(),
        transforms.ColorMapThenTransformSolver(),
        advanced.ScaleUpSolver(),
        advanced.CropSolver(),
        advanced.ShiftSolver(),
        advanced.TileSolver(),
        advanced.KroneckerSolver(),
        advanced.ConcatRepeatSolver(),
        advanced.ConditionalSliceColorMapSolver(),
        patterns.MirrorConcatSolver(),
        cellular.CellularAutomatonSolver(),
        cellular.MultiRuleCASolver(),
        simple.ConstantSolver(),
    ])
    return solvers


def build_submission(output_path="/home/z/my-project/download/submission.zip", verbose=True):
    solvers = get_all_solvers()
    results = []
    solved = 0
    total_score = 0.0
    breakdown = {}
    t0 = time.time()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in range(1, 401):
            task = arc_data.load_task(tid)
            fname = arc_data.task_id_to_filename(tid)
            result = run_solvers(task, solvers, verbose=False)
            if result and result.eligible:
                zf.writestr(f"task{tid:03d}.onnx", result.model.SerializeToString())
                solved += 1
                total_score += result.score
                breakdown[result.solver_name] = breakdown.get(result.solver_name, 0) + 1
                results.append({"task_id": tid, "filename": fname, "solver": result.solver_name,
                                "cost": result.cost, "score": result.score, "eligible": True})
                if verbose and solved <= 50:
                    print(f"  [OK]   task {tid:3d} ({fname}): {result.solver_name:30s} cost={result.cost:5d} score={result.score:.2f}")
            else:
                best = result.solver_name if result else "none"
                results.append({"task_id": tid, "filename": fname, "solver": best,
                                "cost": result.cost if result else 0, "score": result.score if result else 0,
                                "eligible": False})

    elapsed = time.time() - t0
    summary = {"solved": solved, "total": 400, "total_score": total_score,
               "elapsed_sec": elapsed, "breakdown": breakdown,
               "output_path": output_path, "file_size_bytes": os.path.getsize(output_path)}
    with open("/home/z/my-project/data/submission_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== Submission Summary ===")
    print(f"Solved: {solved}/400 ({100*solved/400:.1f}%)")
    print(f"Total expected score: {total_score:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path} ({summary['file_size_bytes']} bytes)")
    print(f"\nSolver breakdown:")
    for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {s:35s}: {c}")
    return summary


if __name__ == "__main__":
    build_submission(verbose=True)
