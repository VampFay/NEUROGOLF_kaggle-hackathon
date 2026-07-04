"""
neurogolf/solvers/base.py — Base classes for per-task solvers.

A Solver takes a task dict and either:
  - returns an ONNX model + diagnostic info, or
  - returns None if it can't solve the task.

The pipeline tries solvers in order of increasing cost (cheap first),
and picks the smallest correct model.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import onnx

from .. import dsl, validator
from ..arc_data import get_pairs, get_train_pairs, get_test_pairs


@dataclass
class SolverResult:
    solver_name: str
    model: onnx.ModelProto
    eligible: bool          # passes structural + functional checks on all task pairs
    params: int
    size_bytes: int
    cost: int
    score: float
    note: str = ""

    def __lt__(self, other: "SolverResult") -> bool:
        # Prefer eligible first, then lower cost
        if self.eligible != other.eligible:
            return self.eligible
        return self.cost < other.cost


class Solver(abc.ABC):
    """Abstract base class for a per-task solver."""

    name: str = "base"

    @abc.abstractmethod
    def attempt(self, task: dict) -> Optional[onnx.ModelProto]:
        """Try to solve the task. Return ONNX model or None."""
        ...

    def solve(self, task: dict) -> Optional[SolverResult]:
        """Try to solve, validate, and return a SolverResult (or None)."""
        try:
            model = self.attempt(task)
        except Exception:
            return None
        if model is None:
            return None
        e = validator.evaluate_model(model, task)
        # Use faithful scorer (onnx-tool) for cost/score
        from .. import faithful_scorer
        ci = faithful_scorer.compute_cost(model)
        return SolverResult(
            solver_name=self.name,
            model=model,
            eligible=e["eligible_for_points"],
            params=ci.get("params", 0),
            size_bytes=ci.get("file_bytes", 0),
            cost=ci.get("cost", 0),
            score=ci.get("score", 1.0),
            note=e.get("structural_msg", ""),
        )


def run_solvers(task: dict, solvers: list[Solver], verbose: bool = False) -> Optional[SolverResult]:
    """Run a list of solvers; return the best (HIGHEST score) eligible result."""
    results: list[SolverResult] = []
    for s in solvers:
        r = s.solve(task)
        if r is None:
            continue
        if verbose:
            status = "OK " if r.eligible else "FAIL"
            print(f"  [{status}] {s.name}: cost={r.cost}, score={r.score:.2f}")
        results.append(r)
    if not results:
        return None
    # Prefer eligible, then HIGHEST score (lowest cost = highest score)
    eligible = [r for r in results if r.eligible]
    if eligible:
        return max(eligible, key=lambda r: r.score)
    return max(results, key=lambda r: r.score)
