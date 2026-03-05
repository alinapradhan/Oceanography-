"""Optimization Agent – refines design parameters to meet performance targets.

Uses gradient-free hill-climbing to iteratively improve a design's
performance metric (e.g. range for AUVs, average power for WECs,
natural-frequency separation for platforms).
"""

from __future__ import annotations

from typing import Any, Callable

from .base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    """Iteratively refines a design to maximise a given objective function.

    The agent performs a simple bounded hill-climbing search that is
    intentionally lightweight so that optimization runs can be embedded
    inside larger agent loops without heavy dependencies.
    """

    def __init__(
        self,
        name: str = "OptimizationAgent",
        max_iterations: int = 50,
        step_size: float = 0.05,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=verbose)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, context: dict[str, Any]) -> dict[str, Any]:
        required = {"design", "objective_fn", "param_bounds"}
        missing = required - context.keys()
        if missing:
            raise KeyError(f"OptimizationAgent context missing keys: {missing}")
        return {
            "design": dict(context["design"]),
            "objective_fn": context["objective_fn"],
            "param_bounds": context["param_bounds"],
            "maximize": context.get("maximize", True),
        }

    def reason(self, observations: dict[str, Any]) -> dict[str, Any]:
        design = observations["design"]
        obj_fn: Callable[[dict], float] = observations["objective_fn"]
        bounds: dict[str, tuple[float, float]] = observations["param_bounds"]
        maximize: bool = observations["maximize"]

        best_design, best_score, history = self._hill_climb(
            design, obj_fn, bounds, maximize
        )
        return {
            "optimized_design": best_design,
            "best_score": best_score,
            "iterations": len(history),
            "score_history": history,
            "converged": len(history) < self.max_iterations,
        }

    def act(self, decision: dict[str, Any]) -> dict[str, Any]:
        decision["agent"] = self.name
        decision["status"] = "optimization_complete"
        return decision

    # ------------------------------------------------------------------
    # Hill-climbing optimizer
    # ------------------------------------------------------------------

    def _hill_climb(
        self,
        initial: dict[str, Any],
        obj_fn: Callable[[dict], float],
        bounds: dict[str, tuple[float, float]],
        maximize: bool,
    ) -> tuple[dict[str, Any], float, list[float]]:
        """Simple coordinate-wise hill climbing.

        Args:
            initial: Starting design parameter dict.
            obj_fn:  Objective function mapping design → scalar score.
            bounds:  Dict of ``{param_name: (lower, upper)}`` bounds.
            maximize: If True, maximise the objective; else minimise.

        Returns:
            Tuple of (best_design, best_score, score_history).
        """
        sign = 1.0 if maximize else -1.0
        current = dict(initial)
        current_score = sign * obj_fn(current)
        history = [obj_fn(current)]

        for _ in range(self.max_iterations):
            improved = False
            for param, (lo, hi) in bounds.items():
                if param not in current:
                    continue
                val = current[param]
                step = (hi - lo) * self.step_size

                # Try increasing
                candidate_up = dict(current)
                candidate_up[param] = min(hi, val + step)
                score_up = sign * obj_fn(candidate_up)

                # Try decreasing
                candidate_dn = dict(current)
                candidate_dn[param] = max(lo, val - step)
                score_dn = sign * obj_fn(candidate_dn)

                best_candidate = current
                best_candidate_score = current_score

                if score_up > best_candidate_score:
                    best_candidate = candidate_up
                    best_candidate_score = score_up

                if score_dn > best_candidate_score:
                    best_candidate = candidate_dn
                    best_candidate_score = score_dn

                if best_candidate is not current:
                    current = best_candidate
                    current_score = best_candidate_score
                    improved = True

            history.append(sign * current_score)
            if not improved:
                break
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.convergence_threshold:
                break

        return current, sign * current_score, history
