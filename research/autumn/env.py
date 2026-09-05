"""`AutumnPlanningEnv`: one curated planning problem behind PRO-LONG's `BaseEnv`.

A state in this benchmark is not a snapshot, it is a replay address -- `(program, seed,
prefix)` -- so `reset()` branches the live interpreter through the prefix and then hard-
checks that the replayed board is byte-identical to the row's recorded START before the
session is allowed to cost anything. A drifted engine that quietly plays a different
board would produce a number, and the number would be wrong.

The goal test is the online evaluator's, called after every executed action with the
trajectory including that action's frame. Success is any-step by construction: the
problem ends the moment the goal holds.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from prolong_agent.environment import BaseEnv
from research.autumn import rig

log = logging.getLogger(__name__)


class StartMismatch(RuntimeError):
    """The replayed prefix did not reproduce the recorded START."""


class AutumnPlanningEnv(BaseEnv):
    def __init__(self) -> None:
        self.problem: dict | None = None
        self._branch = None
        self._goal_test = None
        self.quiescence_waived = False
        self.grids: list = []
        self.executed: list[str] = []
        self.reached_at: int | None = None

    # ---------------------------------------------------------------- lifecycle
    def reset(self, task: dict | None = None) -> dict:
        if not task or "problem" not in task:
            raise ValueError("AutumnPlanningEnv.reset needs task={'problem': row}")
        problem = task["problem"]
        self.problem = problem
        self.budget = int(problem["_eval_action_cap"])
        self._goal_test, self.quiescence_waived = rig.make_goal_test(problem)

        self._branch = rig.Branch(rig.program_for(problem), problem["seed"],
                                  problem["_prefix"], self.budget)
        grid = self._branch.grid()
        if grid != problem["start_grid"]:
            self._branch.close()
            raise StartMismatch(
                f"{problem['task_uid']}: replayed prefix does not reproduce START")

        self.grids = [rig.freeze_grid(problem["start"])]
        self.executed = []
        self.reached_at = None
        return self._observation(grid, reached=False)

    def step(self, action: Any) -> tuple[dict, float, bool]:
        if self._branch is None:
            raise RuntimeError("step() before reset()")
        verb = str(action)
        self._branch.step(verb)
        grid = self._branch.grid()
        self.executed.append(verb)
        self.grids.append(rig.freeze_grid(json.loads(grid)))

        reached = bool(self._goal_test(self.grids, self.executed, grid))
        if reached and self.reached_at is None:
            self.reached_at = len(self.executed)

        done = reached or self._branch.terminated or len(self.executed) >= self.budget
        return self._observation(grid, reached=reached), float(reached), done

    def close(self) -> None:
        if self._branch is not None:
            self._branch.close()
            self._branch = None

    # ------------------------------------------------------------------ helpers
    @property
    def terminated(self) -> bool:
        return bool(self._branch and self._branch.terminated)

    @property
    def actions_used(self) -> int:
        return len(self.executed)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - len(self.executed))

    def _observation(self, grid: str, *, reached: bool) -> dict:
        return {
            "grid": grid,
            "reached_goal": reached,
            "terminated": self.terminated,
            "actions_used": len(self.executed),
            "remaining": self.remaining,
        }

    def outcome(self, failed_reason: str | None = None) -> dict:
        """The row shape the online evaluator writes, so nothing downstream special-cases
        this arm. `success` here is the LIVE verdict; the authoritative one is
        `rig.execute_and_score` re-run on `executed`, which `launch` records alongside."""
        success = self.reached_at is not None
        if not success and failed_reason is None:
            failed_reason = "terminated" if self.terminated else "budget-exhausted"
        return {
            "success": success,
            "reached_at": self.reached_at,
            "actions_used": len(self.executed),
            "failed_reason": None if success else failed_reason,
            "start_match": True,
            "quiescence_waived": self.quiescence_waived,
            "executed": list(self.executed),
        }
