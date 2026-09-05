"""The curated planning battery, as the published evaluators define it.

Nothing here decides what a problem is, what its budget is, or whether a plan worked.
Every one of those answers already exists in the research repo and was used to score
`raw`, `icl` and `lmwm`; a second implementation is a second answer that can disagree
with the published one, which is the one way this arm can be wrong without anyone
noticing. So this module imports and re-exports, and the only code it owns is the glue
that makes those imports reachable from inside the PRO-LONG tree.

What it fixes, deliberately and not as a default:

* goal presentation `nl` -- the goal is an English sentence and a registered checker;
  the goal FRAME is dropped, never merely unrendered.
* cap mode `per-problem` -- 2x the reference reach up to 10 actions, 1.5x above.
* `--max-floor 0.95` -- problems a random policy already solves are not evidence.

`BAI_REPO` overrides where the research repo lives (default: the sibling checkout).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# RGB-Agent/research/autumn/rig.py -> RGB-Agent -> the repo that vendors it
_DEFAULT_REPO = _HERE.parents[2].parent
REPO = Path(os.environ.get("BAI_REPO", _DEFAULT_REPO)).resolve()

if not (REPO / "offline_learning").is_dir():
    raise RuntimeError(
        f"no research repo at {REPO}; set BAI_REPO to the checkout that holds "
        "offline_learning/ and logs/")

for _p in (REPO, REPO / "offline_learning", REPO / "offline_learning/scripts",
           REPO / "cc_autumn/autumn-code/rig"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_coverage_online import Branch  # noqa: E402
from eval_coverage_plan import ACTION_RE  # noqa: E402
from eval_curated_online import make_goal_test, prepare  # noqa: E402
from eval_curated_plan import (  # noqa: E402
    DEFAULT_PROBLEMS, _floor, apply_action_caps, execute_and_score, gstr,
    load_eval_problems, select_goal_presentation,
)
from planning_nl_goals import freeze_grid  # noqa: E402

try:
    from curated import LABELS
except ImportError:                                     # pragma: no cover - rig absent
    LABELS = {}

GOAL_PRESENTATION = "nl"
SUCCESS_MODE = "any"
CAP_MODE = "per-problem"
MAX_FLOOR = 0.95

__all__ = [
    "Branch", "ACTION_RE", "LABELS", "REPO", "freeze_grid", "gstr",
    "execute_and_score", "make_goal_test", "load_problems", "program_for",
    "replay_and_score", "actions_for",
]


def load_problems(path: str | Path | None = None, *,
                  games: list[str] | None = None,
                  max_floor: float = MAX_FLOOR) -> list[dict]:
    """The 86 rows, configured exactly as the online run configured them."""
    _meta, problems = load_eval_problems(path or DEFAULT_PROBLEMS)
    problems = select_goal_presentation(problems, GOAL_PRESENTATION, SUCCESS_MODE)
    apply_action_caps(problems, CAP_MODE)      # returns the caps; stamps the rows
    if games:
        wanted = set(games)
        problems = [p for p in problems if p["game"] in wanted]
    if max_floor is not None:
        # `_floor` is the evaluator's own: it matches the floor to the cap in force and
        # returns None when the only measurement was taken at a different budget. A row
        # whose floor is unmeasured AT THIS CAP is kept -- dropping it would silently
        # change the problem set relative to the runs this arm is compared against.
        problems = [p for p in problems
                    if (_floor(p) is None or _floor(p) <= max_floor)]

    # The evaluator's own preparation: `start_grid`, the replay `_prefix`, `_dims`. It
    # also stamps the perception features the `lmwm` arm plans on, which this arm has
    # none of -- hence the null perceive. Under `nl` that path writes goal_grid="" and
    # _z_goal="", which is the mechanism by which the goal FRAME is dropped rather than
    # merely unrendered; letting `prepare` do it is what keeps that guarantee shared.
    prepare(problems, lambda _grid: ("", None))
    return problems


def replay_and_score(problem: dict, actions: list[str]) -> tuple[bool, int | None]:
    """The authoritative verdict: replay `actions` through a fresh branch and apply the
    ONLINE goal test after each one.

    Not `execute_and_score`. That is the OFFLINE scorer, and it enforces quiescence by
    probing with a hidden noop -- which closed-loop play has no room for, so the online
    evaluator waives the requirement and `raw`, `icl` and `lmwm` were all scored with it
    waived. Measured on the reference plans: 28 of the 86 rows carry the waiver and 11 of
    them flip verdict between the two scorers. Scoring this arm offline would hand it a
    stricter rule than the arms it is compared against and cost it 13% of the battery to
    a scorer mismatch rather than to planning.

    What the plan wanted from `execute_and_score` -- a verdict that a harness bug cannot
    fake -- is preserved: this replays from the state address in a branch the session
    never touched, and re-derives the goal test from the row. It is independent of the
    run, and it is the same rule.
    """
    goal_test, _waived = make_goal_test(problem)
    branch = Branch(program_for(problem), problem["seed"], problem["_prefix"],
                    len(actions) + 1)
    try:
        if branch.grid() != problem["start_grid"]:
            raise RuntimeError(
                f"{problem['task_uid']}: replayed prefix does not reproduce START")
        grids = [freeze_grid(problem["start"])]
        executed: list[str] = []
        import json as _json
        for i, action in enumerate(actions, 1):
            branch.step(action)
            grid = branch.grid()
            executed.append(action)
            grids.append(freeze_grid(_json.loads(grid)))
            if goal_test(grids, executed, grid):
                return True, i
            if branch.terminated:
                break
        return False, None
    finally:
        branch.close()


def program_for(problem: dict) -> str:
    """The Autumn program `Branch` replays. Taken from the problem row, which already
    carries it, rather than from `human_replay.GAMES` -- that registry's second field is
    the world's English name (`ice`, `mario`, `disease`), and this arm must never be one
    import away from the answer."""
    prog = problem.get("program")
    if not prog:
        raise KeyError(f"{problem.get('task_uid')}: row carries no program")
    return prog


def actions_for(game: str) -> list[str]:
    """The action alphabet for a world, as the data pipeline defines it.

    Reads `GAMES[game][2]` and nothing else from that tuple: `[1]` is the English name.
    `click` is expanded to the concrete `click ROW COL` verb the evaluator parses.
    """
    from human_replay import GAMES as HGAMES

    entry = HGAMES.get(game)
    if not entry:
        raise KeyError(f"no action alphabet registered for {game!r}")
    return list(entry[2])
