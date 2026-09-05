"""Autumn's action alphabet, and the queue the runner drains it through.

`click ROW COL` is row-first. This is the trap the arm most easily falls into: the
interpreter's own `click(col, row)` is x,y, and a plan written in x,y order silently
scores as a different click on every world where location matters. `ACTION_RE` in
`eval_coverage_plan` is the reference parser and it reads row first, so this module
matches it rather than restating it -- the regex is imported, not copied.

The queue is upstream's `ActionQueue` with the ARC-specific parts removed: there is no
score to flush on and no RESET, because a curated planning problem has one life.
"""
from __future__ import annotations

import logging
import re
from collections import deque

from research.autumn.rig import ACTION_RE

log = logging.getLogger(__name__)

MOVES = ("left", "right", "up", "down", "noop")


class QueueExhausted(RuntimeError):
    pass


def normalise(entry) -> str | None:
    """One `actions.json` entry -> a verb the evaluator accepts, or None.

    Accepts the string form (`"click 3 4"`) and the dict form
    (`{"action": "click", "row": 3, "col": 4}`); the agent is told the string form, and
    the dict form is here because a model that emits structured JSON is not wrong, only
    different, and rejecting it would score prompt-following rather than planning.
    """
    if isinstance(entry, dict):
        name = str(entry.get("action", "")).strip().lower()
        if name == "click":
            row, col = entry.get("row"), entry.get("col")
            if row is None or col is None:
                return None
            entry = f"click {int(row)} {int(col)}"
        else:
            entry = name
    if not isinstance(entry, str):
        return None
    verb = re.sub(r"\s+", " ", entry.strip().strip("`").lower())
    return verb if ACTION_RE.match(verb) else None


def in_bounds(verb: str, dims: tuple[int, int]) -> bool:
    if not verb.startswith("click "):
        return True
    _, row, col = verb.split()
    return int(row) < dims[0] and int(col) < dims[1]


def parse_actions_json(payload, dims: tuple[int, int], alphabet: set[str],
                       cap: int) -> tuple[list[str], list[str]]:
    """`(actions, rejections)` from a parsed actions.json body.

    An entry is dropped, never silently repaired: a click outside the grid or a verb the
    world does not have is a planning error the log should show the agent, not something
    the harness should guess at.
    """
    entries = payload.get("actions") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return [], ["actions.json: expected a list under 'actions'"]

    actions, rejected = [], []
    for entry in entries:
        if len(actions) >= cap:
            rejected.append(f"beyond the {cap}-action budget: {entry!r}")
            continue
        verb = normalise(entry)
        if verb is None:
            rejected.append(f"unparseable action: {entry!r}")
            continue
        head = verb.split()[0]
        if head not in alphabet:
            rejected.append(f"{head!r} is not an action in this world")
            continue
        if not in_bounds(verb, dims):
            rejected.append(f"click outside the {dims[0]}x{dims[1]} grid: {verb!r}")
            continue
        actions.append(verb)
    return actions, rejected


class ActionQueue:
    """FIFO drain of one plan. No score flush, no RESET: one problem, one life."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self.plan_total = 0
        self.plan_index = 0

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self.plan_total = 0
        self.plan_index = 0

    def load(self, actions: list[str]) -> bool:
        self.clear()
        self._queue.extend(actions)
        self.plan_total = len(self._queue)
        if not self._queue:
            log.warning("ActionQueue.load: nothing to load")
            return False
        log.info("loaded %d-step plan: %s", self.plan_total, actions)
        return True

    def pop(self) -> str:
        if not self._queue:
            raise QueueExhausted("queue empty, no actions from the agent")
        self.plan_index += 1
        return self._queue.popleft()
