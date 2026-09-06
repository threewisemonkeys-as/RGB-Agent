"""`AutumnRunner`: PRO-LONG's turn loop against one curated planning problem.

Upstream's `GameRunner` with the ARC machinery removed -- no levels, no score, no WIN,
no RESET, no scorecard -- and one thing added: the study round.

Upstream treats an empty action list as a malformed response, because in ARC there is
always something worth doing and an agent that emits nothing is broken. Here it is a
legitimate move. Turn one is where megabytes of recorded transitions get read, the
smallest budget in the battery is two actions, and forcing a move to buy a turn spends
the arm's evidence on nothing. So the two cases are told apart:

    actions.json missing or malformed  -> failure, retry with a nudge  (upstream)
    {"actions": []}                    -> study round, budget untouched (added)

Both are bounded. Retries by `agent_retries`, study rounds by `study_rounds`; a session
that exhausts either is out of the corresponding resource, not stuck.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from research.autumn import prompts
from research.autumn.actions import ActionQueue, parse_actions_json

log = logging.getLogger(__name__)

SEP = "=" * 80


class AutumnRunner:
    def __init__(self, env, agent, problem: dict, *, log_path: Path,
                 alphabet, workspace: str = "/workspace",
                 agent_retries: int = 5, study_rounds: int = 5,
                 log_post_board: bool = True) -> None:
        self.env = env
        self.agent = agent
        self.problem = problem
        self.log_path = Path(log_path)
        self.alphabet = set(alphabet)
        self.workspace = workspace
        self.agent_retries = agent_retries
        self.study_rounds = study_rounds
        self.log_post_board = log_post_board

        self.queue = ActionQueue()
        self.rounds: list[dict] = []
        # The replay record, kept beside `rounds` rather than folded into it: `rounds` is
        # the arm's own control flow and the study-round contract is asserted against its
        # shape, while these two exist only so the run can be watched afterwards.
        #   turns -- one per agent call: the prompt, the reasoning, the shell it ran
        #   steps -- one per EXECUTED action: the board it produced
        # A step carries the index of the turn that planned it, which is what lets the
        # page show a turn's thinking beside the board its plan actually produced.
        self.turns: list[dict] = []
        self.steps: list[dict] = []
        self.usage = {"calls": 0, "in": 0, "out": 0, "cache_read": 0,
                      "reasoning": 0, "cost": 0.0}
        self.studies_used = 0
        self._recent: list[str] = []
        self._pending_nudge = ""

    # ----------------------------------------------------------------- the loop
    def run(self) -> dict:
        started = time.time()
        observation = self.env.reset(task={"problem": self.problem})
        dims = (len(self.problem["start"]), len(self.problem["start"][0]))
        self._write_initial(observation["grid"])

        failed = None
        done = False
        while not done and self.env.remaining > 0:
            if not self.queue:
                verdict = self._get_plan(dims)
                if verdict == "exhausted":
                    failed = "no-plan"
                    break
                if verdict == "studied-out":
                    failed = "study-rounds-exhausted"
                    break
                if not self.queue:          # a study round: no actions, ask again
                    continue

            action = self.queue.pop()
            observation, _reward, done = self.env.step(action)
            self._recent.append(action)
            self.steps.append({
                "n": observation["actions_used"],
                "action": action,
                "grid_after": observation["grid"],
                "reached": bool(observation["reached_goal"]),
                "terminated": bool(observation["terminated"]),
                "remaining": observation["remaining"],
                "turn": len(self.turns) - 1,
                "plan_index": self.queue.plan_index,
                "plan_total": self.queue.plan_total,
            })
            self._log_action(action, observation)
            if observation["reached_goal"] or observation["terminated"]:
                self.queue.clear()

        outcome = self.env.outcome(failed)
        outcome.update({
            "rounds": self.rounds, "turns": self.turns, "steps": self.steps,
            "usage": self.usage,
            "study_rounds_used": self.studies_used,
            "wall_s": round(time.time() - started, 1),
        })
        return outcome

    # --------------------------------------------------------------- the agent
    def _get_plan(self, dims) -> str:
        """`loaded` | `studied` | `studied-out` | `exhausted`."""
        for attempt in range(self.agent_retries):
            payload, meta = self._call_agent(attempt)
            self._record_usage(meta)
            turn = self._open_turn(meta, attempt)
            if payload is None:
                turn["kind"] = "malformed"
                self._pending_nudge = prompts.RETRY_NUDGE.format(workspace=self.workspace)
                log.warning("no usable actions.json (attempt %d/%d)",
                            attempt + 1, self.agent_retries)
                continue

            actions, rejected = parse_actions_json(
                payload, dims, self.alphabet, self.env.remaining)
            turn["rejected"] = rejected

            if not actions and self._is_deliberate_pass(payload) and not rejected:
                turn["kind"] = "study"
                if self.studies_used >= self.study_rounds:
                    log.warning("study rounds exhausted; agent still will not act")
                    return "studied-out"
                self.studies_used += 1
                self._log_study()
                self._pending_nudge = ""
                return "studied"

            if not actions:
                turn["kind"] = "rejected"
                self._pending_nudge = prompts.REJECTED_NUDGE.format(
                    rejections="\n".join(f"  - {r}" for r in rejected) or "  - (none parsed)",
                    workspace=self.workspace, rows=dims[0], cols=dims[1],
                    remaining=self.env.remaining)
                log.warning("every action rejected: %s", rejected)
                continue

            if rejected:
                log.info("dropped %d unusable action(s): %s", len(rejected), rejected)
            self.rounds.append({
                "n": self.env.actions_used, "remaining": self.env.remaining,
                "plan": actions, "rejected": rejected, "kind": "plan",
            })
            turn.update({"kind": "plan", "plan": actions})
            self.queue.load(actions)
            self._pending_nudge = ""
            return "loaded"
        return "exhausted"

    def _open_turn(self, meta: dict | None, attempt: int) -> dict:
        """One agent call, recorded whatever it produced.

        A retried turn is kept, not overwritten: a malformed reply followed by a good one
        is the arm working as designed, and a transcript that shows only the good one
        makes the retry budget look untouched.
        """
        meta = meta or {}
        turn = {
            "i": len(self.turns),
            "attempt": attempt,
            "kind": "unknown",
            "n": self.env.actions_used,
            "remaining": self.env.remaining,
            "prompt": meta.get("prompt") or "",
            "events": meta.get("events") or [],
            "plan": [],
            "rejected": [],
            "wall_s": meta.get("wall_s"),
            "tokens": {k: meta.get(k) or 0 for k in
                       ("input_tokens", "cached_tokens", "output_tokens",
                        "reasoning_tokens")},
        }
        self.turns.append(turn)
        return turn

    @staticmethod
    def _is_deliberate_pass(payload) -> bool:
        """An empty list the agent actually wrote, not a body we failed to understand."""
        entries = payload.get("actions") if isinstance(payload, dict) else payload
        return isinstance(entries, list) and len(entries) == 0

    def _call_agent(self, attempt: int):
        first = self.usage["calls"] == 0
        if first:
            body = prompts.FIRST_PROMPT
        elif self.rounds and self.rounds[-1].get("kind") == "study":
            body = prompts.STUDY_PROMPT
        else:
            body = prompts.RESUME_PROMPT
        prompt = body.format(
            workspace=self.workspace, goal=self.problem["nl_goal"],
            remaining=self.env.remaining,
            study_left=max(0, self.study_rounds - self.studies_used),
            last_actions=", ".join(self._recent[-5:]) or "none",
        )
        if self._pending_nudge:
            prompt += f"\n\n{self._pending_nudge}"
        payload, meta = self.agent.analyze(self.log_path, prompt, is_first=first)
        if isinstance(meta, dict):
            meta.setdefault("prompt", prompt)
        return payload, meta

    def _record_usage(self, meta: dict | None) -> None:
        self.usage["calls"] += 1
        if not meta:
            return
        for key, field in (("in", "input_tokens"), ("out", "output_tokens"),
                           ("cache_read", "cached_tokens"),
                           ("reasoning", "reasoning_tokens")):
            self.usage[key] += int(meta.get(field) or 0)
        self.usage["cost"] += float(meta.get("call_cost_usd") or 0.0)

    # ------------------------------------------------------------------ the log
    def _append(self, text: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(text)

    def _write_initial(self, grid: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._append(
            f"{SEP}\nAction 0 | INITIAL STATE | budget {self.env.budget}\n\n"
            f"[INITIAL BOARD STATE]\n{grid}\n\n")

    def _log_action(self, action: str, observation: dict) -> None:
        n = observation["actions_used"]
        step = f" | Plan Step {self.queue.plan_index}/{self.queue.plan_total}" \
            if self.queue.plan_total else ""
        self._append(f"\n{SEP}\nAction {n}{step} | {self.env.remaining} left\n\n"
                     f"Executed: {action}\n")
        if self.log_post_board:
            self._append(f"\n[POST-ACTION BOARD STATE]\n{observation['grid']}\n")
        if observation["reached_goal"]:
            self._append("\n*** GOAL REACHED ***\n")
        elif observation["terminated"]:
            self._append("\n*** EPISODE TERMINATED ***\n")

    def _log_study(self) -> None:
        self.rounds.append({
            "n": self.env.actions_used, "remaining": self.env.remaining,
            "plan": [], "kind": "study",
        })
        self._append(
            f"\n{SEP}\n[STUDY ROUND] {self.studies_used}/{self.study_rounds} "
            f"| no action executed | {self.env.remaining} left\n\n")


def write_actions(path: Path, actions: list[str]) -> None:
    """Test helper: the file the agent is asked to produce."""
    Path(path).write_text(json.dumps({"actions": actions}))
