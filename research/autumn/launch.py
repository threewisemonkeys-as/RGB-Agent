#!/usr/bin/env python3
"""Run the agent arm: one session per problem, 86 of them.

Replaces upstream's `swarm.py`. There is no per-game learning pass and no artifact to
copy between sessions -- a problem's session starts from the exported corpus and the
problem's own start state, and ends when the goal is reached, the episode terminates or
the budget runs out. One session per problem is not an implementation convenience: a
world's problems played as a battery would leave the agent knowing on problem 8 what it
learned on problem 1, which `raw`, `icl` and `lmwm` never do.

Each workspace is built fresh and never shared:

    <out>/<LABEL>/<task_uid>/
        drives/       the exported corpus for this world (copied, not linked)
        AGENTS.md     the system prompt
        logs.txt      the run log the agent greps
        actions.json  cleared before every call
        agent.txt     the session transcript, for the audit

    uv run python -m research.autumn.launch --corpus corpora --out logs/agent --dry-run
    uv run python -m research.autumn.launch --corpus corpora --out logs/agent --games n2ntd
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from research.autumn import prompts, rig
from research.autumn.agent import (
    AutumnCodexAgent, CredentialsExhausted, build_catalog)
from research.autumn.env import AutumnPlanningEnv, StartMismatch
from research.autumn.runner import AutumnRunner

log = logging.getLogger(__name__)


def build_workspace(problem: dict, corpus_root: Path, out_root: Path, *,
                    study_rounds: int, with_data: bool = True) -> Path:
    label = rig.LABELS.get(problem["game"], problem["game"].upper())
    workspace = Path(out_root) / label / problem["task_uid"].replace(":", "_")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    if with_data:
        source = Path(corpus_root) / label / "drives"
        if not source.is_dir():
            raise FileNotFoundError(
                f"no exported corpus at {source}; run export_agent_corpus.py first")
        shutil.copytree(source, workspace / "drives")

    (workspace / "AGENTS.md").write_text(prompts.build_system_prompt(
        goal=problem["nl_goal"], action_cap=problem["_eval_action_cap"],
        alphabet=rig.actions_for(problem["game"]), dims=problem["_dims"],
        workspace=".", study_rounds=study_rounds))
    (workspace / "logs.txt").touch()
    return workspace


def run_problem(problem: dict, workspace: Path, agent, *, study_rounds: int) -> dict:
    env = AutumnPlanningEnv()
    started = time.time()
    try:
        runner = AutumnRunner(env, agent, problem,
                              log_path=workspace / "logs.txt",
                              alphabet=rig.actions_for(problem["game"]),
                              workspace=".", study_rounds=study_rounds)
        outcome = runner.run()
    except StartMismatch as exc:
        return {"task_uid": problem["task_uid"], "status": "start-mismatch",
                "error": str(exc), "wall_s": round(time.time() - started, 1)}
    finally:
        env.close()

    # The verdict of record is an independent replay under the ONLINE rule -- the rule
    # raw/icl/lmwm were scored under. Not `execute_and_score`, which enforces quiescence
    # and would flip 11 of the 86 rows against this arm alone (see rig.replay_and_score).
    verified, reached_at = rig.replay_and_score(problem, outcome["executed"])
    if verified != outcome["success"]:
        log.warning("%s: live verdict %s but replay says %s -- recording the replay",
                    problem["task_uid"], outcome["success"], verified)
    outcome.update({
        "task_uid": problem["task_uid"], "game": problem["game"],
        "status": "done", "success": verified, "reached_at": reached_at,
        "live_success": outcome["success"], "action_cap": problem["_eval_action_cap"],
        "goal_presentation": rig.GOAL_PRESENTATION,
    })
    return outcome


# Everything the replay page needs and the row shape has no place for. Kept as its own
# file per problem, so `rows.jsonl` stays exactly the online evaluator's row shape and
# the report keeps needing no special case.
TRACE_KEYS = ("turns", "steps", "rounds", "executed")


def write_trace(outcome: dict, problem: dict, out_root: Path) -> Path:
    """The turn-by-turn record: what the agent thought, ran, and did to the board.

    This is written per problem rather than accumulated in memory because a 15-game run
    is hours long and a crash at hour nine should not cost the first eight.
    """
    traces = Path(out_root) / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    path = traces / (problem["task_uid"].replace(":", "_") + ".json")
    record = {
        "task_uid": problem["task_uid"],
        "game": problem["game"],
        "label": rig.LABELS.get(problem["game"], problem["game"].upper()),
        "nl_goal": problem["nl_goal"],
        "action_cap": problem["_eval_action_cap"],
        "start_grid": problem["start_grid"],
        "dims": list(problem["_dims"]),
        "alphabet": rig.actions_for(problem["game"]),
        "success": outcome.get("success"),
        "reached_at": outcome.get("reached_at"),
        "live_success": outcome.get("live_success"),
        "failed_reason": outcome.get("failed_reason"),
        "actions_used": outcome.get("actions_used"),
        "study_rounds_used": outcome.get("study_rounds_used"),
        "usage": outcome.get("usage"),
        "wall_s": outcome.get("wall_s"),
        **{k: outcome.get(k) for k in TRACE_KEYS},
    }
    path.write_text(json.dumps(record))
    return path


def emit_row(outcome: dict) -> dict:
    """The online evaluator's row shape, so the report needs no special case."""
    ok = bool(outcome.get("success"))
    return {
        "task_uid": outcome["task_uid"],
        "game": outcome.get("game"),
        "goal_presentation": outcome.get("goal_presentation", rig.GOAL_PRESENTATION),
        "action_cap": outcome.get("action_cap"),
        "agent": {
            "status": outcome.get("status"),
            "attempts": 1,
            "pass_rate": 1.0 if ok else 0.0,
            "pass_any": 1.0 if ok else 0.0,
            "actions_used": outcome.get("actions_used"),
            "reached_at": outcome.get("reached_at"),
            "failed_reason": outcome.get("failed_reason"),
            "study_rounds_used": outcome.get("study_rounds_used"),
            "usage": outcome.get("usage"),
            "wall_s": outcome.get("wall_s"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpora", help="export_agent_corpus.py output")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--games", nargs="*", help="restrict to these worlds")
    ap.add_argument("--limit", type=int, help="first N problems (pilot)")
    ap.add_argument("--study-rounds", type=int, default=5)
    ap.add_argument("--base-url", default="http://127.0.0.1:8788/v1",
                    help="the parity proxy; see rig/agent for why it is not OpenRouter")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="route straight to OpenRouter with NO provider pin (declare it)")
    ap.add_argument("--no-data", action="store_true",
                    help="the agent-nodata condition: no drives/ in the workspace")
    ap.add_argument("--dry-run", action="store_true",
                    help="replay every prefix and reference plan; zero paid calls")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--transcript", default="logs/parity_proxy/reasoning.jsonl",
                    help="the proxy's reasoning log; the replay page has no other "
                         "source for it (codex emits none). Relative to the PRO-LONG "
                         "checkout, which is where proxy_ctl.sh puts it.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    problems = rig.load_problems(games=args.games)
    if args.limit:
        problems = problems[:args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _dry_run(problems)

    catalog = build_catalog(out_root / "ds_catalog.json")
    transcript = Path(args.transcript) if args.transcript else None
    if transcript is not None and not transcript.exists():
        # Not fatal: the arm's result does not depend on it. But a silent absence here
        # is a run whose reasoning is gone for good, so it is said out loud once.
        log.warning("no proxy transcript at %s yet -- reasoning will be missing from the "
                    "replay unless the proxy was started with --transcript", transcript)
    rows_path = out_root / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.open():
            try:
                done.add(json.loads(line)["task_uid"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done)} problem(s) already recorded", flush=True)

    for i, problem in enumerate(problems, 1):
        uid = problem["task_uid"]
        if uid in done:
            continue
        workspace = build_workspace(problem, Path(args.corpus), out_root,
                                    study_rounds=args.study_rounds,
                                    with_data=not args.no_data)
        agent = AutumnCodexAgent(workspace, catalog=catalog, base_url=args.base_url,
                                 timeout=args.timeout,
                                 allow_unpinned=args.allow_unpinned,
                                 transcript=transcript)
        print(f"[{i}/{len(problems)}] {uid} cap={problem['_eval_action_cap']}", flush=True)
        try:
            outcome = run_problem(problem, workspace, agent,
                                  study_rounds=args.study_rounds)
        except CredentialsExhausted as exc:
            # No row is written. A row here would say `budget-exhausted` against a session
            # that never reached the model, and `rows.jsonl` is also the resume ledger --
            # so it would be skipped on the way back and stand as a real miss forever.
            print(f"\nSTOPPING: the API key is finished -- {exc}\n"
                  f"  {len(done)} problem(s) recorded and clean; {uid} was in flight and "
                  f"is NOT recorded.\n"
                  f"  Fix the key, then re-run the same command: it resumes from "
                  f"{rows_path} and replays this problem from the start.", flush=True)
            sys.exit(2)
        try:
            write_trace(outcome, problem, out_root)
        except Exception:                          # noqa: BLE001 - the row is the result;
            log.warning("could not write the trace", exc_info=True)   # the trace is the view
        with rows_path.open("a") as handle:
            handle.write(json.dumps(emit_row(outcome)) + "\n")
        done.add(uid)
        print(f"    -> {outcome.get('status')} success={outcome.get('success')} "
              f"actions={outcome.get('actions_used')} "
              f"calls={(outcome.get('usage') or {}).get('calls')}", flush=True)


def _dry_run(problems: list[dict]) -> None:
    """Phase 4 step 1: prove the 86 starts reproduce and every reference plan still
    satisfies its goal on this path. No agent, no API key, no cost."""
    bad_start, bad_goal = [], []
    for p in problems:
        plan = list(p.get("_eval_oracle_plan") or p["plan"])[:p["_eval_action_cap"]]
        try:
            ok, _at = rig.replay_and_score(p, plan)
        except RuntimeError as exc:
            bad_start.append(f"{p['task_uid']}: {exc}")
            continue
        if not ok:
            bad_goal.append(p["task_uid"])
    print(f"problems: {len(problems)}")
    print(f"start reproduced: {len(problems) - len(bad_start)}/{len(problems)}")
    print(f"reference plan reaches goal: "
          f"{len(problems) - len(bad_start) - len(bad_goal)}/{len(problems)}")
    for line in (bad_start + bad_goal)[:10]:
        print("  FAIL", line)
    if bad_start or bad_goal:
        sys.exit(1)


if __name__ == "__main__":
    main()
