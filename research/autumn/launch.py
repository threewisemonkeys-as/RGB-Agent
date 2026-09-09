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
        nlwm/         the learned world model for this world (only with --nlwm-root)
        AGENTS.md     the system prompt
        logs.txt      the run log the agent greps
        actions.json  cleared before every call
        agent.txt     the session transcript, for the audit

`--workers N` runs N of those sessions at once. It is a scheduling change and nothing
more: the sessions were already independent by design, so the only thing shared between
them is the claim directory that stops two workers taking the same problem.

    uv run python -m research.autumn.launch --corpus corpora --out logs/agent --dry-run
    uv run python -m research.autumn.launch --corpus corpora --out logs/agent --games n2ntd
    uv run python -m research.autumn.launch --corpus corpora --out logs/agent \
        --nlwm-root logs/2026-08-24/human_curated --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.autumn import prompts, rig
from research.autumn.agent import (
    AutumnCodexAgent, CredentialsExhausted, build_catalog)
from research.autumn.env import AutumnPlanningEnv, StartMismatch
from research.autumn.runner import AutumnRunner

log = logging.getLogger(__name__)


# The learned world model, as the `lmwm` arm loads it: `<root>/rexpure/<game>_s1/`,
# seed 1, the same two files `eval_curated_nl_online.py` compiles and prompts with. The
# names are rewritten on the way in -- `best_perception_rexpure_seed1.py` names a search
# this arm is not running -- but the bytes are the bytes, and the manifest records their
# sha256 so a later reader can prove which artifacts a session was handed.
NLWM_FILES = {"beliefs.txt": "best_beliefs_rexpure_seed1.txt",
              "perception.py": "best_perception_rexpure_seed1.py"}

NLWM_README = """\
# A world model learned from `../drives/`

`beliefs.txt` and `perception.py` were fit offline to the same transitions that sit in
`../drives/`, by a different method. They are claims about this world, not its source.
Where they disagree with the recorded transitions, the transitions are right.

## perception.py

    perceive(observation_history: list[str]) -> str

Takes board JSON strings, oldest last, and returns a compact feature summary:

    import json, sys
    sys.path.insert(0, "nlwm")
    from perception import perceive

    d = json.load(open("drives/t000.json"))
    print(perceive([d["state"]]), d["action"], perceive([d["next_state"]]))

The method that learned it always called it with a ONE-element list. Some of these
modules also have multi-frame paths that compare consecutive boards, so passing
`[before, after]` is allowed and may say more than two separate calls do.

It is not meant to crash, but it was learned rather than written -- wrap it if you
depend on it.
"""


def nlwm_dir(root: Path | str, game: str) -> Path:
    return Path(root) / "rexpure" / f"{game}_s1"


def check_world_model(root: Path, problems: list[dict]) -> None:
    """Fail before the first paid call, not on problem 61.

    A missing artifact halfway through is 60 sessions of one condition and the rest of
    another, in one `rows.jsonl`, with nothing on the row saying which.
    """
    missing = [str(nlwm_dir(root, g) / src)
               for g in sorted({p["game"] for p in problems})
               for src in NLWM_FILES.values()
               if not (nlwm_dir(root, g) / src).is_file()]
    if missing:
        raise SystemExit("no world model for {} world(s):\n  {}".format(
            len(missing) // len(NLWM_FILES), "\n  ".join(missing)))


def write_manifest(root: Path, problems: list[dict], out_root: Path) -> Path:
    games = sorted({p["game"] for p in problems})
    entry = {}
    for g in games:
        d = nlwm_dir(root, g)
        entry[g] = {dst: {"source": str(d / src),
                          "sha256": hashlib.sha256((d / src).read_bytes()).hexdigest(),
                          "bytes": (d / src).stat().st_size}
                    for dst, src in NLWM_FILES.items()}
    path = out_root / "nlwm_manifest.json"
    path.write_text(json.dumps({"root": str(root), "games": entry}, indent=1))
    return path


def copy_world_model(problem: dict, root: Path, workspace: Path) -> None:
    src = nlwm_dir(root, problem["game"])
    dst = workspace / "nlwm"
    dst.mkdir()
    for name, source in NLWM_FILES.items():
        shutil.copy(src / source, dst / name)
    (dst / "README.md").write_text(NLWM_README)


def build_workspace(problem: dict, corpus_root: Path, out_root: Path, *,
                    study_rounds: int, with_data: bool = True,
                    nlwm_root: Path | None = None) -> Path:
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

    if nlwm_root is not None:
        copy_world_model(problem, nlwm_root, workspace)

    (workspace / "AGENTS.md").write_text(prompts.build_system_prompt(
        goal=problem["nl_goal"], action_cap=problem["_eval_action_cap"],
        alphabet=rig.actions_for(problem["game"]), dims=problem["_dims"],
        workspace=".", study_rounds=study_rounds,
        world_model=nlwm_root is not None))
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
        # Which condition produced this session. Not on the row: `rows.jsonl` is the
        # online evaluator's shape and a new key there is a special case downstream.
        "world_model": bool(outcome.get("world_model")),
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


def rows_path(out_root: Path, worker_id: int | None) -> Path:
    return out_root / ("rows.jsonl" if worker_id is None else f"rows.w{worker_id}.jsonl")


def recorded(out_root: Path) -> set[str]:
    """Every task_uid already written, across the merged ledger and every worker's own.

    Resume has to read all of them: under `--workers` the rows are written per worker
    and merged only when the run ends, so a run killed mid-flight leaves its record
    spread across `rows.w*.jsonl` and nothing in `rows.jsonl`.
    """
    done = set()
    for f in [out_root / "rows.jsonl", *sorted(out_root.glob("rows.w*.jsonl"))]:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            try:
                done.add(json.loads(line)["task_uid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def merge_rows(out_root: Path) -> int:
    """Fold the workers' rows into `rows.jsonl`, first writer of a uid wins.

    Written through a temp file because the watcher and the report both read this path
    while the run is finishing, and a half-written ledger reads as a shorter run.
    """
    seen: dict[str, str] = {}
    for f in [out_root / "rows.jsonl", *sorted(out_root.glob("rows.w*.jsonl"))]:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                uid = json.loads(line)["task_uid"]
            except (json.JSONDecodeError, KeyError):
                continue
            seen.setdefault(uid, line)
    tmp = out_root / "rows.jsonl.tmp"
    tmp.write_text("".join(v + "\n" for v in seen.values()))
    os.replace(tmp, out_root / "rows.jsonl")
    return len(seen)


def claim(claims: Path, uid: str) -> bool:
    """Take this problem, or report that another worker already holds it.

    `O_EXCL` is the whole of the mutual exclusion -- no lock server, no coordinator. A
    worker that dies mid-problem leaves its claim behind deliberately: the session is
    half-played, and picking it up again would bill the arm twice for one row. Clearing
    `claims/` by hand is how you say you meant to replay it.
    """
    try:
        fd = os.open(str(claims / uid.replace(":", "_")),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
    os.close(fd)
    return True


def order_longest_first(problems: list[dict], order_by: str) -> list[dict]:
    """Sort by a previous run's measured wall, longest first.

    Scheduling only -- with the claim queue the order decides who takes what, never what
    is run. It matters because the battery is badly skewed: one world was a quarter of
    the last run's wall, and a 96-minute problem picked up last is 96 minutes with one
    worker awake.
    """
    walls: dict[str, float] = {}
    for f in (Path(order_by) / "traces").glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("task_uid"):
            walls[d["task_uid"]] = float(d.get("wall_s") or 0.0)
    if not walls:
        log.warning("no measured walls under %s; keeping the battery's own order",
                    order_by)
        return problems
    par = sum(walls.values()) / len(walls)
    return sorted(problems, key=lambda p: -walls.get(p["task_uid"], par))


def ensure_catalog(path: Path, *, rebuild: bool) -> Path:
    """Workers reuse the supervisor's catalog: `build_catalog` shells out to codex and
    rewrites the file, and six of those racing on one path is a torn catalog."""
    if not rebuild and path.is_file() and path.stat().st_size:
        return path
    return build_catalog(path)


def worker_routing(args, i: int) -> list[str]:
    """Where worker `i` sends its calls, and where it reads its own reasoning back.

    With `--proxy-port-base` each worker gets a proxy to itself. That is not about load:
    `agent.py` joins a turn to its reasoning by byte offset into the proxy's transcript,
    which is only that turn's reasoning while one session at a time is writing it. Six
    sessions sharing one transcript would leave every turn holding all six sessions'
    thinking, with nothing in the record to say so.
    """
    if not args.proxy_port_base:
        return ["--base-url", args.base_url, "--transcript", args.transcript]
    return ["--base-url", f"http://127.0.0.1:{args.proxy_port_base + i}/v1",
            "--transcript", f"{args.proxy_log_root}/w{i}/reasoning.jsonl"]


def supervise(args, n: int) -> int:
    """Spawn N copies of this module as workers and wait.

    Processes, not threads: `Branch` drives the Autumn interpreter, whose own evaluator
    funnels every call through a single executor thread, so N of them in one process is
    not something this code gets to assume is safe.
    """
    out_root = Path(args.out)
    (out_root / "claims").mkdir(exist_ok=True)
    ensure_catalog(out_root / "ds_catalog.json", rebuild=True)
    (out_root / "launch.pid").write_text(f"{os.getpid()}\n")

    base = [sys.executable, "-m", "research.autumn.launch",
            "--corpus", args.corpus, "--out", args.out,
            "--study-rounds", str(args.study_rounds),
            "--timeout", str(args.timeout),
            "--workers", str(n), "--order-by", args.order_by]
    if args.games:
        base += ["--games", *args.games]
    if args.limit:
        base += ["--limit", str(args.limit)]
    if args.nlwm_root:
        base += ["--nlwm-root", args.nlwm_root]
    if args.no_data:
        base.append("--no-data")
    if args.allow_unpinned:
        base.append("--allow-unpinned")

    procs = []
    for i in range(n):
        handle = (out_root / f"worker{i}.log").open("a")
        procs.append((i, subprocess.Popen(base + worker_routing(args, i)
                                          + ["--worker-id", str(i)],
                                          stdout=handle, stderr=subprocess.STDOUT),
                      handle))
        print(f"worker {i} pid={procs[-1][1].pid} -> {out_root}/worker{i}.log", flush=True)
        time.sleep(3)          # stagger the codex starts; they each shell out on boot

    codes = {}
    try:
        for i, proc, handle in procs:
            codes[i] = proc.wait()
            handle.close()
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating workers", flush=True)
        for _i, proc, _h in procs:
            proc.terminate()
        for _i, proc, _h in procs:
            proc.wait()
        raise
    finally:
        total = merge_rows(out_root)
        print(f"merged {total} row(s) into {out_root}/rows.jsonl", flush=True)

    dead = {i: c for i, c in codes.items() if c != 0}
    for i, c in dead.items():
        print(f"worker {i} exited {c}"
              + ("  (the API key is finished -- fix it and re-run to resume)"
                 if c == 2 else ""), flush=True)
    return 2 if any(c == 2 for c in dead.values()) else (1 if dead else 0)


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
    ap.add_argument("--nlwm-root", default="",
                    help="hand each session the world model learned for its world: "
                         "<root>/rexpure/<game>_s1/, the seed-1 artifacts the lmwm arm "
                         "plans with, copied into the workspace as nlwm/")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many sessions at once (processes, not threads)")
    ap.add_argument("--worker-id", type=int,
                    help=argparse.SUPPRESS)      # set by the supervisor on its children
    ap.add_argument("--order-by", default="",
                    help="a previous run dir; take its measured per-problem walls and "
                         "hand out the long problems first")
    ap.add_argument("--proxy-port-base", type=int, default=0,
                    help="give worker i its own proxy at 127.0.0.1:<base+i> and its own "
                         "reasoning log; see proxy_fleet.sh for why a shared one cannot "
                         "be used with --workers")
    ap.add_argument("--proxy-log-root", default="logs/parity_proxy",
                    help="where proxy_fleet.sh put the per-worker logs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    problems = rig.load_problems(games=args.games)
    if args.limit:
        problems = problems[:args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _dry_run(problems)

    nlwm_root = Path(args.nlwm_root) if args.nlwm_root else None
    if nlwm_root is not None:
        check_world_model(nlwm_root, problems)

    worker = args.worker_id
    if worker is None and args.workers > 1:
        if nlwm_root is not None:
            print(f"world model: {write_manifest(nlwm_root, problems, out_root)}",
                  flush=True)
        sys.exit(supervise(args, args.workers))

    if args.order_by:
        problems = order_longest_first(problems, args.order_by)
    catalog = ensure_catalog(out_root / "ds_catalog.json", rebuild=worker is None)
    if worker is None:
        (out_root / "launch.pid").write_text(f"{os.getpid()}\n")
        if nlwm_root is not None:
            print(f"world model: {write_manifest(nlwm_root, problems, out_root)}",
                  flush=True)
    transcript = Path(args.transcript) if args.transcript else None
    if transcript is not None and not transcript.exists():
        # Not fatal: the arm's result does not depend on it. But a silent absence here
        # is a run whose reasoning is gone for good, so it is said out loud once.
        log.warning("no proxy transcript at %s yet -- reasoning will be missing from the "
                    "replay unless the proxy was started with --transcript", transcript)
    rows_file = rows_path(out_root, worker)
    claims = out_root / "claims"
    if worker is not None:
        claims.mkdir(exist_ok=True)
    done = recorded(out_root)
    if done:
        print(f"resuming: {len(done)} problem(s) already recorded", flush=True)

    for i, problem in enumerate(problems, 1):
        uid = problem["task_uid"]
        if uid in done:
            continue
        if worker is not None and not claim(claims, uid):
            continue
        workspace = build_workspace(problem, Path(args.corpus), out_root,
                                    study_rounds=args.study_rounds,
                                    with_data=not args.no_data,
                                    nlwm_root=nlwm_root)
        agent = AutumnCodexAgent(workspace, catalog=catalog, base_url=args.base_url,
                                 timeout=args.timeout,
                                 allow_unpinned=args.allow_unpinned,
                                 transcript=transcript)
        tag = "" if worker is None else f"w{worker} "
        print(f"{tag}[{i}/{len(problems)}] {uid} cap={problem['_eval_action_cap']}",
              flush=True)
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
                  f"{rows_file} and replays this problem from the start.", flush=True)
            if worker is not None:      # let the next attempt have it back
                (claims / uid.replace(":", "_")).unlink(missing_ok=True)
            sys.exit(2)
        outcome["world_model"] = nlwm_root is not None
        try:
            write_trace(outcome, problem, out_root)
        except Exception:                          # noqa: BLE001 - the row is the result;
            log.warning("could not write the trace", exc_info=True)   # the trace is the view
        with rows_file.open("a") as handle:
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
