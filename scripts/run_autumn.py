#!/usr/bin/env python3
"""Run the RGB-Agent (opencode analyzer) approach on a single AutumnBench env.

This reuses rgb_agent.agent.OpenCodeAgent — the sandboxed OpenCode "analyzer"
that is the heart of RGB-Agent — but drives bai's AutumnBenchEnvWrapper instead
of ARC-AGI-3. Each batch, the analyzer reads a logs.txt of AutumnBench grid
observations (16x16 JSON color matrices) + phase + the currently-available
actions, and emits a plan of AutumnBench action STRINGS (e.g. "down",
"click 3 4", "choose_option_2"). We drain the plan into env.step(), append the
results to the log, and repeat until the episode terminates.

Run from the bai root (so AutumnBench/MARAProtocol import) with rootless podman
on PATH (so the OpenCode analyzer container runs):

  cd /home/ays57/bai
  PATH="$HOME/.local/bin:$PATH" \
  CONTAINERS_CONF="$HOME/.config/containers/containers.conf" \
  uv run python RGB-Agent/scripts/run_autumn.py \
      --env 7WWW9 --task mfp --max-steps 60 \
      --model openrouter/google/gemini-2.5-flash --interval 6
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

# --- make both packages importable regardless of which venv launched us ---
BAI_ROOT = Path(__file__).resolve().parents[2]      # /home/ays57/bai
RGB_ROOT = Path(__file__).resolve().parents[1]      # /home/ays57/bai/RGB-Agent
for p in (str(RGB_ROOT), str(BAI_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from autumn_env import AutumnBenchEnvWrapper          # noqa: E402
from rgb_agent.agent.opencode_agent import OpenCodeAgent  # noqa: E402
from rgb_agent.agent.text_tool_agent import TextToolAgent  # noqa: E402

# Load provider API keys into os.environ so the OpenCode container (which is
# passed -e OPENROUTER_API_KEY/etc. from os.environ) can authenticate. The
# normal swarm path does this via load_dotenv(); replicate it here.
try:
    from dotenv import load_dotenv
    for _envf in (RGB_ROOT / ".env", BAI_ROOT / ".env"):
        if _envf.exists():
            load_dotenv(_envf)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_autumn")


# --- AutumnBench-specific analyzer prompt templates -------------------------
# These mirror the structure of rgb_agent.agent.prompts but swap the ARC-AGI-3
# ACTION1-6 vocabulary for AutumnBench's string actions.

INITIAL_PROMPT = """\
You are a strategic advisor for an AI agent solving an AutumnBench task.
The agent's full interaction log for this run is at this ABSOLUTE path: {log_path}

You may only access this single file (use its absolute path directly with Read and Grep).

AutumnBench is a 2D grid world (cells are color-name strings). The task has an
INTERACTIVE phase (act to discover the hidden dynamics rules) followed by a
SCORED TEST phase (e.g. fill a masked region by choosing an option, detect a
changed rule, or reach a goal). A reward > 0 means you succeeded.

Deeply analyze the log: what has the agent done, how did the grid respond to
each action, and what rules explain those transitions? Use that model of the
dynamics to plan the next actions.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain the dynamics you've inferred and your reasoning)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

RESUME_PROMPT = """\
The interaction log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest steps (from where you left off). Focus on what changed: which
actions were taken, how the grid/phase responded, and whether your inferred rules
still hold. Update your model of the dynamics and your plan.

Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

PYTHON_ADDENDUM = (
    "\n\nBash (and therefore Python) is available to you. Use Python to parse the "
    "log when helpful — do NOT try to eyeball large grids.\n\n"
    "The log uses these markers:\n"
    "  [AVAILABLE ACTIONS] — the actions you may use RIGHT NOW (one per line)\n"
    "  [OBSERVATION]       — the grid as a JSON 2D list of color-name strings (+ phase/task text)\n"
    "  [ACTION TAKEN]      — the action the agent executed, followed by the resulting [OBSERVATION]\n"
    "\nTo load the latest observation:\n"
    "```python\n"
    "import re\n"
    "data = open('{log_path}').read()\n"
    "obs = re.split(r'\\[OBSERVATION\\]', data)[-1]\n"
    "print(obs[:2000])\n"
    "```\n"
)

ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "down"}}, {{"action": "click 3 4"}}], "reasoning": "why these steps"}}

CRITICAL: every "action" string MUST be copied verbatim from the most recent
"[AVAILABLE ACTIONS]" list in the log (e.g. "left", "right", "up", "down",
"noop", "reset", or for clicks "click ROW COL" with integer ROW COL in range,
and in the test phase actions like "step", "choose_option_2", "Submit choice",
"I found the change!", or "go-to-test" to leave the interactive phase). Any
action not currently available is ignored and wastes a turn.
During the interactive phase, EXPLORE to learn the rules, then use "go-to-test"
when ready. If repeated actions cause no grid change, try a DIFFERENT action.
Plan 1–{plan_size} actions. Shorter plans (2-4 steps) are strongly preferred so
the agent can observe responses and adapt.
"""


def parse_action_plan(text: str) -> list[str]:
    """Extract the list of AutumnBench action strings from the analyzer's [ACTIONS] block."""
    if not text or "[ACTIONS]" not in text:
        return []
    tail = text.rsplit("[ACTIONS]", 1)[1]
    # Find the first balanced {...} JSON object after the marker.
    start = tail.find("{")
    if start < 0:
        return []
    depth = 0
    end = -1
    for i, ch in enumerate(tail[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    try:
        obj = json.loads(tail[start:end])
    except Exception:
        return []
    plan = obj.get("plan", [])
    actions = []
    for item in plan:
        if isinstance(item, dict) and "action" in item:
            a = str(item["action"]).strip()
            # allow {"action": "click", "x": 3, "y": 4} style too
            if a.lower() == "click" and "x" in item and "y" in item:
                a = f"click {item['x']} {item['y']}"
            if a:
                actions.append(a)
        elif isinstance(item, str) and item.strip():
            actions.append(item.strip())
    return actions


def available_actions(env) -> list[str]:
    acts = list(env.language_action_space)
    # composite envs strip go-to-test from the listed space; expose it so the
    # agent can choose to leave the interactive phase for the scored test.
    if getattr(env, "_is_composite", False) and "go-to-test" not in acts:
        acts.append("go-to-test")
    return acts


def write_observation(f, env, obs, *, step: int, reward: float, action: str | None) -> None:
    stats = env.get_stats()
    phase = stats.get("phase", "?")
    if action is not None:
        f.write(f"\n[ACTION TAKEN] {action}  (reward={reward})\n")
    f.write(f"\n{'='*80}\n")
    f.write(f"Step {step} | Phase: {phase} | task={env.task_type} | last_reward={reward}\n")
    f.write(f"{'='*80}\n")
    f.write("[AVAILABLE ACTIONS]\n")
    for a in available_actions(env):
        f.write(f"- {a}\n")
    text = obs.get("text", {}) if isinstance(obs, dict) else {}
    long_term = text.get("long_term_context", "") or ""
    short_term = text.get("short_term_context", "") or ""
    f.write("\n[OBSERVATION]\n")
    if short_term:
        f.write(short_term + "\n")
    f.write(long_term + "\n")
    f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="7WWW9", help="AutumnBench env name (e.g. 7WWW9, DQ8GC)")
    ap.add_argument("--task", default="mfp", choices=["mfp", "cd", "planning", "interactive"])
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--model", default="openrouter/google/gemini-2.5-flash")
    ap.add_argument("--interval", type=int, default=6, help="actions per analyzer batch plan")
    ap.add_argument("--analyzer-retries", type=int, default=8,
                    help="analyzer reissues per batch (absorbs gemini MALFORMED_FUNCTION_CALL churn)")
    ap.add_argument("--analyzer-mode", default="opencode", choices=["opencode", "texttool"],
                    help="opencode = native-function-calling OpenCode sandbox (default); "
                         "texttool = ReAct text-protocol analyzer (no native tools, no MALFORMED)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default: RGB-Agent/evaluation_results_autumn/<ts>)")
    args = ap.parse_args()

    ts = time.strftime("%m%dT%H%M%S")
    out_dir = Path(args.out) if args.out else RGB_ROOT / "evaluation_results_autumn" / f"{ts}_{args.env}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.txt"
    log.info("output dir: %s", out_dir)

    env = AutumnBenchEnvWrapper(
        env_name=args.env, task_type=args.task,
        max_episode_steps=args.max_steps, max_interaction_steps=args.max_steps,
        seed=args.seed, render_mode="text",
        logging_path=str(out_dir / "autumn_inner"),
    )
    AgentCls = TextToolAgent if args.analyzer_mode == "texttool" else OpenCodeAgent
    log.info("analyzer mode: %s (%s)", args.analyzer_mode, AgentCls.__name__)
    agent = AgentCls(
        model=args.model,
        plan_size=args.interval,
        resume_session=True,
        restrict_tools=False,   # opencode only: rootless sandbox relies on OS container isolation
        initial_prompt=INITIAL_PROMPT,
        resume_prompt=RESUME_PROMPT,
        actions_addendum=ACTIONS_ADDENDUM,
        python_addendum=PYTHON_ADDENDUM,
    )

    obs, info = env.reset(seed=args.seed)
    with open(log_path, "w", encoding="utf-8") as f:
        write_observation(f, env, obs, step=0, reward=0.0, action=None)

    total_reward = 0.0
    step = 0
    done = False
    queue: list[str] = []
    action_hist: list[str] = []
    analyzer_calls = 0
    empty_plans = 0

    while step < args.max_steps and not done:
        if not queue:
            # fire the analyzer for a fresh batch plan. gemini-2.5-flash via
            # OpenRouter intermittently returns MALFORMED_FUNCTION_CALL; retry
            # with backoff (mirrors the ls20 swarm, which survives the churn).
            text = None
            for attempt in range(args.analyzer_retries):
                analyzer_calls += 1
                text = agent.analyze(log_path, step, retry_nudge="" if attempt == 0 else
                                     "Your previous response was missing a valid [ACTIONS] JSON block. "
                                     "End with [ACTIONS] and a JSON plan of available action strings.")
                queue = parse_action_plan(text or "")
                if queue:
                    break
                log.warning("analyzer attempt %d/%d produced no actions", attempt + 1, args.analyzer_retries)
                if attempt < args.analyzer_retries - 1:
                    time.sleep(min(2.0 * (2 ** attempt), 20.0))
            if not queue:
                empty_plans += 1
                log.error("no actions after %d retries; stopping at step %d", args.analyzer_retries, step)
                break
            log.info("step %d: analyzer planned %d actions: %s", step, len(queue), queue)

        action = queue.pop(0)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        total_reward += float(reward or 0.0)
        action_hist.append(action)
        done = bool(term or trunc)
        with open(log_path, "a", encoding="utf-8") as f:
            write_observation(f, env, obs, step=step, reward=float(reward or 0.0), action=action)
        log.info("step %d: %-22s reward=%.3f term=%s trunc=%s phase=%s",
                 step, action, float(reward or 0.0), term, trunc, env.get_stats().get("phase"))
        # When a scored phase resolves, reward fires; keep going until terminal.

    # summary
    from collections import Counter
    mix = Counter(a.split()[0] for a in action_hist)
    summary = {
        "env": args.env, "task": args.task, "model": args.model,
        "steps": step, "total_reward": total_reward, "done": done,
        "final_phase": env.get_stats().get("phase"),
        "analyzer_calls": analyzer_calls, "empty_plans": empty_plans,
        "failed_candidates": list(getattr(env, "failed_candidates", []))[:20],
        "action_mix": dict(mix),
        "action_hist": action_hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("=== DONE === steps=%d total_reward=%.3f final_phase=%s action_mix=%s",
             step, total_reward, summary["final_phase"], dict(mix))
    log.info("summary: %s", out_dir / "summary.json")
    env.close()


if __name__ == "__main__":
    main()
