#!/usr/bin/env python3
"""Run the RGB-Agent analyzer approach on a single MiniHack (NetHack) env.

Mirrors scripts/run_autumn.py: it reuses the RGB-Agent analyzer (either the
OpenCode sandbox, the text-protocol ReAct analyzer, or a local mock) but drives
BALROG's MiniHack environment instead of ARC-AGI-3 / AutumnBench. Each batch the
analyzer reads a logs.txt of MiniHack observations (ASCII map + language
description + message + stats + inventory) plus the currently-available actions,
and emits a plan of MiniHack action STRINGS (e.g. "south", "far east", "kick",
"search"). We validate + drain the plan into env.step(), append results to the
log, and repeat until the episode terminates.

Run from the bai root (so BALROG + autumn imports resolve). For the opencode
analyzer mode you also need rootless podman on PATH; texttool/mock need neither:

  cd /home/ays57/bai
  uv run python RGB-Agent/scripts/run_minihack.py \
      --task MiniHack-Quest-Easy-v0 --max-steps 30 \
      --analyzer-mode texttool --model openrouter/google/gemini-2.5-flash

  # validate the whole pipeline with no API calls:
  uv run python RGB-Agent/scripts/run_minihack.py --analyzer-mode mock --max-steps 12
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# --- make every package importable regardless of which venv launched us ---
BAI_ROOT = Path(__file__).resolve().parents[2]      # /home/ays57/bai
RGB_ROOT = Path(__file__).resolve().parents[1]      # /home/ays57/bai/RGB-Agent
for p in (str(RGB_ROOT), str(BAI_ROOT), str(BAI_ROOT / "BALROG")):
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf                       # noqa: E402
from balrog.environments import make_env             # noqa: E402
from rgb_agent.agent.opencode_agent import OpenCodeAgent      # noqa: E402
from rgb_agent.agent.text_tool_agent import TextToolAgent     # noqa: E402

# Load provider API keys into os.environ (texttool uses litellm; opencode passes
# them to the container). Mirrors the swarm/autumn path.
try:
    from dotenv import load_dotenv
    for _envf in (RGB_ROOT / ".env", BAI_ROOT / ".env"):
        if _envf.exists():
            load_dotenv(_envf)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_minihack")


# --- MiniHack-specific analyzer prompt templates ----------------------------
INITIAL_PROMPT = """\
You are a strategic advisor for an AI agent playing MiniHack (a NetHack-based
dungeon game). The agent's full interaction log for this run is at this ABSOLUTE
path: {log_path}

You may only access this single file (use its absolute path directly).

MiniHack is a 2D ASCII dungeon. '@' is the agent; '>' are the stairs down (the
usual goal); '.' is floor, '#' a corridor, '-' and '|' are walls, '+' a door,
'}}' water, '{{' a fountain, letters are monsters/items. The log gives you, each
step: an ASCII map, a language description of nearby features, the latest game
message, the agent's stats, and its inventory.

Deeply analyze the log: where is the agent, what is around it, what has each
action done to its position/state, and what is the shortest sensible route toward
the goal (reach the stairs down '>'). Use that to plan the next actions.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (where the agent is, what you infer, your reasoning)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

RESUME_PROMPT = """\
The interaction log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest steps (from where you left off). Focus on what changed: which
actions were taken, how the map/message/stats responded, and whether the agent
moved as intended (walls and monsters block movement). Update your plan toward
reaching the stairs down ('>').

Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

PYTHON_ADDENDUM = (
    "\n\nBash (and therefore Python) is available to you. Use Python to parse the "
    "log when helpful — do NOT try to eyeball the whole history.\n\n"
    "The log uses these markers:\n"
    "  [AVAILABLE ACTIONS] — the actions you may use RIGHT NOW (one per line)\n"
    "  [OBSERVATION]       — the ASCII map + language description + message + stats + inventory\n"
    "  [ACTION TAKEN]      — the action the agent executed, followed by the resulting [OBSERVATION]\n"
    "\nTo load the latest observation:\n"
    "```python\n"
    "import re\n"
    "data = open('{log_path}').read()\n"
    "obs = re.split(r'\\[OBSERVATION\\]', data)[-1]\n"
    "print(obs[:2500])\n"
    "```\n"
)

ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "south"}}, {{"action": "far east"}}], "reasoning": "why these steps"}}

CRITICAL: every "action" string MUST be copied verbatim from the most recent
"[AVAILABLE ACTIONS]" list in the log (e.g. "north", "south", "east", "west",
the diagonals, "far north"/"far east"/... to travel until blocked, "open",
"kick", "search", "pickup", "eat", "pray", "down" to descend stairs). Any action
not in that list is ignored and wastes a turn.
Prefer the "far <dir>" travel actions to cross open floor quickly; use single
steps near walls, doors, and monsters. If a move causes NO change in the agent's
position, something is blocking it — try a different direction or "open"/"kick".
Plan 1-{plan_size} actions. Shorter plans (2-4 steps) are strongly preferred so
the agent can observe responses and adapt.
"""


def parse_action_plan(text: str) -> list[str]:
    """Extract the list of MiniHack action strings from the analyzer's [ACTIONS] block."""
    if not text or "[ACTIONS]" not in text:
        return []
    tail = text.rsplit("[ACTIONS]", 1)[1]
    start = tail.find("{")
    if start < 0:
        return []
    depth, end = 0, -1
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
    actions = []
    for item in obj.get("plan", []):
        if isinstance(item, dict) and "action" in item:
            a = str(item["action"]).strip()
            if a:
                actions.append(a)
        elif isinstance(item, str) and item.strip():
            actions.append(item.strip())
    return actions


def available_actions(env) -> list[str]:
    return list(env.env.language_action_space)


def write_observation(f, env, obs, *, step: int, reward: float, action: str | None) -> None:
    if action is not None:
        f.write(f"\n[ACTION TAKEN] {action}  (reward={reward})\n")
    stats = env.get_stats()
    f.write(f"\n{'='*80}\n")
    f.write(f"Step {step} | task={env.task_name} | last_reward={reward} | "
            f"return={stats.get('episode_return')} | progression={stats.get('progression')}\n")
    f.write(f"{'='*80}\n")
    f.write("[AVAILABLE ACTIONS]\n")
    for a in available_actions(env):
        f.write(f"- {a}\n")
    text = obs.get("text", {}) if isinstance(obs, dict) else {}
    f.write("\n[OBSERVATION]\n")
    long_term = text.get("long_term_context", "") or ""
    short_term = text.get("short_term_context", "") or ""
    if long_term:
        f.write(long_term + "\n")
    if short_term:
        f.write(short_term + "\n")
    f.flush()


# --- mock analyzer: validates the full pipeline with zero API calls ----------
class MockAgent:
    """Drop-in analyzer that returns canned [ACTIONS] responses (no LLM)."""

    # cycle of plans; includes one invalid action ("teleport") to exercise the
    # action-validity / failed_candidates path.
    _PLANS = [
        ["south", "south"],
        ["far east", "east"],
        ["search", "teleport", "north"],   # "teleport" is NOT a MiniHack action
        ["far south", "west"],
    ]

    def __init__(self, *_, **__):
        self._i = 0
        log.info("MockAgent active (no API calls; canned plans incl. 1 invalid action)")

    def analyze(self, log_path, action_num, retry_nudge: str = ""):
        plan = self._PLANS[self._i % len(self._PLANS)]
        self._i += 1
        plan_json = json.dumps({"plan": [{"action": a} for a in plan],
                                "reasoning": "mock canned plan"})
        text = (
            "[STRATEGIC ANALYSIS FROM LOG REVIEW]\n"
            f"(mock) batch {self._i}: emitting a canned plan to exercise parsing + stepping.\n"
            "[PLAN]\n(mock) follow the canned actions.\n"
            f"[ACTIONS]\n{plan_json}\n"
        )
        try:
            (Path(log_path).parent / (Path(log_path).stem + "_analyzer.txt")).open(
                "a", encoding="utf-8").write(f"\n# MOCK analyze action_num={action_num}\n{text}\n")
        except Exception:
            pass
        return text


def build_agent(mode: str, model: str, plan_size: int):
    common = dict(
        model=model, plan_size=plan_size, resume_session=True, restrict_tools=False,
        initial_prompt=INITIAL_PROMPT, resume_prompt=RESUME_PROMPT,
        actions_addendum=ACTIONS_ADDENDUM, python_addendum=PYTHON_ADDENDUM,
    )
    if mode == "mock":
        return MockAgent(**common)
    if mode == "texttool":
        return TextToolAgent(**common)
    return OpenCodeAgent(**common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="MiniHack-Quest-Easy-v0", help="MiniHack gym task id")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--model", default="openrouter/google/gemini-2.5-flash")
    ap.add_argument("--interval", type=int, default=5, help="actions per analyzer batch plan")
    ap.add_argument("--analyzer-retries", type=int, default=8,
                    help="analyzer reissues per batch (absorbs gemini MALFORMED churn)")
    ap.add_argument("--analyzer-mode", default="texttool", choices=["opencode", "texttool", "mock"],
                    help="opencode = native-tool OpenCode sandbox; texttool = ReAct text protocol "
                         "(no native tools, no MALFORMED); mock = canned plans, no API calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ts = time.strftime("%m%dT%H%M%S")
    short_task = args.task.replace("MiniHack-", "").replace("-v0", "")
    out_dir = Path(args.out) if args.out else RGB_ROOT / "evaluation_results_minihack" / f"{ts}_{short_task}_{args.analyzer_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.txt"
    log.info("output dir: %s", out_dir)

    # Build the BALROG MiniHack env from the default config (text-only).
    cfg = OmegaConf.load(str(BAI_ROOT / "BALROG" / "balrog" / "config" / "config.yaml"))
    cfg.agent.max_image_history = 0                       # no VLM image rendering
    cfg.envs.minihack_kwargs.include_lang_obs = True      # give the analyzer the language obs
    cfg.envs.minihack_kwargs.max_episode_steps = max(args.max_steps + 5, 100)
    env = make_env("minihack", args.task, cfg)

    agent = build_agent(args.analyzer_mode, args.model, args.interval)
    log.info("analyzer mode: %s | task: %s", args.analyzer_mode, args.task)

    obs, info = env.reset(seed=args.seed)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[GOAL] Reach the stairs down ('>').\n\n")
        f.write("[INSTRUCTIONS]\n" + env.get_instruction_prompt() + "\n")
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

        raw = queue.pop(0)
        action = env.check_action_validity(raw)   # invalid -> default_action, recorded
        if action != raw:
            log.warning("invalid action %r -> substituted %r", raw, action)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        total_reward += float(reward or 0.0)
        action_hist.append(raw)
        done = bool(term or trunc)
        with open(log_path, "a", encoding="utf-8") as f:
            write_observation(f, env, obs, step=step, reward=float(reward or 0.0),
                              action=raw if action == raw else f"{raw} (invalid->{action})")
        log.info("step %d: %-22s reward=%.3f term=%s trunc=%s",
                 step, raw, float(reward or 0.0), term, trunc)

    from collections import Counter
    mix = Counter(a.split()[0] for a in action_hist)
    summary = {
        "task": args.task, "analyzer_mode": args.analyzer_mode, "model": args.model,
        "steps": step, "total_reward": total_reward, "done": done,
        "episode_return": env.get_stats().get("episode_return"),
        "progression": env.get_stats().get("progression"),
        "end_reason": str(env.get_stats().get("end_reason")),
        "analyzer_calls": analyzer_calls, "empty_plans": empty_plans,
        "failed_candidates": list(getattr(env, "failed_candidates", []))[:20],
        "action_mix": dict(mix),
        "action_hist": action_hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("=== DONE === steps=%d total_reward=%.3f progression=%s action_mix=%s",
             step, total_reward, summary["progression"], dict(mix))
    log.info("summary: %s", out_dir / "summary.json")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
