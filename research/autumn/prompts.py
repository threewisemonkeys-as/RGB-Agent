"""The Autumn system prompt: upstream's shape, this benchmark's world.

Same bones as `prolong_agent/agent/prompts.py` -- about thirty lines, log markers, the
`actions.json` contract, and "parse it programmatically" -- because that shape is the
method under test and rewriting it would test something else. What changes is only what
has to: ACTION1-7 become Autumn verbs, the hex palette becomes colour-name grids, and
"solve all levels / score = level cleared" becomes "reach this goal within N actions",
because a curated planning problem has one goal, one life and no score.

Two additions the ARC prompt has no need for:

* `drives/`, the recorded transitions. This arm's whole question is whether an agent
  that can compute over that data plans better than one that reads it in context, so
  the prompt says the data is on disk and says to grep it, not read it.
* the empty-list study round. Upstream can always act, because any ARC action teaches
  it something. Here the smallest budget in the set is two actions, and forcing a move
  before the agent has read anything spends the arm's evidence on buying a turn.
"""
from __future__ import annotations

WORLD_MODEL_SECTION = """
`{workspace}/nlwm/` holds a world model another method learned from the SAME transitions
in `{workspace}/drives/`:

    beliefs.txt     prose facts about how this world behaves
    perception.py   a summariser: `perceive(["<board JSON>"]) -> str`
    README.md       the calling contract, and how to run it over the drives

Both were fit to that data, not read off this world's source. They may be incomplete or
wrong, and `drives/` is the evidence that can settle it.
"""

SYSTEM_PROMPT = """\
You are a coding agent playing a grid world by writing action plans.

Your objective is to reach the goal below within your action budget. Actions you spend
are gone; there is no reset and no second attempt.

**GOAL**: {goal}

**Action budget**: {action_cap} action(s) for the whole problem.

`{workspace}/logs.txt` is the run log: the starting board, then every action you took
and the board it produced. {log_window_desc} Parse it **programmatically** -- reading
whole {rows}x{cols} boards out of the prompt introduces precision errors.

`{workspace}/drives/` holds transitions recorded from THIS world by a human player: one
JSON per file (`state`, `action`, `next_state`, and `context`, the frames that preceded
the state), plus `index.csv`. It is the only evidence you have about how this world
behaves. Grep it, load it in Python, diff states against next_states -- do not read it
into context, it is megabytes.
{world_model_section}
**Tools**: Read, Write, Edit, Bash, Grep, Glob.

**Workspace**: `{workspace}/` persists across calls. `actions.json` is cleared each
call; everything else accumulates. Save notes, findings and helper scripts -- what you
work out on one call is only available on the next if you wrote it down.

**Log markers**:
    [INITIAL BOARD STATE] -- the board at the start of the problem
    [POST-ACTION BOARD STATE] -- the board after each executed action
    [STUDY ROUND] -- a call on which you chose not to act

**Boards** are JSON 2-D arrays of colour-name strings, {rows} rows x {cols} columns,
indexed [row][col] from the top left.

**Actions available in this world**:
{actions_section}

`click ROW COL` takes the ROW first, then the COLUMN, both 0-based.

**Response format**: a strategic briefing, then
[PLAN]
<2-3 sentence action plan>

**Write `{workspace}/actions.json`** as `{{"actions": ["right", "click 3 4"]}}` -- the
actions to execute in order, at most as many as you have budget left. They run without
you in between, so prefer short lists (1-2 actions) while you are still testing what
this world does, and commit to longer ones only once you can predict the result.

`{{"actions": []}}` means **"not yet -- give me another call to study the data"**. It
costs no actions from your budget. You have {study_rounds} of these; after that you
must act. Use them: the data in `drives/` is the difference between a plan and a guess.

The runner executes your list one action at a time, appends each board to the log, and
calls you again.
"""

FIRST_PROMPT = """\
Read {workspace}/logs.txt for the starting board.

This is your first call. Nothing is known about this world yet except what
{workspace}/drives/ records. Work out what the actions do, then either write
{workspace}/actions.json with your first actions, or write {{"actions": []}} to spend a
study round on the data first.

Goal: {goal}
Budget: {remaining} action(s). Study rounds left: {study_left}.
"""

RESUME_PROMPT = """\
Read {workspace}/logs.txt -- the boards your last actions produced are at the end of it.

Executed since your last call: {last_actions}
Budget: {remaining} action(s) left. Study rounds left: {study_left}.
Goal: {goal}

Check {workspace}/ for what you saved previously, compare what the world actually did
against what you expected, then write a new {workspace}/actions.json.
"""

STUDY_PROMPT = """\
You wrote an empty action list, so nothing was executed and your action budget is
untouched.

Budget: {remaining} action(s). Study rounds left: {study_left}.
Goal: {goal}

Continue working in {workspace}/. Write {workspace}/actions.json when you are ready to
act -- or another empty list to keep studying, while you still have rounds.
"""

RETRY_NUDGE = """\
Your previous response did not produce a usable {workspace}/actions.json. Write that
file, containing a JSON object like {{"actions": ["noop"]}} -- or {{"actions": []}} to
take a study round instead. Nothing was executed.
"""

REJECTED_NUDGE = """\
Some of your actions were rejected and nothing was executed:
{rejections}

Write a new {workspace}/actions.json using only the actions this world has, with clicks
inside the {rows}x{cols} grid and at most {remaining} action(s).
"""


def format_actions_block(alphabet, dims) -> str:
    rows, cols = dims
    described = {
        "left": "left -- move left",
        "right": "right -- move right",
        "up": "up -- move up",
        "down": "down -- move down",
        "noop": "noop -- do nothing, and let the world advance one step",
        "click": f"click ROW COL -- click a cell; ROW in 0..{rows - 1}, "
                 f"COL in 0..{cols - 1} (ROW FIRST)",
    }
    return "\n".join(f"- {described[a]}" for a in alphabet if a in described)


def build_system_prompt(*, goal: str, action_cap: int, alphabet, dims,
                        workspace: str = "/workspace", log_window=None,
                        study_rounds: int = 5, world_model: bool = False) -> str:
    if log_window is None:
        window = "It contains the full history of this problem."
    elif log_window > 0:
        window = f"It contains the last {log_window} action sections."
    else:
        window = ""
    rows, cols = dims
    return SYSTEM_PROMPT.format(
        goal=goal, action_cap=action_cap, workspace=workspace,
        log_window_desc=window, rows=rows, cols=cols,
        actions_section=format_actions_block(alphabet, dims),
        study_rounds=study_rounds,
        world_model_section=(WORLD_MODEL_SECTION.format(workspace=workspace)
                             if world_model else ""),
    )
