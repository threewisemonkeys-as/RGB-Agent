"""Text-protocol (ReAct-style) analyzer — a drop-in alternative to OpenCodeAgent.

WHY THIS EXISTS
---------------
The OpenCodeAgent analyzer drives the model through OpenCode, which uses the
provider's *native function-calling* API for its read/grep/bash tools. With
gemini-2.5-flash over OpenRouter that path intermittently returns
MALFORMED_FUNCTION_CALL (the model emits malformed JSON for a tool call's
arguments). It clusters and exhausts the retry budget, cutting runs short.

MALFORMED_FUNCTION_CALL is, by construction, a *native-function-calling* failure
— it cannot happen on a plain-text completion. So this analyzer removes native
tools entirely. The model still gets to investigate the log, but through a
TEXT PROTOCOL: it writes `TOOL: read 400 450` (etc.) as ordinary text, we parse
that, run the read ourselves, and feed the result back as the next turn. The
model keeps the same agentic read/grep investigation loop, but every request and
response is plain text — no function-calling surface for the provider to mangle.

The investigation tools are a fixed, read-only set (read/grep/tail/head/wc) that
operate ONLY on the run's own log file. There is no arbitrary code execution, so
no container sandbox is required for the analyzer itself.

The public surface (`analyze(log_path, action_num, retry_nudge) -> Optional[str]`
and the prompt-override kwargs) mirrors OpenCodeAgent so callers can swap the two
without other changes. The returned text still ends in an `[ACTIONS]` block that
the existing plan parsers consume.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Output caps so a single tool result can't blow up the context window.
_READ_MAX_LINES = 400
_READ_MAX_CHARS = 12000
_GREP_MAX_HITS = 100
_GREP_LINE_CHARS = 300
_TAIL_HEAD_MAX = 300


_TOOL_PROTOCOL = """\

HOW YOU INVESTIGATE THE LOG (read carefully)
--------------------------------------------
You do NOT have function/tool calling. Instead you investigate the log file by
writing ONE tool command as plain text, then I run it and reply with the result.
Use as many turns as you need (budget: {max_iters} tool turns), then give your
final answer.

To run a tool, end your message with a single line of exactly this form:

  TOOL: wc                      -> total line count + size of the log
  TOOL: tail <n>                -> the last <n> lines (start here to see the latest state)
  TOOL: head <n>                -> the first <n> lines
  TOOL: read <start> <end>      -> lines <start>..<end> (1-indexed, inclusive)
  TOOL: grep <pattern>          -> every line matching <pattern> (regex), with line numbers

Rules:
- Exactly ONE `TOOL:` line per message, and it must be the LAST line.
- Do NOT include the file path in the command (there is only one file). Just the
  verb and its arguments, e.g. `TOOL: grep ACTION TAKEN` or `TOOL: read 40 60`.
- Do not invent other tools. Only wc/tail/head/read/grep, on this one log file.
- A good loop: `wc` to size it, `tail` to see the latest observation and the
  currently-available actions, `grep` for the markers that show history
  (e.g. action-taken markers), `read` around interesting line ranges.

When you have investigated enough, STOP requesting tools and instead write your
FINAL answer: the required briefing/plan sections AND the closing [ACTIONS] block.
Your message is treated as final as soon as it contains an [ACTIONS] block, so do
NOT mention [ACTIONS] until you are actually done.
"""


def _final_ready(text: str) -> bool:
    """True once the model has emitted a real [ACTIONS] block (marker + JSON)."""
    if "[ACTIONS]" not in text:
        return False
    tail = text.rsplit("[ACTIONS]", 1)[1]
    return "{" in tail


def _extract_tool(text: str) -> Optional[tuple[str, str]]:
    """Return (verb, arg) for the last `TOOL:` request in the text, else None."""
    found = None
    for line in text.splitlines():
        # Drop markdown emphasis/code noise so e.g. "**TOOL:** grep x" still parses.
        s = line.replace("*", "").replace("`", "").strip()
        m = re.match(r"(?i)^TOOL:\s*(\w+)\s*(.*)$", s)
        if m:
            found = (m.group(1).lower(), m.group(2).strip())
    return found


class TextToolAgent:
    """ReAct text-protocol analyzer. Drop-in for OpenCodeAgent.analyze()."""

    def __init__(
        self,
        model: str = "openrouter/google/gemini-2.5-flash",
        plan_size: int = 6,
        *,
        initial_prompt: Optional[str] = None,
        resume_prompt: Optional[str] = None,
        actions_addendum: Optional[str] = None,
        python_addendum: Optional[str] = None,  # accepted for parity; not used (no bash here)
        max_iters: int = 8,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **_ignored,  # swallow OpenCodeAgent-only kwargs (restrict_tools, resume_session, ...)
    ) -> None:
        # Import the ARC default templates lazily so this module doesn't hard-depend
        # on them; callers (e.g. the autumn driver) usually override all of these.
        from rgb_agent.agent import prompts as _p

        self._model = model
        self._plan_size = plan_size
        self._initial_prompt = initial_prompt or _p.INITIAL_PROMPT
        self._resume_prompt = resume_prompt or _p.RESUME_PROMPT
        self._actions_addendum = actions_addendum or _p.ACTIONS_ADDENDUM
        self._max_iters = max_iters
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Remember which logs we've already analyzed once (for first-vs-resume framing).
        self._seen: set[str] = set()
        log.info("TextToolAgent (text-protocol, no native tools) model=%s", model)

    # ---- the public interface (mirrors OpenCodeAgent) ---------------------
    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        log_path = Path(log_path)
        if not log_path.exists():
            return None
        key = str(log_path)
        is_first = key not in self._seen
        self._seen.add(key)

        base = (self._resume_prompt if (not is_first and self._resume_prompt) else self._initial_prompt)
        system = (
            base.format(log_path=str(log_path))
            + _TOOL_PROTOCOL.format(max_iters=self._max_iters)
            + self._actions_addendum.format(plan_size=self._plan_size)
        )
        if retry_nudge:
            system += f"\n\nIMPORTANT: {retry_nudge}\n"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"Begin. The log is at {log_path} (it is the only file you can inspect, "
                f"via the TOOL protocol above). Investigate the latest state, then give "
                f"your final briefing, [PLAN], and [ACTIONS]."},
        ]

        transcript: list[str] = [f"# text-protocol analyzer | model={self._model} | action_num={action_num}\n"]
        final_text: Optional[str] = None
        try:
            for it in range(self._max_iters + 1):
                last_turn = it == self._max_iters
                if last_turn:
                    messages.append({"role": "user", "content":
                        "Investigation budget exhausted. Do NOT request another tool. "
                        "Output your final briefing, [PLAN], and the [ACTIONS] JSON now."})
                reply = self._complete(messages)
                if reply is None:
                    transcript.append("\n[provider returned no text]\n")
                    break
                messages.append({"role": "assistant", "content": reply})
                transcript.append(f"\n===== ASSISTANT (turn {it}) =====\n{reply}\n")

                if _final_ready(reply):
                    final_text = reply
                    break

                tool = None if last_turn else _extract_tool(reply)
                if tool is None:
                    # No tool and no final answer — nudge toward a decision.
                    messages.append({"role": "user", "content":
                        "That message had neither a `TOOL:` request nor an [ACTIONS] block. "
                        "Either issue one TOOL command (last line) or give your final answer "
                        "ending with [ACTIONS]."})
                    transcript.append("\n[no tool / no actions -> nudge]\n")
                    continue

                verb, arg = tool
                result = self._run_tool(log_path, verb, arg)
                messages.append({"role": "user", "content": f"TOOL RESULT ({verb} {arg}):\n{result}"})
                transcript.append(f"\n----- TOOL {verb} {arg} -----\n{result[:2000]}\n")
        except Exception as e:  # provider/transport error — let the caller retry
            log.warning("text-protocol analyze failed: %s", e)
            transcript.append(f"\n[exception: {e}]\n")

        # Persist the full reasoning transcript next to the log for inspection.
        try:
            (log_path.parent / (log_path.stem + "_analyzer.txt")).open("a", encoding="utf-8").write(
                "".join(transcript) + ("\n" if final_text else "\n[no final ACTIONS this call]\n") + "\n")
        except Exception:
            pass
        return final_text

    # ---- LLM call (no tools => no MALFORMED_FUNCTION_CALL) ----------------
    def _complete(self, messages: list[dict]) -> Optional[str]:
        import litellm
        resp = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return None

    # ---- the read-only text tools ----------------------------------------
    def _run_tool(self, log_path: Path, verb: str, arg: str) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"(could not read log: {e})"
        lines = text.splitlines()
        n = len(lines)

        def fmt(rng) -> str:
            out = []
            total = 0
            for i in rng:
                if i < 1 or i > n:
                    continue
                ln = lines[i - 1]
                piece = f"{i}: {ln}"
                total += len(piece) + 1
                if total > _READ_MAX_CHARS:
                    out.append(f"... (truncated at {_READ_MAX_CHARS} chars)")
                    break
                out.append(piece)
            return "\n".join(out) if out else "(no lines in range)"

        if verb == "wc":
            return f"{n} lines, {len(text)} chars"
        if verb in ("tail", "head"):
            try:
                k = int(arg.split()[0]) if arg.split() else 40
            except ValueError:
                k = 40
            k = max(1, min(k, _TAIL_HEAD_MAX))
            rng = range(n - k + 1, n + 1) if verb == "tail" else range(1, k + 1)
            return fmt(rng)
        if verb == "read":
            nums = re.findall(r"-?\d+", arg)
            if len(nums) < 2:
                return "usage: read <start> <end>"
            a, b = int(nums[0]), int(nums[1])
            if b < a:
                a, b = b, a
            b = min(b, a + _READ_MAX_LINES - 1)
            return fmt(range(a, b + 1))
        if verb == "grep":
            if not arg:
                return "usage: grep <pattern>"
            try:
                pat = re.compile(arg)
            except re.error:
                pat = re.compile(re.escape(arg))
            hits = []
            for i, ln in enumerate(lines, 1):
                if pat.search(ln):
                    hits.append(f"{i}: {ln[:_GREP_LINE_CHARS]}")
            if not hits:
                return "(no matches)"
            shown = hits[-_GREP_MAX_HITS:]
            head = f"({len(hits)} matches; showing last {len(shown)})\n" if len(hits) > len(shown) else ""
            return head + "\n".join(shown)
        return f"(unknown tool '{verb}'; use wc/tail/head/read/grep)"
