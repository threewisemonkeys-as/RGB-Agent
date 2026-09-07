"""The codex backend, wired to deepseek-v4-flash on the planner arms' own routing.

Upstream's `CodexAgent` with three measured corrections, each of which fails silently if
you get it wrong -- the run completes, the numbers look fine, and the arm is not the one
you meant to run:

* **reasoning.** Leaving `model_reasoning_effort` unset does NOT inherit the provider's
  default thinking; it turns reasoning OFF. Measured: unset -> 0 reasoning tokens,
  `"medium"` -> reasoning is ~80% of output. The planner arms run with provider-default
  thinking on, so an unset effort would race a non-thinking agent against thinking
  baselines. `REASONING_EFFORT` is explicit and recorded.

* **the context window.** `-c model_context_window` is accepted and ignored. Codex falls
  back to `gpt-5.6-sol`'s metadata -- 272,000 x 95% = an effective 258,400 -- against a
  real window of 1,000,000, so a long session auto-compacts and discards its own
  analysis a quarter of the way in. The fix is `model_catalog_json`, a file path with a
  ~35-field schema; `build_catalog` clones a real entry rather than hand-rolling one,
  which also preserves codex's own `base_instructions`.

* **which entry gets cloned.** The clone carries a tool surface as well as a context
  window, and `tool_mode="code_mode_only"` -- what every gpt-5.6 entry sets -- offers the
  model exactly one way to act: a CUSTOM tool whose argument is raw JavaScript, run in a
  V8 isolate. DeepSeek calls it with JSON (`{"code": "..."}`) instead of freeform text,
  codex cannot parse the call, and the tool silently does nothing: no output item, no
  error, the turn just ends having accomplished nothing. Measured on a one-line "write
  this file" task, cloning `gpt-5.6-sol` succeeded 1 time in 5 and ran 0 shell commands
  at 73k-107k input tokens a call; cloning `gpt-5.5`, whose `tool_mode` is unset and
  which therefore offers ordinary `exec_command` function calls, succeeded 3 of 3 at 27k.
  The 46k-token difference is the code-mode preamble plus a sub-agent namespace this arm
  has no use for -- charged on every call, against the context the corpus needs.

* **the provider.** Codex cannot send OpenRouter's `provider` body field, and nothing it
  CAN send substitutes: a model-slug suffix, a `?provider=` query param and an
  `X-OpenRouter-Provider` header were each measured returning 200 and routing somewhere
  else entirely. Unpinned, consecutive calls landed on four different hosts and the
  catalogue includes an fp4 one. So `base_url` points at the injecting proxy, and if it
  is not there the run refuses to start rather than quietly measuring random hardware.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from prolong_agent.agent.codex_events import CodexEventParser

log = logging.getLogger(__name__)

MODEL = "deepseek/deepseek-v4-flash"
REASONING_EFFORT = "medium"
CONTEXT_WINDOW = 1_000_000          # min across parasail/novita/alibaba fp8, not the
MAX_OUTPUT_TOKENS = 393_216         # 1,048,576 headline: the pin's smallest host wins
AUTO_COMPACT_LIMIT = 900_000
CLONE_FROM = "gpt-5.5"


# What a turn's transcript keeps. Codex\'s own log is prose written for a human tail-ing
# a file; these are the same events kept as data, so the replay page can show a turn as
# what the agent THOUGHT and what it DID rather than as a wall of text.
MAX_REASONING = 20_000          # per item; deepseek\'s medium-effort blocks run ~2-6k
MAX_COMMAND = 2_000
MAX_OUTPUT = 8_000              # a `cat` of a 60-transition drive file is bigger than
                                # anything worth showing, and there are hundreds of them

# Errors that mean the ACCOUNT is finished, not that this turn went badly. Codex retries a
# 403 five times and then reports an ordinary failed turn, so without this the runner
# treats a dead key exactly like a model that would not answer: it retries, records
# `budget-exhausted` against a session that never reached the model, moves to the next
# problem and does it again. Measured: the key hit its spend limit 12.9h into the 86 and
# the run recorded one such row before it was stopped by hand -- and because `launch.py`
# skips any task_uid already in `rows.jsonl`, that row would have survived the resume and
# gone into the paper as a real miss.
FATAL_PATTERNS = (
    "key limit exceeded", "insufficient_quota", "insufficient credits",
    "quota exceeded", "payment required", "billing",
    "invalid api key", "no auth credentials", "unauthorized",
    "403 forbidden", "401 unauthorized",
)


class CredentialsExhausted(RuntimeError):
    """The key is out of money or invalid. Every remaining problem would fail the same
    way, so the run stops instead of writing 55 rows nobody can use."""


class _UsageParser(CodexEventParser):
    """Upstream\'s parser plus two things it drops on the floor.

    * **the reasoning tokens.** Upstream reports a hard 0. They are the whole point of
      the effort setting, so the arm has to be able to show they were actually spent
      rather than asserting that they were.

    * **the turn transcript.** Upstream writes reasoning, commands and file edits to a
      text log and keeps only a list of command STRINGS in memory -- no outputs, no exit
      codes, no reasoning at all (`item.completed` for a `reasoning` item is handled by
      marking the clock and discarding `text`). That is unrecoverable after the run: the
      agent arm\'s whole claim is that it read the corpus and worked something out, and a
      transcript that cannot show the reading or the working cannot support it. So the
      events are kept structurally, capped, in the order they arrived.
    """

    def __init__(self, output):
        super().__init__(output)
        self.last_tokens_reasoning = 0
        self.events: list[dict] = []
        self._open: dict | None = None
        self.fatal_error: str | None = None

    def handle(self, event):
        usage = (event or {}).get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        reasoning = (usage.get("reasoning_output_tokens")
                     or details.get("reasoning_tokens"))
        if reasoning is not None:
            self.last_tokens_reasoning = int(reasoning)
        try:
            self._capture(event or {})
        except Exception:                          # noqa: BLE001 - never lose a turn to
            log.debug("event capture failed", exc_info=True)   # the transcript
        return super().handle(event)

    # The pairing is `item.started` -> `item.completed`; a command\'s text arrives on the
    # first and its output on the second, so the record is opened early and filled late.
    def _capture(self, event: dict) -> None:
        kind = event.get("type", "")
        item = event.get("item") or {}
        itype = item.get("type", "")

        if kind == "item.started" and itype == "command_execution":
            self._open = {"kind": "command",
                          "command": str(item.get("command") or "")[:MAX_COMMAND]}
            self.events.append(self._open)

        elif kind == "item.completed":
            if itype == "reasoning":
                text = _text_of(item)
                if text:
                    self.events.append({"kind": "reasoning",
                                        "text": text[:MAX_REASONING]})
            elif itype == "agent_message":
                text = str(item.get("text") or "")
                if text:
                    self.events.append({"kind": "message", "text": text[:MAX_REASONING]})
            elif itype == "command_execution":
                out = str(item.get("aggregated_output") or item.get("output") or "")
                rec = self._open if self._open is not None else {
                    "kind": "command",
                    "command": str(item.get("command") or "")[:MAX_COMMAND]}
                if rec is not self._open or rec not in self.events:
                    self.events.append(rec)
                rec["exit_code"] = item.get("exit_code")
                rec["output"] = out[:MAX_OUTPUT]
                rec["truncated"] = len(out) > MAX_OUTPUT
                self._open = None
            elif itype == "file_change":
                changes = item.get("changes") or []
                self.events.append({
                    "kind": "file_change",
                    "changes": [{"path": str(c.get("path") or c.get("file") or ""),
                                 "type": str(c.get("type") or c.get("kind") or "edit")}
                                for c in changes if isinstance(c, dict)][:50],
                    "n": len(changes),
                })

        elif kind in ("error", "turn.failed"):
            message = event.get("message") or event.get("error") or ""
            if isinstance(message, dict):
                message = message.get("message") or str(message)
            message = str(message)
            self.events.append({"kind": "error", "text": message[:MAX_OUTPUT]})
            low = message.lower()
            if any(pat in low for pat in FATAL_PATTERNS):
                self.fatal_error = message[:500]


def _interleave(events: list[dict], reasoning: list[list[str]]) -> list[dict]:
    """Put each upstream call's reasoning in front of the tool call it asked for.

    Codex does not tell us which model call produced which tool call, so this is an
    alignment BY ORDER, not a hard link: within one turn the calls are strictly
    sequential -- call k reasons, requests a tool, the tool runs, call k+1 follows -- so
    the k-th reasoning block belongs in front of the k-th recorded action. Any surplus
    (the final call, which answers instead of acting) is appended at the end, where it
    belongs. Getting this wrong misplaces a block by one action; it never invents one.
    """
    if not reasoning:
        return list(events)
    actionable = [i for i, e in enumerate(events)
                  if e.get("kind") in ("command", "file_change")]
    out: list[dict] = []
    used = 0
    for i, e in enumerate(events):
        if used < len(reasoning) and i in set(actionable[used:used + 1]):
            for block in reasoning[used]:
                out.append({"kind": "reasoning", "text": block[:MAX_REASONING]})
            used += 1
        out.append(e)
    for block in (b for call in reasoning[used:] for b in call):
        out.append({"kind": "reasoning", "text": block[:MAX_REASONING]})
    return out


def _text_of(item: dict) -> str:
    """A reasoning item\'s text, wherever this codex build put it.

    The shape has moved between releases (`text`, `summary` as a list of parts, `content`
    as typed blocks), and a reasoning trace that silently comes back empty looks exactly
    like a model that did not think -- which is the thing F12 exists to detect. So try
    every shape rather than the current one.
    """
    for key in ("text", "reasoning", "content", "summary"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            parts = []
            for entry in val:
                if isinstance(entry, str):
                    parts.append(entry)
                elif isinstance(entry, dict):
                    for k in ("text", "content", "summary"):
                        if isinstance(entry.get(k), str):
                            parts.append(entry[k])
                            break
            if parts:
                return "\n".join(parts)
    return ""


def build_catalog(out_path: Path, *, codex: str = "codex") -> Path:
    """Write a model catalog for `MODEL` by cloning a real entry from codex's own.

    Hand-rolling is not viable: the schema demands ~35 fields (`visibility`,
    `truncation_policy`, `supported_reasoning_levels` as structs, `shell_type` enums)
    and, critically, `base_instructions` -- 17,730 characters of codex's harness prompt.
    A catalog without it is rejected; a catalog with an empty one silently removes the
    agent's tool conventions.
    """
    raw = subprocess.run([codex, "debug", "models"], capture_output=True, text=True,
                         timeout=120, check=True).stdout
    models = json.loads(raw)["models"]
    source = next((m for m in models if m["slug"] == CLONE_FROM), None)
    if source is None:
        # Fall back on the PROPERTY, never on models[0]: the thing that matters about the
        # clone is that it is not code-mode, and a silent fallback to a code-mode entry
        # produces an agent that reasons, calls its one tool, is ignored, and stops.
        source = next((m for m in models if not m.get("tool_mode")), None)
        if source is None:
            raise RuntimeError(
                f"{CLONE_FROM!r} is gone from codex's catalog and every remaining entry "
                "is tool_mode=code_mode_only, which this model cannot drive. Re-measure "
                "before running the arm.")
        log.warning("%s is gone from codex's catalog; cloning %s instead",
                    CLONE_FROM, source["slug"])
    if source.get("tool_mode") == "code_mode_only":
        raise RuntimeError(
            f"{source['slug']} is tool_mode=code_mode_only: its only tool takes freeform "
            "JavaScript, which this model emits as JSON, so every tool call is dropped "
            "without an error. Measured 1/5 success against 3/3 for a non-code-mode "
            "clone.")

    entry = json.loads(json.dumps(source))
    entry.update({
        "slug": MODEL,
        "display_name": "DeepSeek v4 Flash",
        "description": "DeepSeek v4 Flash via the parity proxy",
        "context_window": CONTEXT_WINDOW,
        "max_context_window": CONTEXT_WINDOW,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "auto_compact_token_limit": AUTO_COMPACT_LIMIT,
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"models": [entry]}))
    return out_path


class AutumnCodexAgent:
    """One codex session per problem. `analyze` is one turn of it."""

    BACKEND_ID = "codex"

    def __init__(self, workspace: Path, *, catalog: Path, base_url: str,
                 api_key_env: str = "OPENROUTER_API_KEY", codex_home: Path | None = None,
                 model: str = MODEL, reasoning_effort: str = REASONING_EFFORT,
                 timeout: int = 1800, codex: str = "codex",
                 allow_unpinned: bool = False,
                 transcript: Path | None = None) -> None:
        self.workspace = Path(workspace)
        self.catalog = Path(catalog)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.codex = codex
        self.session_id: str | None = None
        self.calls = 0

        if not allow_unpinned and "openrouter.ai" in self.base_url:
            raise RuntimeError(
                "base_url points straight at OpenRouter, so the provider pin cannot be "
                "injected and each call routes to an arbitrary host (fp4 included). "
                "Point it at the parity proxy, or pass allow_unpinned=True and say so "
                "in the run manifest.")

        self.codex_home = Path(codex_home) if codex_home else self.workspace / ".codex"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.agent_log = self.workspace / "agent.txt"
        # The proxy's per-call reasoning log. A turn is bracketed by this file's size
        # before and after the codex process runs -- calls are strictly sequential
        # within a turn and the launcher runs one problem at a time, so the rows that
        # appear in between are exactly this turn's.
        self.transcript = Path(transcript) if transcript else None

    # ------------------------------------------------------------------ codex
    def _args(self, prompt: str, is_first: bool) -> list[str]:
        common = [
            "--json", "--skip-git-repo-check", "--ignore-user-config",
            "-o", "last_message.txt",
            "-m", self.model,
            "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c", f'model_catalog_json="{self.catalog}"',
            "-c", f'model_providers.parity={{name="parity",base_url="{self.base_url}",'
                  f'env_key="{self.api_key_env}",wire_api="responses"}}',
            "-c", 'model_provider="parity"',
        ]
        if not is_first and self.session_id:
            # `exec resume` rejects -C; the cwd is the workspace instead
            return ["exec", "resume", *common,
                    "--dangerously-bypass-approvals-and-sandbox",
                    self.session_id, prompt]
        return ["exec", *common, "-s", "danger-full-access", prompt]

    def analyze(self, log_path: Path, prompt: str, *, is_first: bool = False):
        """One turn. Returns `(actions_json_payload | None, meta)`.

        The payload is returned unparsed-into-actions on purpose: the runner has to tell
        `{"actions": []}` (a study round) apart from a missing or malformed file (a
        retry), and a normalised action list cannot express that difference.
        """
        is_first = is_first or self.session_id is None
        self._clear("actions.json", "last_message.txt")

        with open(self.agent_log, "a", encoding="utf-8") as handle:
            handle.write(f"\n--- call={self.calls} | "
                         f"{datetime.now().strftime('%H:%M:%S')} | codex ---\n"
                         f"[USER PROMPT]\n{prompt}\n\n")

        env = os.environ.copy()
        env["CODEX_HOME"] = str(self.codex_home)
        started = time.monotonic()
        meta = {"model": self.model, "reasoning_effort": self.reasoning_effort}
        mark = self._transcript_offset()

        try:
            with open(self.agent_log, "a", encoding="utf-8") as handle:
                parser = _UsageParser(handle)
                payload = self._pump([self.codex, *self._args(prompt, is_first)],
                                     env, parser, handle)
        except Exception as exc:                       # noqa: BLE001 - one turn, not the run
            log.error("codex call failed: %s", exc, exc_info=True)
            self.session_id = None
            return None, meta

        if parser.fatal_error:
            # Not this turn's problem, and not recoverable by retrying: stop the run
            # before it manufactures failures for every problem that is left.
            raise CredentialsExhausted(parser.fatal_error)

        self.calls += 1
        if parser.session_id:
            self.session_id = parser.session_id
        elif not is_first:
            log.warning("codex session dropped; the next call starts fresh and loses "
                        "everything this one worked out")
            self.session_id = None

        meta.update({
            "input_tokens": parser.last_tokens_input or 0,
            "cached_tokens": parser.last_tokens_cache_read or 0,
            "output_tokens": parser.last_tokens_output or 0,
            "reasoning_tokens": parser.last_tokens_reasoning or 0,
            "commands": list(parser.commands),
            # the turn as data: what it thought, what it ran, what came back
            "events": _interleave(parser.events, self._reasoning_since(mark)),
            "wall_s": round(time.monotonic() - started, 1),
            "session_id": self.session_id,
        })
        if parser.overflow_error:
            log.warning("codex overflow (%s): session reset", parser.overflow_error)
            self.session_id = None
        return payload, meta

    def _pump(self, cmd, env, parser, handle):
        proc = subprocess.Popen(cmd, cwd=self.workspace, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        stderr: list[str] = []
        drain = threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True)
        drain.start()
        deadline = time.monotonic() + self.timeout if self.timeout else None
        try:
            for line in proc.stdout:
                if deadline and time.monotonic() > deadline:
                    proc.kill()
                    handle.write("[TIMEOUT]\n")
                    log.warning("codex turn exceeded %ss", self.timeout)
                    self.session_id = None
                    return None
                line = line.strip()
                if not line:
                    continue
                try:
                    parser.handle(json.loads(line))
                except json.JSONDecodeError:
                    handle.write(f"[RAW] {line}\n")
            proc.wait()
        finally:
            drain.join(timeout=5)
        if proc.returncode not in (0, None):
            tail = " | ".join(x.strip() for x in stderr[-3:] if x.strip())
            log.warning("codex exited %s: %s", proc.returncode, tail)
        return self._read_actions()

    def _read_actions(self):
        path = self.workspace / "actions.json"
        if not path.exists():
            log.info("no actions.json written")
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            log.warning("actions.json is malformed: %s", exc)
            return None

    # ------------------------------------------------------------- the reasoning
    def _transcript_offset(self) -> int:
        if self.transcript is None:
            return 0
        try:
            return self.transcript.stat().st_size
        except OSError:
            return 0

    def _reasoning_since(self, offset: int) -> list[list[str]]:
        """This turn's reasoning blocks, one list per upstream call, in call order."""
        if self.transcript is None:
            return []
        try:
            with self.transcript.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read().decode("utf-8", "replace")
        except OSError:
            return []
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:      # a row still being written; the next turn
                continue                      # will not re-read it, which is the right
            blocks = [b for b in (row.get("reasoning") or []) if b.strip()]
            if blocks:                        # trade for never showing half a thought
                out.append(blocks)
        return out

    def _clear(self, *names: str) -> None:
        for name in names:
            path = self.workspace / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
