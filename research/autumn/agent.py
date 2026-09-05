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
  which also preserves codex's own 17k-character `base_instructions`.

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
CLONE_FROM = "gpt-5.6-sol"


class _UsageParser(CodexEventParser):
    """Upstream's parser plus the reasoning tokens, which it reports as a hard 0.

    They are the whole point of the effort setting, so the arm has to be able to show
    they were actually spent rather than asserting that they were.
    """

    def __init__(self, output):
        super().__init__(output)
        self.last_tokens_reasoning = 0

    def handle(self, event):
        usage = (event or {}).get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        reasoning = (usage.get("reasoning_output_tokens")
                     or details.get("reasoning_tokens"))
        if reasoning is not None:
            self.last_tokens_reasoning = int(reasoning)
        return super().handle(event)


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
    source = next((m for m in models if m["slug"] == CLONE_FROM), None) or models[0]

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
                 allow_unpinned: bool = False) -> None:
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

        try:
            with open(self.agent_log, "a", encoding="utf-8") as handle:
                parser = _UsageParser(handle)
                payload = self._pump([self.codex, *self._args(prompt, is_first)],
                                     env, parser, handle)
        except Exception as exc:                       # noqa: BLE001 - one turn, not the run
            log.error("codex call failed: %s", exc, exc_info=True)
            self.session_id = None
            return None, meta

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

    def _clear(self, *names: str) -> None:
        for name in names:
            path = self.workspace / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
