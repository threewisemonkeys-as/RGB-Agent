"""OpenCodeAgent: runs OpenCode in a sandboxed Docker container to produce action plans."""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

from rgb_agent.agent.prompts import (
    INITIAL_PROMPT,
    RESUME_PROMPT,
    ACTIONS_ADDENDUM,
    PYTHON_ADDENDUM,
)

log = logging.getLogger(__name__)

_DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "rgb-agent/opencode-sandbox:latest")


def _docker_image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


class _EventStreamParser:
    """Parses nd-JSON events from OpenCode and writes to an analyzer log."""

    def __init__(self, f: IO[str]):
        self._f = f
        self.accumulated_text = ""
        self.session_id: str | None = None

    def _write(self, label: str, content: str) -> None:
        if content:
            self._f.write(f"[{label}]\n{content}\n\n")
            self._f.flush()

    def _write_tool(self, name: str, state: dict) -> None:
        status = state.get("status", "?")
        if status in ("running", "completed", "done"):
            input_data = state.get("input", {})
            input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
            self._write(f"TOOL CALL: {name}", input_str)
        if status in ("completed", "done"):
            output = state.get("output", state.get("result", ""))
            is_error = state.get("is_error", False) or state.get("error", False)
            label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
            self._write(label, str(output)[:4000])

    def handle(self, event: dict) -> None:
        etype = event.get("type")
        log.debug("event type=%s", etype)

        if etype == "step_start":
            sid = event.get("sessionID")
            if sid and not self.session_id:
                self.session_id = sid

        elif etype == "text":
            text = event.get("part", {}).get("text", "")
            if text:
                self.accumulated_text += text
                self._write("ASSISTANT", text)

        elif etype == "tool_use":
            part = event.get("part", {})
            self._write_tool(part.get("tool", "?"), part.get("state", {}))

        elif etype == "message.part.updated":
            part = event.get("part", {})
            ptype = part.get("type")
            if ptype in ("thinking", "reasoning"):
                self._write("THINKING", part.get("text", ""))
            elif ptype == "tool":
                name = part.get("name", "?")
                pstate = part.get("state", "?")
                if pstate == "running":
                    input_data = part.get("input", {})
                    input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
                    self._write(f"TOOL CALL: {name}", input_str)
                elif pstate in ("completed", "done"):
                    result = part.get("result", part.get("output", ""))
                    text = result if isinstance(result, str) else str(result)
                    is_error = part.get("is_error", False) or part.get("error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "error":
            err = event.get("error", {})
            name = err.get("name", "UnknownError")
            msg = err.get("data", {}).get("message", str(err))
            self._write(f"ERROR: {name}", msg)
            log.error("API error: %s: %s", name, msg)
            if "overflow" in name.lower() or "too long" in msg.lower():
                self.session_id = None

        elif etype == "step_finish":
            cost = event.get("part", {}).get("cost")
            self._write("RESULT", f"cost=${cost}")

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    self._write("THINKING", block.get("thinking", ""))
                elif btype == "text":
                    text = block["text"]
                    self.accumulated_text += text
                    self._write("ASSISTANT", text)
                elif btype == "tool_use":
                    self._write(f"TOOL CALL: {block['name']}", json.dumps(block.get("input", {}), indent=2))

        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = str(content)
                    is_error = block.get("is_error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "result":
            result_text = event.get("result", "").strip()
            if result_text and not self.accumulated_text.strip():
                self.accumulated_text = result_text
            cost = event.get("total_cost_usd")
            self._write("RESULT", f"cost=${cost}")

        else:
            self._f.write(f"[RAW:{etype}] {json.dumps(event)[:500]}\n")
            self._f.flush()


class _ContainerPool:
    """Manages persistent Docker containers running `opencode serve`."""

    def __init__(self, config_path: Path, permission: dict, docker_image: str, sandbox_prefix: str):
        self._config_path = config_path
        self._permission = permission
        self._image = docker_image
        self._prefix = sandbox_prefix
        self._containers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int, str]:
        with self._lock:
            if key in self._containers:
                info = self._containers[key]
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", info["name"]],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and "true" in check.stdout.lower():
                    return info["name"], info["port"], info["sandbox_dir"]
                log.warning("server container %s died, recreating", info["name"])
                subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                del self._containers[key]

            return self._create(key)

    def _create(self, key: str) -> tuple[str, int, str]:
        sandbox = tempfile.mkdtemp(prefix=self._prefix)
        os.chmod(sandbox, 0o777)
        name = f"oc_{uuid.uuid4().hex[:12]}"
        port = 4096

        shutil.copy2(self._config_path, Path(sandbox) / "opencode.json")

        env_flags: list[str] = []
        for key_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            val = os.environ.get(key_name)
            if val:
                env_flags.extend(["-e", f"{key_name}={val}"])

        # Rootless single-UID (no subuid range) compatibility: drop `--user 1000:1000`
        # and the uid=/gid= tmpfs options (unmappable without a subuid range), run as
        # root-in-userns (= host uid), and give that root a writable HOME on tmpfs.
        # The /home/opencode tmpfs keeps nosuid but allows exec (bun needs it).
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--memory=4g", "--cpus=2",
            "--pids-limit=128",
            "--shm-size=8m",
            "-e", "HOME=/home/opencode",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", "/home/opencode:rw,nosuid,size=128m",
            "-v", f"{os.path.realpath(sandbox)}:/workspace:rw",
            "-e", "OPENCODE_CONFIG=/workspace/opencode.json",
            *(["-e", f"OPENCODE_PERMISSION={json.dumps(self._permission)}"] if self._permission else []),
            *env_flags,
            self._image,
            "serve", "--port", str(port), "--hostname", "0.0.0.0",
        ]

        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        for _ in range(15):
            time.sleep(1)
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=15,
            )
            if "listening" in logs.stdout or "listening" in logs.stderr:
                break
        else:
            log.warning("server %s may not be ready (timeout)", name)

        self._containers[key] = {"name": name, "port": port, "sandbox_dir": sandbox}
        log.info("container ready: %s", name)
        return name, port, sandbox

    def cleanup(self) -> None:
        with self._lock:
            for info in self._containers.values():
                try:
                    log.info("stopping container: %s", info["name"])
                    subprocess.run(["docker", "stop", "-t", "3", info["name"]], capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                except Exception as e:
                    log.warning("failed to cleanup container %s: %s", info["name"], e)
                if info.get("sandbox_dir"):
                    shutil.rmtree(info["sandbox_dir"], ignore_errors=True)
            self._containers.clear()


class OpenCodeAgent:
    """Runs OpenCode in a sandboxed Docker container to analyze game logs and produce action plans."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-6",
        plan_size: int = 5,
        timeout: Optional[int] = None,
        resume_session: bool = True,
        restrict_tools: bool = True,
        initial_prompt: Optional[str] = None,
        resume_prompt: Optional[str] = None,
        actions_addendum: Optional[str] = None,
        python_addendum: Optional[str] = None,
    ) -> None:
        if not shutil.which("docker"):
            raise FileNotFoundError("'docker' CLI not found. Install Docker Desktop to use the analyzer.")
        if not _docker_image_exists(_DOCKER_IMAGE):
            raise FileNotFoundError(
                f"Docker image '{_DOCKER_IMAGE}' not found. Build with:\n"
                f"  cd docker/opencode-sandbox && bash build.sh"
            )
        log.info("using Docker sandbox: %s", _DOCKER_IMAGE)

        self._oc_model = model if "/" in model else f"anthropic/{model}"
        self._plan_size = plan_size
        # Prompt templates default to the ARC-AGI-3 set; callers (e.g. an
        # AutumnBench driver) may override them to swap in a different action
        # vocabulary. The ARC path passes nothing and is unaffected.
        self._initial_prompt = initial_prompt or INITIAL_PROMPT
        self._resume_prompt = resume_prompt or RESUME_PROMPT
        self._actions_addendum = actions_addendum or ACTIONS_ADDENDUM
        self._python_addendum = python_addendum if python_addendum is not None else PYTHON_ADDENDUM
        self._timeout = timeout
        self._resume_session = resume_session
        # How many extra times to re-issue the opencode call when a turn ends in a
        # provider error (finish=unknown/error or nonzero rc) with no usable output.
        # These transient errors are common with gemini-2.5-flash via openrouter.
        self._error_retries = 2

        oc_provider = self._oc_model.split("/")[0]

        # NOTE: an opencode `permission` ruleset with any "deny"/"ask" entry makes some
        # models (e.g. gemini-2.5-flash via openrouter) abort the turn with
        # finish="unknown" and empty output the moment they call a restricted tool —
        # they don't gracefully handle the denial the way Claude does. With
        # restrict_tools=False we omit the opencode permission layer entirely and rely
        # on the OS-level container sandbox (read-only rootfs, --cap-drop=ALL,
        # no-new-privileges, ephemeral tmpfs/workspace) to constrain the analyzer.
        permission: dict = {
            "*": "deny",
            "read": "allow",
            "grep": "allow",
            "bash": {
                "*": "deny",
                "python3 *": "allow",
                "python *": "allow",
            },
            "external_directory": "deny",
            "doom_loop": "allow",
            "question": "deny",
            "edit": "deny",
            "write": "deny",
            "patch": "deny",
            "glob": "deny",
            "list": "deny",
            "lsp": "deny",
            "skill": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "todowrite": "deny",
            "todoread": "deny",
        } if restrict_tools else {}

        config = {
            "model": self._oc_model,
            "provider": {oc_provider: {}},
            "agent": {"build": {"steps": 50}},
        }
        if permission:
            config["permission"] = permission

        config_dir = tempfile.mkdtemp(prefix="opencode_analyzer_")
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2))
        atexit.register(shutil.rmtree, config_dir, True)

        self._pool = _ContainerPool(config_path, permission, _DOCKER_IMAGE, f"oc_sandbox_{uuid.uuid4().hex[:8]}_")
        atexit.register(self._pool.cleanup)

        self._session_ids: dict[str, str] = {}
        self._session_lock = threading.Lock()

    def _build_prompt(self, log_name: str, is_first: bool) -> str:
        if self._resume_session and not is_first:
            prompt = self._resume_prompt.format(log_path=log_name)
        else:
            prompt = self._initial_prompt.format(log_path=log_name)
        prompt += self._python_addendum.format(log_path=log_name)
        prompt += self._actions_addendum.format(plan_size=self._plan_size)
        return prompt

    def _try_recover_text(self, container_name: str, sid: str, sandbox_dir: str,
                          max_wait: float = 30.0, poll: float = 1.0) -> str:
        """Recover the assistant's response from the persisted session via `opencode export`.

        POLLS the export because opencode's `run` client often exits on a premature idle
        event while the server is still streaming — the server keeps generating and
        persists the full reply (incl. the trailing [ACTIONS]) shortly after, even though
        the client already disconnected (verified: client-disconnect does not abort the
        server). We poll until [ACTIONS] appears, or the newest assistant turn finishes
        without it (genuine miss → let the caller's retry-nudge handle it), or timeout.
        """
        export_path = Path(sandbox_dir) / "_export.json"
        deadline = time.monotonic() + max_wait
        best = ""
        while True:
            data = {}
            try:
                subprocess.run(
                    ["docker", "exec", container_name, "sh", "-c",
                     f"opencode export {sid} > /workspace/_export.json 2>/dev/null"],
                    capture_output=True, text=True, timeout=30,
                )
                if export_path.exists():
                    data = json.loads(export_path.read_text())
            except Exception as e:
                log.debug("export recovery poll failed: %s", e)

            assistants = [m for m in data.get("messages", []) if m.get("info", {}).get("role") == "assistant"]
            actions_text = ""
            newest_text = ""
            newest_finish = None
            for i, msg in enumerate(reversed(assistants)):
                text = "".join(
                    p.get("text", "") for p in msg.get("parts", []) if p.get("type") == "text"
                ).strip()
                if "[ACTIONS]" in text and not actions_text:
                    actions_text = text
                if i == 0:
                    newest_text = text
                    newest_finish = msg.get("info", {}).get("finish")

            if actions_text:
                return actions_text, (newest_finish or "stop")
            if newest_text:
                best = newest_text
            # "tool-calls"/unset finish mean the turn is still going; any other finish
            # (incl. "unknown"/"error" provider failures) means nothing more is coming.
            terminal = newest_finish in ("stop", "length", "content-filter", "error", "unknown")
            if terminal or time.monotonic() >= deadline:
                return best, newest_finish
            time.sleep(poll)

    def _latest_session_id(self, container_name: str) -> Optional[str]:
        """Read the most recent session id straight from the server's opencode.db.

        Needed because opencode's `run` CLI mirrors LIVE server events to stdout and
        exits on the idle event without reading persisted state; over the --attach SSE
        transport those events frequently arrive after teardown, so we get zero events
        (and thus no streamed session id) even though the turn was generated and stored.
        Falling back to the DB lets export-based recovery run regardless.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 "import sqlite3,glob;"
                 "p=glob.glob('/home/opencode/.local/share/opencode/opencode.db')"
                 "+glob.glob('/root/.local/share/opencode/opencode.db');"
                 "c=sqlite3.connect(p[0]);"
                 "print(list(c.execute('select id from session order by rowid desc limit 1'))[0][0])"],
                capture_output=True, text=True, timeout=15,
            )
            sid = result.stdout.strip()
            return sid or None
        except Exception as e:
            log.debug("latest-session lookup failed: %s", e)
            return None

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        """Analyze the game log and return the agent's response text, or None on failure.

        Re-issues the underlying opencode call on transient provider errors
        (finish=unknown/error or nonzero rc with no usable output) with backoff,
        which are common with gemini-2.5-flash via openrouter.
        """
        attempts = self._error_retries + 1
        for i in range(attempts):
            hint, provider_error = self._analyze_once(log_path, action_num, retry_nudge)
            if hint or not provider_error or i == attempts - 1:
                return hint
            backoff = min(3.0 * (2 ** i), 20.0)
            log.warning("action=%d provider error; reissuing opencode in %.0fs (%d/%d)",
                        action_num, backoff, i + 1, self._error_retries)
            time.sleep(backoff)
        return None

    def _analyze_once(self, log_path: Path, action_num: int,
                      retry_nudge: str = "") -> tuple[Optional[str], bool]:
        """One opencode call + recovery. Returns (hint, provider_error)."""
        if not log_path.exists():
            return None, False

        provider_error = False
        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")
        path_key = str(log_path)

        is_first = True
        current_sid = None
        if self._resume_session:
            with self._session_lock:
                if path_key in self._session_ids:
                    current_sid = self._session_ids[path_key]
                    is_first = False

        container_name, server_port, sandbox_dir = self._pool.get(path_key)
        sandbox = Path(sandbox_dir)

        try:
            shutil.copy2(log_path, sandbox / log_path.name)

            prompt = self._build_prompt(log_path.name, is_first)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            oc_args = ["run", "--attach", f"http://127.0.0.1:{server_port}"]
            if self._resume_session and not is_first and current_sid:
                oc_args.extend(["--session", current_sid, "--continue"])
            oc_args.extend(["--model", self._oc_model])
            oc_args.extend(["--format", "json", "--dir", "/workspace"])
            oc_args.append(prompt)

            cmd = ["docker", "exec", container_name, "opencode", *oc_args]
            log.info("exec %s model=%s%s", container_name, self._oc_model,
                     f" session={current_sid}" if current_sid else "")

            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            stderr_lines: list[str] = []
            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip("\n"))
                    log.debug("STDERR: %s", line[:300].rstrip())

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | opencode ---\n")
                if is_first or not self._resume_session:
                    f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                parser = _EventStreamParser(f)
                deadline = time.monotonic() + self._timeout if self._timeout is not None else None

                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if deadline is not None and time.monotonic() > deadline:
                        proc.kill()
                        f.write("[TIMEOUT]\n")
                        log.warning("timed out at action %d", action_num)
                        return None, False

                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parser.handle(json.loads(line))
                    except json.JSONDecodeError:
                        f.write(f"[RAW] {line}\n")
                        f.flush()

                proc.wait()
                stderr_thread.join(timeout=5)
                if stderr_lines:
                    f.write(f"\n--- STDERR ---\n{''.join(l + chr(10) for l in stderr_lines)}")
                    f.flush()

                needs_recovery = (
                    not parser.accumulated_text.strip()
                    or "[ACTIONS]" not in parser.accumulated_text
                )
                if needs_recovery:
                    # Resolve a session id even when the stream delivered no events:
                    # prefer the streamed id, then the resumed id we already hold, then
                    # the latest session straight from the server DB.
                    recovery_sid = parser.session_id or current_sid
                    if not recovery_sid:
                        recovery_sid = self._latest_session_id(container_name)
                    if recovery_sid:
                        recovered, finish = self._try_recover_text(container_name, recovery_sid, sandbox_dir)
                        if finish in ("unknown", "error"):
                            provider_error = True
                        if recovered:
                            parser.accumulated_text = recovered
                            # The live event stream drops text/thinking events, so the
                            # model's briefing + reasoning never reached the log. Now that
                            # we've recovered the full assistant text from the session DB,
                            # write it so the analyzer log shows the actual reasoning.
                            parser._write("ASSISTANT (recovered)", recovered)
                            # Adopt the id so session resume keeps working and the
                            # context-overflow guard below doesn't wrongly clear it.
                            if not parser.session_id:
                                parser.session_id = recovery_sid
                            log.info("recovered %d chars via session export (sid=%s, finish=%s)",
                                     len(recovered), recovery_sid, finish)

                if self._resume_session and parser.session_id is None and not is_first:
                    log.warning("context overflow — clearing session for %s", path_key)
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)

                f.flush()

            hint = parser.accumulated_text.strip() or None

            if proc.returncode != 0 or not hint:
                if proc.returncode not in (0, None):
                    provider_error = True
                log.warning("action=%d failed: rc=%d, hint_len=%d, provider_error=%s",
                            action_num, proc.returncode, len(hint) if hint else 0, provider_error)
                if self._resume_session:
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)
                return None, provider_error

            if self._resume_session and parser.session_id:
                with self._session_lock:
                    self._session_ids[path_key] = parser.session_id

            log.info("action=%d OK (%d chars)", action_num, len(hint))
            return hint, False

        except Exception as e:
            log.error("unexpected error: %s", e, exc_info=True)
            return None, False
