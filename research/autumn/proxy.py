#!/usr/bin/env python3
"""The parity proxy: the one place the agent arm's requests can be made to match the
planner arms' requests.

Codex builds its own request body and cannot reach into it. OpenRouter reads the provider
pin ONLY from that body -- a model-slug suffix, a `?provider=` query param and an
`X-OpenRouter-Provider` header were each measured returning 200 and routing somewhere else
(plan F15). Unpinned, four consecutive calls landed on Baidu, DeepInfra, SiliconFlow and
Phala, and this model's catalogue contains an fp4 host (`atlas-cloud/fp4`) beside the fp8
ones. So "same model as the planner arms" without the pin is not the same hardware, not
the same quantisation, and not a comparison.

This is a transparent passthrough to OpenRouter that rewrites exactly the JSON fields the
planner's `llm_call` sets and nothing else:

    provider = {"only": [...]}   the pin (eval_coverage_plan.py::llm_call)
    usage    = {"include": true} so cost and native token counts come back

and removes the sampling knobs the planner never sends, so neither arm gets a decoding
distribution the other didn't have.

Every call is audited. The proxy sniffs the `gen-...` id out of the response, asks
OpenRouter's `/generation` endpoint who actually served it, and appends the answer to
`parity.jsonl`. An arm that claims a provider pin should be able to show it held for all
N calls rather than assert it, and the failure this guards against is silent by
construction: a dropped pin returns 200 and plausible text.

    uv run python -m research.autumn.proxy --port 8788
    uv run python -m research.autumn.proxy --verify        # one real call, then report
    uv run python -m research.autumn.proxy --audit logs/agent/parity.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("parity")

UPSTREAM = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash"       # only used to look up provider display names

# The planner arms' default pin (offline_learning/launch/launch_planning_v2_online.py).
# `tests/test_parity_proxy.py` asserts these are still the same string, because the whole
# arm's comparability is this tuple.
PIN = ("parasail/fp8", "novita/fp8", "alibaba/fp8")

# The planner sends `model`, `messages`, `provider` and `usage` -- no decoding parameters
# at all, so the provider's own defaults apply. Codex sends some of these; dropping them
# is what makes the two arms' sampling identical rather than merely similar.
STRIP = ("temperature", "top_p", "top_k", "seed", "max_tokens", "max_output_tokens",
         "frequency_penalty", "presence_penalty")

GEN_ID_RE = re.compile(rb"\"(gen-[A-Za-z0-9_-]{8,})\"")
# Codex's own `--json` event stream carries commands and messages but NOT reasoning:
# measured on this build, `show_raw_agent_reasoning` and `model_reasoning_summary` change
# nothing and no `reasoning` item is ever emitted. The proxy is the only place the
# reasoning is visible, because it alone sees the raw SSE, where it arrives as
# `response.reasoning_text.done` frames carrying the completed block. The arm's claim is
# that the agent read a corpus and worked something out; throwing this away would leave
# the replay page unable to show any of the working.
REASONING_DONE = b"response.reasoning_text.done"
MAX_REASONING_PER_CALL = 200_000

HEAD_BYTES = 8192          # the id is in the first SSE frame (`response.created`)
TAIL_BYTES = 65536         # ...and the usage block sits at the end of the last one
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade",
              "content-encoding", "content-length", "host"}


def _pin_names(model: str = "") -> dict[str, str]:
    """Map OpenRouter's provider display names to endpoint tags, from OpenRouter.

    `/generation` reports `provider_name` ("Alibaba"); the pin is written in endpoint tags
    ("alibaba/fp8"). Normalising one into the other by hand would be a guess that happens
    to work; the model's own endpoint listing states the correspondence, so ask it.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    model = model or os.environ.get("PARITY_MODEL", MODEL)
    try:
        r = httpx.get(f"{UPSTREAM}/models/{model}/endpoints",
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        endpoints = r.json()["data"]["endpoints"]
    except Exception as exc:                                     # noqa: BLE001
        log.warning("could not fetch the endpoint listing (%s); the audit will compare "
                    "normalised names instead of exact tags", exc)
        return {}
    return {e["name"].split(" | ")[0]: e.get("tag", "") for e in endpoints}


def _usage_from_tail(tail: bytes) -> dict:
    """Pull the last `usage` object out of the tail of an SSE stream.

    The terminal `response.completed` frame carries it, a few dozen bytes from the end
    (`..."usage":{...}},"sequence_number":N}`), so a rolling tail finds it whatever the
    frame's total size.
    """
    idx = tail.rfind(b'"usage":')
    if idx < 0:
        return {}
    try:
        obj, _end = json.JSONDecoder().raw_decode(
            tail[idx + len(b'"usage":'):].decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    details = obj.get("output_tokens_details") or {}
    return {"tokens_prompt": obj.get("input_tokens"),
            "tokens_completion": obj.get("output_tokens"),
            "tokens_reasoning": details.get("reasoning_tokens"),
            "tokens_cached": (obj.get("input_tokens_details") or {}).get("cached_tokens"),
            "cost": obj.get("cost")}


def _served_by_pinned(provider_name: str, tags: dict[str, str], pin=PIN) -> bool:
    if provider_name in tags:
        return tags[provider_name] in pin
    flat = re.sub(r"[^a-z0-9]", "", (provider_name or "").lower())
    return any(flat == re.sub(r"[^a-z0-9]", "", p.split("/")[0]) for p in pin)


class Parity:
    def __init__(self, *, pin=PIN, strip=STRIP, audit_path: Path | None = None,
                 dump_dir: Path | None = None, upstream: str = UPSTREAM,
                 model: str = MODEL, transcript: Path | None = None) -> None:
        self.pin, self.strip, self.upstream = tuple(pin), tuple(strip), upstream.rstrip("/")
        self.model = model
        self.audit_path = Path(audit_path) if audit_path else None
        self.dump_dir = Path(dump_dir) if dump_dir else None
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        # One row per upstream call, in call order. The agent brackets a codex turn by
        # this file's byte offsets, which is what lets a turn's reasoning be recovered
        # even though nothing codex emits carries it.
        self.transcript = Path(transcript) if transcript else None
        if self.transcript:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.tags: dict[str, str] = {}
        self.calls = 0
        self.audited = 0
        self.off_pin: list[str] = []
        self.stripped_seen: set[str] = set()
        self.client: httpx.AsyncClient | None = None
        # asyncio keeps only a weak reference to a bare create_task, so an audit still
        # sleeping out its backoff can be collected mid-poll and lose the record.
        self._audits: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ the body
    def rewrite(self, body: bytes) -> tuple[bytes, dict]:
        """Inject the pin; drop the sampling knobs. Anything we cannot parse is passed
        through untouched and flagged -- silently swallowing a body would turn a wiring
        bug into a mystery."""
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body, {"pinned": False, "reason": "unparseable body"}
        if not isinstance(payload, dict):
            return body, {"pinned": False, "reason": "body is not an object"}

        removed = [k for k in self.strip if k in payload]
        for key in removed:
            payload.pop(key)
        self.stripped_seen.update(removed)
        payload["provider"] = {"only": list(self.pin)}
        payload["usage"] = {"include": True}
        return json.dumps(payload).encode(), {
            "pinned": True, "stripped": removed,
            "model": payload.get("model"), "stream": bool(payload.get("stream")),
        }

    # ----------------------------------------------------------------- the audit
    async def audit(self, gen_id: str, note: dict) -> None:
        """Ask OpenRouter who served `gen_id`. The record lands a beat after the stream
        closes, so poll briefly rather than accepting the first 404 as an answer.

        Only the PROVIDER is taken from here. `/generation` reports zeroes for every
        token and cost field of a streamed responses-API call -- measured, permanently,
        not a lag -- so the accounting is read out of the stream instead, where the
        injected `usage.include` puts it.
        """
        assert self.client is not None
        # The audit is the proxy's OWN call, not a forwarded one: the caller's
        # Authorization header is not on it, and /generation answers 401 for an
        # anonymous request. Measured -- every retry 401'd and the row recorded a null
        # provider, which reads exactly like "the pin could not be confirmed".
        key = os.environ.get("OPENROUTER_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        record = None
        # Measured: the record is not queryable the instant the stream closes -- a 15s
        # budget gave up at 404 on a call that resolved at ~17s. This runs detached, so
        # patience is free and a null provider is a hole in the arm's evidence.
        for delay in (2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0):
            await asyncio.sleep(delay)
            try:
                r = await self.client.get(f"{self.upstream}/generation",
                                          params={"id": gen_id}, headers=headers,
                                          timeout=30)
                if r.status_code == 200:
                    record = r.json().get("data") or {}
                    break
                if r.status_code in (401, 403):
                    self._write({**note, "gen_id": gen_id, "provider": None, "ok": None,
                                 "error": f"/generation refused the audit key "
                                          f"({r.status_code}); the pin is UNVERIFIED"})
                    return
            except Exception:                                    # noqa: BLE001
                continue
        if record is None:
            self._write({**note, "gen_id": gen_id, "provider": None,
                         "ok": None, "error": "no /generation record"})
            return

        provider = record.get("provider_name")
        ok = _served_by_pinned(provider, self.tags, self.pin)
        self.audited += 1
        if not ok:
            self.off_pin.append(f"{gen_id}: {provider}")
            log.error("OFF-PIN: %s served by %r, which is not in %s",
                      gen_id, provider, list(self.pin))
        self._write({**note, "gen_id": gen_id, "provider": provider,
                     "tag": self.tags.get(provider or ""), "ok": ok,
                     "model": record.get("model")})

    def _write(self, row: dict) -> None:
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a") as handle:
            handle.write(json.dumps({"t": round(time.time(), 3), **row}) + "\n")

    def _write_transcript(self, started: float, reasoning: list[str],
                          status: int) -> None:
        """One line per call, appended and flushed, so a reader tailing the file by
        offset sees whole rows and never a half-written one."""
        if self.transcript is None:
            return
        self.seq += 1
        row = {"seq": self.seq, "t0": round(started, 3), "t1": round(time.time(), 3),
               "status": status, "reasoning": reasoning}
        try:
            with self.transcript.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        except OSError:                                # the transcript is a view, never
            log.warning("could not append to the transcript", exc_info=True)  # the run


def build_app(state: Parity) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):                           # noqa: ANN202
        if state.client is None:
            # No read timeout: a reasoning turn can be quiet for minutes and OpenRouter
            # sends SSE keepalives; the agent's per-turn deadline is what bounds a call.
            state.client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=30.0, read=600.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        if not state.tags:
            state.tags = await asyncio.to_thread(_pin_names, state.model)
        log.info("pin=%s | %d provider tags resolved | audit=%s",
                 ",".join(state.pin), len(state.tags), state.audit_path)
        yield
        # Drain the detached audits: they are the run's evidence that the pin held, and
        # they outlive the responses they describe by design.
        if state._audits:
            log.info("waiting on %d outstanding audit(s)", len(state._audits))
            await asyncio.wait(set(state._audits), timeout=150)
        if state.client:
            await state.client.aclose()

    app = FastAPI(title="parity proxy", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:                         # noqa: ANN202
        return JSONResponse({
            "ok": True, "pin": list(state.pin), "calls": state.calls,
            "audited": state.audited, "off_pin": state.off_pin[:10],
            "stripped_seen": sorted(state.stripped_seen),
            "provider_tags": len(state.tags),
        })

    @app.api_route("/{path:path}",
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def passthrough(path: str, request: Request):          # noqa: ANN202
        assert state.client is not None
        body = await request.body()
        note: dict = {"path": path}
        if request.method == "POST" and body:
            body, note = state.rewrite(body)
            note["path"] = path

        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        if state.dump_dir:
            stamp = f"{time.time():.3f}-{path.replace('/', '_') or 'root'}"
            (state.dump_dir / f"{stamp}.req.json").write_bytes(body or b"{}")

        # Callers point at `http://host:port/v1` and codex appends `/responses`, while
        # UPSTREAM already ends in `/api/v1`; without this the join would be `/v1/v1/...`.
        rel = path.lstrip("/")
        if rel == "v1" or rel.startswith("v1/"):
            rel = rel[3:]
        url = f"{state.upstream}/{rel}"
        stack = AsyncExitStack()
        try:
            upstream = await stack.enter_async_context(state.client.stream(
                request.method, url, content=body or None, headers=headers,
                params=dict(request.query_params)))
        except Exception as exc:                                 # noqa: BLE001
            await stack.aclose()
            log.error("upstream %s %s failed: %s", request.method, path, exc)
            return JSONResponse({"error": {"message": f"parity proxy: {exc}",
                                           "type": "upstream_error"}}, status_code=502)

        state.calls += 1
        head, tail = bytearray(), bytearray()
        dump = state.dump_dir / f"{time.time():.3f}.resp.txt" if state.dump_dir else None
        # SSE frames are line-delimited but chunks are not, so a frame straddles chunks;
        # `pending` carries the unterminated tail into the next one.
        pending = bytearray()
        reasoning: list[str] = []
        started = time.time()

        def scan(buf: bytearray) -> None:
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[:nl + 1]
                if REASONING_DONE not in line or not line.startswith(b"data: "):
                    continue
                try:
                    frame = json.loads(line[6:])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                text = frame.get("text")
                if frame.get("type") == "response.reasoning_text.done" and text:
                    if sum(len(x) for x in reasoning) < MAX_REASONING_PER_CALL:
                        reasoning.append(str(text))

        async def stream():                                      # noqa: ANN202
            try:
                async for chunk in upstream.aiter_bytes():
                    if len(head) < HEAD_BYTES:
                        head.extend(chunk[:HEAD_BYTES - len(head)])
                    tail.extend(chunk)
                    if len(tail) > TAIL_BYTES:
                        del tail[:-TAIL_BYTES]
                    if state.transcript is not None:
                        pending.extend(chunk)
                        scan(pending)
                        if len(pending) > 1_000_000:   # not a frame; do not grow forever
                            del pending[:-4096]
                    if dump:
                        with dump.open("ab") as handle:
                            handle.write(chunk)
                    yield chunk
            finally:
                state._write_transcript(started, reasoning, upstream.status_code)
                await stack.aclose()
                match = GEN_ID_RE.search(bytes(head))
                note.update({"status": upstream.status_code,
                             **_usage_from_tail(bytes(tail))})
                if match and upstream.status_code < 400:
                    task = asyncio.create_task(state.audit(
                        match.group(1).decode(), dict(note)))
                    state._audits.add(task)
                    task.add_done_callback(state._audits.discard)
                else:
                    state._write({**note, "gen_id": None, "ok": None,
                                  "error": None if match else "no gen id in response"})

        out = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(stream(), status_code=upstream.status_code,
                                 headers=out,
                                 media_type=upstream.headers.get("content-type"))

    return app


# ------------------------------------------------------------------------ the checks
def verify(port: int, model: str) -> int:
    """One real call through the proxy, then the two facts that matter: who served it,
    and that an impossible pin fails instead of quietly routing elsewhere."""
    base = f"http://127.0.0.1:{port}/v1"
    key = os.environ.get("OPENROUTER_API_KEY", "")
    hdr = {"Authorization": f"Bearer {key}"}

    r = httpx.post(f"{base}/responses", headers=hdr, timeout=180, json={
        "model": model, "input": "Reply with the single word: ok.",
        # deliberately sent, and expected to be stripped on the way through
        "temperature": 0.9, "max_output_tokens": 64})
    if r.status_code != 200:
        print(f"FAIL: proxy returned {r.status_code}: {r.text[:300]}")
        return 1
    payload = r.json()
    gen_id = payload.get("id", "")
    print(f"call ok: id={gen_id} temperature_echoed={payload.get('temperature')} "
          f"max_output_tokens_echoed={payload.get('max_output_tokens')}")

    tags = _pin_names(model)
    served = None
    for delay in (1.0, 2.0, 4.0, 8.0):
        time.sleep(delay)
        g = httpx.get(f"{UPSTREAM}/generation", params={"id": gen_id},
                      headers=hdr, timeout=30)
        if g.status_code == 200:
            served = (g.json().get("data") or {}).get("provider_name")
            break
    ok = _served_by_pinned(served or "", tags)
    print(f"served by: {served!r} tag={tags.get(served or '')!r} in-pin={ok}")

    # The control. "Served by a pinned host" on its own is also what luck looks like:
    # three of the fifteen endpoints are in the pin. So send a DECOY pin straight to
    # OpenRouter and check the routing follows it. If it does, the body field is causal,
    # and the call above landing in-pin is the injection working rather than a coin flip.
    decoy = os.environ.get("PARITY_DECOY", "atlas-cloud/fp4")
    d = httpx.post(f"{UPSTREAM}/responses", headers=hdr, timeout=180, json={
        "model": model, "input": "Reply with the single word: ok.",
        "provider": {"only": [decoy]}, "usage": {"include": True}})
    if d.status_code != 200:
        print(f"control INCONCLUSIVE: decoy pin {decoy} returned {d.status_code}")
        return 0 if ok else 1
    served_decoy = None
    for delay in (2.0, 3.0, 5.0, 8.0, 15.0, 30.0):
        time.sleep(delay)
        g = httpx.get(f"{UPSTREAM}/generation", params={"id": d.json().get("id", "")},
                      headers=hdr, timeout=30)
        if g.status_code == 200:
            served_decoy = (g.json().get("data") or {}).get("provider_name")
            break
    steered = tags.get(served_decoy or "") == decoy
    print(f"control: decoy pin {decoy!r} -> served by {served_decoy!r} steered={steered}")
    if not steered:
        print("FAIL: the provider field did not steer routing; the pin proves nothing")
    return 0 if (ok and steered) else 1


def audit_report(path: Path) -> int:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    served: dict[str, int] = {}
    off, unknown, cost = [], 0, 0.0
    for row in rows:
        if row.get("ok") is True:
            served[row.get("provider") or "?"] = served.get(row.get("provider") or "?", 0) + 1
            cost += float(row.get("cost") or 0.0)
        elif row.get("ok") is False:
            off.append(row)
        else:
            unknown += 1
    print(f"calls recorded: {len(rows)}")
    for name, n in sorted(served.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {name}")
    print(f"off-pin: {len(off)}   unaudited: {unknown}   cost: ${cost:.4f}")
    for row in off[:10]:
        print(f"  OFF-PIN {row.get('gen_id')} -> {row.get('provider')}")
    return 1 if off else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--audit", default="", help="parity.jsonl to append to (or to report)")
    ap.add_argument("--dump-dir", default="", help="write every request/response body")
    ap.add_argument("--pin", default=",".join(PIN))
    ap.add_argument("--verify", action="store_true", help="probe a running proxy and exit")
    ap.add_argument("--report", action="store_true", help="summarise --audit and exit")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--transcript", default="",
                    help="reasoning.jsonl: one row per upstream call, carrying the "
                         "reasoning codex does not emit. Defaults to sitting beside "
                         "--audit, because a run that records the pin and not the "
                         "working is only half a record.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.report:
        sys.exit(audit_report(Path(args.audit)))
    if args.verify:
        sys.exit(verify(args.port, args.model))
    if not os.environ.get("OPENROUTER_API_KEY"):
        # The proxy forwards the caller's Authorization header, but the audit is its own
        # call; without a key the run would look pinned and be unverified.
        log.warning("OPENROUTER_API_KEY is unset: the /generation audit will not work")

    import uvicorn
    state = Parity(pin=tuple(x.strip() for x in args.pin.split(",") if x.strip()),
                   audit_path=Path(args.audit) if args.audit else None,
                   dump_dir=Path(args.dump_dir) if args.dump_dir else None,
                   model=args.model,
                   transcript=(Path(args.transcript) if args.transcript
                               else (Path(args.audit).with_name("reasoning.jsonl")
                                     if args.audit else None)))
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
