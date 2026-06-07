#!/usr/bin/env python3
"""Render an RGB-Agent run into a single self-contained HTML file you can open
in a browser and step through with the arrow keys.

For each action it shows: the colored board grid, score/state/level, the action
the agent took, and the strategic reasoning that led there (parsed from the
[STRATEGIC ANALYSIS FROM LOG REVIEW] / [CURRENT PLAN] blocks in logs.txt, and,
when present, the [ASSISTANT (recovered)] blocks in logs_analyzer.txt).

Usage:
    uv run scripts/visualize.py evaluation_results/<run>/ls20
    uv run scripts/visualize.py evaluation_results/<run>/ls20 -o /tmp/run.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

# One distinct color per board glyph. Unknown glyphs fall back to a neutral grey.
CELL_COLORS = {
    "O": "#2b2b3a",  # fixed border / wall
    "q": "#11151c",  # background fill
    "h": "#3a5a7a",  # structure
    "f": "#c8a23a",  # feature / target
    "-": "#7a3a6a",  # door / gap
    "$": "#3ac88a",  # token / reward
    "8": "#c85a3a",  # agent / mover
    "G": "#5ac8c8",
    "n": "#9a5ac8",
}
DEFAULT_COLOR = "#444"

SEP = re.compile(r"^={20,}\s*$")
HEADER = re.compile(r"^Action\s+(\d+)\s*\|\s*Level\s+(\d+)\s*\|\s*Attempt\s+(\d+)\s*\|\s*(.+)$")
SCORE_STATE = re.compile(r"^Score:\s*(\d+)\s*\|\s*State:\s*(\S+)")
BOARD_MARK = re.compile(r"^\[(?:INITIAL|POST-ACTION) BOARD STATE\]")
TOOLCALL = re.compile(r"Tool Call:\s*(ACTION\d+)\((\{.*?\})\)")
SECTION = re.compile(r"^\[([A-Z][A-Z _/]+)\]\s*$")

# Sections worth surfacing as "reasoning" for a step.
REASON_SECTIONS = {
    "STRATEGIC ANALYSIS FROM LOG REVIEW",
    "CURRENT PLAN",
    "OBSERVATION_PHASE",
    "ACTION_PHASE",
}


def parse_run(log_dir: Path) -> list[dict]:
    text = (log_dir / "logs.txt").read_text(errors="replace")
    lines = text.split("\n")

    # Split into action blocks delimited by the ==== separators.
    blocks: list[list[str]] = []
    cur: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if SEP.match(ln) and i + 1 < len(lines) and HEADER.match(lines[i + 1]):
            if cur:
                blocks.append(cur)
            cur = []
            i += 1
            continue
        cur.append(ln)
        i += 1
    if cur:
        blocks.append(cur)

    steps: list[dict] = []
    for blk in blocks:
        if not blk or not HEADER.match(blk[0]):
            continue
        m = HEADER.match(blk[0])
        action_n, level, attempt, desc = m.group(1), m.group(2), m.group(3), m.group(4)
        score, state = "0", "?"
        if len(blk) > 1 and SCORE_STATE.match(blk[1]):
            sm = SCORE_STATE.match(blk[1])
            score, state = sm.group(1), sm.group(2)

        # Board: rows following a BOARD_MARK (skip a leading "Score:" line).
        board: list[str] = []
        j = 0
        while j < len(blk):
            if BOARD_MARK.match(blk[j]):
                k = j + 1
                if k < len(blk) and blk[k].startswith("Score:"):
                    k += 1
                while k < len(blk) and blk[k].strip() and not blk[k].startswith("["):
                    board.append(blk[k])
                    k += 1
                # keep the LAST board in the block (post-action state)
                j = k
            else:
                j += 1

        # Action taken
        tool = None
        for ln in blk:
            tm = TOOLCALL.search(ln)
            if tm:
                try:
                    args = json.loads(tm.group(2))
                except Exception:
                    args = {}
                tool = {"action": tm.group(1), **args}
                break

        # Reasoning: collect text under interesting sections.
        reasoning_parts: list[str] = []
        cur_sec = None
        buf: list[str] = []
        for ln in blk:
            sm = SECTION.match(ln)
            if sm:
                if cur_sec in REASON_SECTIONS and buf:
                    body = "\n".join(buf).strip()
                    if body:
                        reasoning_parts.append(f"### {cur_sec}\n{body}")
                cur_sec = sm.group(1).strip()
                buf = []
            else:
                buf.append(ln)
        if cur_sec in REASON_SECTIONS and buf:
            body = "\n".join(buf).strip()
            if body:
                reasoning_parts.append(f"### {cur_sec}\n{body}")

        steps.append({
            "n": int(action_n),
            "level": level,
            "attempt": attempt,
            "desc": desc,
            "score": score,
            "state": state,
            "board": board,
            "tool": tool,
            "reasoning": "\n\n".join(reasoning_parts).strip(),
        })
    return steps


def render_board_html(board: list[str], tool: dict | None) -> str:
    hx = hy = None
    if tool and tool.get("action") == "ACTION6":
        hx, hy = tool.get("x"), tool.get("y")
    rows = []
    for r, line in enumerate(board):
        cells = []
        for c, ch in enumerate(line):
            color = CELL_COLORS.get(ch, DEFAULT_COLOR)
            ring = "box-shadow:0 0 0 1px #fff inset;" if (r == hx and c == hy) else ""
            cells.append(f'<i style="background:{color};{ring}" title="{r},{c} {ch}"></i>')
        rows.append('<div class="brow">' + "".join(cells) + "</div>")
    return '<div class="board">' + "".join(rows) + "</div>"


def build_html(steps: list[dict], title: str) -> str:
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{html.escape(g)}</span>'
        for g, c in CELL_COLORS.items()
    )
    frames = []
    for s in steps:
        tool_str = "—"
        if s["tool"]:
            t = s["tool"]
            if t.get("action") == "ACTION6":
                tool_str = f'ACTION6 click ({t.get("x")},{t.get("y")})'
            else:
                tool_str = t.get("action", "?")
        delta = ""
        meta = (
            f'<b>Action {s["n"]}</b> · Level {s["level"]} · Attempt {s["attempt"]} · '
            f'Score <b>{s["score"]}</b> · {html.escape(s["state"])}<br>'
            f'<span class="desc">{html.escape(s["desc"])}</span><br>'
            f'Took: <code>{html.escape(tool_str)}</code>{delta}'
        )
        frames.append({
            "meta": meta,
            "board": render_board_html(s["board"], s["tool"]),
            "reasoning": html.escape(s["reasoning"]) or "<em>(no reasoning captured for this step)</em>",
        })
    data = json.dumps(frames)
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>
 body{{margin:0;background:#0c0e12;color:#cdd3da;font:13px/1.5 ui-monospace,Menlo,monospace}}
 header{{padding:8px 14px;background:#151921;border-bottom:1px solid #222;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
 .wrap{{display:flex;gap:16px;padding:14px;align-items:flex-start}}
 .board{{line-height:0;border:1px solid #222;background:#000}}
 .brow{{display:flex}}
 .brow i{{width:8px;height:8px;display:inline-block}}
 .side{{flex:1;min-width:320px;max-height:88vh;overflow:auto}}
 .meta{{padding:8px 10px;background:#151921;border:1px solid #222;border-radius:6px;margin-bottom:10px}}
 .desc{{color:#8a93a0}}
 pre{{white-space:pre-wrap;word-break:break-word;background:#10141a;border:1px solid #1d232c;border-radius:6px;padding:10px;margin:0}}
 .lg{{display:inline-flex;align-items:center;gap:4px;margin-right:6px}}
 .lg i{{width:11px;height:11px;display:inline-block;border:1px solid #000}}
 button{{background:#222a35;color:#cdd3da;border:1px solid #333;border-radius:5px;padding:4px 10px;cursor:pointer}}
 input[type=range]{{width:340px}}
 code{{color:#9ad}}
 h3{{margin:.4em 0 .2em;color:#7fd}}
</style>
<header>
 <span>{html.escape(title)}</span>
 <button id=prev>◀ prev</button>
 <input type=range id=slider min=0 value=0>
 <button id=next>next ▶</button>
 <span id=pos></span>
 <span style="margin-left:auto">{legend}</span>
</header>
<div class=wrap>
 <div id=boardbox></div>
 <div class=side><div class=meta id=meta></div><h3>Reasoning</h3><pre id=reason></pre></div>
</div>
<script>
const F={data};
const slider=document.getElementById('slider');
slider.max=F.length-1;
function md(t){{return t.replace(/^### (.+)$/gm,'<h3>$1</h3>');}}
function show(i){{
  i=Math.max(0,Math.min(F.length-1,i));slider.value=i;
  document.getElementById('boardbox').innerHTML=F[i].board;
  document.getElementById('meta').innerHTML=F[i].meta;
  document.getElementById('reason').innerHTML=md(F[i].reasoning);
  document.getElementById('pos').textContent=(i+1)+' / '+F.length;
}}
slider.oninput=()=>show(+slider.value);
document.getElementById('prev').onclick=()=>show(+slider.value-1);
document.getElementById('next').onclick=()=>show(+slider.value+1);
document.onkeydown=e=>{{if(e.key==='ArrowLeft')show(+slider.value-1);if(e.key==='ArrowRight')show(+slider.value+1);}};
show(0);
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", help="path to a run's per-suite dir (contains logs.txt), e.g. evaluation_results/<run>/ls20")
    ap.add_argument("-o", "--out", default=None, help="output HTML path (default: <log_dir>/visualize.html)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not (log_dir / "logs.txt").exists():
        raise SystemExit(f"no logs.txt in {log_dir}")
    steps = parse_run(log_dir)
    out = Path(args.out) if args.out else log_dir / "visualize.html"
    out.write_text(build_html(steps, title=str(log_dir)))
    print(f"{len(steps)} steps -> {out}")


if __name__ == "__main__":
    main()
