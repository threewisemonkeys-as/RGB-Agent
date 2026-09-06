#!/usr/bin/env bash
# Start / stop / status for the parity proxy (research/autumn/proxy.py), the injecting
# passthrough the agent arm's codex sessions must route through (plan F15).
#
#   bash research/autumn/proxy_ctl.sh start [audit-path] | stop | restart | status | verify
#
# Process identity is a PID FILE, not `pgrep -f`. A pattern that names the module also
# matches the command line of whatever shell invoked this script, so `pgrep -f` reports
# "already running" for its own caller and then kills it -- measured here, and the same
# way round as the Claude reflection proxy's gotcha.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PARITY_PROXY_PORT:-8788}"
LOG_DIR="${PARITY_PROXY_LOG_DIR:-$ROOT/logs/parity_proxy}"
AUDIT="${2:-$LOG_DIR/parity.jsonl}"
PIDFILE="$LOG_DIR/proxy.pid"

alive() {
  [ -f "$PIDFILE" ] || return 1
  local p; p="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -qa "autumn.proxy" "/proc/$p/cmdline" 2>/dev/null
}

stop() {
  if alive; then
    local p; p="$(cat "$PIDFILE")"
    pkill -TERM -P "$p" 2>/dev/null   # uv spawns python as a child
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 1 10); do alive || break; sleep 1; done
    alive && { pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; }
    rm -f "$PIDFILE"; echo "stopped: $p"
  else
    rm -f "$PIDFILE"; echo "not running"
  fi
}

start() {
  if alive; then echo "already running: $(cat "$PIDFILE")"; curl -s "http://127.0.0.1:$PORT/healthz"; echo; return 0; fi
  mkdir -p "$LOG_DIR"
  cd "$ROOT" || exit 1
  set -a; [ -f "$ROOT/../.env" ] && . "$ROOT/../.env"; set +a
  nohup uv run python -m research.autumn.proxy --port "$PORT" --audit "$AUDIT" \
    >> "$LOG_DIR/proxy.log" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do
    curl -s "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
    sleep 1
  done
  echo "started pid=$(cat "$PIDFILE") port=$PORT audit=$AUDIT"
  curl -s "http://127.0.0.1:$PORT/healthz"; echo
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  if alive; then echo "running: $(cat "$PIDFILE")"; curl -s "http://127.0.0.1:$PORT/healthz"; echo; else echo "not running"; fi ;;
  verify)  cd "$ROOT" && set -a && . "$ROOT/../.env" && set +a && uv run python -m research.autumn.proxy --verify --port "$PORT" ;;
  *)       echo "usage: $0 start|stop|restart|status|verify"; exit 2 ;;
esac
