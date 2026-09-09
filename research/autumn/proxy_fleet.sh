#!/usr/bin/env bash
# One parity proxy per worker, for a run launched with --workers.
#
# A turn's reasoning is joined to that turn by BYTE OFFSET into the proxy's
# reasoning.jsonl: mark the size before the call, read what was appended after it. That
# is exact while one session runs at a time and wrong the moment two do -- each turn
# would carry every other session's thinking as well as its own, and the replay page
# would show, per turn, the wrong working. The scores would not move (they come from an
# independent replay), which is what makes it worth guarding: it is a corruption of the
# evidence that leaves the numbers looking fine.
#
# A worker plays one session at a time, so one proxy per worker restores exactly the
# property the join needs.
#
#   bash research/autumn/proxy_fleet.sh start  [N]     (default N=6)
#   bash research/autumn/proxy_fleet.sh stop   [N]
#   bash research/autumn/proxy_fleet.sh status [N]
#   bash research/autumn/proxy_fleet.sh verify [N]
#
# Ports are PORT_BASE..PORT_BASE+N-1 (default 8790, clear of the single-session 8788).
# Each proxy keeps its own audit, reasoning log and pidfile under logs/parity_proxy/w<i>,
# so cost accounting globs the fleet instead of reading one file.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BASE="${PORT_BASE:-8790}"
CMD="${1:-status}"
N="${2:-6}"

for i in $(seq 0 $((N - 1))); do
  export PARITY_PROXY_PORT=$((BASE + i))
  export PARITY_PROXY_LOG_DIR="$ROOT/logs/parity_proxy/w$i"
  printf '%-6s w%-2d port %s  ' "$CMD" "$i" "$PARITY_PROXY_PORT"
  bash "$HERE/proxy_ctl.sh" "$CMD" | tr '\n' ' '
  echo
done
