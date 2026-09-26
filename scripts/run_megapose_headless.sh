#!/usr/bin/env bash
set -euo pipefail

display_number="${AUTPOSETRACK_X_DISPLAY:-99}"
Xvfb ":${display_number}" -screen 0 1280x1024x24 -nolisten tcp \
  >"/tmp/autoposetrack-xvfb-${display_number}.log" 2>&1 &
xvfb_pid=$!
cleanup() {
  kill "${xvfb_pid}" 2>/dev/null || true
  wait "${xvfb_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
export DISPLAY=":${display_number}"
sleep 2
"$@"
