#!/bin/bash
# usage: run_ox.sh <brief-file> <log-file>
# Waits for the Zen endpoint, retries transient failures, runs Ox Alpha at max effort.
BRIEF=$1; LOG=$2
for attempt in $(seq 1 60); do
  echo "=== attempt $attempt $(date)" >> "$LOG"
  if ! timeout 180 opencode run --variant max --model opencode/x-preview-f-free "Reply with one word: PONG" 2>&1 | grep -q PONG; then
    echo "=== ping failed, sleeping 180s" >> "$LOG"; sleep 180; continue
  fi
  echo "=== ping ok, launching" >> "$LOG"
  timeout 21600 opencode run --auto --variant max --model opencode/x-preview-f-free \
    "Read the file $BRIEF and carry out the task it describes, in full, including every verification step it specifies. It is a complete brief; follow it." >> "$LOG" 2>&1
  rc=$?
  if grep -q -E "Endpoint is unavailable|network_error|Unexpected server error" <(tail -c 2000 "$LOG"); then
    echo "=== transient failure, retrying in 180s" >> "$LOG"; sleep 180; continue
  fi
  echo "OX EXIT $rc" >> "$LOG"; exit $rc
done
echo "OX GAVE UP" >> "$LOG"; exit 1
