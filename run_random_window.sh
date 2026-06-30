#!/usr/bin/env bash
# Run hinge-auto on odd hours (9am-9pm).
# Sleeps 0-20min so actual run lands between :00-:20.
# Guarantees 40+ min buffer before next hour's job.

set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/ada/.local/bin:/usr/bin:/bin:$PATH"

# Max 20 min random delay — still guarantees 40+ min runtime before next cron
delay=$((RANDOM % 1200))
start_time=$(date -d "+${delay} seconds" '+%H:%M')
echo "[$(date)] Cron fired. Will run at ~${start_time} (${delay}s delay)"
sleep "$delay"

echo "[$(date)] Starting run..."
source .venv/bin/activate
python -u main.py --mode carlos 2>&1
EXIT_CODE=$?
echo "[$(date)] Done (exit $EXIT_CODE)"
exit $EXIT_CODE
