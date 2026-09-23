#!/usr/bin/env bash
# Run hinge-auto on odd hours (9am-9pm).
# Sleeps 0-20min so actual run lands between :00-:20.
# Guarantees 40+ min buffer before next hour's job.

set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/ada/.local/bin:/usr/bin:/bin:$PATH"

# DEBUG_DIR lives on the USB SSD at /mnt/sda1. If that drive is absent,
# /mnt/sda1 is still a real directory on the SD card's root filesystem —
# so writes don't fail, they quietly land on the SD and burn write cycles.
# Checked here (fail fast, before the 20 min sleep) and again below.
require_ssd() {
  if ! mountpoint -q /mnt/sda1; then
    echo "[$(date)] ERROR: /mnt/sda1 is not mounted — refusing to run." >&2
    echo "        Debug output would land on the SD card instead." >&2
    exit 1
  fi
}
require_ssd

# Max 20 min random delay — still guarantees 40+ min runtime before next cron
delay=$((RANDOM % 1200))
start_time=$(date -d "+${delay} seconds" '+%H:%M')
echo "[$(date)] Cron fired. Will run at ~${start_time} (${delay}s delay)"
sleep "$delay"

echo "[$(date)] Starting run..."
require_ssd  # re-check: the sleep above can outlast an unplug
source .venv/bin/activate
python -u main.py --mode carlos 2>&1
EXIT_CODE=$?
echo "[$(date)] Done (exit $EXIT_CODE)"
exit $EXIT_CODE
