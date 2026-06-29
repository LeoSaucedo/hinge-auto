#!/usr/bin/env bash
# Run hinge-auto at :13 past each hour (9am-9pm).
# No random jitter — cron handles the timing directly.

set -euo pipefail
cd "$(dirname "$0")"

echo "[$(date)] Starting run..."
source .venv/bin/activate
python -u main.py --mode carlos 2>&1
EXIT_CODE=$?
echo "[$(date)] Done (exit $EXIT_CODE)"
exit $EXIT_CODE
