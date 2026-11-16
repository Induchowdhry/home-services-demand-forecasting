#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" src/train.py --epochs 4
"$PYTHON_BIN" src/drift_monitor.py --data data/processed_daily_bookings.csv || echo "Drift alert raised; review before promoting the new model."
# Production: validate the new artifact, version it, and restart/reload the API service.
