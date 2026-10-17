#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/run.sh CONFIG.json RUN_DIRECTORY" >&2
    exit 1
fi
python3 -m qlg prepare "$1" "$2"
python3 -m qlg run "$2" --jobs "${JOBS:-1}" --threads "${THREADS:-2}"
python3 -m qlg analyze "$2"
