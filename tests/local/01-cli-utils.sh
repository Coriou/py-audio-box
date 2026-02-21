#!/usr/bin/env bash
# tests/local/01-cli-utils.sh
# CLI utility commands that don't load the model (fast smoke tests).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 01: CLI utilities (no model load) ==="

run_cmd "list-voices (human)" \
  ./run voice-synth list-voices \
    --cache "$CACHE"

run_cmd "list-voices --json" \
  ./run voice-synth list-voices \
    --cache "$CACHE" --json

run_cmd "capabilities --json --skip-speaker-probe" \
  ./run voice-synth capabilities \
    --threads "$THREADS" --dtype "$DTYPE" \
    --skip-speaker-probe --json

run_cmd "list-speakers --json" \
  ./run voice-synth list-speakers \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE" \
    --json

print_summary "01-cli-utils"
