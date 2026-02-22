#!/usr/bin/env bash
# tests/remote/01-basic-synth.sh
# Smoke tests: CLI utilities + single-voice basic synthesis.
# Mirrors tests/local/01-cli-utils.sh + first speaks from 02-voice-clone.sh
set -euo pipefail
cd /app 2>/dev/null || true
source tests/remote/lib/common.sh

echo "=== 01: basic-synth (${TARGET}) ==="

run_cmd "list-voices" \
  $RUNNER voice-synth list-voices \
    --cache "$CACHE"

run_cmd "list-voices --json" \
  $RUNNER voice-synth list-voices \
    --cache "$CACHE" --json

run_cmd "capabilities --json --skip-speaker-probe" \
  $RUNNER voice-synth capabilities \
    --dtype "$DTYPE" \
    --skip-speaker-probe --json

run_cmd "list-speakers --json" \
  $RUNNER voice-synth list-speakers \
    --cache "$CACHE" \
    --dtype "$DTYPE" \
    --json

speak "rascar-capac / balanced / FR" \
  --voice rascar-capac \
  --text "Bienvenue dans ce test de synthèse vocale sur GPU. La qualité est bien meilleure qu'en CPU." \
  --language French \
  --profile balanced \
  --seed 1

speak "chalamet-en / stable / EN" \
  --voice chalamet-en \
  --text "This is a remote GPU synthesis smoke test. Everything should be significantly faster than CPU." \
  --language English \
  --profile stable \
  --seed 2

speak "david-attenborough / balanced / EN" \
  --voice david-attenborough \
  --text "In the vast expanse of the cosmos, every star tells a story billions of years in the making." \
  --language English \
  --profile balanced \
  --seed 3

print_summary "01-basic-synth"
