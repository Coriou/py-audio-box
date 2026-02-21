#!/usr/bin/env bash
# tests/local/02-voice-clone.sh
# voice-clone: self-test + synth via named voice + synth via slug.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 02: voice-clone pipeline ==="

run_cmd "voice-clone self-test" \
  ./run voice-clone self-test \
    --cache "$CACHE" \
    --out "$OUT/voice-clone-self-test" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

run_cmd "voice-clone synth: chalamet-en / balanced / EN" \
  ./run voice-clone synth \
    --voice chalamet-en \
    --text "The world is full of extraordinary things worth discovering." \
    --language English \
    --profile balanced \
    --seed 1 \
    --cache "$CACHE" \
    --out "$OUT/voice-clone-synth" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

run_cmd "voice-clone synth: rascar-capac / stable / FR" \
  ./run voice-clone synth \
    --voice rascar-capac \
    --text "Le monde est rempli de choses extraordinaires qui meritent d'etre decouvertes." \
    --language French \
    --profile stable \
    --seed 2 \
    --cache "$CACHE" \
    --out "$OUT/voice-clone-synth" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

run_cmd "voice-clone prepare-ref: chalamet-en" \
  ./run voice-clone prepare-ref \
    --voice chalamet-en \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

print_summary "02-voice-clone"
