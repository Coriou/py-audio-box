#!/usr/bin/env bash
# tests/local/10-custom-voice.sh
# CustomVoice: direct built-in speaker synthesis without a clone prompt.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 10: CustomVoice — direct speaker, no instruct ==="

# list-speakers acts as a model-load smoke test for CustomVoice
run_cmd "list-speakers --json" \
  ./run voice-synth list-speakers \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE" \
    --json

# Direct speaker synthesis — no instruct flag at all
speak "Ryan / stable / no instruct / EN" \
  --speaker Ryan \
  --text "Hello. This is Ryan speaking without any style instruction set." \
  --language English \
  --profile stable \
  --seed 1

speak "Ryan / balanced / no instruct / EN" \
  --speaker Ryan \
  --text "The information you requested is available on the main dashboard." \
  --language English \
  --profile balanced \
  --seed 2

speak "Ryan / expressive / no instruct / EN" \
  --speaker Ryan \
  --text "Honestly I was not expecting that at all. What a surprise!" \
  --language English \
  --profile expressive \
  --seed 3

print_summary "10-custom-voice"
