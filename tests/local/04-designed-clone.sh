#!/usr/bin/env bash
# tests/local/04-designed-clone.sh
# voice-synth speak: designed_clone engine (pre-registered designed voices).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 04: designed_clone engine ==="

speak "smoke-designed / stable / EN" \
  --voice smoke-designed \
  --text "A gentle breeze carried the scent of pine and cedar through the valley below." \
  --language English \
  --profile stable \
  --seed 11

speak "smoke-designed / expressive / EN" \
  --voice smoke-designed \
  --text "The signal is faint but unmistakably clear. Something has changed out there." \
  --language English \
  --profile expressive \
  --seed 12

speak "smoke-designed / balanced / qa / EN" \
  --voice smoke-designed \
  --text "Each line of code carries the intention of the person who wrote it." \
  --language English \
  --profile balanced \
  --seed 13 \
  --qa

print_summary "04-designed-clone"
