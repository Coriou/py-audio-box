#!/usr/bin/env bash
# tests/local/16-design-voice.sh
# design-voice: generate a brand-new voice from a text description, then speak.
# Skipped when SKIP_DESIGN=1 (uses the 1.7B VoiceDesign model — much slower).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 16: design-voice + speak (requires VoiceDesign model) ==="

if [[ "$SKIP_DESIGN" == "0" ]]; then

  run_cmd "design-voice: test-designed-broadcaster" \
    ./run voice-synth design-voice \
      --instruct "A seasoned radio broadcaster, deep authoritative timbre, slight warmth and gravitas." \
      --ref-text "Good evening. You are listening to the late-night programme on public radio." \
      --voice-name test-designed-broadcaster \
      --cache "$CACHE" \
      --out "$OUT/design" \
      --threads "$THREADS" \
      --dtype "$DTYPE"

  speak "test-designed-broadcaster / stable / EN" \
    --voice test-designed-broadcaster \
    --text "The microphone is live. Everything you say from this point will be recorded and broadcast." \
    --language English \
    --profile stable \
    --seed 1

  speak "test-designed-broadcaster / expressive / EN" \
    --voice test-designed-broadcaster \
    --text "And with that, we close another chapter. Thank you for being with us tonight." \
    --language English \
    --profile expressive \
    --seed 2

  run_cmd "design-voice: test-designed-narrator (2nd design)" \
    ./run voice-synth design-voice \
      --instruct "A young female scientist, clear and enthusiastic, approachable and curious tone." \
      --ref-text "What we discovered changes everything we thought we knew about this molecule." \
      --voice-name test-designed-narrator \
      --cache "$CACHE" \
      --out "$OUT/design" \
      --threads "$THREADS" \
      --dtype "$DTYPE"

  speak "test-designed-narrator / balanced / EN" \
    --voice test-designed-narrator \
    --text "The experiment yielded results far beyond what the original hypothesis predicted." \
    --language English \
    --profile balanced \
    --seed 3

else
  skip_test "design-voice: test-designed-broadcaster (SKIP_DESIGN=1)"
  skip_test "test-designed-broadcaster / stable / EN (SKIP_DESIGN=1)"
  skip_test "test-designed-broadcaster / expressive / EN (SKIP_DESIGN=1)"
  skip_test "design-voice: test-designed-narrator (SKIP_DESIGN=1)"
  skip_test "test-designed-narrator / balanced / EN (SKIP_DESIGN=1)"
fi

print_summary "16-design-voice"
