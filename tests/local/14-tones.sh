#!/usr/bin/env bash
# tests/local/14-tones.sh
# register-builtin --tone + speak --tone: per-voice tone variants.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 14: tone registration + speak --tone ==="

# Register base voice first
run_cmd "register-builtin: test-ryan (base)" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

# Register calm tone
run_cmd "register-builtin: test-ryan --tone calm" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --tone calm \
    --tone-instruct "Speak in a calm measured tone, like a meditation guide leading a breathing exercise." \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

# Register energetic tone
run_cmd "register-builtin: test-ryan --tone energetic" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --tone energetic \
    --tone-instruct "Speak with high energy and enthusiasm, like a sports commentator at the decisive moment." \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

# Register serious tone
run_cmd "register-builtin: test-ryan --tone serious" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --tone serious \
    --tone-instruct "Speak in a serious authoritative tone, like a news anchor reporting breaking news." \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

# Synthesise with each tone
speak "test-ryan / --tone calm / stable" \
  --voice test-ryan \
  --tone calm \
  --text "Take a deep breath. Let your thoughts settle. There is nowhere you need to be right now." \
  --language English \
  --profile stable \
  --seed 20

speak "test-ryan / --tone energetic / expressive" \
  --voice test-ryan \
  --tone energetic \
  --text "And the crowd goes absolutely wild! What an incredible finish! Nobody saw that coming!" \
  --language English \
  --profile expressive \
  --seed 21

speak "test-ryan / --tone serious / stable" \
  --voice test-ryan \
  --tone serious \
  --text "Breaking news: significant developments have been reported in the past hour. We go live." \
  --language English \
  --profile stable \
  --seed 22

# No --tone should still use base voice (no tone override)
speak "test-ryan / no tone / balanced" \
  --voice test-ryan \
  --text "This synthesis uses the base voice with no tone override applied." \
  --language English \
  --profile balanced \
  --seed 23

print_summary "14-tones"
