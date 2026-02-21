#!/usr/bin/env bash
# tests/local/13-register-builtin.sh
# register-builtin: turn a built-in speaker into a named custom_voice voice.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 13: register-builtin — named custom_voice ==="

# Register without a default instruct
run_cmd "register-builtin: test-ryan (no default instruct)" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

speak "test-ryan / stable / no instruct" \
  --voice test-ryan \
  --text "This voice was registered as a named custom voice with no default instruction." \
  --language English \
  --profile stable \
  --seed 10

# Register with an --instruct-default string
run_cmd "register-builtin: test-ryan (instruct-default string)" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan \
    --speaker Ryan \
    --instruct-default "Speak clearly and professionally, like a tech conference presenter." \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

speak "test-ryan / balanced / uses instruct-default" \
  --voice test-ryan \
  --text "The default instruction is applied automatically whenever this voice is used." \
  --language English \
  --profile balanced \
  --seed 11

# --instruct-style override should take precedence over the registered default
speak "test-ryan / expressive / --instruct-style excited overrides default" \
  --voice test-ryan \
  --text "Incredible! The override is working perfectly, ignoring the registered default!" \
  --instruct-style excited \
  --language English \
  --profile expressive \
  --seed 12

# Register with --instruct-default-style (template name instead of raw string)
run_cmd "register-builtin: test-ryan-warm (instruct-default-style warm)" \
  ./run voice-synth register-builtin \
    --voice-name test-ryan-warm \
    --speaker Ryan \
    --instruct-default-style warm \
    --cache "$CACHE" \
    --threads "$THREADS" \
    --dtype "$DTYPE"

speak "test-ryan-warm / balanced / uses warm default style" \
  --voice test-ryan-warm \
  --text "It is so lovely to have you here with us today. We are thrilled to share this with you." \
  --language English \
  --profile balanced \
  --seed 13

print_summary "13-register-builtin"
