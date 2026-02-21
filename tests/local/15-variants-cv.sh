#!/usr/bin/env bash
# tests/local/15-variants-cv.sh
# CustomVoice variants + select-best (slow — skipped when SKIP_SLOW=1).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 15: CustomVoice variants + select-best (slow) ==="

if [[ "$SKIP_SLOW" == "0" ]]; then

  # Ensure test-ryan with tone calm is registered (dependency from 14-tones)
  run_cmd "register-builtin: test-ryan --tone calm (ensure registered)" \
    ./run voice-synth register-builtin \
      --voice-name test-ryan \
      --speaker Ryan \
      --tone calm \
      --tone-instruct "Speak in a calm measured tone, like a meditation guide leading a breathing exercise." \
      --cache "$CACHE" \
      --threads "$THREADS" \
      --dtype "$DTYPE"

  speak "test-ryan / --tone calm / 2-variants select-best / balanced" \
    --voice test-ryan \
    --tone calm \
    --text "Consistency is what separates the good from the truly great over the long run." \
    --language English \
    --profile balanced \
    --variants 2 \
    --select-best \
    --seed 30

  speak "Ryan / expressive / 3-variants select-best + --instruct / EN" \
    --speaker Ryan \
    --text "We only get one shot at this. Let us make it count." \
    --instruct "Speak with intense focus, like an athlete before a decisive competition." \
    --language English \
    --profile expressive \
    --variants 3 \
    --select-best \
    --seed 31

  speak "Ryan / stable / 2-variants qa scoreboard / EN" \
    --speaker Ryan \
    --text "The data shows a clear and consistent improvement across all metrics measured." \
    --instruct-style serious_doc \
    --language English \
    --profile stable \
    --variants 2 \
    --qa \
    --seed 32

else
  skip_test "test-ryan --tone calm 2-variants select-best (SKIP_SLOW=1)"
  skip_test "Ryan 3-variants select-best --instruct (SKIP_SLOW=1)"
  skip_test "Ryan 2-variants qa scoreboard (SKIP_SLOW=1)"
fi

print_summary "15-variants-cv"
