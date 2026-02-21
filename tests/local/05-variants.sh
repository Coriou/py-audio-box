#!/usr/bin/env bash
# tests/local/05-variants.sh
# --variants + --select-best and --variants + --qa scoreboard.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 05: variants + select-best + qa ==="

speak "chalamet-en / expressive / 3-variants select-best" \
  --voice chalamet-en \
  --text "This is the moment everything changes. Are you ready?" \
  --language English \
  --profile expressive \
  --variants 3 \
  --select-best \
  --seed 100

speak "david-attenborough / stable / 2-variants qa scoreboard" \
  --voice david-attenborough \
  --text "The forest was silent except for the distant call of birds." \
  --language English \
  --profile stable \
  --variants 2 \
  --qa \
  --seed 7

speak "chalamet-en-2 / balanced / qa single-take" \
  --voice chalamet-en-2 \
  --text "Sometimes the simplest questions are the hardest ones to answer honestly." \
  --language English \
  --profile balanced \
  --seed 7 \
  --qa

if [[ "$SKIP_SLOW" == "0" ]]; then
  speak "rascar-capac / stable / 2-variants select-best / FR" \
    --voice rascar-capac \
    --text "Le secret d'un bon recit, c'est de savoir ce qu'on ne dit pas." \
    --language French \
    --profile stable \
    --variants 2 \
    --select-best \
    --seed 200
else
  skip_test "rascar-capac 2-variants select-best / FR (SKIP_SLOW=1)"
fi

print_summary "05-variants"
