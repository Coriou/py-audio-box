#!/usr/bin/env bash
# tests/local/03-clone-profiles.sh
# voice-synth speak: clone_prompt engine across profiles / languages / voices.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 03: clone_prompt - profiles x languages x voices ==="

# English voices

speak "david-attenborough / stable / EN" \
  --voice david-attenborough \
  --text "The open ocean remains one of the least explored environments on Earth." \
  --language English \
  --profile stable \
  --seed 1

speak "chalamet-en / expressive / EN" \
  --voice chalamet-en \
  --text "I did not sleep at all. I kept thinking what if this is actually real?" \
  --language English \
  --profile expressive \
  --seed 42

speak "chalamet-en-2 / balanced / EN" \
  --voice chalamet-en-2 \
  --text "Sometimes the simplest questions are the hardest ones to answer honestly." \
  --language English \
  --profile balanced \
  --seed 7

speak "hikaru-nakamura / balanced / temperature override / EN" \
  --voice hikaru-nakamura \
  --text "Every move has a purpose. Think three moves ahead to stay competitive." \
  --language English \
  --profile balanced \
  --temperature 0.8 \
  --seed 55

# French voices

speak "rascar-capac / stable / FR" \
  --voice rascar-capac \
  --text "Les secrets du passe ne disparaissent jamais. Ils attendent dans l'ombre." \
  --language French \
  --profile stable \
  --seed 3

speak "blitzstream / balanced / FR" \
  --voice blitzstream \
  --text "Voila, vous avez compris la strategie. On passe au niveau suivant." \
  --language French \
  --profile balanced \
  --seed 5

speak "blitzstream-2 / expressive / FR" \
  --voice blitzstream-2 \
  --text "Franchement c'est incroyable. J'aurais jamais pense que ca allait se passer comme ca." \
  --language French \
  --profile expressive \
  --seed 20

print_summary "03-clone-profiles"
