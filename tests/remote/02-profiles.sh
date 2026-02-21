#!/usr/bin/env bash
# tests/remote/02-profiles.sh
# Generation profiles: stable / balanced / expressive across multiple voices + languages.
# Mirrors tests/local/03-clone-profiles.sh + parts of 05-variants.sh
set -euo pipefail
cd /app
source tests/remote/lib/common.sh

echo "=== 02: profiles (stable · balanced · expressive) ==="

# English voices
speak "chalamet-en / stable / EN" \
  --voice chalamet-en \
  --text "Not every story has a happy ending. But every story has something worth telling." \
  --language English \
  --profile stable \
  --seed 10

speak "chalamet-en / balanced / EN" \
  --voice chalamet-en \
  --text "Not every story has a happy ending. But every story has something worth telling." \
  --language English \
  --profile balanced \
  --seed 10

speak "chalamet-en / expressive / EN" \
  --voice chalamet-en \
  --text "Not every story has a happy ending. But every story has something worth telling." \
  --language English \
  --profile expressive \
  --seed 10

# French voice across all profiles
speak "rascar-capac / stable / FR" \
  --voice rascar-capac \
  --text "Le monde appartient à ceux qui se lèvent tôt et qui osent rêver grand." \
  --language French \
  --profile stable \
  --seed 11

speak "rascar-capac / balanced / FR" \
  --voice rascar-capac \
  --text "Le monde appartient à ceux qui se lèvent tôt et qui osent rêver grand." \
  --language French \
  --profile balanced \
  --seed 11

speak "rascar-capac / expressive / FR" \
  --voice rascar-capac \
  --text "Le monde appartient à ceux qui se lèvent tôt et qui osent rêver grand." \
  --language French \
  --profile expressive \
  --seed 11

# David Attenborough: known for stable deep prose
speak "david-attenborough / stable / EN  (nature narration)" \
  --voice david-attenborough \
  --text "The forest floor is a place of quiet drama. Hidden beneath every leaf is a world of extraordinary complexity." \
  --language English \
  --profile stable \
  --seed 12

# Hikaru — another available voice
speak "hikaru-nakamura / balanced / EN" \
  --voice hikaru-nakamura \
  --text "The position is equal but the game is far from over. White has some interesting tries here." \
  --language English \
  --profile balanced \
  --seed 13

print_summary "02-profiles"
