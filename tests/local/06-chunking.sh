#!/usr/bin/env bash
# tests/local/06-chunking.sh
# --chunk flag: long text split into chunks and concatenated.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 06: --chunk (long text auto-split) ==="

speak "david-attenborough / stable / --chunk / EN" \
  --voice david-attenborough \
  --text "The ocean is the lifeblood of our planet. It covers seventy percent of the surface. It drives weather and regulates temperature. Life began in its waters three billion years ago. Today it faces extraordinary pressure from warming seas, plastic pollution, and overfishing. Yet it still holds secrets we have barely begun to uncover." \
  --language English \
  --profile stable \
  --chunk \
  --seed 10

speak "rascar-capac / stable / --chunk / FR" \
  --voice rascar-capac \
  --text "Le langage est bien plus qu'un outil de communication. Il facon notre pensee, structure notre memoire, et donne forme a nos emotions les plus intimes. Chaque mot porte en lui une histoire, un contexte, une culture entiere. C'est pour cela que certaines choses ne se traduisent pas. Elles restent, irreductibles, dans la langue qui les a fait naitre." \
  --language French \
  --profile stable \
  --chunk \
  --seed 11

speak "chalamet-en / expressive / --chunk + --qa / EN" \
  --voice chalamet-en \
  --text "I never expected any of this. Not the camera, not the lights, not the crowd. All I wanted was to tell stories worth listening to. And somehow that turned into a career. Into a life. Into something I can barely explain but feel in every single take." \
  --language English \
  --profile expressive \
  --chunk \
  --qa \
  --seed 12

print_summary "06-chunking"
