#!/usr/bin/env bash
# tests/local/11-instruct.sh
# CustomVoice: explicit --instruct string for delivery control.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 11: CustomVoice — explicit --instruct ==="

speak "Ryan / expressive / --instruct podcast host / EN" \
  --speaker Ryan \
  --text "Welcome back! Today we cover the future of voice technology and what it means for all of us." \
  --instruct "Speak with warm energy, like a podcast host welcoming listeners back after a break." \
  --language English \
  --profile expressive \
  --seed 2

speak "Ryan / stable / --instruct technical presenter / EN" \
  --speaker Ryan \
  --text "The attention mechanism computes a weighted sum of value vectors guided by query and key similarity." \
  --instruct "Speak clearly and precisely, like an engineer presenting at a technical conference. No filler words." \
  --language English \
  --profile stable \
  --seed 3

speak "Ryan / stable / --instruct audiobook narrator / EN" \
  --speaker Ryan \
  --text "The night was dark and still. Only the distant rattle of a passing train broke the silence of the village." \
  --instruct "Read as a warm, unhurried audiobook narrator. Slight dramatic pause between sentences." \
  --language English \
  --profile stable \
  --seed 4

speak "Ryan / expressive / --instruct sports commentator / EN" \
  --speaker Ryan \
  --text "And he scores! Right in the final second! The stadium erupts!" \
  --instruct "High energy sports commentary. Fast paced, excited, loud energy." \
  --language English \
  --profile expressive \
  --seed 5

speak "Ryan / balanced / --instruct meditation guide / EN" \
  --speaker Ryan \
  --text "Take a deep breath. Let your thoughts settle. There is nowhere you need to be right now." \
  --instruct "Calm, slow, and soothing, like a meditation guide. Long natural pauses." \
  --language English \
  --profile balanced \
  --seed 6

print_summary "11-instruct"
