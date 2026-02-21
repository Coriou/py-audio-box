#!/usr/bin/env bash
# tests/remote/04-instruct.sh
# CustomVoice: --instruct and --instruct-style delivery control.
# Mirrors tests/local/11-instruct.sh + tests/local/12-instruct-style.sh
set -euo pipefail
cd /app
source tests/remote/lib/common.sh

echo "=== 04: CustomVoice instruct + instruct-style ==="

# ── Explicit --instruct strings ────────────────────────────────────────────────

speak "Ryan / expressive / --instruct podcast-host / EN" \
  --speaker Ryan \
  --text "Welcome back! Today we cover the future of voice technology and what it means for all of us." \
  --instruct "Speak with warm energy, like a podcast host welcoming listeners back after a break." \
  --language English \
  --profile expressive \
  --seed 30

speak "Ryan / stable / --instruct technical-presenter / EN" \
  --speaker Ryan \
  --text "The attention mechanism computes a weighted sum of value vectors guided by query and key similarity." \
  --instruct "Speak clearly and precisely, like an engineer presenting at a technical conference. No filler words." \
  --language English \
  --profile stable \
  --seed 31

speak "Ryan / stable / --instruct audiobook-narrator / EN" \
  --speaker Ryan \
  --text "The night was dark and still. Only the distant rattle of a passing train broke the silence of the village." \
  --instruct "Read as a warm, unhurried audiobook narrator. Slight dramatic pause between sentences." \
  --language English \
  --profile stable \
  --seed 32

speak "Ryan / expressive / --instruct sports-commentator / EN" \
  --speaker Ryan \
  --text "And he scores! Right in the final second! The stadium erupts!" \
  --instruct "High energy sports commentary. Fast paced, excited, loud energy." \
  --language English \
  --profile expressive \
  --seed 33

speak "Ryan / balanced / --instruct meditation-guide / EN" \
  --speaker Ryan \
  --text "Take a deep breath. Let your thoughts settle. There is nowhere you need to be right now." \
  --instruct "Calm, slow, and soothing, like a meditation guide. Long natural pauses." \
  --language English \
  --profile balanced \
  --seed 34

# ── --instruct-style template shortcuts ───────────────────────────────────────

speak "Ryan / stable / --instruct-style serious_doc / EN" \
  --speaker Ryan \
  --text "The following section outlines the key principles of acoustic signal processing." \
  --instruct-style serious_doc \
  --language English \
  --profile stable \
  --seed 35

speak "Ryan / balanced / --instruct-style warm / EN" \
  --speaker Ryan \
  --text "It has been a wonderful journey together and we are grateful you joined us for every step." \
  --instruct-style warm \
  --language English \
  --profile balanced \
  --seed 36

speak "Ryan / expressive / --instruct-style excited / EN" \
  --speaker Ryan \
  --text "This is unbelievable! Everything we worked so hard for is finally happening right now!" \
  --instruct-style excited \
  --language English \
  --profile expressive \
  --seed 37

speak "Ryan / stable / --instruct-style audiobook / EN" \
  --speaker Ryan \
  --text "Chapter one. The night was dark and perfectly still, save for the soft sound of distant rainfall." \
  --instruct-style audiobook \
  --language English \
  --profile stable \
  --seed 38

# Non-Ryan voice with instruct
speak "chalamet-en / expressive / --instruct dramatic / EN" \
  --voice chalamet-en \
  --text "This is the moment everything changes. Are you ready?" \
  --instruct "Intense dramatic delivery. Slow, measured, with weight on every word." \
  --language English \
  --profile expressive \
  --seed 39

print_summary "04-instruct"
