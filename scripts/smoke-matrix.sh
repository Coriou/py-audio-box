#!/usr/bin/env bash
# scripts/smoke-matrix.sh
#
# Container smoke matrix for Phase 5 operational hardening.
# Runs:
#   1) capability probe (strict)
#   2) clone smoke (voice-clone self-test)
#   3) built-in direct smoke (voice-synth speak --speaker ...)
#   4) designed voice smoke (design-voice -> speak --voice ...)
#
# Usage:
#   ./scripts/smoke-matrix.sh
#   TOOLBOX_VARIANT=gpu ./scripts/smoke-matrix.sh
#   SMOKE_SKIP_DESIGN=1 ./scripts/smoke-matrix.sh
#   SMOKE_ENABLE_CUSTOMVOICE=1 ./scripts/smoke-matrix.sh

set -euo pipefail

SMOKE_CACHE="${SMOKE_CACHE:-/cache}"
SMOKE_WORK_DIR="${SMOKE_WORK_DIR:-/work/smoke}"
SMOKE_DTYPE="${SMOKE_DTYPE:-auto}"
SMOKE_THREADS="${SMOKE_THREADS:-8}"
SMOKE_ENABLE_CUSTOMVOICE="${SMOKE_ENABLE_CUSTOMVOICE:-1}"

SMOKE_BUILTIN_MODEL="${SMOKE_BUILTIN_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
SMOKE_BUILTIN_SPEAKER="${SMOKE_BUILTIN_SPEAKER:-Ryan}"
SMOKE_BUILTIN_TEXT="${SMOKE_BUILTIN_TEXT:-Hello from built-in voice smoke test.}"

SMOKE_DESIGN_MODEL="${SMOKE_DESIGN_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
SMOKE_CLONE_MODEL="${SMOKE_CLONE_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
SMOKE_DESIGN_VOICE="${SMOKE_DESIGN_VOICE:-smoke-designed}"
SMOKE_DESIGN_INSTRUCT="${SMOKE_DESIGN_INSTRUCT:-Calm documentary narrator, clear diction, warm timbre.}"
SMOKE_DESIGN_REF_TEXT="${SMOKE_DESIGN_REF_TEXT:-A steady breeze carried the scent of rain across the valley.}"
SMOKE_DESIGN_SPEAK_TEXT="${SMOKE_DESIGN_SPEAK_TEXT:-This line verifies designed voice synthesis in the smoke matrix.}"
SMOKE_SKIP_DESIGN="${SMOKE_SKIP_DESIGN:-0}"

run_tool() {
  export QWEN3_ENABLE_CUSTOMVOICE="${SMOKE_ENABLE_CUSTOMVOICE}"
  if [[ "${TOOLBOX_VARIANT:-cpu}" == "gpu" ]]; then
    TOOLBOX_VARIANT=gpu ./run "$@"
  else
    ./run "$@"
  fi
}

echo "==> [1/4] Capability probe (strict)"
run_tool voice-synth capabilities \
  --model "${SMOKE_BUILTIN_MODEL}" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}" \
  --require-runtime-speakers \
  --strict \
  --json

echo
echo "==> [2/4] Clone smoke (voice-clone self-test)"
run_tool voice-clone self-test \
  --cache "${SMOKE_CACHE}" \
  --out "${SMOKE_WORK_DIR}/clone" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}"

echo
echo "==> [3/4] Built-in direct smoke"
run_tool voice-synth list-speakers \
  --model "${SMOKE_BUILTIN_MODEL}" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}" \
  --json
run_tool voice-synth speak \
  --speaker "${SMOKE_BUILTIN_SPEAKER}" \
  --model "${SMOKE_BUILTIN_MODEL}" \
  --text "${SMOKE_BUILTIN_TEXT}" \
  --profile stable \
  --seed 0 \
  --variants 1 \
  --cache "${SMOKE_CACHE}" \
  --out "${SMOKE_WORK_DIR}/builtin" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}"

if [[ "${SMOKE_SKIP_DESIGN}" == "1" ]]; then
  echo
  echo "==> [4/4] Designed voice smoke skipped (SMOKE_SKIP_DESIGN=1)"
  echo "Smoke matrix completed (design phase skipped)."
  exit 0
fi

echo
echo "==> [4/4] Designed voice smoke (design -> speak)"
run_tool voice-synth design-voice \
  --instruct "${SMOKE_DESIGN_INSTRUCT}" \
  --ref-text "${SMOKE_DESIGN_REF_TEXT}" \
  --voice-name "${SMOKE_DESIGN_VOICE}" \
  --design-model "${SMOKE_DESIGN_MODEL}" \
  --clone-model "${SMOKE_CLONE_MODEL}" \
  --cache "${SMOKE_CACHE}" \
  --out "${SMOKE_WORK_DIR}/design" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}"
run_tool voice-synth speak \
  --voice "${SMOKE_DESIGN_VOICE}" \
  --text "${SMOKE_DESIGN_SPEAK_TEXT}" \
  --profile stable \
  --seed 0 \
  --variants 1 \
  --cache "${SMOKE_CACHE}" \
  --out "${SMOKE_WORK_DIR}/design" \
  --threads "${SMOKE_THREADS}" \
  --dtype "${SMOKE_DTYPE}"

echo
echo "Smoke matrix completed successfully."
