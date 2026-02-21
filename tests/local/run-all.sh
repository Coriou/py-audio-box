#!/usr/bin/env bash
# tests/local/run-all.sh
# Main orchestrator: run all local test scripts in sequence.
#
# Usage:
#   ./tests/local/run-all.sh                   # full suite
#   SKIP_SLOW=1   ./tests/local/run-all.sh     # skip slow variant/clone tests
#   SKIP_DESIGN=1 ./tests/local/run-all.sh     # skip design-voice (VoiceDesign model)
#   ONLY="01 03 10" ./tests/local/run-all.sh   # run specific suite numbers only
#   SKIP="18 19"    ./tests/local/run-all.sh   # skip specific suite numbers
#   TEST_AUDIO=/work/recording.wav ./tests/local/run-all.sh  # enable audio-source tests
#
# Exit code: 0 if all executed suites pass, 1 if any fail.

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

SUITES=(
  "01-cli-utils.sh"
  "02-voice-clone.sh"
  "03-clone-profiles.sh"
  "04-designed-clone.sh"
  "05-variants.sh"
  "06-chunking.sh"
  "07-text-file.sh"
  "08-save-profile.sh"
  "09-export-import.sh"
  "10-custom-voice.sh"
  "11-instruct.sh"
  "12-instruct-style.sh"
  "13-register-builtin.sh"
  "14-tones.sh"
  "15-variants-cv.sh"
  "16-design-voice.sh"
  "17-slow-clones.sh"
  "18-voice-split.sh"
  "19-voice-register.sh"
)

# ── Filtering helpers ─────────────────────────────────────────────────────────

_should_run() {
  local name="$1"
  local num="${name%%[-_]*}"  # "01" from "01-cli-utils.sh"

  # ONLY filter: run ONLY these numbers
  if [[ -n "$ONLY" ]]; then
    for o in $ONLY; do
      [[ "$num" == "$o" ]] && return 0
    done
    return 1
  fi

  # SKIP filter: skip these numbers
  for s in $SKIP; do
    [[ "$num" == "$s" ]] && return 1
  done

  return 0
}

# ── Run loop ──────────────────────────────────────────────────────────────────

total_suites=0
pass_suites=0
fail_suites=0
skip_suites=0
failed_names=()
start_all=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         py-audio-box  —  local test suite                ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  SKIP_SLOW=%s  SKIP_DESIGN=%s\n" "${SKIP_SLOW:-0}" "${SKIP_DESIGN:-0}"
[[ -n "$ONLY" ]] && printf "  ONLY: %s\n" "$ONLY"
[[ -n "$SKIP" ]] && printf "  SKIP: %s\n" "$SKIP"
echo ""

for suite_file in "${SUITES[@]}"; do
  suite_path="tests/local/$suite_file"

  if ! _should_run "$suite_file"; then
    printf "  ── SKIP (filter)  %s\n" "$suite_file"
    (( skip_suites++ )) || true
    continue
  fi

  if [[ ! -f "$suite_path" ]]; then
    printf "  ── MISSING        %s\n" "$suite_path"
    (( fail_suites++ )) || true
    failed_names+=("MISSING  $suite_file")
    continue
  fi

  (( total_suites++ )) || true

  t0=$(date +%s)
  printf "\n──────────────────────────────────────────────────────────\n"
  printf "  SUITE START: %s\n" "$suite_file"
  printf "──────────────────────────────────────────────────────────\n"

  # Export env flags so child scripts pick them up
  export SKIP_SLOW="${SKIP_SLOW:-0}"
  export SKIP_DESIGN="${SKIP_DESIGN:-0}"
  [[ -n "${TEST_AUDIO:-}" ]] && export TEST_AUDIO
  [[ -n "${TEST_AUDIO_LONG:-}" ]] && export TEST_AUDIO_LONG

  if bash "$suite_path"; then
    t1=$(date +%s)
    printf "\n  ✓ SUITE PASS  %s  (%ds)\n" "$suite_file" "$(( t1 - t0 ))"
    (( pass_suites++ )) || true
  else
    t1=$(date +%s)
    printf "\n  ✗ SUITE FAIL  %s  (%ds)\n" "$suite_file" "$(( t1 - t0 ))"
    (( fail_suites++ )) || true
    failed_names+=("FAIL     $suite_file")
  fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
end_all=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  FULL SUITE RESULTS                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  suites run:   %d\n" "$total_suites"
printf "  pass:         %d\n" "$pass_suites"
printf "  fail:         %d\n" "$fail_suites"
printf "  skipped:      %d\n" "$skip_suites"
printf "  total time:   %ds\n" "$(( end_all - start_all ))"
echo ""

if [[ ${#failed_names[@]} -gt 0 ]]; then
  echo "  Failed suites:"
  for n in "${failed_names[@]}"; do
    echo "    $n"
  done
  echo ""
fi

[[ $fail_suites -eq 0 ]] && exit 0 || exit 1
