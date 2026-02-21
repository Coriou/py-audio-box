#!/usr/bin/env bash
# tests/remote/run-all.sh
# Remote GPU test suite orchestrator.
# Mirrors tests/local/run-all.sh but runs on a GPU instance via ./run-direct.
#
# Usage (called via vast-deploy.sh as a raw command):
#   bash /app/tests/remote/run-all.sh
#
# Env overrides:
#   ONLY="01 03"      — run only these suite numbers
#   SKIP="05"         — skip these suite numbers
#   SKIP_SLOW=1       — skip long variant/parallel tests inside each suite
#   DTYPE=float16     — override dtype (default: bfloat16)
#   OUT=/work/foo     — override output directory (default: /work/remote-tests)
#   PARALLEL_N=4      — number of parallel jobs in suite 05 (default: 6)
#
# Exit code: 0 if all executed suites pass, 1 if any fail.

set -euo pipefail
cd /app

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

SUITES=(
  "01-basic-synth.sh"
  "02-profiles.sh"
  "03-chunking.sh"
  "04-instruct.sh"
  "05-parallel-stress.sh"
)

# ── Filtering helpers ─────────────────────────────────────────────────────────
_should_run() {
  local name="$1"
  local num="${name%%[-_]*}"  # "01" from "01-basic-synth.sh"

  if [[ -n "$ONLY" ]]; then
    for o in $ONLY; do
      [[ "$num" == "$o" ]] && return 0
    done
    return 1
  fi

  for s in $SKIP; do
    [[ "$num" == "$s" ]] && return 1
  done

  return 0
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     py-audio-box  —  remote GPU test suite               ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  DTYPE=%s  SKIP_SLOW=%s  PARALLEL_N=%s\n" \
  "${DTYPE:-bfloat16}" "${SKIP_SLOW:-0}" "${PARALLEL_N:-6}"
[[ -n "$ONLY" ]] && printf "  ONLY: %s\n" "$ONLY"
[[ -n "$SKIP" ]] && printf "  SKIP: %s\n" "$SKIP"
echo ""

# Export overridable vars so child scripts inherit them
export DTYPE="${DTYPE:-bfloat16}"
export SKIP_SLOW="${SKIP_SLOW:-0}"
export OUT="${OUT:-/work/remote-tests}"
export PARALLEL_N="${PARALLEL_N:-6}"
export CACHE="${CACHE:-/cache}"

mkdir -p "$OUT"

# Print GPU info upfront
echo "  GPU info:"
python3 -c "
import torch
print(f'    torch      : {torch.__version__}')
print(f'    cuda       : {torch.version.cuda}')
print(f'    device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'    vram       : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB' if torch.cuda.is_available() else '    vram       : N/A')
try:
    import flash_attn
    print(f'    flash-attn : {flash_attn.__version__}')
except ImportError:
    print('    flash-attn : not available (sdpa will be used)')
" 2>/dev/null || echo "    (torch not importable)"
echo ""

# ── Run loop ──────────────────────────────────────────────────────────────────
total_suites=0
pass_suites=0
fail_suites=0
skip_suites=0
failed_names=()
start_all=$(date +%s)

for suite_file in "${SUITES[@]}"; do
  suite_path="tests/remote/$suite_file"

  if ! _should_run "$suite_file"; then
    printf "  ── SKIP (filter)  %s\n" "$suite_file"
    (( skip_suites++ )) || true
    continue
  fi

  if [[ ! -f "$suite_path" ]]; then
    printf "  ── MISSING        %s  (expected at %s)\n" "$suite_file" "$suite_path"
    (( fail_suites++ )) || true
    failed_names+=("MISSING  $suite_file")
    continue
  fi

  (( total_suites++ )) || true

  t0=$(date +%s)
  printf "\n──────────────────────────────────────────────────────────\n"
  printf "  SUITE START: %s\n" "$suite_file"
  printf "──────────────────────────────────────────────────────────\n"

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
total_sec=$(( end_all - start_all ))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              REMOTE GPU SUITE RESULTS                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  suites run:   %d\n" "$total_suites"
printf "  pass:         %d\n" "$pass_suites"
printf "  fail:         %d\n" "$fail_suites"
printf "  skipped:      %d\n" "$skip_suites"
printf "  total time:   %dm%02ds\n" "$(( total_sec / 60 ))" "$(( total_sec % 60 ))"
echo ""

if [[ ${#failed_names[@]} -gt 0 ]]; then
  echo "  Failed suites:"
  for n in "${failed_names[@]}"; do
    echo "    $n"
  done
  echo ""
fi

echo "  Output: $OUT"
echo ""

[[ $fail_suites -eq 0 ]] && exit 0 || exit 1
