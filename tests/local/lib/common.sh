#!/usr/bin/env bash
# tests/local/lib/common.sh
# Shared helpers for all local test scripts.
# Source this file at the top of each test script:
#   source "$(dirname "$0")/lib/common.sh"  # from tests/local/
#   source "$(dirname "$0")/../lib/common.sh"  # from a nested dir

# ── Runtime configuration ─────────────────────────────────────────────────────
CACHE="${CACHE:-/cache}"
OUT="${OUT:-/work/local-tests}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
DTYPE="${DTYPE:-float32}"
SKIP_SLOW="${SKIP_SLOW:-0}"
SKIP_DESIGN="${SKIP_DESIGN:-0}"

# ── Result counters (each script owns its own) ────────────────────────────────
pass=0
fail=0
skip=0
results=()

# ── Helpers ───────────────────────────────────────────────────────────────────

# run_cmd LABEL CMD [ARGS...]
#   Run an arbitrary command, track timing + pass/fail.
run_cmd() {
  local label="$1"; shift
  echo ""
  echo "============================================================"
  echo "  TEST: $label"
  echo "  CMD:  $*"
  echo "============================================================"
  local t0 t1
  t0=$(date +%s)
  if "$@" 2>&1; then
    t1=$(date +%s)
    echo "  PASS  ($(( t1 - t0 ))s)"
    results+=("PASS  $(( t1 - t0 ))s    $label")
    (( pass++ )) || true
  else
    t1=$(date +%s)
    echo "  FAIL  ($(( t1 - t0 ))s)"
    results+=("FAIL  $(( t1 - t0 ))s    $label")
    (( fail++ )) || true
  fi
}

# speak LABEL [voice-synth speak flags...]
#   Convenience wrapper: injects --cache, --threads, --dtype, --out.
speak() {
  local label="$1"; shift
  run_cmd "$label" \
    ./run voice-synth speak \
      --cache   "$CACHE" \
      --threads "$THREADS" \
      --dtype   "$DTYPE" \
      --out     "$OUT" \
      "$@"
}

# skip_test LABEL
#   Record a skip without running anything.
skip_test() {
  echo ""; echo "  SKIP  $1"
  results+=("SKIP         $1")
  (( skip++ )) || true
}

# print_summary [SUITE_LABEL]
#   Print the results table and exit 1 if any failures.
print_summary() {
  local suite="${1:-}"
  echo ""
  echo "===================================================="
  [[ -n "$suite" ]] && echo "  SUITE: $suite"
  printf "  RESULTS  pass=%d  fail=%d  skip=%d  total=%d\n" \
    "$pass" "$fail" "$skip" "$(( pass + fail + skip ))"
  echo "===================================================="
  for r in "${results[@]}"; do echo "  $r"; done
  echo ""
  [[ $fail -eq 0 ]] && exit 0 || exit 1
}
