#!/usr/bin/env bash
# tests/remote/lib/common.sh
# Shared helpers for remote test scripts running on a GPU instance.
#
# Key differences from tests/local/lib/common.sh:
#   - Uses ./run-direct instead of ./run (no Docker wrapper)
#   - dtype defaults to bfloat16 (GPU-optimised)
#   - No --threads flag (irrelevant on GPU)
#   - Output lands in /work/remote-tests/<suite>/

# ── Runtime configuration ─────────────────────────────────────────────────────
CACHE="${CACHE:-/cache}"
OUT="${OUT:-/work/remote-tests}"
DTYPE="${DTYPE:-bfloat16}"
SKIP_SLOW="${SKIP_SLOW:-0}"

# Ensure we always run from /app (where the repo is cloned on the instance)
cd /app 2>/dev/null || true

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
#   Convenience wrapper: injects --cache, --dtype, --out.
#   Unlike the local wrapper there is no --threads flag.
speak() {
  local label="$1"; shift
  run_cmd "$label" \
    ./run-direct voice-synth speak \
      --cache   "$CACHE" \
      --dtype   "$DTYPE" \
      --out     "$OUT" \
      "$@"
}

# speak_bg LABEL [voice-synth speak flags...]
#   Same as speak but launches in the background.  Returns the PID via stdout.
#   Used by the parallel stress test.
speak_bg() {
  local label="$1"; shift
  local log_dir="$OUT/parallel-logs"
  mkdir -p "$log_dir"
  local safe_label
  safe_label="$(echo "$label" | tr '/ ' '__')"
  local logfile="$log_dir/${safe_label}.log"

  echo ""
  echo "  [BG] $label"
  echo "  CMD: ./run-direct voice-synth speak --cache $CACHE --dtype $DTYPE --out $OUT $*"

  (
    local t0 t1 rc=0
    t0=$(date +%s)
    ./run-direct voice-synth speak \
      --cache "$CACHE" \
      --dtype "$DTYPE" \
      --out   "$OUT" \
      "$@" > "$logfile" 2>&1 || rc=$?
    t1=$(date +%s)
    echo "exit=$rc elapsed=$(( t1 - t0 ))s label=$label" >> "$logfile"
    exit $rc
  ) &
  echo $!
}

# skip_test LABEL
skip_test() {
  echo ""; echo "  SKIP  $1"
  results+=("SKIP         $1")
  (( skip++ )) || true
}

# print_summary [SUITE_LABEL]
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
