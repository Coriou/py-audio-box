#!/usr/bin/env bash
# tests/remote/lib/common.sh
# Shared helpers for the py-audio-box synthesis benchmark suite.
#
# Supports three runtimes — set TARGET before sourcing (or let run-all.sh do it):
#   TARGET=gpu   ./run-direct, bfloat16   (default – inside vast.ai container)
#   TARGET=cpu   ./run-direct, float32    (inside local Mac CPU Docker container)
#   TARGET=rog   ./run-direct, bfloat16   (ROG / other GPU machine, run directly)
#
# Key env vars:
#   TARGET        — benchmark label + driver preset (gpu / cpu / rog / custom)
#   RUNNER        — command prefix (inferred from TARGET; override freely)
#   DTYPE         — model dtype   (inferred from TARGET; override freely)
#   CACHE         — cache directory
#   OUT           — output directory
#   BENCH_DIR     — per-test JSON records (default: $OUT/.bench_records)
#   SKIP_SLOW     — skip slow variant tests (default: 0)
#   CURRENT_SUITE — set by run-all.sh before sourcing each suite

# ── Target presets ────────────────────────────────────────────────────────────
TARGET="${TARGET:-gpu}"

case "$TARGET" in
  cpu)
    RUNNER="${RUNNER:-./run-direct}"
    DTYPE="${DTYPE:-float32}"
    CACHE="${CACHE:-/cache}"
    OUT="${OUT:-/work/bench-tests}"
    ;;
  rog|gpu|*)
    RUNNER="${RUNNER:-./run-direct}"
    DTYPE="${DTYPE:-bfloat16}"
    CACHE="${CACHE:-/cache}"
    OUT="${OUT:-/work/remote-tests}"
    ;;
esac

BENCH_DIR="${BENCH_DIR:-$OUT/.bench_records}"
SKIP_SLOW="${SKIP_SLOW:-0}"

# Run from repo root (/app inside container, cwd on host)
if [[ -d /app ]]; then
  cd /app 2>/dev/null || true
fi

mkdir -p "$OUT" "$BENCH_DIR"

# ── Result counters (each suite script owns its own) ──────────────────────────
pass=0
fail=0
skip=0
results=()

# ── Internal benchmark helpers ────────────────────────────────────────────────

# _bench_mark — place a sentinel file so we can find take meta files written
# during the subsequent test.
_bench_mark() {
  touch /tmp/.bench_mark_$$
}

# _extract_take_meta MARK_FILE OUT_DIR
#   Finds the takes.meta.json written since MARK_FILE under OUT_DIR.
#   Echoes TAB-separated: rtf  audio_s  synth_s  load_s  n_chunks
_extract_take_meta() {
  local mark="$1" out_dir="$2"
  local meta
  meta=$(find "$out_dir" -name "takes.meta.json" -newer "$mark" 2>/dev/null \
         | head -1)
  if [[ -z "$meta" ]]; then
    echo -e "n/a\tn/a\tn/a\tn/a\tn/a"
    return
  fi
  python3 - "$meta" <<'PYEOF'
import json, sys
try:
    d     = json.load(open(sys.argv[1]))
    takes = d.get("takes", [])
    dur   = sum(t.get("duration_sec", 0) for t in takes)
    syn   = sum(t.get("synth_sec",    0) for t in takes)
    rtf   = round(syn / dur, 3) if dur > 0 else 0
    load  = d.get("load_sec", 0)
    nch   = d.get("n_chunks", 1)
    print(f"{rtf}\t{round(dur,2)}\t{round(syn,2)}\t{round(load,2)}\t{nch}")
except Exception:
    print("n/a\tn/a\tn/a\tn/a\tn/a")
PYEOF
}

# _bench_record SUITE LABEL PASS_01 ELAPSED_S MARK_FILE OUT_DIR [VOICE] [DTYPE_OVERRIDE]
_bench_record() {
  local suite="$1" label="$2" pass_val="$3" elapsed="$4"
  local mark="$5" out_dir="$6"
  local voice="${7:-}"
  local dtype_val="${8:-$DTYPE}"

  local IFS_OLD="$IFS"
  IFS=$'\t'
  read -r rtf audio_s synth_s load_s n_chunks \
        <<< "$(_extract_take_meta "$mark" "$out_dir")"
  IFS="$IFS_OLD"

  local safe_label
  safe_label="$(echo "${suite}__${label}" | tr '/ ' '___' | tr -cd '[:alnum:]_-')"
  local record_path="$BENCH_DIR/${safe_label}.json"

  python3 - \
    "$record_path" "$suite" "$label" "$pass_val" "$elapsed" \
    "$rtf" "$audio_s" "$synth_s" "$load_s" "$n_chunks" \
    "$voice" "$dtype_val" "$TARGET" \
    <<'PYEOF'
import json, sys
_, path, suite, label, pass_s, elapsed, rtf, audio, syn, load, nch, voice, dtype, target = sys.argv

def _f(v):
    try: return float(v)
    except: return None

def _i(v):
    try: return int(float(v))
    except: return None

record = {
    "suite":     suite,
    "label":     label,
    "pass":      pass_s == "1",
    "elapsed_s": _i(elapsed),
    "rtf":       _f(rtf),
    "audio_s":   _f(audio),
    "synth_s":   _f(syn),
    "load_s":    _f(load),
    "n_chunks":  _i(nch),
    "voice":     voice or None,
    "dtype":     dtype,
    "target":    target,
}
with open(path, "w") as f:
    json.dump(record, f, indent=2)
PYEOF
}

# ── Public test helpers ───────────────────────────────────────────────────────

# run_cmd LABEL CMD [ARGS...]
#   Run an arbitrary command, track timing + pass/fail, write bench record.
run_cmd() {
  local label="$1"; shift
  local suite="${CURRENT_SUITE:-unknown}"
  echo ""
  echo "============================================================"
  echo "  TEST: $label"
  echo "  CMD:  $*"
  echo "============================================================"

  _bench_mark
  local t0 t1 rc=0
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
    rc=1
    (( fail++ )) || true
  fi

  _bench_record "$suite" "$label" "$(( 1 - rc ))" "$(( t1 - t0 ))" \
    "/tmp/.bench_mark_$$" "$OUT"
}

# speak LABEL [voice-synth speak flags...]
#   Wrapper: injects --cache, --dtype, --out; extracts --voice/--speaker for tagging.
speak() {
  local label="$1"; shift
  local suite="${CURRENT_SUITE:-unknown}"

  # Extract voice/speaker label from flags for benchmark tagging
  local _v=""
  local _args=("$@")
  for (( _i=0; _i<${#_args[@]}; _i++ )); do
    if [[ "${_args[$_i]}" == "--voice" || "${_args[$_i]}" == "--speaker" ]]; then
      _v="${_args[$(( _i+1 ))]:-}"
      break
    fi
  done

  echo ""
  echo "============================================================"
  echo "  TEST: $label"
  echo "  CMD:  $RUNNER voice-synth speak --cache $CACHE --dtype $DTYPE --out $OUT $*"
  echo "============================================================"

  _bench_mark
  local t0 t1 rc=0
  t0=$(date +%s)

  if $RUNNER voice-synth speak \
      --cache   "$CACHE" \
      --dtype   "$DTYPE" \
      --out     "$OUT" \
      "$@" 2>&1; then
    t1=$(date +%s)
    echo "  PASS  ($(( t1 - t0 ))s)"
    results+=("PASS  $(( t1 - t0 ))s    $label")
    (( pass++ )) || true
  else
    t1=$(date +%s)
    echo "  FAIL  ($(( t1 - t0 ))s)"
    results+=("FAIL  $(( t1 - t0 ))s    $label")
    rc=1
    (( fail++ )) || true
  fi

  _bench_record "$suite" "$label" "$(( 1 - rc ))" "$(( t1 - t0 ))" \
    "/tmp/.bench_mark_$$" "$OUT" "$_v" "$DTYPE"
}

# launch_parallel LABEL LOG_FILE [voice-synth speak flags...]
#   Background synthesis job for the parallel stress test.
#   Writes bench record on completion; returns PID via stdout.
launch_parallel() {
  local label="$1" logfile="$2"; shift 2
  local suite="${CURRENT_SUITE:-05-parallel-stress}"

  echo "  [launch] $label"
  (
    local mark t0 t1 rc=0
    mark=$(mktemp /tmp/.bench_mark_XXXXXX)
    t0=$(date +%s)
    echo "=== $label ===" > "$logfile"
    echo "cmd: $RUNNER voice-synth speak --cache $CACHE --dtype $DTYPE --out $OUT $*" >> "$logfile"
    $RUNNER voice-synth speak \
      --cache "$CACHE" \
      --dtype "$DTYPE" \
      --out   "$OUT" \
      "$@" >> "$logfile" 2>&1 || rc=$?
    t1=$(date +%s)
    echo "" >> "$logfile"
    echo "exit=$rc  elapsed=$(( t1 - t0 ))s" >> "$logfile"
    _bench_record "$suite" "[parallel] $label" "$(( 1 - rc ))" "$(( t1 - t0 ))" \
      "$mark" "$OUT"
    rm -f "$mark"
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

# print_summary SUITE_NAME
#   Prints the pass/fail table.  Exits 0 if all pass, 1 if any fail.
print_summary() {
  local name="$1"
  echo ""
  echo "===================================================="
  printf "  SUITE: %s\n" "$name"
  printf "  RESULTS  pass=%d  fail=%d  skip=%d  total=%d\n" \
    "$pass" "$fail" "$skip" "$(( pass + fail + skip ))"
  echo "===================================================="
  for r in "${results[@]}"; do
    echo "  $r"
  done
  echo ""
  [[ $fail -eq 0 ]]
}
