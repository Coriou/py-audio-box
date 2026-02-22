#!/usr/bin/env bash
# tests/remote/05-parallel-stress.sh
# Stress test: run N synthesis tasks simultaneously on the GPU.
#
# Each job is launched as a background process.  All PIDs are collected,
# then we wait for every job and track pass/fail per-PID.
#
# This exercises:
#   - GPU memory contention (multiple models sharing the same GPU)
#   - Concurrent disk I/O (reading voice profiles, writing WAVs)
#   - Scheduler fairness under load
#
# Set PARALLEL_N env var to control concurrency (default: 6).
# Set SKIP_SLOW=1 to run only the first 3 jobs.
set -euo pipefail
cd /app
source tests/remote/lib/common.sh

PARALLEL_N="${PARALLEL_N:-6}"

# Cap OpenMP threads per job to avoid exhausting the host thread limit.
# Default: floor(nproc / N), minimum 4, max 32.
_ncpu=$(nproc 2>/dev/null || echo 8)
_threads_per_job=$(( _ncpu / PARALLEL_N ))
[[ $_threads_per_job -lt 4  ]] && _threads_per_job=4
[[ $_threads_per_job -gt 32 ]] && _threads_per_job=32
export OMP_NUM_THREADS=$_threads_per_job
export MKL_NUM_THREADS=$_threads_per_job

echo "=== 05: parallel stress test (N=${PARALLEL_N}) ==="
echo "     Launching ${PARALLEL_N} synthesis jobs simultaneously…"
echo "     OMP_NUM_THREADS=${OMP_NUM_THREADS}  (nproc=${_ncpu})"
echo ""

# ── Job definitions ────────────────────────────────────────────────────────────
# Each entry: "LABEL|voice-synth-speak-args…"
# speak_bg handles background launch + per-job logging.

declare -a JOB_LABELS=()
declare -a JOB_PIDS=()
declare -a JOB_LOGS=()

LOG_DIR="$OUT/parallel-logs"
mkdir -p "$LOG_DIR"

launch_job() {
  local label="$1"; shift
  local safe="$(echo "$label" | tr '/ ' '__')"
  local logfile="$LOG_DIR/${safe}.log"

  JOB_LABELS+=("$label")
  JOB_LOGS+=("$logfile")

  echo "  [launch] $label"
  (
    local t0 t1 rc=0
    t0=$(date +%s)
    echo "=== $label ===" > "$logfile"
    echo "cmd: ./run-direct voice-synth speak --cache $CACHE --dtype $DTYPE --out $OUT $*" >> "$logfile"
    ./run-direct voice-synth speak \
      --cache "$CACHE" \
      --dtype "$DTYPE" \
      --out   "$OUT" \
      "$@" >> "$logfile" 2>&1 || rc=$?
    t1=$(date +%s)
    echo "" >> "$logfile"
    echo "exit=$rc  elapsed=$(( t1 - t0 ))s" >> "$logfile"
    exit $rc
  ) &
  JOB_PIDS+=($!)
}

# ── Launch all jobs ────────────────────────────────────────────────────────────

# Job 1 — rascar-capac, FR, stable
launch_job "rascar-capac/stable/FR/stress1" \
  --voice rascar-capac \
  --text "La synthèse vocale parallèle est un exercice fascinant. Plusieurs voix s'expriment simultanément sur le même processeur graphique." \
  --language French \
  --profile stable \
  --seed 50

# Job 2 — chalamet-en, EN, balanced
launch_job "chalamet-en/balanced/EN/stress2" \
  --voice chalamet-en \
  --text "Running multiple synthesis jobs in parallel tests the GPU scheduler and memory allocator under real concurrency pressure." \
  --language English \
  --profile balanced \
  --seed 51

# Job 3 — david-attenborough, EN, stable
launch_job "david-attenborough/stable/EN/stress3" \
  --voice david-attenborough \
  --text "The natural world operates in parallel at every scale, from the firing of neurons to the collapse of distant stars." \
  --language English \
  --profile stable \
  --seed 52

if [[ "${SKIP_SLOW:-0}" != "1" ]]; then

  # Job 4 — hikaru-nakamura, EN, expressive
  launch_job "hikaru-nakamura/expressive/EN/stress4" \
    --voice hikaru-nakamura \
    --text "Genius is not about knowing all the moves. It is about seeing the position clearly when time is running out." \
    --language English \
    --profile expressive \
    --seed 53

  # Job 5 — rascar-capac, FR, expressive (different profile from job 1)
  launch_job "rascar-capac/expressive/FR/stress5" \
    --voice rascar-capac \
    --text "C'est incroyable! Jamais je n'aurais imaginé que cette technologie serait aussi rapide, aussi naturelle, aussi convaincante!" \
    --language French \
    --profile expressive \
    --seed 54

  # Job 6 — chalamet-en, EN, stable (different profile from job 2)
  if [[ $PARALLEL_N -ge 6 ]]; then
    launch_job "chalamet-en/stable/EN/stress6" \
      --voice chalamet-en \
      --text "Stability is the foundation of every great performance. Without it, even the most expressive art collapses." \
      --language English \
      --profile stable \
      --seed 55
  fi

fi  # SKIP_SLOW

# ── Wait for all jobs + collect results ───────────────────────────────────────

echo ""
echo "  Waiting for ${#JOB_PIDS[@]} background jobs…"
echo ""

T0=$SECONDS
declare -i par_pass=0 par_fail=0

for i in "${!JOB_PIDS[@]}"; do
  pid="${JOB_PIDS[$i]}"
  label="${JOB_LABELS[$i]}"
  logfile="${JOB_LOGS[$i]}"
  rc=0

  wait "$pid" || rc=$?

  # Extract elapsed from log tail
  elapsed=""
  if [[ -f "$logfile" ]]; then
    elapsed=$(grep -oE 'elapsed=[0-9]+s' "$logfile" | tail -1 | sed 's/elapsed=//')
  fi

  if [[ $rc -eq 0 ]]; then
    echo "  ✓ PASS  ${elapsed:+[${elapsed}]  }$label"
    results+=("PASS  ${elapsed:-?}    [parallel] $label")
    (( par_pass++ )) || true
    (( pass++     )) || true
  else
    echo "  ✗ FAIL  (rc=$rc)  $label"
    echo "    log: $logfile"
    # Print last 10 lines of log for quick diagnosis
    [[ -f "$logfile" ]] && tail -10 "$logfile" | sed 's/^/      /'
    results+=("FAIL  ${elapsed:-?}    [parallel] $label")
    (( par_fail++ )) || true
    (( fail++      )) || true
  fi
done

echo ""
printf "  Parallel batch complete in %dm%02ds  (pass=%d  fail=%d)\n" \
  "$(( (SECONDS - T0) / 60 ))" "$(( (SECONDS - T0) % 60 ))" \
  "$par_pass" "$par_fail"

# ── Show per-job logs on failure ───────────────────────────────────────────────
if [[ $par_fail -gt 0 ]]; then
  echo ""
  echo "  ── Failed job logs ──"
  for i in "${!JOB_PIDS[@]}"; do
    logfile="${JOB_LOGS[$i]}"
    label="${JOB_LABELS[$i]}"
    if grep -q "exit=[^0]" "$logfile" 2>/dev/null; then
      echo ""
      echo "  === $label ==="
      cat "$logfile"
    fi
  done
fi

print_summary "05-parallel-stress"
