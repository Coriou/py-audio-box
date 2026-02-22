#!/usr/bin/env bash
# tests/remote/run-all.sh
# py-audio-box synthesis benchmark suite orchestrator.
#
# Runs on any supported target:
#   TARGET=gpu   (default) — inside a vast.ai / GPU container via run-direct
#   TARGET=cpu             — inside the CPU Docker container via ./run
#   TARGET=rog             — on the ROG GPU machine via run-direct
#
# Usage (inside container, called via vast-deploy.sh):
#   bash /app/tests/remote/run-all.sh
#
# Usage (local CPU benchmark from repo root):
#   TARGET=cpu bash tests/remote/run-all.sh
#
# Env overrides:
#   TARGET=cpu|gpu|rog    hardware profile (affects RUNNER + DTYPE defaults)
#   ONLY="01 03"          run only specific suite numbers
#   SKIP="05"             skip specific suite numbers
#   SKIP_SLOW=1           skip slow sections inside individual suites
#   DTYPE=float16         override default dtype
#   OUT=/work/foo         override output root directory
#   PARALLEL_N=4          parallel jobs in suite 05 (default: 6)
#
# Exit code: 0 if all executed suites pass, 1 if any fail.

set -euo pipefail

# Resolve repo root (works both inside container at /app and from host)
if [[ -d /app ]]; then
  cd /app
fi

# ── Target + env setup ────────────────────────────────────────────────────────
TARGET="${TARGET:-gpu}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "$TARGET" in
  cpu)
    export RUNNER="${RUNNER:-./run}"
    export DTYPE="${DTYPE:-float32}"
    export CACHE="${CACHE:-./cache}"
    export OUT="${OUT:-./work/bench-tests}"
    ;;
  rog|gpu|*)
    export RUNNER="${RUNNER:-./run-direct}"
    export DTYPE="${DTYPE:-bfloat16}"
    export CACHE="${CACHE:-/cache}"
    export OUT="${OUT:-/work/remote-tests}"
    ;;
esac

export TARGET SKIP_SLOW="${SKIP_SLOW:-0}" PARALLEL_N="${PARALLEL_N:-6}"
export BENCH_DIR="${BENCH_DIR:-$OUT/.bench_records}"

mkdir -p "$OUT" "$BENCH_DIR"
# Clear stale bench records from a previous run
rm -f "$BENCH_DIR"/*.json 2>/dev/null || true

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

SUITES=(
  "01-basic-synth.sh"
  "02-profiles.sh"
  "03-chunking.sh"
  "04-instruct.sh"
  "05-parallel-stress.sh"
  "06-long-form.sh"
)

# ── Filtering helpers ─────────────────────────────────────────────────────────
_should_run() {
  local name="$1"
  local num="${name%%[-_]*}"   # "01" from "01-basic-synth.sh"

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
echo "║     py-audio-box  —  synthesis benchmark suite           ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  TARGET=%s  DTYPE=%s  SKIP_SLOW=%s  PARALLEL_N=%s\n" \
  "$TARGET" "$DTYPE" "$SKIP_SLOW" "$PARALLEL_N"
[[ -n "$ONLY" ]] && printf "  ONLY: %s\n" "$ONLY"
[[ -n "$SKIP" ]] && printf "  SKIP: %s\n" "$SKIP"
printf "  Output → %s\n" "$OUT"
echo ""

# ── Full system snapshot ──────────────────────────────────────────────────────
echo "  Hardware / software:"
python3 - <<'PYEOF'
import os, platform, multiprocessing, subprocess

def _s(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"

# Software
try:
    import torch
    print(f"    torch      : {torch.__version__}")
    print(f"    cuda       : {torch.version.cuda or 'none'}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"    device     : {p.name}")
        print(f"    vram       : {p.total_memory//1024**3} GB")
        print(f"    sm_count   : {p.multi_processor_count}  (compute {p.major}.{p.minor})")
    else:
        print("    device     : CPU (no CUDA)")
    try:
        import flash_attn
        print(f"    flash-attn : {flash_attn.__version__}  (attn=flash_attention_2)")
    except Exception as e:
        print(f"    flash-attn : not available — {str(e).split(chr(10))[0][:80]}")
        print(f"    attn       : sdpa")
except ImportError:
    print("    torch      : not importable")

print(f"    python     : {platform.python_version()}")
print(f"    dtype      : {os.environ.get('DTYPE','?')}")

# Hardware
ncpu = multiprocessing.cpu_count()
cpu_model = "unknown"
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                cpu_model = line.split(":",1)[1].strip()
                break
except Exception:
    cpu_model = platform.processor()

try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024**2, 1)
                break
    print(f"    ram        : {ram_gb} GB")
except Exception:
    pass

print(f"    cpu        : {cpu_model[:60]}")
print(f"    cores      : {ncpu} logical")

# nvidia-smi details
smi = _s(["nvidia-smi", "--query-gpu=driver_version,power.default_limit,clocks.max.gr,clocks.max.mem",
           "--format=csv,noheader,nounits"])
if smi != "n/a":
    parts = [x.strip() for x in smi.split(",")]
    print(f"    driver     : {parts[0] if len(parts)>0 else '?'}")
    if len(parts) >= 4:
        print(f"    gpu_tdp    : {parts[1]} W  (max clocks: gr={parts[2]} MHz  mem={parts[3]} MHz)")
PYEOF

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
  suite_num="${suite_file%%[-_]*}"

  if ! _should_run "$suite_file"; then
    printf "  ── SKIP (filter)  %s\n" "$suite_file"
    (( skip_suites++ )) || true
    continue
  fi

  if [[ ! -f "$suite_path" ]]; then
    printf "  ── MISSING  %s  (expected at %s)\n" "$suite_file" "$suite_path"
    (( fail_suites++ )) || true
    failed_names+=("MISSING  $suite_file")
    continue
  fi

  (( total_suites++ )) || true

  # Export suite ID for benchmark record tagging
  export CURRENT_SUITE="$suite_num"

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

end_all=$(date +%s)
wall_s=$(( end_all - start_all ))

# ── Generate benchmark report ─────────────────────────────────────────────────
echo ""
python3 /app/tests/remote/lib/report.py \
  --bench-dir  "$BENCH_DIR" \
  --out-dir    "$OUT" \
  --target     "$TARGET" \
  --run-id     "$RUN_ID" \
  --wall-s     "$wall_s" \
  || true   # report errors are non-fatal

# ── Suite-level summary ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  SUITE RESULTS                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  suites run:   %d\n" "$total_suites"
printf "  pass:         %d\n" "$pass_suites"
printf "  fail:         %d\n" "$fail_suites"
printf "  skipped:      %d\n" "$skip_suites"
printf "  total time:   %dm%02ds\n" "$(( wall_s / 60 ))" "$(( wall_s % 60 ))"
echo ""

if [[ ${#failed_names[@]} -gt 0 ]]; then
  echo "  Failed suites:"
  for n in "${failed_names[@]}"; do
    echo "    $n"
  done
  echo ""
fi

printf "  Output: %s\n" "$OUT"
echo ""

[[ $fail_suites -eq 0 ]] && exit 0 || exit 1