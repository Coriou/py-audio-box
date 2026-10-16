#!/usr/bin/env bash
# Intentional real CPU inference: same fixed public pair; no model downloads.
set -euo pipefail
task_here="$(cd "$(dirname "$0")" && pwd)"
task_root="$(git -C "$task_here" rev-parse --show-toplevel)"
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash run-spread.sh pab-stt-debug-UNIQUE OWNED_FRESH_RECEIPT_DIR' >&2
  exit 2
fi
task_name="$1"
task_out="$2"
exec python3 "$task_here/supervise.py" --name "$task_name" --seconds 115 --out "$task_out" -- \
  docker run --name "$task_name" --network none --cpus 4 --memory 6g --pids-limit 256 \
  -e CT2_VERBOSE=1 -e CT2_PACKED_GEMM=1 -e OMP_PROC_BIND=spread -e OMP_PLACES=cores \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$task_here:/diagnostic:ro" -v "$task_root/work/stt-evaluation:/assets:ro" \
  sha256:fd86a754708e099477a8e624babe5e9880df365a100f18cdc784f55f917a038e \
  python /diagnostic/diagnose.py
