#!/usr/bin/env bash
# Run from the repository root. Only task-owned container names and paths.
set -uo pipefail
task_repo_root="$(pwd)"
task_harness="$task_repo_root/docs/agent-interface/evaluation/stt"
task_assets="$task_repo_root/work/stt-evaluation"
task_results="$task_harness/results"
task_image="py-audio-box:stt-evaluation-20261015"
mkdir -p "$task_results"
docker image inspect "$task_image" --format '{{json .}}' | jq '{Id,Size,Architecture,Os,Created}' > "$task_results/image.json"
for task_model in small turbo; do
  task_container="pab-stt-${task_model}-20261015"
  if docker container inspect "$task_container" >/dev/null 2>&1; then
    echo "Task container already exists; refusing to repeat: $task_container" >&2
    exit 1
  fi
  if [ -e "$task_results/$task_model.jsonl" ]; then
    echo "Result already exists; refusing to append/repeat: $task_model" >&2
    exit 1
  fi
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$task_results/$task_model.started"
  docker run --name "$task_container" --network none --cpus 4 --memory 6g --pids-limit 256 \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$task_harness:/evaluation:ro" -v "$task_assets:/assets:ro" -v "$task_results:/results" \
    "$task_image" timeout --signal=TERM --kill-after=5s 295s \
    python /evaluation/benchmark.py --model "$task_model" --threads 4 \
    > "$task_results/$task_model.stdout.log" 2> "$task_results/$task_model.stderr.log"
  task_exit=$?
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$task_results/$task_model.ended"
  docker inspect "$task_container" --format '{{json .}}' | jq '{Id,State,limits:{NanoCpus:.HostConfig.NanoCpus,Memory:.HostConfig.Memory,PidsLimit:.HostConfig.PidsLimit,NetworkMode:.HostConfig.NetworkMode},image:.Image}' > "$task_results/$task_model.container.json"
  printf '%s\n' "$task_exit" > "$task_results/$task_model.exit-code"
  docker rm "$task_container" >/dev/null
  echo "$task_model exit=$task_exit"
done
