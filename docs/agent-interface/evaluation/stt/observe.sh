#!/usr/bin/env bash
# External sampler survives an inference Python timeout. Run beside run.sh.
set -uo pipefail
task_out="$(pwd)/docs/agent-interface/evaluation/stt/results"
task_seen=0
for task_tick in $(seq 1 65); do
  task_active=0
  for task_model in small turbo; do
    task_name="pab-stt-${task_model}-20261015"
    if [ "$(docker inspect "$task_name" --format '{{.State.Running}}' 2>/dev/null)" = true ]; then
      task_active=1
      task_seen=1
      date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$task_out/$task_model.external-memory.log"
      docker exec "$task_name" sh -c 'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/cpu.stat' >> "$task_out/$task_model.external-memory.log" 2>/dev/null || true
    fi
  done
  if [ "$task_active" -eq 0 ] && [ "$task_seen" -eq 1 ] && [ -f "$task_out/turbo.exit-code" ]; then exit 0; fi
  sleep 10
done
