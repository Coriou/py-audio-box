#!/usr/bin/env bash
# Optional third diagnostic, only after the two serial faster-whisper attempts.
set -uo pipefail
task_root="$(pwd)"
task_eval="$task_root/docs/agent-interface/evaluation/stt"
task_assets="$task_root/work/stt-evaluation"
task_out="$task_eval/results"
task_image="ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647"
task_name=pab-stt-cpp-20261015
if [ ! -f "$task_out/turbo.exit-code" ] || [ -e "$task_out/cpp.container.json" ]; then exit 1; fi
docker image inspect "$task_image" --format '{{json .}}' | jq '{Id,RepoDigests,Size,Architecture,Os,Created}' > "$task_out/cpp.image.json" || exit 1
task_hash=$(shasum -a 256 "$task_assets/models/whisper-cpp/ggml-tiny-q5_1.bin" | cut -d ' ' -f 1)
if [ "$task_hash" != 818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7 ]; then exit 1; fi
mkdir -p "$task_out/cpp"
jq -r '.samples as $s | ([$s[]|select(.language=="fr")][0]), ([$s[]|select(.language=="en")][0]), ($s[]|select(.kind!="natural_read_speech")), ([$s[]|select(.language=="fr")][1:]), ([$s[]|select(.language=="en")][1:]) | if type=="array" then .[] else . end | [.file,.id,.duration]|@tsv' "$task_assets/manifest.json" > "$task_out/cpp.input-order.tsv"
task_args=()
while IFS=$'\t' read -r task_file task_id task_duration; do
  task_args+=(--file "$task_file" --output-file "/results/cpp/$task_id")
done < "$task_out/cpp.input-order.tsv"
task_repeat_file=$(jq -r '.samples[]|select(.language=="fr")|.file' "$task_assets/manifest.json" | head -1)
task_args+=(--file "$task_repeat_file" --output-file /results/cpp/fr-warm-repeat)
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$task_out/cpp.started"
(
  for task_tick in $(seq 1 30); do
    if [ "$(docker inspect "$task_name" --format '{{.State.Running}}' 2>/dev/null)" = true ]; then
      date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$task_out/cpp.external-memory.log"
      docker exec "$task_name" sh -c 'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/cpu.stat' >> "$task_out/cpp.external-memory.log" 2>/dev/null || true
    fi
    sleep 10
  done
) &
task_observer=$!
docker run --name "$task_name" --network none --cpus 4 --memory 6g --pids-limit 256 \
  -v "$task_assets:/assets:ro" -v "$task_out:/results" \
  --entrypoint /usr/bin/timeout "$task_image" --signal=TERM --kill-after=5s 270s \
  /app/build/bin/whisper-cli --model /assets/models/whisper-cpp/ggml-tiny-q5_1.bin \
  --no-gpu --threads 4 --beam-size 5 --best-of 5 --language auto --no-context \
  --output-json "${task_args[@]}" 2>&1 \
  | perl -MTime::HiRes=time -ne 'printf "%.6f\t%s", time, $_' > "$task_out/cpp.timestamped.log"
task_exit=${PIPESTATUS[0]}
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$task_out/cpp.ended"
kill "$task_observer" 2>/dev/null || true
wait "$task_observer" 2>/dev/null || true
docker inspect "$task_name" --format '{{json .}}' | jq '{Id,State,limits:{NanoCpus:.HostConfig.NanoCpus,Memory:.HostConfig.Memory,PidsLimit:.HostConfig.PidsLimit,NetworkMode:.HostConfig.NetworkMode},image:.Image}' > "$task_out/cpp.container.json"
printf '%s\n' "$task_exit" > "$task_out/cpp.exit-code"
docker rm "$task_name" >/dev/null
echo "cpp exit=$task_exit"
