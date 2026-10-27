#!/usr/bin/env bash
# Single owned container per job; no host pytest, mounts, registry or fallback.
set -euo pipefail
mode=$1
diagnostics=$2
transfer=$3
shard=${4:-01}
mkdir -p "$diagnostics" "$transfer"
helper=scripts/ci_unit_shards.py
canonical_dir=${5:-$diagnostics}
# WORK_STARTED is monotonic and recorded before checkout/download by the workflow.
remaining() {
  python3 - "$WORK_STARTED" "$1" <<'TIME'
import math, sys, time
left=math.floor(float(sys.argv[1])+int(sys.argv[2])-time.monotonic())
assert left>0, 'shared work deadline expired'
print(left)
TIME
}
stamp() {
  python3 - "$diagnostics/phases.jsonl" "$1" <<'STAMP'
import json, os, sys, time
with open(sys.argv[1], 'a') as f:
    f.write(json.dumps(dict(phase=sys.argv[2],elapsed=time.monotonic()-float(os.environ['WORK_STARTED'])))+'\n')
STAMP
}
stamp runner-start
container_id=''
follower_pid=''
producer_pid=''
finish() {
  result=$?
  cleanup_result=0
  trap - EXIT INT TERM
  set +e
  if [[ -n "${producer_pid:-}" ]]; then
    kill -TERM "$producer_pid" 2>/dev/null
    wait "$producer_pid"
  fi
  if [[ -n "$container_id" ]]; then
    # Cancellation: stop only our container, then collect terminal state.
    running=$(timeout 10s docker inspect --format '{{.State.Running}}' "$container_id" 2>>"$diagnostics/diagnostics.stderr")
    if [[ "$running" == true ]]; then
      timeout 25s docker stop --time 10 "$container_id" >>"$diagnostics/cleanup.log" 2>&1
      running=$(timeout 10s docker inspect --format '{{.State.Running}}' "$container_id" 2>>"$diagnostics/diagnostics.stderr")
      if [[ "$running" == true ]]; then
        timeout 10s docker kill "$container_id" >>"$diagnostics/cleanup.log" 2>&1
      fi
      timeout 15s docker wait "$container_id" >>"$diagnostics/cleanup.log" 2>&1
    fi
    timeout 10s docker inspect --format '{"Status":{{json .State.Status}},"Running":{{json .State.Running}},"OOMKilled":{{json .State.OOMKilled}},"ExitCode":{{json .State.ExitCode}},"Error":{{json .State.Error}},"Pid":{{json .State.Pid}}}' "$container_id" >"$diagnostics/state.json" 2>>"$diagnostics/diagnostics.stderr"
    cat "$diagnostics/state.json"
    running=$(timeout 10s docker inspect --format '{{.State.Running}}' "$container_id" 2>>"$diagnostics/diagnostics.stderr")
  fi
  if [[ -n "$follower_pid" ]]; then
    # Terminal container state does not imply that log replay reached EOF.
    # Spend at most five seconds of cleanup/upload reserve on natural drain.
    drain_status=not-terminal
    if [[ "${running:-}" == false ]]; then
      drain_status=eof
      drain_deadline=$((SECONDS + 5))
      while kill -0 "$follower_pid" 2>/dev/null; do
        if (( SECONDS >= drain_deadline )); then
          drain_status=expired-truncated
          break
        fi
        sleep 0.1
      done
    fi
    if kill -0 "$follower_pid" 2>/dev/null; then
      kill -TERM "$follower_pid" 2>/dev/null
      # retain.py gives its owned logs child two seconds before KILL/reap.
      follower_deadline=$((SECONDS + 4))
      while kill -0 "$follower_pid" 2>/dev/null && (( SECONDS < follower_deadline )); do
        sleep 0.1
      done
      if kill -0 "$follower_pid" 2>/dev/null; then
        drain_status="$drain_status-helper-killed"
        kill -KILL "$follower_pid" 2>/dev/null
      fi
    fi
    wait "$follower_pid"
    follower_result=$?
    printf 'drain=%s follower_exit=%s\n' "$drain_status" "$follower_result" >"$diagnostics/log-drain.txt"
  fi
  if [[ -n "$container_id" && "${running:-}" == false ]]; then
    timeout 10s docker rm "$container_id" >>"$diagnostics/cleanup.log" 2>&1 || cleanup_result=125
  fi
  if [[ -n "$container_id" ]]; then
    [[ "${drain_status:-}" == eof && "${follower_result:-1}" == 0 && "$cleanup_result" == 0 ]] || cleanup_result=125
    python3 - "$diagnostics/state.json" <<'STATE'
import json, sys
s=json.load(open(sys.argv[1]))
assert s['Running'] is False and s['Pid']==0 and s['OOMKilled'] is False and not s['Error']
STATE
    [[ $? == 0 ]] || cleanup_result=125
    if [[ "$cleanup_result" == 0 ]]; then printf 'yes\n' >"$diagnostics/cleanup-ok.txt"; fi
  fi
  if [[ "$result" == 0 && "$cleanup_result" != 0 ]]; then result=$cleanup_result; fi
  python3 - "$diagnostics" <<'SIZE'
from pathlib import Path
import sys
assert sum(p.stat().st_size for p in Path(sys.argv[1]).rglob('*') if p.is_file()) <= 8*1024*1024
SIZE
  if [[ $? != 0 && "$result" == 0 ]]; then result=125; fi
  stamp cleanup-finished
  printf '%s\n' "$result" >"$diagnostics/step-exit-code.txt"
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

image="pab-unit-${GITHUB_SHA}"
retain() { python3 "$helper" retain "$diagnostics" "$@"; }
if [[ "$mode" == prepare ]]; then
  budget=$(remaining 1020)
  (( budget > 600 )) || exit 124
  python3 "$helper" retain "$diagnostics" build timeout --signal=TERM --kill-after=5s 600s docker build -f Dockerfile.tests -t "$image" . &
  producer_pid=$!
  wait "$producer_pid"
  producer_pid=''
  stamp build-finished
  iid=$(timeout 10s docker image inspect --format '{{.Id}}' "$image")
  export IMAGE_ID="$iid"
  # Both sides of the gzip pipe are capped before bytes reach disk/Docker.
  budget=$(remaining 1020); (( budget > 120 )) || exit 124
  python3 "$helper" retain "$diagnostics" save timeout --signal=TERM --kill-after=5s 120s bash -c '
    set -euo pipefail
    docker save "$1" | python3 scripts/ci_unit_shards.py limit 3221225472 "$3/archive-bytes.txt" | gzip -1 | python3 scripts/ci_unit_shards.py limit 1073741824 >"$2/image.tar.gz"
  ' _ "$iid" "$transfer" "$diagnostics" &
  producer_pid=$!; wait "$producer_pid"; producer_pid=''
  stamp transfer-finished
  python3 - "$transfer/image.tar.gz" "$diagnostics/identity.json" <<'IDENTITY'
import hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0,'.')
from scripts.ci_unit_shards import VERSION, SHARDS, SELECTION, digest, write, source_hash
p=Path(sys.argv[1]); h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(65536),b''):
        assert time.monotonic() < float(os.environ['WORK_STARTED'])+1020
        h.update(b)
write(sys.argv[2],dict(sha=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],image=os.environ['IMAGE_ID'],archive_sha=h.hexdigest(),archive_bytes=p.stat().st_size,version=VERSION,shards=len(SHARDS),selection_hash=digest(SELECTION),source_hash=source_hash()))
IDENTITY
  stamp identity-finished
  setup_limit=1020
  pytest_mode=collect
  test_limit=120
else
  cp "$canonical_dir/identity.json" "$diagnostics/identity.json"
  budget=$(remaining 300)
  # Verify compressed identity before loading; never pull or rebuild.
  python3 - "$transfer/image.tar.gz" "$diagnostics/identity.json" <<'VERIFY'
import hashlib, json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1]); identity=json.load(open(sys.argv[2])); h=hashlib.sha256()
assert identity['sha']==os.environ['GITHUB_SHA'] and identity['run']==os.environ['GITHUB_RUN_ID'] and identity['attempt']==os.environ['GITHUB_RUN_ATTEMPT']
assert p.stat().st_size==identity['archive_bytes']<=1073741824
with p.open('rb') as f:
    for b in iter(lambda:f.read(65536),b''):
        assert time.monotonic() < float(os.environ['WORK_STARTED'])+300
        h.update(b)
assert h.hexdigest()==identity['archive_sha']
VERIFY
  budget=$(remaining 300)
  python3 "$helper" retain "$diagnostics" load timeout --signal=TERM --kill-after=5s "${budget}s" bash -c '
    set -euo pipefail
    gzip -dc "$1/image.tar.gz" | python3 scripts/ci_unit_shards.py limit 3221225472 "$2/archive-bytes.txt" | docker load
  ' _ "$transfer" "$diagnostics" &
  producer_pid=$!; wait "$producer_pid"; producer_pid=''
  stamp transfer-finished
  iid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$diagnostics/identity.json")
  [[ $(timeout 10s docker image inspect --format '{{.Id}}' "$iid") == "$iid" ]] || exit 125
  remaining 300 >"$diagnostics/setup-remaining.txt"
  setup_limit=300
  pytest_mode=run
  test_limit=720
fi
identity=$(cat "$diagnostics/identity.json")
args=("$pytest_mode" --identity "$identity" --shard "$shard")
if [[ "$mode" == shard ]]; then
  hashes=$(python3 - "$canonical_dir" <<'HASH'
import sys
sys.path.insert(0,'.')
from scripts.ci_unit_shards import read,digest
from pathlib import Path
p=Path(sys.argv[1]); print(digest(read(p/'manifest.json')),digest(read(p/'plan.json')))
HASH
)
  read -r manifest_hash plan_hash <<<"$hashes"
  args+=(--manifest-hash "$manifest_hash" --plan-hash "$plan_hash")
fi
name="pab-unit-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${mode}-${shard}-${RANDOM}-${RANDOM}"
budget=$(remaining "$setup_limit"); (( budget <= 10 )) || budget=10
container_id=$(timeout "${budget}s" docker create --name "$name" --network none --cpus 2 --memory 1g --memory-swap 1g \
  --log-driver local --log-opt max-size=1m --log-opt max-file=2 \
  --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  "$iid" python -u -m scripts.ci_unit_shards "${args[@]}" 2>"$diagnostics/create.stderr")
budget=$(remaining "$setup_limit"); (( budget <= 10 )) || budget=10
timeout "${budget}s" docker start "$container_id" >"$diagnostics/start.stdout" 2>"$diagnostics/start.stderr"
stamp tests-started
remaining "$setup_limit" >/dev/null
python3 "$helper" retain "$diagnostics" container docker logs --follow "$container_id" &
follower_pid=$!
test_deadline=$((SECONDS + test_limit))
while :; do
  running=$(timeout 10s docker inspect --format '{{.State.Running}}' "$container_id" 2>>"$diagnostics/diagnostics.stderr") || exit 125
  [[ "$running" == true || "$running" == false ]] || exit 125
  [[ "$running" == true ]] || break
  if (( SECONDS >= test_deadline )) || ! remaining 1020 >/dev/null; then
    printf 'test watchdog expired; exit 124\n' >"$diagnostics/timeout.txt"
    exit 124
  fi
  kill -0 "$follower_pid" 2>/dev/null || exit 125
  sleep 2
done
timeout 15s docker wait "$container_id" >"$diagnostics/container-exit-code.txt" 2>"$diagnostics/wait.stderr"
stamp tests-finished
result=$(cat "$diagnostics/container-exit-code.txt")
[[ "$result" =~ ^[0-9]+$ && "$result" -le 255 ]] || exit 125
exit "$result"
