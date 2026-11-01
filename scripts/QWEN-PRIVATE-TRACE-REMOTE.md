# One-use private Qwen exception capture

Operator diagnostic only. The overlay captures `runtime_child` and `qwen_resident` stderr separately in the worker's 64 KiB `/tmp` tmpfs. Each 0600 tail is at most 8192 bytes; state is `open`, `complete`, or `error`. A live reader updates bytes and truncation. A retained descendant stderr FD can keep a reader alive; another launch of that role is refused until it retires. The preflight requires 32768 free bytes and four free inodes before each launch, accounting for both tails, states, and allocation overhead. Concurrent tmpfs consumers can still exhaust space. Never enlarge the quota. A pipe or sink fault can perturb the workload. This is not runtime, audio, CI, or release qualification.

## Inputs and gates

On the source machine, export the **clean HEAD** file, not the dirty R33 copy, to private storage:

```sh
set -eu
umask 077
HEAD=35899c25d3d8e4866f869ade1ec151be6f0a7090
git show "$HEAD:worker/runtime_supervisor.py" > "$PRIVATE_SOURCE_EXPORT"
test "$(git hash-object "$PRIVATE_SOURCE_EXPORT")" = "$(git rev-parse "$HEAD:worker/runtime_supervisor.py")"
shasum -a 256 "$PRIVATE_SOURCE_EXPORT" scripts/qwen_private_trace_overlay.py
```

Transfer those two exact files privately. Record their SHA-256 values on source and remote and compare expected values byte for byte. Keep raw traces, paths, credentials, transcripts, reference audio, and request text out of product logs and public artifacts.

On the authorized remote host, supply absolute private paths `PY311`, `TRACE_ROOT` (new), `SOURCE_FILE`, `OVERLAY_TOOL`, `CLONE_BUNDLE`, `RYAN_BUNDLE`, `REFERENCE_WAV`, `AUDITION_WAV`, `INSTALL_ROOT` (new), `CUSTODY` (new), and `PROVENANCE_RECEIPT`. Supply exact existing `CONTROL_IMAGE`, `HELPER_IMAGE`, `BASE_ID`, `BASE_TAG` (new), `HANDLE` (new), `PORT`, `RAM_BYTES`, `THREADS`, `REFERENCE_TEXT`, `AUDITION_TEXT`, `EXPECTED_SOURCE_SHA256`, and `EXPECTED_OVERLAY_SHA256`. The private receipt must contain two exact lines, `source_commit=35899c25d3d8e4866f869ade1ec151be6f0a7090` and `image_id=<BASE_ID>`, backed by the retained prior build output. Inspect that output as well as the receipt before proceeding. An image ID or label alone does not establish source provenance. If that receipt or the existing image is unavailable, use the source-build fallback below before continuing. Do not fabricate a receipt.

```sh
set -eu
umask 077
for v in PY311 TRACE_ROOT SOURCE_FILE OVERLAY_TOOL CLONE_BUNDLE RYAN_BUNDLE REFERENCE_WAV AUDITION_WAV INSTALL_ROOT CUSTODY PROVENANCE_RECEIPT CONTROL_IMAGE HELPER_IMAGE BASE_ID BASE_TAG HANDLE PORT RAM_BYTES THREADS REFERENCE_TEXT AUDITION_TEXT EXPECTED_SOURCE_SHA256 EXPECTED_OVERLAY_SHA256; do
  eval 'test -n "${'"$v"'-}"' || exit 1
done
for v in PY311 TRACE_ROOT SOURCE_FILE OVERLAY_TOOL CLONE_BUNDLE RYAN_BUNDLE REFERENCE_WAV AUDITION_WAV INSTALL_ROOT CUSTODY PROVENANCE_RECEIPT HANDLE; do
  eval 'value=${'"$v"'}'
  case "$value" in /*) ;; *) exit 1 ;; esac
done
test ! -e "$TRACE_ROOT" && test ! -e "$INSTALL_ROOT"
test ! -e "$HANDLE" && test ! -e "$CUSTODY"
test "$(git rev-parse HEAD)" = 35899c25d3d8e4866f869ade1ec151be6f0a7090
test -z "$(git status --porcelain --untracked-files=no)"
test "$PORT" -gt 8010 && test "$PORT" -le 65535
"$PY311" -c 'import socket,sys; assert sys.version_info[:2]==(3,11); s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1])))' "$PORT"
for p in "$PY311" "$SOURCE_FILE" "$OVERLAY_TOOL" "$CLONE_BUNDLE" "$RYAN_BUNDLE" "$REFERENCE_WAV" "$AUDITION_WAV" "$PROVENANCE_RECEIPT"; do test -f "$p" && test -r "$p" || exit 1; done
test "$(sha256sum "$SOURCE_FILE" | cut -d ' ' -f 1)" = "$EXPECTED_SOURCE_SHA256"
test "$(sha256sum "$OVERLAY_TOOL" | cut -d ' ' -f 1)" = "$EXPECTED_OVERLAY_SHA256"
test "$(docker image inspect "$BASE_ID" --format '{{.Id}}')" = "$BASE_ID"
"$PY311" - "$PROVENANCE_RECEIPT" "$BASE_ID" <<'PY'
import pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
assert lines == ["source_commit=35899c25d3d8e4866f869ade1ec151be6f0a7090", "image_id=" + sys.argv[2]]
PY
test "$(docker image inspect "$BASE_TAG" --format '{{.Id}}' 2>/dev/null || true)" = ""
"$PY311" -c 'import os,sys; assert sys.version_info[:2]==(3,11); print(os.path.realpath(sys.executable))' > /dev/null
test "$(sudo "$PY311" -c 'import os,sys; assert sys.version_info[:2]==(3,11); print(os.path.realpath(sys.executable))')" = "$("$PY311" -c 'import os,sys; print(os.path.realpath(sys.executable))')"
"$PY311" -m service.qwen_service_setup --help
"$PY311" -m clients.cli --help
"$PY311" -m clients.cli enroll --help
mkdir -m 700 "$TRACE_ROOT"
docker tag "$BASE_ID" "$BASE_TAG"
"$PY311" "$OVERLAY_TOOL" --source "$SOURCE_FILE" --image "$BASE_TAG" --output "$TRACE_ROOT/context"
"$PY311" -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$TRACE_ROOT/context/runtime_supervisor.py"
NEW_TAG="pab-worker-qwen:private-trace-$("$PY311" -c 'import secrets; print(secrets.token_hex(8))')"
test "$(docker image inspect "$NEW_TAG" --format '{{.Id}}' 2>/dev/null || true)" = ""
docker build --pull=false --no-cache -t "$NEW_TAG" "$TRACE_ROOT/context"
DEBUG_ID=$(docker image inspect "$NEW_TAG" --format '{{.Id}}')
test -n "$DEBUG_ID"
```

`docker build --network=none` limits `RUN` networking; it does not prevent `FROM` resolution. The explicit local base ID check and `--pull=false` gate the base. The generated Dockerfile must begin with `FROM`, without a backslash.

### Source-build fallback

If the retained source/image build evidence is unavailable, use this fallback in the existing **clean remote repository**, before the main gate block. Set `BUILD_EVIDENCE` to a new absolute private directory. This may download pinned dependencies on the authorized remote host; it does not download models. Existing sealed model bundles remain required. Do not run this locally.

```sh
set -eu
umask 077
test "$(git rev-parse HEAD)" = 35899c25d3d8e4866f869ade1ec151be6f0a7090
test -z "$(git status --porcelain)"
case "$BUILD_EVIDENCE" in /*) ;; *) exit 1 ;; esac
test ! -e "$BUILD_EVIDENCE"
mkdir -m 700 "$BUILD_EVIDENCE"
BUILD_TAG="pab-worker-qwen:trace-base-$("$PY311" -c 'import secrets; print(secrets.token_hex(8))')"
sha256sum service/Dockerfile.release worker/runtime_supervisor.py > "$BUILD_EVIDENCE/source.sha256"
git rev-parse HEAD > "$BUILD_EVIDENCE/source.commit"
printf '%s\n' 'docker build --platform linux/amd64 --pull=false --target worker-qwen -f service/Dockerfile.release' > "$BUILD_EVIDENCE/command"
docker build --platform linux/amd64 --pull=false --target worker-qwen \
  -f service/Dockerfile.release -t "$BUILD_TAG" . > "$BUILD_EVIDENCE/build.private" 2>&1
BASE_ID=$(docker image inspect "$BUILD_TAG" --format '{{.Id}}')
test -n "$BASE_ID"
PROVENANCE_RECEIPT="$BUILD_EVIDENCE/provenance.private"
printf 'source_commit=35899c25d3d8e4866f869ade1ec151be6f0a7090\nimage_id=%s\n' "$BASE_ID" > "$PROVENANCE_RECEIPT"
```

Return to the gates above. Control and helper must still be the verified images from the same source; this fallback only builds the worker.

## Single installation and retrieval

Use the actual setup options shown by help and source. Keep the prior measurement classification: use `--measurement "$MEASUREMENT"` only with an actual earlier receipt, otherwise use `--local-unqualified`. The following command shows the unqualified case. Ensure the cgroup, handle, root, and port are unused before invoking it; retain the setup output privately and stop on failure.

```sh
sudo "$PY311" -m service.qwen_service_setup \
  --handle "$HANDLE" --root "$INSTALL_ROOT" --custody "$CUSTODY" --port "$PORT" \
  --clone-bundle "$CLONE_BUNDLE" --ryan-bundle "$RYAN_BUNDLE" \
  --context default --control-image "$CONTROL_IMAGE" --worker-image "$DEBUG_ID" \
  --helper-image "$HELPER_IMAGE" --reference-wav "$REFERENCE_WAV" \
  --reference-text "$REFERENCE_TEXT" --ram-bytes "$RAM_BYTES" --threads "$THREADS" \
  --local-unqualified > "$TRACE_ROOT/setup.private" 2> "$TRACE_ROOT/setup-error.private"
"$PY311" -m clients.cli --installation "$INSTALL_ROOT" status > "$TRACE_ROOT/status.private"
NEW_WORKER_CONTAINER_ID=$("$PY311" - "$TRACE_ROOT/setup.private" <<'PYWORKER'
import json,re,sys
records=[]
for line in open(sys.argv[1]):
    try: row=json.loads(line)
    except ValueError: continue
    if isinstance(row,dict) and row.get('phase')=='ready': records.append(row)
assert len(records)==1
worker=records[0]['containers']['worker']
assert isinstance(worker,str) and re.fullmatch(r'[0-9a-f]{64}',worker)
print(worker)
PYWORKER
)
test "$(docker inspect "$NEW_WORKER_CONTAINER_ID" --format '{{.Image}}')" = "$DEBUG_ID"
test "$(docker inspect "$NEW_WORKER_CONTAINER_ID" --format '{{.State.Running}}')" = true
set +e
"$PY311" -m clients.cli --installation "$INSTALL_ROOT" enroll \
  --name private-trace-once --wait-seconds 600 --wav "$AUDITION_WAV" \
  --reference-text "$REFERENCE_TEXT" --audition-text "$AUDITION_TEXT" \
  > "$TRACE_ROOT/enroll.private" 2> "$TRACE_ROOT/enroll-error.private"
ENROLL_EXIT=$?
set -e
printf "%s\n" "$ENROLL_EXIT" > "$TRACE_ROOT/enroll-exit.private"
```

Use the exact previous audition text when reproducing. There is one enrollment and no automatic retry. Inspect setup output and Docker handle membership to identify `NEW_WORKER_CONTAINER_ID`; verify it belongs to the **new** handle before retrieval. Capture while the worker is running, before any stop. A stopped tmpfs is inaccessible; never claim `docker cp` from a stopped container recovered it.

```sh
for role in resident attempt; do
  if docker exec --user 0:0 "$NEW_WORKER_CONTAINER_ID" sh -c \
    'set -e; stat -c "%a %s %U:%G" "/tmp/qwen-private-$1.state" "/tmp/qwen-private-$1.tail"; cat "/tmp/qwen-private-$1.state"; cat "/tmp/qwen-private-$1.tail"' sh "$role" \
    > "$TRACE_ROOT/$role.private" 2> "$TRACE_ROOT/$role-read-error.private"; then
    chmod 600 "$TRACE_ROOT/$role.private" "$TRACE_ROOT/$role-read-error.private"
  else
    chmod 600 "$TRACE_ROOT/$role.private" "$TRACE_ROOT/$role-read-error.private"
    printf 'error retrieval incomplete\n' > "$TRACE_ROOT/$role-retrieval.state"
  fi
done
```

A missing file, `open` state, retrieval error, or `error` state means incomplete evidence. Summarize only sanitized exception type and frames, trace state/truncation, verified image/commit and new handle identities, and timing. Preserve existing ports 8000–8010, handles, volumes, and logs.
