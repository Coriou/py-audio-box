#!/usr/bin/env bash
# Task-owned named-volume/socket smoke only. No host sockets or user data.
set -euo pipefail
task_image='sha256:42008ccbab95ca5e7afbe52dd2335f041c0fc87e8fd7620c844b88b44bbbdece'
task_suffix="$(date +%s)-${RANDOM}"
task_volume="pab-research-uds-${task_suffix}"
task_server="pab-research-uds-server-${task_suffix}"
cleanup() { docker rm -f "$task_server" >/dev/null 2>&1 || true; docker volume rm "$task_volume" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker volume create "$task_volume" >/dev/null
docker run -d --name "$task_server" --network none --cpus 1 --memory 128m --read-only --cap-drop ALL --security-opt no-new-privileges -v "$task_volume:/ipc" --entrypoint python "$task_image" -u -c 'import socket; s=socket.socket(socket.AF_UNIX); s.bind("/ipc/probe.sock"); s.listen(1); s.settimeout(10); c,_=s.accept(); message=c.recv(128); assert message==b"synthetic-ping-v1"; c.sendall(b"synthetic-pong-v1"); c.close(); s.close(); print("SERVER_OK",flush=True)' >/dev/null
docker run --rm --network none --cpus 1 --memory 128m --read-only --cap-drop ALL --security-opt no-new-privileges -v "$task_volume:/ipc:ro" --entrypoint python "$task_image" -c 'import socket,time,json; deadline=time.monotonic()+8
while True:
 s=socket.socket(socket.AF_UNIX)
 try: s.connect("/ipc/probe.sock"); break
 except (FileNotFoundError,ConnectionRefusedError):
  s.close()
  if time.monotonic()>deadline: raise
  time.sleep(.1)
s.sendall(b"synthetic-ping-v1"); assert s.recv(128)==b"synthetic-pong-v1"; s.close(); print(json.dumps({"client":"ok","transport":"unix-domain socket","network":"none","client_volume":"read-only"}))'
task_exit="$(docker wait "$task_server")"
docker logs "$task_server"
test "$task_exit" = '0'
printf 'SERVER_EXIT=%s\n' "$task_exit"
