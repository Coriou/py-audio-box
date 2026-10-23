# P1B23 pinned Linux runtime comparison — 2026-10-23

**PASS for the predeclared bounded SYNTHETIC comparison only. Root and WIP are eligible for this small Linux health/focused comparison; no native-cause clearance, candidate acceptance, T2/C1/C2 or release GO.**

Sole repository write: this report. Entire ISOLATED-RUNTIME report/exact Darwin blocker, ENV-HEALTH summary and current ledger were read; release Dockerfile and both full locks inspected. No applicable ancestor/root/report-path AGENTS.md found. Initial broad parent find was interrupted before producing content. No delegation, code/test/dependency edits, commit, production service, ports, paid E2E, model/audio/GPU/assets download, fault-generation, native/heap/allocator/GC diagnostic, fork stress, host installation or host-limit change. Existing synthetic test bodies were unchanged. Memory search returned no relevant exact-lane record and supplied no facts.

## Frozen sources and provenance

- Root exact HEAD: `14fdf06c5f5994c1ded3b97724c255bdf58d5f34`; source diff against independently reviewed T1 `7b9cafdc2073eba52149ba46fa9253706ace3015` empty across service/worker/scripts/tests/clients/contracts. Root executable source only; no C candidate overlay.
- WIP: `f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf` at `/Users/ben/.txcode/worktrees/py-audio-box/lifetime-bounded-core-20261023`; clean, incomplete and unintegrated. Exact source diff to `9d09a627897c098c87a6fd6ef0672dec7e2dc12c` empty across the same scope.
- Guard `4ff625669024c33b0eeb3ecad40f26a2a71bfc99` and T2 `1d741cb328228e67f99a096e9d6a9c1acd1f01b6` not executed or modified.
- Git archives retained at `/tmp/pab-linux-runtime-comparison-20261023/corrected-setup`. Exact tracked service/tests/worker/scripts/clients/contracts plus pyproject.toml extracted from each committed archive, copied using `docker cp`, never a user repository bind mount. Per-file capture manifests: root 188 files, WIP 189 files; current working-source rehash has zero mismatches, HEADs unchanged, root original untracked status identical before report, WIP clean. Full git archives also retained, but only this source selection entered the container.
- Service lock SHA-256 `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea`: 32 pins/480 hashes. Release lock `1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad`: 42 pins/860 hashes. Provision used only release lock and normal PyPI, `--isolated --no-cache-dir --only-binary=:all: --require-hashes`; no build dependencies or source builds. All 42 installed versions match; pip check passes. Base image pip remains provisioning tooling, not an extra project pin. Archive authentication here is fresh pip hash enforcement, not repair of missing Darwin original archives.
- Tokenizer cache sole file `fb374d419588a4632f3f557e76b4b70aebbca790`, verified before copy: SHA-256 `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`. Copied from `/tmp/pab-k-prefifth-env-20261022/tokenizer-cache`; no tokenizer/model download.

## Docker preflight and owned targets

Docker context `orbstack`; client/server 29.8.2; client Darwin amd64, server Linux amd64, kernel `7.2.9-orbstack-00391-g2b298d05fbd4`, cgroup v2. Docker VM total RAM 16843059200 bytes, 12 CPUs. Read-only initial info reports 194 images/17 containers/1 running: contextual snapshot only, no global preservation claim. Existing workloads were never stopped or modified.

Host initial df: 14,280,580 KiB available (~13.62 GiB); free RAM 4,274,245 × 4096 bytes (~16.31 GiB), load 1.63/1.83/1.97. Conservative 2 GiB pull/dependency estimate leaves >4 GiB reserve. Exact pinned pull reused four existing layers, no cleanup. VM capacity measured inside the sole owned provision container before installing dependencies: 14,236,224 KiB available; MemAvailable 15,795,952 KiB, load 0/0/0. The earlier `orb -m docker ...` probe returned no output and established no VM capacity; container measurement is the actual receipt. Post-work host free space 13,962,924 KiB (~13.32 GiB), still >4 GiB.

`/tmp/pab-worker-w1-docker-lane.lock` initially absent; each driver acquired it atomically with mkdir and PID/evidence owner receipt. It was removed only by its owner after cleanup; absent at finish. At most one owned container existed at any time. Owned name `pab-linux-comparison-20261023` was absent before/after. Three sequential container instances used that same name: initial setup, corrected setup, final runtime. Full IDs and pre/post states are in inspect receipts. No volumes were created. Only exact owned containers removed with `docker rm` (no `-v`). Images retained; no prune/eviction/global maintenance.

Exact base: `python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83`. Requested index digest equals pull/RepoDigests receipt. Resolved platform `linux/amd64`; platform manifest `sha256:a66d3a463d0be3cc537c269b0d3418746dda512417a975559a0eaa8a6406baaf` from pinned index inspection; actual image ID `sha256:82cd6bca489be723b62b192752c133eb7c6acbcecaeca797628725d994998c5b`, 129014125 bytes. No fallback version/platform.

Owned derived image retained: `pab-linux-comparison:20261023`, ID `sha256:3a921bd0951289f29c6697dba7271adf42788b19628eac0213b856c196fdbda1`, 213264321 bytes; docker commit of exact pinned base plus hash-locked packages and both source snapshots. Local derived image has no registry RepoDigest; its ID is the local identity. This bypasses release Dockerfile's unrelated Node/studio build and network tokenizer acquisition, not its Python/package pins.

All container instances: `--cpus 2 --memory 2g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges`; lifetime command bounded to 1800 seconds, provisioning dependency command at most 600 seconds, each focused/collection command at most 180 seconds (health at most 60). No host/agent limit lowered. Final runtime adds `--network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m,mode=1777 --user 10001:10001`; no bind mounts, secrets, SSH/home/socket, host networking or ports. Only /tmp is workload-writable (Docker also supplies standard special mounts); source and packages are on the read-only root. Provision network was default bridge for normal PyPI only; final runtime had no network. Docker inspect confirms resource flags, empty user mounts and no published ports. MemorySwap was left at daemon default, not claimed swap-free.

## Setup failure retained and concrete correction

Initial package install succeeded in 54.554s. Setup `chmod -R a-w /snapshots /tokenizer-cache` exited 1 with `Operation not permitted`: copied files retained non-root ownership while all capabilities were dropped. No health/test command ran in that attempt. Driver stopped, preserved logs/inspect, stopped and removed only that owned container. Initial sleep PID1 stop took 10.557s; no post-stop inspect was captured, so do not infer its exit signal. This setup error is not a native crash.

One explicit setup correction used read-only runtime root as the source immutability boundary, removing the unnecessary chmod step. Dependency setup was repeated from the same pinned base with identical hash lock, with its own retained receipts (69.599s). No test/native failure was retried or hidden; no native/GC/environment waiver. Corrected idle PID1 handles SIGTERM; provision stopped with captured ExitCode 0/OOMKilled false. Runtime graceful stop returned in 0.131s, followed by exact-owned rm; its post-stop exit status was not captured.

## Predeclared matrix and actual results

`plan.json` was written before health execution: runtime identity/imports/42 pins/pip check, tiny owned SQLite CRUD/close, normal-plugin trivial pytest, root Core collection, WIP Core/Dispatcher/recovery collection; only if healthy, same small root/WIP selection and a WIP bounded-lifetime subset, maximum 200 executed cases. Exact focused selection was frozen in `focused-plan.json` before focused execution. No plugin-autoload suppression, tests changed, xfail/skip added, or environment plugin waiver. Bytecode and cacheprovider disabled only for read-only operation. Trivial pytest trace loads normal anyio plugin.

| Row | Actual result | Wall seconds including Docker exec |
| --- | --- | --- |
| health-identity | imports + 42 exact version checks PASS — exit 0 | 0.412 |
| health-pip-check | PASS, no broken requirements — exit 0 | 0.359 |
| health-sqlite | owned SQLite CRUD/close PASS — exit 0 | 0.136 |
| health-trivial | 1 passed — exit 0 | 0.909 |
| root-core-collection | 35 collected, no bodies — exit 0 | 1.916 |
| wip-collection | 1806 collected, no bodies — exit 0 | 2.824 |
| root-focused | 5 passed (pytest 2.15s) — exit 0 | 2.828 |
| wip-focused | 5 passed (pytest 2.33s) — exit 0 | 3.041 |
| wip-bounded-focused | 4 passed, 62 deselected (pytest 0.29s) — exit 0 | 0.949 |

Executed bodies: 14 project tests + 1 trivial test, all pass, no skips/failures or abnormal test exits/signals/timeouts. Collection of 1806 cases is not execution of a full campaign. Root/WIP common selection is identical; source working directory is the only changed input between those focused commands, with the same image/container/resources/network/dependencies. Separate fresh pytest processes avoid importing one snapshot into another. WIP extra covers capacity-before-effects, atomic final slot, nested/alias/sequential close lifecycle and identity/reentrant/stale ownership. It deliberately does not complete the missing matrix or rerun the historical 30-pass/1-corrected-unrerun harness campaign. No RLIMIT, ext4, SHM, peer-credential or other 19-mechanism platform row was selected; zero such rows qualified.

Runtime inventory (actual Linux; normal imports include jsonschema/rpds/pytest/tiktoken/cryptography/pydantic_core):

```json
{
  "python": "3.14.8 (main, Oct  6 2026, 02:01:53) [GCC 12.2.0]",
  "platform": "Linux-7.2.9-orbstack-00391-g2b298d05fbd4-x86_64-with-glibc2.36",
  "machine": "x86_64",
  "sqlite": "3.40.1",
  "ssl": "OpenSSL 3.0.22 25 Aug 2026",
  "config": "'--build=x86_64-linux-gnu' '--enable-loadable-sqlite-extensions' '--enable-optimizations' '--enable-option-checking=fatal' '--enable-shared' '--with-lto' '--with-ensurepip' 'build_alias=x86_64-linux-gnu'",
  "pins": {
    "annotated-doc": "0.0.5",
    "annotated-types": "0.8.0",
    "anyio": "4.15.1",
    "attrs": "26.1.0",
    "certifi": "2026.7.22",
    "cffi": "2.1.1",
    "charset-normalizer": "3.5.2",
    "click": "8.5.0",
    "cryptography": "50.0.2",
    "fastapi": "0.143.2",
    "h11": "0.16.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "httpx-sse": "0.4.3",
    "idna": "3.20",
    "iniconfig": "2.3.1",
    "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1",
    "mcp": "1.30.0",
    "opentelemetry-api": "1.45.1",
    "packaging": "26.3",
    "pluggy": "1.7.0",
    "pycparser": "3.11",
    "pydantic": "2.14.1",
    "pydantic-core": "2.50.1",
    "pydantic-settings": "2.16.0",
    "pygments": "2.21.0",
    "pyjwt": "2.15.1",
    "pytest": "9.1.1",
    "python-dotenv": "1.2.4",
    "python-multipart": "0.0.32",
    "referencing": "0.37.0",
    "regex": "2026.9.29",
    "requests": "2.34.2",
    "rpds-py": "2026.9.1",
    "sse-starlette": "3.5.0",
    "starlette": "1.8.0",
    "tiktoken": "0.14.0",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.4",
    "urllib3": "2.8.0",
    "uvicorn": "0.54.0"
  },
  "limits": {
    "7": [
      20480,
      1048576
    ],
    "6": [
      64228,
      64228
    ],
    "9": [
      -1,
      -1
    ]
  }
}
```

RLIMIT_NOFILE is key 7, NPROC key 6, AS key 9; these were observed, never changed. Container cgroup memory/CPU/PID bounds are separate from process RLIMIT.

## Evidence, exact commands and preservation

Evidence root `/tmp/pab-linux-runtime-comparison-20261023`; corrected attempt `/tmp/pab-linux-runtime-comparison-20261023/corrected-setup`. Drivers `/tmp/pab-linux-comparison-driver.py`, `/tmp/pab-linux-comparison-run.py`, `/tmp/pab-linux-comparison-corrected-setup.py` retained. `commands.json` and each `run-commands.json` preserve exact argv, exits and monotonic elapsed seconds; stdout/stderr separate. Complete small-evidence SHA-256 manifest `evidence-hashes.json`; source has separate per-file capture manifests. No secret environment dump or native crash report read.

Key log/capture SHA-256:

- `pull.stdout`: `c1e286a33515297f56f194a647529d406371ff9deac08572ad39515282af86c5`.
- `base-after.stdout`: `b51f0a098879ba5606bb32b1d529870b0c10c7e17e11c55444e81552c27201ae`.
- `platform-manifest.txt`: `05fb0379a0c6b48346fc9bb040c9e535edefa49b3b99d8d1916d98e7476772fb`.
- `locks.json`: `f1d83ff942caf988d4afe97d63483f8c50a1d6adc6e46a6734a0475396b66dd8`.
- `source-readonly.stderr`: `f5ecdb75f6842cd385dbb0909c8b8308ff61ebbccd3655fc790310311015b1f7`.
- `corrected-setup/root-manifest.json`: `0b6283ebb22d7595286f72a4c18affc92c55c82c405ef58e9ef714672f4dc3a5`.
- `corrected-setup/wip-manifest.json`: `57af2d343966cdd48a8c7fbfbd21a5946a88603fa55737239d1d7f755ee11aca`.
- `corrected-setup/provision-deps.stdout`: `bd9bbc017e8a89e3d9be8159767a8487962b0f709ba0e56cb87ad3f4726c4655`.
- `corrected-setup/health-identity.stdout`: `b15bf382083a006796ff3ff5d38b445091ba176bb062a61159726f33dcdcdec1`.
- `corrected-setup/health-trivial.stdout`: `fdd983d6a35e2c643e76d87bc07646af38c8bc69b7d9bd71f3d9d09b7a052249`.
- `corrected-setup/root-core-collection.stdout`: `10321ce19d9783b14002dc1090f3df414f56806944594298e408133eba73fa3b`.
- `corrected-setup/wip-collection.stdout`: `ef8114a15b70b4e8f5a40c4895a5a7a5010db58c7482d46114a888fea06193d2`.
- `corrected-setup/root-focused.stdout`: `f1a9bc6b96dda32c346e72d8d7b9451e43210a4aa9074bb137c8ea3577756233`.
- `corrected-setup/wip-focused.stdout`: `28cdcde17d75c7b7f512c611d564b7a48d4ec330ed01fd4138544e05400bcfb0`.
- `corrected-setup/wip-bounded-focused.stdout`: `395140bf91524428a16d1e77d222277add757d36b860e7af39527cad5fff9813`.
- `corrected-setup/run-commands.json`: `cc67b0d6adc457d38e86eb46a416f55f51742c5c6f2d5d51505936b730bf9c30`.
- `preservation-final.json`: `37d30cd1eeec5b580f68038464aaba5983e38b06d3d88217611144a654219385`.

Exact orchestrated commands (full Python inline health programs remain verbatim in command JSON/driver):

```sh
# preflight
git rev-parse HEAD  # exit=0, seconds=0.013
git status --porcelain=v1 -uall  # exit=0, seconds=0.089
docker context show  # exit=0, seconds=0.066
docker version  # exit=0, seconds=0.079
docker info --format '{{json .}}'  # exit=0, seconds=0.153
df -k /tmp  # exit=0, seconds=0.007
vm_stat  # exit=0, seconds=0.007
uptime  # exit=0, seconds=0.006
docker image inspect python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83  # exit=1, seconds=0.046
docker ps -a --filter name=pab-linux-comparison-20261023 --no-trunc  # exit=0, seconds=0.04
docker pull python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83  # exit=0, seconds=3.679
docker image inspect python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83  # exit=0, seconds=0.078
# initial setup
git -C /Users/ben/Projects/py-audio-box archive --format=tar -o /tmp/pab-linux-runtime-comparison-20261023/root.tar 14fdf06c5f5994c1ded3b97724c255bdf58d5f34  # exit=0, seconds=0.204
git -C /Users/ben/.txcode/worktrees/py-audio-box/lifetime-bounded-core-20261023 archive --format=tar -o /tmp/pab-linux-runtime-comparison-20261023/wip.tar f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf  # exit=0, seconds=0.201
git -C /Users/ben/.txcode/worktrees/py-audio-box/lifetime-bounded-core-20261023 diff 9d09a627897c098c87a6fd6ef0672dec7e2dc12c f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf -- service tests worker scripts clients contracts  # exit=0, seconds=0.01
docker image inspect pab-linux-comparison:20261023  # exit=1, seconds=0.074
docker create --name pab-linux-comparison-20261023 --cpus 2 --memory 2g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 sleep 1800  # exit=0, seconds=0.137
docker start pab-linux-comparison-20261023  # exit=0, seconds=0.288
docker exec pab-linux-comparison-20261023 sh -c 'df -k /; head -5 /proc/meminfo; cat /proc/loadavg'  # exit=0, seconds=0.131
docker cp /tmp/pab-linux-runtime-comparison-20261023/root/tests/release/requirements.lock pab-linux-comparison-20261023:/tmp/requirements.lock  # exit=0, seconds=0.048
docker exec pab-linux-comparison-20261023 python -m pip --isolated install --index-url https://pypi.org/simple --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock  # exit=0, seconds=54.554
docker exec pab-linux-comparison-20261023 mkdir -p /snapshots /tokenizer-cache  # exit=0, seconds=0.082
docker cp /tmp/pab-linux-runtime-comparison-20261023/root pab-linux-comparison-20261023:/snapshots/root  # exit=0, seconds=0.071
docker cp /tmp/pab-linux-runtime-comparison-20261023/wip pab-linux-comparison-20261023:/snapshots/wip  # exit=0, seconds=0.084
docker cp /tmp/pab-k-prefifth-env-20261022/tokenizer-cache/fb374d419588a4632f3f557e76b4b70aebbca790 pab-linux-comparison-20261023:/tokenizer-cache/fb374d419588a4632f3f557e76b4b70aebbca790  # exit=0, seconds=0.078
docker exec pab-linux-comparison-20261023 chmod -R a-w /snapshots /tokenizer-cache  # exit=1, seconds=0.08
docker inspect pab-linux-comparison-20261023  # exit=0, seconds=0.038
docker stop --time 10 pab-linux-comparison-20261023  # exit=0, seconds=10.557
docker rm pab-linux-comparison-20261023  # exit=0, seconds=0.141
docker ps -a --filter name=pab-linux-comparison-20261023 --no-trunc  # exit=0, seconds=0.046
df -k /tmp  # exit=0, seconds=0.007
# corrected setup and runtime
git -C /Users/ben/Projects/py-audio-box archive --format=tar -o /tmp/pab-linux-runtime-comparison-20261023/corrected-setup/root.tar 14fdf06c5f5994c1ded3b97724c255bdf58d5f34  # exit=0, seconds=0.128
git -C /Users/ben/.txcode/worktrees/py-audio-box/lifetime-bounded-core-20261023 archive --format=tar -o /tmp/pab-linux-runtime-comparison-20261023/corrected-setup/wip.tar f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf  # exit=0, seconds=0.144
git -C /Users/ben/.txcode/worktrees/py-audio-box/lifetime-bounded-core-20261023 diff 9d09a627897c098c87a6fd6ef0672dec7e2dc12c f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf -- service tests worker scripts clients contracts  # exit=0, seconds=0.013
docker image inspect pab-linux-comparison:20261023  # exit=1, seconds=0.078
docker create --name pab-linux-comparison-20261023 --cpus 2 --memory 2g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 python -c 'import signal,time; signal.signal(signal.SIGTERM, lambda *a: exit(0)); time.sleep(1800)'  # exit=0, seconds=0.288
docker start pab-linux-comparison-20261023  # exit=0, seconds=0.186
docker exec pab-linux-comparison-20261023 sh -c 'df -k /; head -5 /proc/meminfo; cat /proc/loadavg'  # exit=0, seconds=0.132
docker cp /tmp/pab-linux-runtime-comparison-20261023/corrected-setup/root/tests/release/requirements.lock pab-linux-comparison-20261023:/tmp/requirements.lock  # exit=0, seconds=0.079
docker exec pab-linux-comparison-20261023 python -m pip --isolated install --index-url https://pypi.org/simple --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock  # exit=0, seconds=69.599
docker exec pab-linux-comparison-20261023 mkdir -p /snapshots /tokenizer-cache  # exit=0, seconds=0.082
docker cp /tmp/pab-linux-runtime-comparison-20261023/corrected-setup/root pab-linux-comparison-20261023:/snapshots/root  # exit=0, seconds=0.082
docker cp /tmp/pab-linux-runtime-comparison-20261023/corrected-setup/wip pab-linux-comparison-20261023:/snapshots/wip  # exit=0, seconds=0.133
docker cp /tmp/pab-k-prefifth-env-20261022/tokenizer-cache/fb374d419588a4632f3f557e76b4b70aebbca790 pab-linux-comparison-20261023:/tokenizer-cache/fb374d419588a4632f3f557e76b4b70aebbca790  # exit=0, seconds=0.084
docker stop --time 10 pab-linux-comparison-20261023  # exit=0, seconds=0.255
docker commit pab-linux-comparison-20261023 pab-linux-comparison:20261023  # exit=0, seconds=2.52
docker inspect pab-linux-comparison-20261023  # exit=0, seconds=0.073
docker rm pab-linux-comparison-20261023  # exit=0, seconds=0.129
docker image inspect pab-linux-comparison:20261023  # exit=0, seconds=0.04
docker create --name pab-linux-comparison-20261023 --cpus 2 --memory 2g --pids-limit 256 --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m,mode=1777 --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges -e PYTHONDONTWRITEBYTECODE=1 -e TIKTOKEN_CACHE_DIR=/tokenizer-cache pab-linux-comparison:20261023 python -c 'import signal,time; signal.signal(signal.SIGTERM, lambda *a: exit(0)); time.sleep(1800)'  # exit=0, seconds=0.254
docker start pab-linux-comparison-20261023  # exit=0, seconds=0.139
docker inspect pab-linux-comparison-20261023  # exit=0, seconds=0.04
docker exec pab-linux-comparison-20261023 python -c '<verbatim inline program in run-commands.json>'  # exit=0, seconds=0.412
docker exec pab-linux-comparison-20261023 python -m pip check  # exit=0, seconds=0.359
docker exec pab-linux-comparison-20261023 python -c 'import sqlite3; c=sqlite3.connect('"'"'/tmp/health.sqlite'"'"'); c.execute('"'"'create table t (v)'"'"'); c.execute('"'"'insert into t values (1)'"'"'); c.commit(); assert c.execute('"'"'select v from t'"'"').fetchone()==(1,); c.execute('"'"'delete from t'"'"'); c.commit(); c.close(); print('"'"'CRUD close PASS'"'"')'  # exit=0, seconds=0.136
docker exec pab-linux-comparison-20261023 python -c 'from pathlib import Path; Path('"'"'/tmp/test_health.py'"'"').write_text('"'"'def test_health(): assert 2+2==4\n'"'"')'  # exit=0, seconds=0.135
docker exec pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider --trace-config -q /tmp/test_health.py  # exit=0, seconds=0.909
docker exec -w /snapshots/root pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider --collect-only -q tests/service/test_core.py  # exit=0, seconds=1.916
docker exec -w /snapshots/wip pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider --collect-only -q tests/service/test_core.py tests/service/test_dispatcher.py tests/service/test_recovery.py  # exit=0, seconds=2.824
docker exec -w /snapshots/root pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider -q -rs tests/service/test_core.py::test_persisted_idempotency_conflict_scope tests/service/test_core.py::test_one_owner_and_lane tests/service/test_core.py::test_cas_and_history_no_inference tests/service/test_core.py::test_queue_cancel_without_worker tests/service/test_dispatcher.py::test_startup_normal_exact_receipt_two_restarts  # exit=0, seconds=2.828
docker exec -w /snapshots/wip pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider -q -rs tests/service/test_core.py::test_persisted_idempotency_conflict_scope tests/service/test_core.py::test_one_owner_and_lane tests/service/test_core.py::test_cas_and_history_no_inference tests/service/test_core.py::test_queue_cancel_without_worker tests/service/test_dispatcher.py::test_startup_normal_exact_receipt_two_restarts  # exit=0, seconds=3.041
docker exec -w /snapshots/wip pab-linux-comparison-20261023 python -m pytest -p no:cacheprovider -q -rs tests/service/test_core.py -k 'bounded_core_capacity_before_effects or bounded_core_atomic_last_slot or bounded_core_nested_roots_aliases_and_sequential or bounded_core_identity_active_reentrant_and_stale'  # exit=0, seconds=0.949
docker inspect pab-linux-comparison-20261023  # exit=0, seconds=0.044
docker stop --time 10 pab-linux-comparison-20261023  # exit=0, seconds=0.131
docker rm pab-linux-comparison-20261023  # exit=0, seconds=0.082
docker ps -a --filter name=pab-linux-comparison-20261023 --no-trunc  # exit=0, seconds=0.043
df -k /tmp  # exit=0, seconds=0.008
```

Ancillary read-only commands: exact-path instruction checks; rg/cat/sed/tail of reports/Dockerfile/locks; git status/rev-parse/diff/archive; shasum/hashlib; df/vm_stat/uptime; docker image inspect and pinned-index `docker buildx imagetools inspect`; no unrelated target mutations. Final git diff --check passes. Full tracked-source working-file hashes match the frozen archive manifests after all test execution. No claim of global Docker inventory preservation beyond owned-target before/after receipts: A1 unexplained drift remains unexplained.

## Eligibility and remaining gates

Ordinary Linux startup, imports, tiny SQLite, root collection, WIP collection and the declared focused subset are healthy on this exact Linux runtime. These receipts permit assessing this small comparison only; they do not authorize a broad campaign or source integration. Original archive provenance for Darwin remains incomplete. Exact Darwin 3.14.8 managed runtime remains unavailable under the prior owned-directory constraints; existing Darwin 3.14.7 and venv were never executed or modified here.

Linux changes OS/libc/process/filesystem envelope, Python patch/build/compiler, SQLite (3.40.1 here versus prior Darwin 3.53.4), SSL (3.0.22 here versus prior Darwin 3.6.4) and extension binaries. Therefore Linux PASS cannot identify or clear Darwin native causes. Prior V1/V2/V3, collection/fresh-helper SIGSEGV and unknown child exits remain unresolved. WIP remains incomplete/unqualified; guard HOLD and T2 NO-GO retained. All unrun actual Linux mechanisms, Apple Silicon, C1 mandatory `b2_pinned_launcher_context_unavailable`, B3/C1/C2, final independent/global Astra, hardware/browser/acoustic/activation and A1 inventory-drift gates remain mandatory. No production deployment, installed C1 host qualification or native diagnostic was performed.
