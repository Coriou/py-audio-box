# Actual CPU environment inventory — r233 sol

Observed 2026-10-24T18:48Z. **NO-GO for installed-host qualification here; conditional candidate for one isolated legacy CPU speech smoke. Actual host qualification NOT RUN.** No inference, network, installation, Docker mutation, remote connection, protected-proof reading or HELD resource operation occurred.

## Provenance

Root `/Users/ben/Projects/py-audio-box`; actual HEAD `3ebabd3549ab276b71b7d9ca2a620baad98e650f`, not requested `493a56e3d29904ff65717a3f4ce100b063afa9ea`. Requested commit is an ancestor. Difference is three documentation files, 112 insertions; release Dockerfile, requirements, service, worker, lib and HEADLESS-INSTALL are unchanged between these commits. Tracked working tree was clean; existing untracked evaluation artifacts were listed by status only, not opened. No checkout change or commit.

## Existing resources and limits

- Host: Darwin x86_64, 12 logical CPUs, 64 GiB RAM. Workspace filesystem: 466 GiB total, 447 GiB used, **3.9 GiB available**. This is host filesystem headroom, not measured Docker VM free space; no volume or VM-content inspection performed.
- Active Docker context `orbstack`; client/server 29.8.2. Daemon reports OrbStack, x86_64, 12 CPUs, 16,843,051,008 bytes RAM, cgroup v2, overlay2, seccomp and cgroup namespaces. This is a macOS VM daemon, **not an established local rootful Linux installer host**. Docker Desktop is also explicitly refused by the installer; OrbStack must not be treated as qualification-equivalent merely because its daemon reports Linux-compatible facilities.
- Read-only image listing found `py-audio-box:kokoro-cpu` / `:memory-release`, short ID `7e4b229332c3`, reported 779 MB; also older Kokoro performance/audit images. These are candidates only: no image contents, immutable full ID, runtime versions, source provenance or absence of historical ownership established. No release control/worker image pair was enrolled by this inventory. Historical images/containers/volumes remain unexamined and unavailable for implicit reuse. Docker calls were individually bounded to 12 seconds and completed successfully.
- `/Users/ben/.local/bin/python3.11` executes as CPython 3.11.16, x86_64; metadata reports pip 26.2.1 and no kokoro-onnx, onnxruntime, soundfile, mcp or fastapi in that interpreter. This is a Mac interpreter, not Linux wheel execution evidence. `/usr/local/bin/python3` also exists.
- Explicit wheelhouse checks: repository `wheelhouse/`, `wheels/`, `~/.cache/pip` absent; `~/Library/Caches/pip` contains only `selfcheck` at its top level, no wheels directory. No matching offline Linux CPython 3.11 wheelhouse identified. No opaque HTTP cache or broad home scan used for asset discovery.

## Cached Kokoro and current requirements

Documented cache root is repository `cache/` (README and docker-compose.yml); targeted `cache/tts/kokoro` contains only manifest plus model and voices. File stat sizes: `model.onnx` 325,532,387 bytes; `voices.bin` 28,214,398 bytes (353,746,785 combined). No large-file hashing performed; cached bytes are **not reverified**.

The 953-byte manifest records French, checkpoint `Kokoro-82M-v1.0-onnx-fp32`, kokoro-onnx 0.6.1, onnxruntime **1.29.0**, espeakng-loader 0.2.4, phonemizer 3.4.0. Recorded sizes match stat. Recorded hashes match `requirements/tts-assets.json`:

- model: `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5`
- voices: `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d`

Recorded origins are the official upstream `thewh1teagle/kokoro-onnx` GitHub release assets `model-files-v1.0/kokoro-v1.0.onnx` and `voices-v1.0.bin`. This is local metadata, not a fresh upstream verification. License text records Apache-2.0 weights / MIT runtime and requires selected voice provenance verification. Top-level revision and access metadata are absent. This legacy manifest is **not a managed expected manifest**.

Current `requirements/kokoro.lock` pins kokoro-onnx 0.6.1, onnxruntime **1.31.0**, espeakng-loader 0.2.4, phonemizer 3.4.0, soundfile 0.12.1, numpy 2.4.6 plus hashed transitive dependencies. Do not silently substitute cached manifest runtime 1.29.0. `service/requirements.in` pins fastapi 0.143.2, pydantic 2.14.1, uvicorn 0.54.0, httpx 0.28.1, mcp 1.30.0 and jsonschema 4.26.0; installed host dependencies need the hashed service lock and matching offline wheels.

Both `Dockerfile.cpu` and `service/Dockerfile.release` use Python 3.11 slim bookworm base digest `sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89`. CPU runtime requires libsndfile1, espeak-ng, ffmpeg; legacy image also tini. Release worker combines Kokoro and service locks. Dockerfile presence proves neither image availability nor successful build/runtime.

`service/provisioning.py:validate_expected` requires exact schema/backend/language/checkpoint/revision/license/access/versions/files, two exact assets with reviewed positive byte sizes, SHA256, HTTPS origin and per-file revision, and four exact runtime package versions. `requirements/tts-assets.json` explicitly says existing hashes do not qualify a real profile. Missing immutable top-level source revision, independent size/hash/origin review, license/access and preset provenance must be obtained from official upstream model/runtime sources later; no pin was invented here. Legacy `scripts/provision-tts.py` output alone does not meet this managed contract.

## Existing host references

`HEADLESS-INSTALL.md` requires trusted Python 3.11 on x86_64 Linux, local rootful Docker at `unix:///var/run/docker.sock`, ordinary local volumes, cgroup v2, no remapping and private PID/cgroup namespaces. It also requires exact images, sealed model/config/credential inputs, four distinct empty root-owned volume roots, protected authority and actual mount/ownership evidence. None is established here.

Makefile generic `REMOTE_HOST` defaults empty. Its concrete ROG configuration documents **Windows + WSL2 + Docker**, CUDA benchmarks and SSH transport, not a qualified rootful standalone Linux host. `.env.example` documents Vast GPU provisioning, which would risk paid resources and is outside scope. SSH config exists (symlink); only directive counts inspected: 10 Host, 8 HostName, one Include, 10 IdentityFile. Alias/endpoint/key values withheld; included files and keys not read. No host connected and no existing qualifying Linux rootful host identified from these references. A configured alias does not establish availability, cost, authorization or qualification.

## Next ONE bounded actual test proposal — not executed

Prefer a **single offline legacy CLI French sentence** using a newly named isolated Docker container, the existing Kokoro CPU image only after exact full image ID/provenance and non-HELD ownership are established. This is a proposal requiring a separately authorized execution lane; no historical container/volume reuse. If the candidate image belongs to HELD resources, refuse it and stop. Do not build/pull as fallback.

Prerequisites: verify cached asset hashes in the authorized lane; confirm image runtime metadata against the chosen legacy manifest (1.29.0 is not current 1.31.0 release evidence); confirm preset `ff_siwis` exists in these voices; confirm no protected cache or HELD ownership. Pin the actual source/image identity. Preserve the existing manifest; resolve mismatch explicitly rather than rewriting it. Require at least 2 GiB host free before starting, no image layers/downloads, at most 64 MiB new output and 256 MiB tmpfs. Current 3.9 GiB suggests a small no-build smoke may fit, but VM free disk and concurrent load are unmeasured; feasibility remains conditional.

Execution budget: one container, CPU-only, 2 CPUs / 2 engine threads, 4 GiB RAM, no swap allowance, 128 PIDs, network disabled, nonroot, read-only root/source/assets, fresh writable output only, no Docker socket or privileged mounts. Use existing `apps/voice-cpu/voice-cpu.py speak --backend kokoro --language fr --preset ff_siwis --threads 2 --timeout 120 --json`, one short French sentence and a fresh WAV path. Outer wall budget 180 seconds; stop at first unexpected failure with no retry/tuning, retain exact new-resource identity and bounded raw evidence. No automatic cleanup of retained failures.

Evidence: full image ID and source SHA; actual versions/provider (`CPUExecutionProvider` only); verified asset identities; exact text/preset/command and resource/network policy; start/end, raw exit/stdout/stderr; WAV hash, sample rate, channels, duration, finite nonzero samples/peak/RMS; generated metadata; explicit human listening result for intelligibility and clipping. WAV existence alone is not useful-speech acceptance. Failure/timeout remains failure/unknown, not qualification.

This would establish only one real CPU synthesis through the legacy CLI on the Mac-hosted Linux VM. It would **not** qualify current release image 1.31.0, installed interface/authenticated journeys, retained artifact reuse, cancellation/reap, clean/failed restart or protected Linux host policy. Full installed-host qualification still needs a separately established compatible Linux host and exact release resources. No ongoing watch.
