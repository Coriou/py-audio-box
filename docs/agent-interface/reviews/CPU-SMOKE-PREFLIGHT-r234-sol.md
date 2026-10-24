# CPU smoke preflight — r234 sol

**NO-GO for container execution: headroom below 6 GiB; legacy source/runtime/provider and ownership clearance remain incomplete.** Static preflight only, not actual inference. Observed 2026-10-24T18:51:08.271656+00:00. Root `/Users/ben/Projects/py-audio-box`, HEAD `7d5ffe703a1397b314b58a7d73311f42d29e7659`.

Read repository r233 inventory (`docs/agent-interface/reviews/ACTUAL-CPU-INVENTORY-r233-sol.md`), current Dockerfile.cpu, service/Dockerfile.release, docker-compose.yml, pyproject.toml, headless distribution manifest, Kokoro lock/input and asset manifests, and legacy CLI/backend source. r233's installed-host NO-GO on the Mac VM remains distinct from this legacy CPU smoke assessment.

## Cached bytes — PASS against existing pins

One streaming SHA256 pass per asset, 1 MiB chunks; **353,746,785 bytes total read**, below 354 MB. Sizes and hashes equal both cached manifest and repository requirements/tts-assets.json; no manifest rewrite.

- `cache/tts/kokoro/model.onnx`: 325,532,387 bytes; `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5`.
- `cache/tts/kokoro/voices.bin`: 28,214,398 bytes; `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d`.

Cached manifest records kokoro-onnx 0.6.1 / onnxruntime 1.29.0 / espeakng-loader 0.2.4 / phonemizer 3.4.0. Current Kokoro lock pins ONNX Runtime 1.31.0. Cached metadata establishes neither installed image versions nor current managed-profile qualification; preserve this difference.

## Immutable image metadata — bounded observation only

`docker image inspect py-audio-box:kokoro-cpu` and `docker history --no-trunc --format '{{json .}}' py-audio-box:kokoro-cpu` each completed exit 0 with a 12-second subprocess timeout.

Full image ID: **`sha256:7e4b229332c348e6f52ded724b023796c3246ea4d1b1d41c99b5a67404347059`**. Tags: kokoro-cpu and memory-release. RepoDigests empty; labels absent. Linux/amd64; 778,715,218 bytes; created `2026-10-15T09:50:50.937441096+02:00`; BuildKit comment. Config has no explicit User, no recorded source revision, WorkingDir `/app`, entrypoint `/usr/bin/tini --`, command `python apps/voice-cpu/voice-cpu.py --help`. Environment records Python **3.11.17**, TORCH_DEVICE=cpu, HF_HOME=/cache/huggingface and MALLOC_MMAP_THRESHOLD_=131072. Python version is static config metadata, not an executed interpreter check.

Compact history: Debian bookworm amd64 base; Python 3.11.17 build; BACKEND=kokoro; apt install libsndfile1/espeak-ng/ffmpeg/tini (~468 MB); COPY requirements (~109 kB) and asset pins (193 bytes); hashed pip install `/requirements/${BACKEND}.lock` (~173 MB); COPY lib, legacy voice-cpu/audio-export, provision script and benchmarks; tini/legacy help startup. This resembles the current CPU Dockerfile recipe but does **not** identify copied source bytes or the historical lock contents. No git/OCI revision, lock digest, package inventory or installed Kokoro/ONNX Runtime versions appears. Current source requires CPUExecutionProvider only; the image's copied backend and actual selected provider remain UNKNOWN. Neither tag nor TORCH_DEVICE proves an ONNX provider.

Exact full-ID search across public `docs/**/*.md` found no occurrence. A separate comparison against all 20 exact SHA256 tokens in public review reports containing “held” also found no equality (that conservative set includes non-image tokens). Known historical image IDs include `sha256:14b0cf52e7337b449938701c0256fe1f32fe846fa25d62f8d18413a190edbbcb` (r75), `sha256:e62008ff63f8a8eb48f7f7fe9f20f1c73af4a58f5f8da25a65f885e1c7dcc30a` (r106), and `sha256:36b96bf2750e09da6dd2489ae1f5ce2515d166b252d61fabc476232af80efa0e` (r113): all differ. **No exact public match found is not proof of non-HELD ownership or reuse permission.** No protected proofs, held lane, historical containers or volumes were opened or operated on. Image graph paths were returned as metadata only; none was accessed.

## Headroom and budget — BLOCK

Host `df -k .`: 488,245,288 KiB total; 469,082,140 KiB used; **4,062,244 KiB available = 4,159,737,856 bytes = 3.874 GiB**. Historical operational floor **6 GiB = 6,442,450,944 bytes**; shortfall **2,282,713,088 bytes (2.126 GiB)**. No floor waiver or cleanup.

`docker system df` completed exit 0 under 12 seconds: images 234 / 26.46 GB (5.244 GB reported reclaimable), containers 44 / 211.9 MB, local volumes 153 / 10.13 GB, build cache 1258 / 17.41 GB. These are aggregate usage/reclaimability, **not Docker VM filesystem free space** and not cleanup authorization. VM free capacity remains UNKNOWN; no volume access occurred.

A future existing-image, zero-new-image-layer container could nominally budget at most **64 MiB persistent output + 256 MiB tmpfs**, plus bounded Docker writable-layer/log/metadata overhead that is not yet measured. Tmpfs is RAM-backed, not a substitute for disk admission. This small payload budget may fit arithmetically within 3.874 GiB, but cannot meet the required 6 GiB starting headroom and is not a feasible authorized execution today. New image-layer budget is zero, not zero total disk writes. r233's proposed 2 GiB minimum does not override this task's 6 GiB floor.

## Next bounded step — separately authorized, after blockers close

First establish host >=6 GiB and sufficient measured VM free capacity, plus explicit clearance for this exact immutable image and intended cache inputs. Do not reclaim held resources or infer clearance from this inspection. Establish source identity through an existing owner/build provenance record; a new-container probe alone cannot reconstruct an absent Git SHA.

Then separately authorize ONE fresh isolated read-only/no-network metadata probe pinned to the full ID: no mounts of held resources, no model imports or inference, bounded output/time/resources; inspect Python and installed distribution versions (kokoro-onnx, onnxruntime, espeakng-loader, phonemizer, soundfile, numpy), historical embedded lock/source identities, and available ONNX providers using only a specifically authorized runtime import. Available providers still do not prove an actual session's selected provider. Resolve 1.29.0 versus 1.31.0 explicitly; do not rewrite the cached manifest. Actual CPU-only session/provider, preset availability and useful-speech evidence require a later separately authorized ONE smoke. Stop at first unexpected failure, retain new evidence/resources, no retry or automatic cleanup.

No run/create/build/pull/exec/stop, remote connection, network/download, model import, inference, volume access, HELD operation, source edit, commit or delegation occurred. Sole write: this report. Static inspection grants no reuse permission or installed-host/release qualification.
