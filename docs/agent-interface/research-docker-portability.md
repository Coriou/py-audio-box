# Docker portability and agent installation qualification

Source audit/design, **15 October 2026**. No image pulls/builds, model downloads,
inference, new containers, host configuration changes or secret inspection were
performed in this lane. The separate evaluation owner runs authorized model
experiments. Commands and JSON below are **proposed product interfaces**, not
commands already implemented by this checkout.

## Release promise

Aim for **one small bootstrap, an explicit runtime profile and a useful diagnosis
on every supported host**. "Basically any box with Docker" should mean a stable
control/UI/API experience plus compatible CPU engines where qualified, with clear
limits elsewhere. It cannot mean every bundled model runs acceptably on every
CPU/GPU or that Docker supplies missing host drivers, native microphone access,
unlimited RAM and model rights.

Publish CPU and GPU qualification separately. Choose the profile once, persist
the exact image/model identities, and let agents use capabilities rather than
hardware guesses. A missing GPU never silently selects a huge CPU job; a slow CPU
never silently sends private audio to a cloud provider. Existing Qwen CLI defaults
remain unchanged unless a separately approved product-default decision changes
them.

## Current checkout: what exists and what needs packaging work

| Evidence | Current behavior | Product implication |
|---|---|---|
| `scripts/publish.sh` | Explicit `PLATFORM=linux/amd64`; clean builds gain commit-derived tags | No current arm64 release claim; use immutable digests, because commit-looking tags remain registry-mutable |
| `Dockerfile` | Broad toolbox, root runtime, mutable Python base, Poetry lock, separate unpinned tools and latest CUDA branches | Not the small reproducible service/control image; runtime closure differs from the lock after wheel replacement |
| `Dockerfile.cuda-base`, `Dockerfile.cuda` | Heavy CUDA base/thin code split; mutable `cuda-base` reference; ABI-coupled flash-attn wheel | Good layering idea, but source SHA alone does not identify the parent image/dependency closure |
| `Dockerfile.cpu`, `requirements/*.lock` | Pinned Python base digest; backend-specific hashed locks generated for Linux x86_64; narrow app copies | Best starting packaging discipline; arm64 needs an intentionally resolved and tested closure |
| `docker-compose.yml`, `run`, `run.ps1` | Interactive source-mounted toolbox, writable cache/work, optional GPU and host-network overlays | Useful developer flow, not a production install; no small always-on API/UI or publishable install contract yet |
| `run-cpu`, `scripts/provision-tts.py` | Explicit provisioning network; inference network disabled; assets read-only; bundle manifests/hashes | Preserve and extend into service workers; provisioning is distinct from readiness/discovery |
| GPU overlay / `SETUP.local.md` | Maxwell/cu124 and speed claims conflict with more careful README scope | Rewrite from exact release evidence; do not implement doctor using those comments as truth |
| `.github/workflows/unit.yml` | One Ubuntu CPU contract job invoking isolated unit tests | Valuable correctness gate; not a CPU ISA, arm64, Desktop, GPU, install/upgrade or acoustic matrix |
| Dockerfile license comments | Project redistribution license explicitly undeclared | Owner must choose the project's redistribution terms before calling a distributable release ready; dependency/model terms remain separate |

The general build copies a broad repository context; `.dockerignore` excludes
major runtime data and `.env`, but the release should use narrow stage copies
and an inventory rather than depend on an ever-growing denylist. Do not ship
private-work directories, environment files, source mounts or cloud orchestration
credentials as an incidental dependency of a local speech service.

## Practical qualification matrix

**Observed host:** this session is Intel i7-8700B, Darwin x86_64, macOS 15.7.4.
The parent measured six physical/twelve logical cores, 64 GiB host RAM, Docker
29.4 and a Linux VM with about 15.7 GiB RAM/12 vCPU. These are session facts, not
new tests of a release image. Current local Kokoro evidence is bounded to the
frozen runs documented in `docs/TTS_MEMORY.md`; the new service itself is unbuilt.

The parent separately verified the transport primitive on this host: two Linux
containers, both with networking disabled, exchanged synthetic ping/pong through
a Unix-domain socket in a Docker-managed Linux named volume; the client mounted
that volume read-only. Both exited successfully and task-owned resources were
removed. This is a **narrow observed Mac Docker result**, not qualification of the
production protocol, non-root mappings or another platform.
[Probe and limits](evaluation/ipc/README.md), [actual output](evaluation/ipc/result.txt),
[reproduction source](evaluation/ipc/probe.sh).

| Proposed profile | Initial status | Required qualification / explicit exclusions |
|---|---|---|
| `cpu-amd64-avx2` | First release baseline; current Intel machine is target hardware | Native Linux/amd64 image; record the complete required ISA set including any FMA/F16C requirements; inspect VM CPU flags, actual image/dependency imports and model execution; test Linux Engine, Intel Mac Desktop and Windows WSL2 separately |
| `cpu-amd64-sse41` | Candidate, not advertised as current supported | Separate conservative closure/binary configuration, all dependency imports, tiny/short real jobs and memory/latency test on hardware without AVX2; limit capability list to engines actually proven; do not offer full Qwen stack merely because STT supports SSE4.1 |
| `cpu-arm64` | Candidate | Native arm64 locks/builds for each selected backend; native ARM execution and memory/quality checks; Linux ARM and Apple Silicon Desktop are distinct host tests; do not hide amd64 QEMU emulation as normal support |
| `cuda-linux-amd64` | GPU candidate, exact device/model tuple required | Supported NVIDIA driver/container toolkit, selected device exposure, matching CUDA/cuDNN/framework/CT2/flash-attn kernel/ABI and checkpoint; test both actual load and inference, not `nvidia-smi` alone |
| `cuda-windows-wsl2` | Separate GPU candidate | Windows NVIDIA/WSL2/Desktop prerequisites plus Linux-container device execution, mounts and lifecycle; no claim for Windows native containers or Windows ARM from this profile |
| `rocm-linux-amd64` | Optional experimental lane | AMD-supported GPU/driver/ROCm/framework and per-engine implementation; explicit device nodes/runtime permissions; do not use architecture spoofing as automatic repair |
| `vulkan-linux-amd64` | Optional per-engine lane | Actual Vulkan ICD/device, compiled backend and workload tests, initially plausible for whisper.cpp; not an acceleration switch for PyTorch/Qwen/faster-whisper generally |
| `native-macos-worker` | Later explicit host/remote extension | A separately packaged native worker can use a proven native backend; Apple Silicon Metal/Core ML/MLX are not Linux-container GPU passthrough and do not apply to this Intel CPU profile |
| 32-bit x86/ARM, unsupported OS/ISA, emulated ML runtime | Unsupported until separately qualified | Refuse clearly; offer a configured compatible remote worker or metadata-only control, never fake success |

AVX2 here is a deliberately conservative product baseline, **not a claim that
all STT requires AVX2**. Current CTranslate2 documents prebuilt CPU support from
SSE4.1 and for ARM64, with runtime instruction dispatch. That says nothing about
all other wheels in our image or their latency. Its broad upstream GPU statement
also does not prove our particular CUDA wheel contains older-SM kernels.
[CTranslate2 hardware contract](https://opennmt.net/CTranslate2/hardware_support.html).

whisper.cpp's source exposes CPU-variant/ISA build machinery; portability is a
compile/dependency property, not just a Docker architecture label. Set intentional
ISA targets rather than compiling a redistributable binary for the build host's
`native` CPU. Inspect the exact binary and execute the oldest claimed profile.
[CPU CMake implementation](https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/ggml/src/ggml-cpu/CMakeLists.txt).

Docker multi-platform manifests select OS/architecture images; they do not certify
instruction-set compatibility or native performance. Emulation may help builds,
but cannot replace real ARM or old-x86 runtime evidence.
[Docker multi-platform builds](https://docs.docker.com/build/building/multi-platform/).

NVIDIA Linux needs a compatible host driver and container toolkit. The installer
must diagnose their absence and link the official owner-run setup rather than
install drivers, edit `daemon.json` or restart Docker automatically. Compose can
reserve an explicit device ID; do not default to claiming every GPU.
[NVIDIA toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[Compose GPU selection](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows/WSL2 GPU passthrough has its own platform prerequisites.
[Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/).

ROCm has host-driver/device/runtime requirements, and AMD documents selected
render nodes as an alternative to exposing all GPUs. An optional image must not
silently add privileged mode, disable isolation or spoof a GPU architecture to
pass a compatibility check. Vulkan support likewise belongs to a named engine.
[AMD Docker requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html),
[whisper.cpp accelerator options](https://github.com/ggml-org/whisper.cpp).

## Docker Model Runner is a different extension

Docker Model Runner documents Apple Silicon Metal through its inference engines,
with host/container API routes. That native acceleration path is distinct from
passing a Mac GPU into our Linux worker. Its current API/engine documentation
covers LLM/chat/embedding and image-generation use cases; it does **not establish
compatibility with our TTS models or an STT endpoint**. Do not propose it as a
drop-in Whisper/Qwen speech solution without a separate model/API proof.
[Inference engines](https://docs.docker.com/ai/model-runner/inference-engines/),
[Model Runner API](https://docs.docker.com/ai/model-runner/api-reference/).

A future native Mac worker or configured remote GPU host can satisfy the same
job/artifact contract over authenticated RPC, with explicit network permission
and model ownership. Do not make a Linux-container Unix socket depend on mounting
a native Mac process's socket through host sharing.

## Packaging lessons from actual alternatives

**Voicebox:** current canonical Compose binds localhost, persists data/model cache
in named volumes and sets four CPU/eight-GiB limits. The Dockerfile builds the web
UI, bundles Python dependencies and uses an entrypoint to drop privileges; its
current source installs some dependencies/Git sources without immutable revisions.
Its ROCm overlay forwards GPU nodes and contains hardware-specific overrides.
Borrow the easy web/container path and persistence clarity; do not inherit mutable
dependency resolution, broad device access or override-based compatibility as a
release guarantee. A Dockerfile health comment saying auto-restart is not proof
of that behavior.
[Compose](https://raw.githubusercontent.com/jamiepine/voicebox/main/docker-compose.yml),
[Dockerfile](https://raw.githubusercontent.com/jamiepine/voicebox/main/Dockerfile),
[ROCm overlay](https://raw.githubusercontent.com/jamiepine/voicebox/main/docker-compose.rocm.yml).

**LocalAI:** its Dockerfile and Compose have substantial architecture/accelerator
build machinery and persistent models/backends/configuration. It supports CPU
flag detection, explicit backend configuration, external gRPC backends and model
installation/preload actions. This is useful evidence for capability-directed
runtime selection. Its general AI breadth is much larger than our audio service;
do not import all those engines or startup preparation paths merely for hardware
coverage. Current Compose explicitly disables one remote metadata-warming path,
illustrating why seemingly cheap discovery/startup can still contact providers.
[Dockerfile](https://raw.githubusercontent.com/mudler/LocalAI/master/Dockerfile),
[Compose](https://raw.githubusercontent.com/mudler/LocalAI/master/docker-compose.yaml),
[Advanced runtime/backend controls](https://localai.io/docs/advanced/index.html).

**Speaches:** an especially relevant speech-service packaging reference. Current
Compose advertises amd64/arm64 builds and CPU/CUDA overlays with persistent Hub
cache; the image runs non-root, uses frozen uv dependency sync, disables telemetry
and exposes health. The CUDA overlay selects a CUDA base/version; that is more
concrete than one "GPU" switch. Still qualify the selected backend/architecture:
a build-platform list and a health URL do not prove every speech model or offline
behavior. Its examples publish broadly; our first install should bind loopback.
[Base Compose](https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml),
[CPU overlay](https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cpu.yaml),
[CUDA overlay](https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cuda.yaml),
[Dockerfile](https://raw.githubusercontent.com/speaches-ai/speaches/master/Dockerfile).

**whisper.cpp:** narrow native STT engine with CPU and multiple accelerator build
paths, attractive for a smaller/older-CPU worker. The inspected main Dockerfile
actually invokes `make base.en` during build; its README says that target downloads
and runs a model. Do not treat that image recipe as a weights-free generic base.
Build our own pinned engine-only artifact if adopted, then provision a multilingual
model explicitly for French/English/mixed work. Its server is not a full production
studio or durable executor.
[Dockerfile](https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/.devops/main.Dockerfile),
[Upstream engine/build behavior](https://github.com/ggml-org/whisper.cpp).

No alternative is reuse-approved by this source audit. Source/API suitability,
artifact manifests, licenses, clean install and actual runtime measurements remain
separate gates. The full-product adaptation decision remains with the parent audit.

## Minimal install and profile selection

Ship a versioned release bundle with small POSIX and PowerShell bootstraps, Compose
files, signed/checksummed release metadata, a capability/model manifest and a short
agent guide. A repo checkout should support the same path, while deployment uses
prebuilt images and does not need Python, Poetry, Node, ffmpeg, `make` or source
builds on the host. Keep an explicit developer-build path separate.

Proposed agent journey:

1. `pab doctor --metadata-only --json`: inspect only; no pull, create, provision,
   model imports, inference or remote discovery. Emit a compatible profile or
   actionable block. On a clean machine, missing runtime image/model is a valid
   state, not an attempt to download them behind the check.
2. `pab install --plan --profile cpu-amd64-avx2 --capabilities stt,tts --json`:
   name exact images/asset bundles, sizes where known, license/access conditions,
   data location, bind address and resource limits. Select engines deliberately;
   do not provision every TTS family or silently change the legacy default.
3. Apply that frozen plan with an explicit download/install action. This may pull
   images and run the provisioning task; gated terms remain an owner action.
   Atomic activation happens only after complete verification.
4. Start the pinned Compose release with pull policy disabled after installation;
   return a local UI URL, redacted connection configuration and service state.
   A stored scoped token is referenced by CLI/MCP config, not printed into logs.
5. Metadata discovery lists capabilities and bundle state. An explicit smoke job
   proves one useful result and its artifact; startup/liveness alone does not.

The common successful install should require few decisions: CPU or a detected
qualified GPU, desired capabilities/languages, and data location. Advanced CPU
flags, model repositories and socket paths stay in diagnostics. The default bind
is loopback; private-server access is an explicit HTTPS/auth deployment choice.

## Metadata-only doctor contract

Separate **host metadata**, **daemon metadata**, **configured capability**, **local
artifact verification**, **runtime smoke** and **acoustic acceptance**. An unknown
value stays unknown; avoid a green pass because a library or driver name exists.

Read-only inputs may include platform/CPU flags, Docker client/server API and
Compose versions, daemon OS/architecture/VM limits, locally present image labels,
deployment settings and previously verified asset manifests. Do not run `docker
run` to "inspect" a missing image; Docker's normal pull-if-missing behavior would
break this contract. Do not import torch/ONNX/Whisper or call model constructors.
Use cached model verification timestamps; full rehash is an explicit bounded
verification step, not misleadingly claimed by a manifest-only read.

Registry/network metadata checks are separately opt-in and labelled. Default
doctor should not contact Hugging Face, a model gallery, Docker Hub, telemetry or
a configured cloud account. Docker daemon connection is still a deliberate
read-only local operation; respect the selected Docker context and report if it
is remote rather than silently changing the deployment target.

Example on a clean compatible Intel host (illustrative values):

```json
{
  "schema_version": 1,
  "mode": "metadata_only",
  "status": "install_required",
  "host": {"os": "darwin", "arch": "amd64", "cpu_isa": ["sse4.1", "avx2"]},
  "docker": {"reachable": true, "daemon_os": "linux", "daemon_arch": "amd64"},
  "selection": {"profile": "cpu-amd64-avx2", "compatible": true},
  "artifacts": {"images": "missing", "models": "missing"},
  "verification": {"model_loaded": false, "inference_run": false},
  "next": [{"code": "INSTALL_PLAN", "command": "pab install --plan --json"}]
}
```

Stable exit classes: `0` usable/compatible at the requested evidence tier; `2`
configuration/argument error; `3` install/provision action needed; `4` unsupported
profile; `5` daemon/runtime unavailable; `6` storage/resource block. Exact numbers
are a proposed contract to freeze with the CLI, not existing behavior.

Errors carry code, phase, retryability, evidence and one safe next action. Bound
the output; redact tokens, private endpoints, proxy credentials, user paths,
raw environment and transcript contents. Offer explicit expanded diagnostics.

## Safe failure and repair recipes

| Failure code | Agent response | Deliberate owner/agent repair path |
|---|---|---|
| `DOCKER_NOT_FOUND` / `DAEMON_UNREACHABLE` | Stop install before mutation; retain plan | Install/start the chosen container runtime; rerun doctor; no automatic root/sudo retry |
| `DOCKER_ACCESS_DENIED` | Identify access problem, not model failure | Owner fixes runtime access policy; do not chmod daemon sockets or add privileged mode |
| `WRONG_DAEMON_OS` / `ARCH_UNSUPPORTED` | Name requested and observed target | Select Linux containers/qualified daemon explicitly; do not switch contexts or emulate silently |
| `CPU_ISA_UNSUPPORTED` | Offer only a genuinely qualified smaller capability profile | Choose SSE4.1 candidate only after its release exists, or configure compatible remote compute; no instruction-spoof env vars |
| `GPU_RUNTIME_MISSING` / `GPU_KERNEL_UNSUPPORTED` | Distinguish host, exposure and wheel/kernel mismatch | Link exact driver/toolkit/image support; explicit CPU re-plan or owner maintenance; no driver install/reboot fallback |
| `MODEL_NOT_PROVISIONED` / `ACCESS_REQUIRED` | Return exact asset identity/owner access action | Explicit provision with ephemeral scoped credential; do not accept terms or expose tokens |
| `ASSET_HASH_MISMATCH` / `PROVISION_INTERRUPTED` | Quarantine incomplete staging; retain active version | Retry verified staged download or obtain reviewed pin update; never overwrite active bundle or trust changed upstream bytes |
| `DOWNLOAD_NETWORK_ERROR` | Report attempted authorized phase and bounded diagnostic | Owner reviews DNS/proxy/CA/network; explicit retry; no host networking, disabled TLS, alternate mirror or proxy bypass automatically |
| `DATA_PERMISSION_DENIED` | Name affected volume class and required identity | Repair only designated volume ownership through explicit maintenance; never recursive chmod/chown of the repo/home |
| `INSUFFICIENT_MEMORY` / `DISK_FULL` | Refuse new work while preserving accepted jobs/artifacts | Reduce configured concurrency/model profile, raise VM limit deliberately, or clean named reproducible caches; no global Docker prune |
| `WORKER_TRANSPORT_UNVERIFIED` | Keep profile unavailable | Run the explicit transport qualification or select an already qualified transport; no silent networking/isolation change |
| `PORT_IN_USE` | Retain data and configuration | Choose a port in a new install plan; do not kill the current owner |
| `SCHEMA_NEWER_THAN_BINARY` | Stop with data untouched | Use compatible image or supported restore/migration; never force downgrade/reset |

An automatic bounded retry is reasonable only for a known transient operation
inside the same authorized phase. It must not expand the network destinations,
permissions, model choice, download budget or data mutation scope.

## Storage, worker transport and offline inference

Use explicit named volumes for control data, immutable model bundles, artifacts
and worker runtime/IPC. The single control owner holds the SQLite database;
workers do not open it. Keep the live database on the daemon's local persistent
filesystem, not SMB/NFS or a cross-machine shared folder. SQLite documents WAL's
shared-memory/local-host assumptions.
[SQLite WAL](https://www.sqlite.org/wal.html),
[Docker volumes](https://docs.docker.com/engine/storage/volumes/).

The preferred same-host worker transport can be a Unix-domain socket in a
**Linux named volume shared by Linux containers**, with narrow permissions and
authenticated/versioned RPC. The parent smoke above establishes synthetic
ping/pong for this Intel Mac's Docker Linux VM, including a read-only client IPC
mount and no Docker network. It does **not** establish Windows/WSL2 portability,
non-root UID mapping, production authentication or failure/reconnect behavior.
Windows' native Docker named pipe,
a socket on an NTFS bind mount, a socket in the WSL filesystem and a socket
inside a Docker-managed Linux volume are different cases. Do not conflate them.
[Docker daemon connection surfaces](https://docs.docker.com/desktop/troubleshoot-and-support/faqs/general/),
[WSL filesystem guidance](https://docs.docker.com/desktop/features/wsl/best-practices/).

Transport gate, on each advertised daemon: two pinned minimal Linux containers,
non-root UIDs, short socket path, shared named volume, handshake/request/cancel/
reconnect after restart, stale-socket cleanup and confirmed peer death. Check
payload limits, generation fencing and permissions. This explicit smoke performs
no model download/inference, but creates test containers/volume and therefore is
separate from metadata-only doctor. The recorded Mac primitive smoke covers only
part of this gate; the remaining cases still require qualification.

Once proven, workers can use `network_mode: none` plus the IPC volume and read-only
models. The control plane can serve HTTP without giving the worker internet.
Docker's `none` driver leaves only loopback in that container.
[None network](https://docs.docker.com/engine/network/drivers/none/).
If a platform needs authenticated internal TCP instead, ship it as an explicit
separately qualified profile and state its different network boundary. Remote
workers/native host extensions likewise require explicit network credentials.

Provisioning runs with a writable staging model volume and bounded network access.
Record source URL, immutable revision, size/hash, license/access receipt and
runtime compatibility. Activate by atomic manifest/version switch after all
files verify. Runtime mounts model versions read-only, uses offline library
settings and has independently proven network isolation. Keep secrets out of
images, persistent model manifests, CLI receipts and ordinary worker environments.
Existing Pocket access conditions remain owner-controlled.

## Resource, health, update and uninstall behavior

Use one inference job per worker initially. Configure CPU threads, container CPU/
memory/pid limits, bounded scratch space, input/duration ceilings and queue limits.
Control/UI memory is separately budgeted. Advertised model tiers need observed
container peaks and headroom, not parameter-count arithmetic. The measured Kokoro
four-CPU/four-GiB setup is a starting experiment, not a universal clone/STT budget.
Docker does not impose useful CPU/memory limits by default; enforce the proposed
caps and inspect their actual application under Compose.
[Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/).

Health levels:

- **Live:** control loop responds; no model imports or provider requests.
- **Ready:** schema/storage/scheduler available and at least the configured routes
  can accept/reject jobs truthfully; sleeping models are not unhealthy.
- **Runtime verified:** explicit frozen-image/model smoke passed on this host.
- **Qualified output:** text fidelity/acoustic/listening evidence for the named
  profile; never inferred from liveness or successful WAV output.

A Compose healthcheck marks health; a restart policy restarts an exited process.
It does not by itself repair a running unhealthy service. Use supervised worker
deadlines/reap, process exit for unrecoverable owner failure and bounded restart
policy with durable job fencing. Do not auto-restart indefinitely around a bad
asset/configuration or delete state to make the health icon green.
[Docker restart behavior](https://docs.docker.com/engine/containers/start-containers-automatically/),
[Healthcheck contract](https://docs.docker.com/reference/compose-file/services/#healthcheck).

Pin the release manifest to OCI index and per-architecture image digests, source
commit, dependency lock hashes, SBOM/notices, exact model assets and schema
compatibility. No startup pip/apt install, unpinned Git dependency, background
backend download or automatic model promotion. Registry tags may be convenient
selectors during an explicit upgrade plan; the installed record uses digests.

Automatic update **checks** can be opt-in metadata operations. Applying updates
is a controlled sequence: plan compatibility/disk cost, stop new scheduling,
drain or preserve jobs, take a consistent SQLite/artifact/config backup, migrate,
start exact new digests, check readiness/reopen a project and explicitly smoke.
Retain previous image/model identities. Rollback after a schema migration needs
a supported reverse migration or matching backup; changing a tag is insufficient.

Default uninstall stops/removes only this installation's labelled containers and
network, retaining named data/model/artifact volumes and printed recovery paths.
Separate explicit purge can remove named resources after an inventory/backup
choice. Never use `docker system prune`, unqualified `down -v`, shared cache deletion
or broad home/repository cleanup as ordinary uninstall. Removing Docker Desktop
itself belongs to the owner, not this app's bootstrap.

## Release acceptance, in evidence order

1. **Metadata:** clean CPU/GPU/unsupported/offline/missing-image/remote-context
   cases produce stable JSON and no download, container creation or secrets.
2. **Packaging:** published digests resolve to the claimed architectures/ISA and
   dependency closure; no private context leaks; non-root volumes work from fresh
   roots; POSIX/PowerShell paths and CRLF checkouts behave intentionally.
3. **Transport/storage:** named-volume UDS (or explicit alternative) works on
   Linux, Mac Desktop and Windows WSL2; sole-owner SQLite survives crash/disk-full;
   backup restores to a fresh installation; worker never mounts the live DB.
4. **Runtime:** exact chosen CPU/GPU image, asset and hardware passes bounded
   STT/TTS/jobs/cancel/restart/publication tests with inference network isolation.
   Run SSE4.1 and ARM on real matching hosts before promotion.
5. **Product:** a fresh agent installs the selected profile, discovers capabilities,
   obtains one useful artifact, opens it in the shared studio, and recovers from
   the listed failures using short structured recipes. Native dictation remains a
   separate optional adapter qualification.
6. **Release:** upgrade/rollback/uninstall preservation and measured resource/
   latency/output quality meet the advertised profile. Keep failed/unknown cells
   visible; a broad host list is not a substitute for these receipts.

**Judgment:** GO for a narrow amd64/AVX2-first release design with explicit
qualification expansion. SSE4.1, arm64, each GPU family and Windows/WSL2 UDS remain
candidates. Mac Docker named-volume UDS has the narrow primitive receipt above;
the production protocol and non-root lifecycle remain unqualified. Docker-only
Mac GPU speech, universal GPU acceleration and current
"runs everywhere" claims are unsupported by this audit.
