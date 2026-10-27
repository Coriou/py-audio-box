# py-audio-box

A headless, local, CPU-first speech box: preset voices, reference voice cloning and
a voice catalogue behind one authenticated HTTP API, with a CLI and an MCP server
for agents that call the same operations. Everything runs in Linux Docker
containers on your machine. No GPU is needed, and there is no browser UI.

> **Status: pre-release.** The CPU Kokoro (preset) and Pocket (French cloning)
> installs passed end-to-end runs on Linux/amd64 Docker at an earlier source
> revision (2026-10-25); these runs were not repeated on this commit. Nothing is
> published, source is licensed under AGPL-3.0-only, and audio quality has
> not been accepted. Check the [readiness table](#what-works-today) before
> relying on any feature.

## What works today

| Capability | State | Notes |
| --- | --- | --- |
| Preset speech, CPU (Kokoro, French bundle) | **Works**: installed run passed at an earlier source revision (2026-10-25); not repeated on this commit | HTTP, CLI and MCP; speech, PCM24 export, download, restart |
| Reference voice cloning, CPU (Pocket, French) | **Works, with quality caveats**: installed run passed at an earlier source revision (2026-10-25); not repeated on this commit | Gated Hugging Face model; the pilot showed material text-fidelity failures |
| Voice catalogue: save, versions, aliases, archive | **Implemented** | Save and reuse exercised in installed runs; aliases and archive are unit-tested only |
| Jobs: wait, events, cancel, retry, list | **Implemented** | Wait, cancel and download exercised in installed runs |
| "Default voice for X" | **Convention, not a feature** | Set an alias such as `default.narration`, then resolve it; see the [agent guide](docs/AGENT-GUIDE.md#make-it-the-default-voice-for-x) |
| Qwen 0.6B cloning, CPU (EN/FR) | **Not ready** | Implemented in source; installed startup and inference not yet passed ([status](docs/agent-interface/QWEN-CPU.md#current-status--2026-10-27)) |
| Apple Silicon (Metal), NVIDIA CUDA | **Unqualified** | Native CPU/Metal unqualified; amd64 Docker on Apple Silicon uses untested emulation |
| Listening / perceptual quality | **Partial** | Owner listened to earlier comparison samples and accepted three spoken-text pairs; French liaison unresolved. No per-engine perceptual release acceptance; installed-run outputs not listened |
| Projects, blocks, clips, render (contract) | **Not implemented** | Defined in `contracts/v1`; not advertised by `discover` |
| Published images / packages | **None** | Build from source |
| Web UI | **Not planned for this release** | A future UI would use the same HTTP API |

Preview scope: cloning is French only through gated Pocket, with retained
text-fidelity failures. Qwen English cloning is not ready; native CPU/Metal and
CUDA remain unqualified. On Apple Silicon, existing linux/amd64 Docker images
use emulation; this path is untested. Import-only CI is not runtime readiness.
For executable native prerequisite checks and the remaining installation gates,
see [Apple Silicon diagnostics](docs/agent-interface/APPLE-CI.md).

Each installation runs **one engine**, and `discover` reports what that
installation can actually do. Treat `discover`, not this table, as the source of
truth for a running service.

## Quick start (CPU, Kokoro presets)

Requirements: Python 3.11, a local Docker daemon with `linux/amd64` and cgroup v2
(Docker Engine, Docker Desktop or OrbStack, selected through a Docker context),
about 2 GiB of free memory and several GiB of disk for images and models.

```sh
python3.11 -m venv .venv-client
.venv-client/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
.venv-client/bin/python scripts/cpu-kokoro-export.py \
  --context YOUR_DOCKER_CONTEXT \
  --output /absolute/path/to/new-install-dir \
  --port 18765 --principal operator
```

This builds the images, verifies the pinned model, runs one real synthesis, then
installs and starts the service at `http://127.0.0.1:18765`. The output directory
must not exist yet. It ends up holding a private `credential.json`, so keep it
private. Full details, stop/start commands and troubleshooting are in
[CPU-QUICKSTART.md](docs/agent-interface/CPU-QUICKSTART.md).

For **voice cloning**, install the Pocket engine instead. It needs gated Hugging
Face model files and a short reference WAV:
[POCKET-QUICKSTART.md](docs/agent-interface/POCKET-QUICKSTART.md).

## Use it

Point the client at the service. Pass the credential through the environment,
never on the command line:

```sh
export SPEECH_URL=http://127.0.0.1:18765
export SPEECH_CREDENTIAL="$(.venv-client/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["credentials"][0]["token"])' /absolute/path/to/install-dir/credential.json)"
.venv-client/bin/python -m clients.cli discover          # what this install can do
.venv-client/bin/python -m clients.cli status            # presets, model identity, readiness
.venv-client/bin/python -m clients.cli --help
```

- **HTTP**: `/v1/...` routes with bearer auth. Each operation's schema is in
  [`contracts/v1/operations.json`](contracts/v1/operations.json), and the CLI
  command `operation-detail ID` prints it.
- **CLI**: `python -m clients.cli`. Write operations take `--request '<json>'` and
  `--idempotency-key`. Audio is saved to files with `asset content ID --output FILE`.
- **MCP** (stdio): `python -m clients.mcp`, with tool names `speech_*`. It needs the
  separate environment described in
  [CPU-QUICKSTART.md § MCP adapter](docs/agent-interface/CPU-QUICKSTART.md#mcp-adapter).

The **[agent guide](docs/AGENT-GUIDE.md)** walks the whole flow: discover, upload
a reference, clone, save, set a default, generate, wait and download, with the
same steps in CLI and MCP form.

## Model choice and repository declarations

| Engine | Use | Model and declared license | Notes |
| --- | --- | --- | --- |
| Kokoro | Preset voices | [Kokoro-82M v1.0 ONNX](requirements/tts/kokoro-fr.expected.json); Apache-2.0 weights, MIT runtime; public download | No cloning, instructions or seed |
| Pocket | French voice cloning | [pocket-tts-fr 3.3.0](requirements/tts/pocket-fr.expected.json); CC-BY-4.0; **gated** HF access you obtain yourself | Short (≤ 8 s) reference; [pilot notes](docs/TTS_POCKET_CPU_PILOT.md) |
| Qwen | English/French cloning | [Qwen3-TTS Base 0.6B](requirements/tts/qwen-0.6b-base.expected.json); Apache-2.0; public | Not installable yet (see above) |

Models are pinned by revision and SHA-256 and are downloaded only when you
explicitly run a recipe. These are the repository's declarations, not a live
check of upstream terms. You remain responsible for model licenses, gated-access
conditions and consent for any voice you clone.

## Documentation

- [Agent guide](docs/AGENT-GUIDE.md): task recipes for agents and scripts
- [CPU quick start (Kokoro)](docs/agent-interface/CPU-QUICKSTART.md) ·
  [Pocket cloning quick start](docs/agent-interface/POCKET-QUICKSTART.md) ·
  [Installation reference](docs/agent-interface/HEADLESS-INSTALL.md)
- [Audio export](docs/agent-interface/AUDIO-EXPORT.md) ·
  [Docker volumes and data](docs/agent-interface/DOCKER-VOLUMES.md) ·
  [Qwen CPU status](docs/agent-interface/QWEN-CPU.md)
- Design history, reviews and evidence: `docs/agent-interface/README.md` (source checkout).
  You don't need these to use the service.
- The original `./run voice-*` Docker toolbox, job runner and Vast.ai workflow:
  `docs/LEGACY-TOOLBOX.md` (source checkout)

## License

Project source is licensed under **AGPL-3.0-only**: see [LICENSE](LICENSE) and
[practical source guidance](docs/SOURCE-LICENSE.md). Dependencies and model weights
keep their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and the model table above.
