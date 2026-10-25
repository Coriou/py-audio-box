# Pocket French CPU, without a UI

This create-only worked recipe builds the public `control` and `worker-pocket`
Linux/amd64 Docker targets, checks a supplied pinned model cache, measures one
actual contained Pocket preparation, installs the retained Docker-volume family,
and performs upload → prepare → save → generate → verified WAV download through
HTTP. It stops and restarts the same containers and leaves the service running.
HTTP is canonical; CLI and MCP use the same operations. No host ML environment,
UI, implicit model download, GPU or fallback is involved.

Use a clean checkout, Python 3.11, and a local Unix Docker endpoint selected by an
existing Docker context. The complete pinned worker closure is
`requirements/pocket-installed.lock`. Image construction needs network access;
the model acquisition and license/access acceptance are separate operator work.
Build and model storage need several GiB beyond runtime memory. The installed
budget is 4 GiB independently for control and worker; the contained ML address
space ceiling is 3 GiB, two threads and 180 seconds per operation. These are
fixed bounded recipe limits, not a promise that every host will complete.
The installed disk quota is 1 GiB, with 64 MiB free-space admission and a 4 MiB
reservation per retained job. The latter covers two copies of the maximum
720044-byte output plus the full 1 MiB admitted input (3537240 bytes total).
Retained jobs still consume quota; ordinary Core capacity checks remain active.

Acquire authorized gated Hugging Face files with your own authenticated HF CLI:
repository `kyutai/pocket-tts`, immutable revision
`d0ebc5deb50bc6b225f686782f124dd37e2dca0a`, files
`languages/french/model.safetensors` and `languages/french/tokenizer.json`.
Place the two files directly in a private cache directory as `model.safetensors`
(219029196 bytes) and `tokenizer.json` (246403 bytes). Exact SHA-256 pins and
provenance are in `requirements/tts/pocket-fr.expected.json`. The recipe derives
and verifies `config.yaml` locally. It never reads or transfers your HF token.

Supply a reference voice WAV you are authorized to use: canonical PCM16, mono,
24 kHz or 48 kHz, nonempty and at most eight seconds and 1 MiB. Standard PCM WAV
headers without ancillary chunks are required. No trimming or resampling happens
silently. The complete supplied reference is selected; ticks always use 48000
units per second. The audition and generated text are `Bonjour.`.

```sh
python3.11 -m venv .venv-client
.venv-client/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
.venv-client/bin/python scripts/cpu-pocket.py \
  --context YOUR_LOCAL_DOCKER_CONTEXT \
  --model-cache /absolute/private/pocket-cache \
  --reference /absolute/private/reference.wav \
  --output /absolute/existing-parent/fresh-pocket-install \
  --port 18766 --principal operator
```

The output directory must not exist. It contains private credentials, model
copies, reference and generated audio, exact image IDs, command wait receipts,
probe inspection and HTTP records. Keep it private. The engine probe uses
networkless Docker, bounded copies into Linux tmpfs, sealed model files,
permanent UID 20000 without groups or capabilities, and resource limits before
ML import. Its measurement is accepted only after exact-image exit-zero/no-OOM
inspection. That engine receipt precedes the independently recorded installed
HTTP journey; neither records listening acceptance.

The canonical journey replays idempotency keys, waits for terminal job reap and
idle acknowledgement, reads `voice_results`, uses `voice_save` with revision zero,
and generates using the returned real voice-version identity. Audio downloads
are chunked, bounded and SHA-256 checked. Listen to `audition.wav` and
`generated.wav` yourself; perceptual acceptance is not automated.

Inspect or stop this same retained installation:

```sh
.venv-client/bin/python scripts/headless-install.py docker-pocket status \
  --handle /absolute/existing-parent/fresh-pocket-install/handle
.venv-client/bin/python scripts/headless-install.py docker-pocket stop \
  --handle /absolute/existing-parent/fresh-pocket-install/handle
```

Use `up` with the same handle to start again. Failures retain logs, process wait
receipts, containers, handles and volumes. There is no cleanup, repair or retry
against a partial output directory. Retain unfinished owners until actual reap.

For ordinary CLI operations, use `python -m clients.cli --help` from the client
environment and point it at the printed URL. Supply credentials privately through
`SPEECH_CREDENTIAL`, and `SPEECH_URL` for the service URL; never put tokens in
command arguments or share the output directory. The named operation schemas and
`discover` response describe the same HTTP operations used by the recipe.

For stdio MCP on a supported Linux/amd64 Python 3.11 client, create a separate
virtual environment and install `service/requirements.lock` with
`--only-binary=:all: --require-hashes`. Configure its command as
`python -m clients.mcp`, working directory this checkout, and inject
`SPEECH_URL` and `SPEECH_CREDENTIAL` privately. MCP loads no model. macOS MCP SDK
wheel availability is a separate client prerequisite. This recipe establishes
no CUDA, Metal, cancellation, listening or release-wide qualification.
