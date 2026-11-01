# Pocket French CPU, without a UI

**Status: preview; French cloning only.** This recipe passed end to end on Linux/amd64 Docker
at an earlier source revision (2026-10-25), not repeated on this commit: upload, prepare,
save, generate, verified download, cancellation, restart and reuse of saved voices.
Listening quality for these installed runs is unmeasured, the retained pilot shows text-fidelity failures,
and CUDA and Apple Metal are unqualified. Evidence:
[CPU release report](reviews/CPU-RELEASE-r407-parent.md).

This create-only worked recipe builds the public `control` and `worker-pocket`
Linux/amd64 Docker targets, checks a supplied pinned model cache, measures one
actual contained Pocket preparation, installs the retained Docker-volume family,
and performs upload → prepare → save → generate → verified WAV download through
HTTP. It stops and restarts the same containers and leaves the service running.
HTTP is canonical; CLI and MCP use the same operations. No host ML environment,
UI, implicit model download, GPU or fallback is involved.

The [retained backend pilot](../TTS_POCKET_CPU_PILOT.md) records material
text-fidelity failures; clean/exact-text cloning has not passed acceptance.
See the [model comparison](../../README.md#model-choice-and-repository-declarations).

This is a Git source-checkout recipe: the script runs `git rev-parse HEAD`, so
it is not a repo-free extracted-archive recipe. Use a clean checkout, Python 3.11, and a local Unix Docker endpoint selected by an
existing Docker context. The complete pinned worker closure is
`requirements/pocket-installed.lock`. Image construction needs network access;
the model acquisition and license/access acceptance are separate operator work.
Build and model storage need several GiB beyond runtime memory. The installed
budget is 4 GiB independently for control and worker; the contained ML address
space ceiling is 3 GiB, two threads and 180 seconds per operation. These are
fixed bounded recipe limits, not a promise that every host will complete.
The installed disk quota is 1 GiB, with 64 MiB free-space admission and an 8 MiB
reservation per retained job. The latter covers the preparation protocol’s two output copies and six
reference/segment copies at the full 1 MiB input ceiling (7731544 bytes total).
Retained jobs still consume quota; ordinary Core capacity checks remain active.

The [pinned descriptor](../../requirements/tts/pocket-fr.expected.json) declares
CC-BY-4.0 and operator-authorized HF gated access. Obtain HF access and accept
upstream conditions before acquisition. The recipe neither logs in nor accepts
conditions on your behalf; a metadata field does not authenticate acceptance.
The eight-second reference bound below belongs to this recipe, not a universal
upstream model limit. Empty `reference_text` and separately authored audition
text have different roles; saving retains the reference, not the audition.

With the HF CLI already installed separately, first accept the upstream gate
as the operator, then log in interactively. Never put a token in argv or automate
acceptance. Choose fresh private download and cache paths. This fail-fast block
stops if the cache directory already exists, before moving either file:

```sh
set -eu
hf auth login
hf download kyutai/pocket-tts \
  languages/french/model.safetensors languages/french/tokenizer.json \
  --revision d0ebc5deb50bc6b225f686782f124dd37e2dca0a \
  --local-dir /absolute/private/fresh-pocket-download
mkdir -m 700 /absolute/private/fresh-pocket-cache
mv /absolute/private/fresh-pocket-download/languages/french/model.safetensors \
  /absolute/private/fresh-pocket-cache/model.safetensors
mv /absolute/private/fresh-pocket-download/languages/french/tokenizer.json \
  /absolute/private/fresh-pocket-cache/tokenizer.json
python3 - <<'PYHASH'
import hashlib
from pathlib import Path
cache = Path('/absolute/private/fresh-pocket-cache')
expected = {
    'model.safetensors': (219029196, 'dd5fe98f5577ecfd56db9af8057cd05fe20cbc91560042aec8a56cfb01c34995'),
    'tokenizer.json': (246403, '6211acffe9cb95172f71f0051f18e26ea305b2b32ea00b331ee976551c70887f'),
}
for name, (size, digest) in expected.items():
    path = cache / name
    if path.stat().st_size != size:
        raise SystemExit('size mismatch: ' + name)
    with path.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != digest:
        raise SystemExit('SHA-256 mismatch: ' + name)
print('Pinned descriptor sizes and hashes match.')
PYHASH
```

The sizes and hashes above match `requirements/tts/pocket-fr.expected.json`.
Use `/absolute/private/fresh-pocket-cache` as `--model-cache` below. The existing
recipe derives and verifies `config.yaml` locally; do not download it.
These are future operator acquisition instructions, not download evidence.

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
  --model-cache /absolute/private/fresh-pocket-cache \
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
`SPEECH_URL` and `SPEECH_CREDENTIAL` privately. MCP loads no model. Apple CI on macos-15 arm64 installs the same lock and
passes CLI/MCP fixture and stdio protocol calls; no synthesis is qualified. This recipe establishes
no CUDA, Metal, listening or release-wide qualification. The ordinary Docker-client
cancellation/reap check passed as recorded above.

For reference uploads from MCP, optionally set `SPEECH_UPLOAD_ROOT` in the agent
host to an absolute operator-authorized directory, for example
`/absolute/private/authorized-references`. This grants host filesystem read scope
within that root and exposes `speech_asset_upload_file` with `{"path":"reference.wav"}`.
Paths must be relative, with no `..` or symlink following. Without this setting,
use CLI `asset upload FILE` or authenticated HTTP first and supply the asset ID.
MCP direct audio content is limited to 3072 decoded bytes per call. Keep complete
audio outside model context: use `python -m clients.cli asset content ID --output FILE`
in the configured client environment for a complete hash-verified exclusive file.

Next: [agent guide](../AGENT-GUIDE.md).

Qwen English cloning is not ready; native CPU/Metal and CUDA are unqualified.
Existing linux/amd64 Docker images use emulation on Apple Silicon; this is
untested. Import-only CI does not establish runtime readiness.
