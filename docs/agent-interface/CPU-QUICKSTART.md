# CPU Kokoro and PCM24 export

This worked recipe builds the control and CPU worker images from this checkout,
verifies the pinned French Kokoro bundle, runs actual contained synthesis and
FFmpeg export, then installs the fresh `docker-kokoro-export-split-v2` family.
HTTP, CLI and MCP share the canonical service; no browser or GPU is required.
The expected manifest and build pins are inputs, not runtime qualification.

**Status:** this recipe has passed end to end on Linux/amd64 (OrbStack): HTTP, CLI
and MCP, speech, PCM24 export, verified download and restart. Listening quality for these installed runs is
unmeasured, and CUDA and Apple Metal are unqualified. Evidence:
[installed run report](reviews/CPU-KOKORO-INSTALLED-r407-parent.md).

Use Python 3.11 and a local rootful Docker daemon with cgroup v2 and
`linux/amd64` support. Docker Desktop/OrbStack must expose an explicit local Unix
socket context. Allow at least 2 GiB worker memory plus control/build overhead,
and enough disk for the images, models, retained build logs and private volumes.
The installed disk quota is 1 GiB. Speech reserves 64 MiB per job and export
reserves 128 MiB per job; retained reservations share that quota across jobs.
This quota does not cover the host's image, model and build storage needs.
Builds acquire Debian packages and hash-locked Python wheels; model acquisition
uses bounded downloads from the pinned upstream release.

The [pinned descriptor](../../requirements/tts/kokoro-fr.expected.json) declares
Apache-2.0 weights and MIT kokoro-onnx runtime, from a public upstream release.
This is preset speech only, without reference cloning, instructions or seed.
See the [three-engine comparison](../../README.md#model-choice-and-repository-declarations)
and [current Qwen status](QWEN-CPU.md#current-status--2026-10-27).

From the source checkout:

```sh
python3.11 -m venv .venv-client
.venv-client/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
.venv-client/bin/python scripts/cpu-kokoro-export.py \
  --context YOUR_LOCAL_DOCKER_CONTEXT \
  --output /absolute/existing-parent/fresh-cpu-install \
  --port 18765 --principal operator
```

The output directory must not exist. The command builds serially, inspects the
finished image IDs, provisions and seals models, and runs a networkless probe
under production engine limits with separate unprivileged synthesis and converter
children. Only a successful container exit, Docker inspection, actual media and
matching runtime identity permit qualified profiles to be written. It then
registers a real preset through HTTP, synthesizes speech, exports stereo 48 kHz
PCM24, downloads and hashes `export.wav`, and stops/restarts the same installation.
The final service remains running on `http://127.0.0.1:18765`.

To reuse existing model downloads, append
`--model-cache /absolute/path/to/cache`. That directory must contain `model.onnx`
and `voices.bin` matching `requirements/tts/kokoro-fr.expected.json`; local copies
still pass the existing size/hash and bundle verification. No ambient cache is
trusted.

The output contains private `credential.json` (mode 0600), exact `request.json`
and `install.json`, the ownership handle, build/probe/HTTP receipts and the WAV.
Never publish that directory or print its credential. The script prints only the
service URL and evidence location. To stop or inspect the same installation:

```sh
.venv-client/bin/python -c 'from service.daemon_client import export_main; export_main()' \
  status --handle /absolute/existing-parent/fresh-cpu-install/handle
.venv-client/bin/python -c 'from service.daemon_client import export_main; export_main()' \
  stop --handle /absolute/existing-parent/fresh-cpu-install/handle
```

Use `up` with that handle to start it again. Failures retain exact container IDs,
logs, handles and volumes for review; there is no repair, deletion or automatic
retry path. If a child does not exit after TERM, the command retains its process
owner and waits for actual reap; it may require manual intervention. Repeated
interrupts do not release that owner. Do not rerun against a partial output directory. Build acquisition,
actual probe success and the installed HTTP journey are distinct evidence stages;
a failed stage confers no authority on a later stage. This recipe makes no CUDA,
Metal, Pocket or release-wide qualification claim.

## MCP adapter

The minimal `requirements/client.lock` installs the HTTP/CLI recipe dependencies.
The stdio MCP adapter also needs the pinned MCP SDK. On the supported Linux/amd64
Python 3.11 client, install it in a separate environment:

```sh
python3.11 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install --only-binary=:all: --require-hashes -r service/requirements.lock
```

Configure the agent's MCP server command as `.venv-mcp/bin/python -m clients.mcp`,
with this checkout as its working directory. Supply `SPEECH_URL` and
`SPEECH_CREDENTIAL` privately through the agent host's secret configuration.
The adapter calls the same authenticated HTTP service and named operation schemas;
it loads no model. Do not put the credential in command arguments or publish the
installation output directory. Native macOS SDK wheel availability is a separate
client prerequisite; the CPU runtime itself remains in Linux Docker.

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
