# CPU Kokoro and PCM24 export

This worked recipe builds the control and CPU worker images from this checkout,
verifies the pinned French Kokoro bundle, runs actual contained synthesis and
FFmpeg export, then installs the fresh `docker-kokoro-export-split-v2` family.
HTTP, CLI and MCP share the canonical service; no browser or GPU is required.
The expected manifest and build pins are inputs, not runtime qualification.

Use Python 3.11 and a local rootful Docker daemon with cgroup v2 and
`linux/amd64` support. Docker Desktop/OrbStack must expose an explicit local Unix
socket context. Allow at least 2 GiB worker memory plus control/build overhead,
and enough disk for the images, models, retained build logs and private volumes.
Builds acquire Debian packages and hash-locked Python wheels; model acquisition
uses bounded downloads from the pinned upstream release.

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
