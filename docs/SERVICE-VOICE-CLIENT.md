# Named voice service client

These commands compose the existing HTTP service operations. They require a
configured, running service and an authorized local installation. For the
Qwen local path, `qwen-local-init` creates `client.json` and `control.key` in
the selected private root; pass that root to `--installation`. The client also
accepts the native CPU `install.json`/`credential.json` format. They do not
prove Qwen runtime readiness; check `status` and the installation's qualification
before using it for production speech.

```sh
python -m clients.cli --installation /absolute/local/installation status
python -m clients.cli --installation /absolute/local/installation enroll --name bruce --wav /absolute/trusted/bruce.wav --reference-text 'Exact words in the WAV.' --audition-text 'A short audition.'
python -m clients.cli --installation /absolute/local/installation speak --voice bruce --text 'Hello from Bruce.' --output ./bruce.wav
```

The reference must be a complete mono PCM16 WAV, within the service upload
budget. `--reference-text` must transcribe that reference exactly;
`--audition-text` is spoken separately during preparation. Enrollment submits
one preparation job, waits at most 120 seconds, then saves its one candidate.
Speech uses the saved version's identity and publishes the output file only
after all downloaded bytes match the service SHA-256. An existing output file
is refused. The CLI also accepts `--url` and `--credential`; those flags take
priority over an explicit `--installation` for their respective values.
Without that flag, `SPEECH_URL` and `SPEECH_CREDENTIAL` take priority over
the root selected by `SPEECH_INSTALLATION`. No credential is printed.

Each workflow has `--wait-seconds` (0–600), one deadline for HTTP connect,
headers, body reads, upload writes, polling, and output transfer. The client
closes a timed-out response, limits decoded JSON bodies to 32 MiB, and never
publishes output after expiry. A timeout can leave a submitted job running;
the result returns its ID only when confirmed. The service
`job cancel ID --idempotency-key KEY` command explicitly cancels.

Deadline-bearing calls require a numeric IP address in `--url` or
`SPEECH_URL`: use `http://127.0.0.1:PORT` for the local service, or
`http://[::1]:PORT` for IPv6. `localhost` and other hostnames are rejected
before connection because system name resolution has no reliable deadline.
Low-level calls without a workflow deadline retain hostname support. For
HTTPS, use a certificate valid for the numeric IP address; the URL host remains
the TLS verification name. Credentials are never sent to a redirect or proxy.

Resume enrollment with the confirmed `--asset-id`, the same name, exact
reference text and audition text, and `--job-id` only when confirmed. A bare
`enroll --wav` repeat uploads another asset and can prepare a different job.
If an upload response is lost before an asset ID is confirmed, inspect service
assets before attempting another upload. If a submission or save response is
lost, retain the returned idempotency key and confirmed handles; the workflow
checks a stable saved version before saving again. Resume speech with the
returned `--voice-version-id`, the same text, and `--job-id` only when
confirmed, so a later voice update cannot change the request. Every resumed
job is checked against its operation and idempotency key before results are
used. Existing voices can be generated directly. A configured preset such as
Ryan is registered once under its lower-case name on first use.

MCP exposes `speech_enroll` for an already uploaded asset and `speech_speak`
for a named voice. The existing `speech_asset_upload_file` tool is available
only when the MCP operator sets `SPEECH_UPLOAD_ROOT`; it accepts relative paths
within that no-follow root. MCP convenience tools never read arbitrary host
paths or return inline audio. `speech_speak` returns an asset handle; use the
bounded asset-content tool or CLI `asset content ID --output FILE` to retrieve
bytes. Low-level operations remain available for explicit inspection and
recovery.
