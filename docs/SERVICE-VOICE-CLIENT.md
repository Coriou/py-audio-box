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

Each workflow has `--wait-seconds` (0–600). A timeout leaves the job running
and returns its ID. Repeat the command with `--job-id ID` to resume; the
service `job cancel ID --idempotency-key KEY` command explicitly cancels. If
submission's HTTP outcome is unknown, repeat with the same arguments: the
workflow uses a deterministic idempotency key and returns that key. For speech,
also pass the returned `--voice-version-id` so a later voice update cannot
change the replay request. If upload
outcome is unknown, retain the error and recover a confirmed asset ID from
service/operator evidence before resuming with `--asset-id ID`; uploads are
never retried automatically. Existing voices can
be generated directly. A configured preset such as Ryan is registered once
under its lower-case name on first use; its registration is also idempotent.
If a save response is lost, the workflow returns the preparation job ID and
save key; repeating the command first checks the stable saved version ID.

MCP exposes `speech_enroll` for an already uploaded asset and `speech_speak`
for a named voice. The existing `speech_asset_upload_file` tool is available
only when the MCP operator sets `SPEECH_UPLOAD_ROOT`; it accepts relative paths
within that no-follow root. MCP convenience tools never read arbitrary host
paths or return inline audio. `speech_speak` returns an asset handle; use the
bounded asset-content tool or CLI `asset content ID --output FILE` to retrieve
bytes. Low-level operations remain available for explicit inspection and
recovery.
