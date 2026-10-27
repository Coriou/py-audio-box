# Agent guide

How an agent (or a script) gets from *"use py-audio-box, clone this voice and make
it the default voice for X"* to downloaded audio. Every step is a named operation
that HTTP, CLI and MCP all share. CLI commands are shown first and the MCP tool
names follow. Each operation's exact schema is available from
`python -m clients.cli operation-detail OPERATION_ID` (MCP: `speech_operation_detail`).

## 0. Connect

An operator installs the service once, using the
[CPU quick start](agent-interface/CPU-QUICKSTART.md) for presets or the
[Pocket quick start](agent-interface/POCKET-QUICKSTART.md) for cloning. The agent
then needs only:

- `SPEECH_URL`: for example `http://127.0.0.1:18765`
- `SPEECH_CREDENTIAL`: a bearer token from the install's private `credential.json`.
  Inject it through your agent host's secret configuration and never put it in
  prompts, arguments or logs.

```sh
python -m clients.cli discover     # MCP: speech_discover
python -m clients.cli status       # MCP: speech_skeleton_status
```

`discover` lists the **operations this installation supports** and its profiles.
Preparation and generation requests carry an `identity` object (`engine`, `model`, `bundle_sha256`,
`device`, `precision`); other operations accept only the fields in their schema. Copy it from
`profiles[].compatibility[].identity` in `discover`, choosing the entry whose
`operation` matches the call. If `voice_prepare` is missing from `operations`,
this installation cannot clone (a Kokoro preset install, for example). Report
that to the user. Don't retry.

## 1. Upload a reference recording

Pocket requires mono PCM16 WAV at 24 or 48 kHz, at most 8 s and 1 MiB, with no
extra chunks. Nothing is trimmed or resampled for you. Make sure the user is
entitled to clone the voice.

```sh
python -m clients.cli asset upload /path/to/reference.wav   # → {"id": ASSET_ID, ...}
```

MCP: `speech_asset_upload_file {"path": "reference.wav"}`. This tool exists only
when the operator sets `SPEECH_UPLOAD_ROOT`, and paths are relative to that root.

## 2. Prepare (clone) and save the voice

```sh
python -m clients.cli voice prepare --idempotency-key prep-1 --request '{
  "asset_id": "ASSET_ID", "in_tick": 0, "out_tick": OUT_TICK,
  "reference_text": "", "audition_text": "Bonjour.", "identity": IDENTITY}'
```

Ticks are 48000 per second. `out_tick` covers the selected reference, for
example `frames * 48000 / sample_rate` for the whole file. The call returns a
`job_id`. Wait for the job (step 4), then:

```sh
python -m clients.cli voice results JOB_ID          # → candidates[0].id, audition audio
python -m clients.cli voice save --idempotency-key save-1 --request '{
  "job_id": "JOB_ID", "candidate_id": "CANDIDATE_ID",
  "voice_id": "maya", "expected_revision": 0}'    # → version.id
```

MCP: put `idempotency_key` inside the arguments for `speech_voice_prepare` and
`speech_voice_save`; `speech_voice_results` takes only `job_id`. Saving a voice creates an immutable
**voice version**, and later saves add new versions to the same `voice_id`.

On a preset-only (Kokoro) install, replace this step with

```sh
python -m clients.cli voice preset-register --idempotency-key preset-1 --request '{"preset":"ff_siwis","voice_id":"maya","expected_revision":0}'
```
Use a preset selector advertised by `status`.

## Make it the default voice for X

The service has no special "default voice" setting. A default is a **voice
alias** that you name by convention, for example `default.narration` or
`default.support-bot`. Aliases are scoped to the credential, are unique, and
always point at one voice:

```sh
python -m clients.cli voice get maya                       # → revision
python -m clients.cli voice aliases --idempotency-key alias-1 --request '{
  "voice_id": "maya", "aliases": ["default.narration"], "expected_revision": REVISION}'
```

`aliases` replaces the voice's full alias list, so include any aliases you want
to keep. To move a default to another voice, remove the alias from the old voice
first. Otherwise the call fails with `alias_conflict`. Archiving a voice releases
all its aliases for reuse; the immutable voice ID and versions remain reserved.
Resolving a released alias returns `not_found` until assigned again; resolving
the archived ID returns `voice_archived`. Later, in any workflow:

```sh
python -m clients.cli voice resolve default.narration      # MCP: speech_voice_resolve
```

The call returns the voice, with `version_ids` ordered oldest to newest. Use the
last ID, unless the user pinned a particular version, then fetch that version's
`identity` with `voice version get VERSION_ID`.

## 3. Generate speech

```sh
python -m clients.cli speech generate --idempotency-key gen-1 --request '{
  "voice_version_id": "VERSION_ID", "text": "Bonjour.", "identity": VERSION_IDENTITY,
  "fallback": "forbid", "variants": 1, "reuse": "never"}'
```

MCP: `speech_speech_generate`. Text and variant limits come from the profile's
`limits` in `discover`.

## 4. Wait, inspect and cancel jobs

```sh
python -m clients.cli job wait JOB_ID --wait-seconds 5   # repeat until state is terminal
python -m clients.cli job get JOB_ID                     # state: succeeded | failed | cancelled | ...
python -m clients.cli job events JOB_ID
python -m clients.cli job cancel JOB_ID --idempotency-key cancel-1
python -m clients.cli jobs list
```

Only one expensive job runs at a time on CPU. Others queue, up to the profile's
`limits.queue`.

## 5. Download the audio

```sh
python -m clients.cli job get JOB_ID                     # → artifact_ids
python -m clients.cli artifact get ARTIFACT_ID           # → asset_ids
python -m clients.cli asset content ASSET_ID --output /path/new-file.wav
```

The CLI writes the file only after the full download matches its SHA-256, and it
never overwrites an existing file. MCP can return at most 3072 bytes inline
(`speech_asset_content`), so keep audio out of model context and use the CLI for
files. To export stereo 48 kHz PCM24, use
`audio export` ([details](agent-interface/AUDIO-EXPORT.md)).

## Rules that keep agents out of trouble

- **Idempotency keys.** Every write takes a key. Replaying the same key with the
  same request returns the same result. The same key with a different request
  returns `409 idempotency_conflict`, so use a new key for new intent.
- **Revisions.** `expected_revision` provides optimistic concurrency. Read the
  voice first, then pass its current `revision`.
- **Errors.** Errors have the form `{code, message, field, action}`. Follow
  `action`, and don't parse `message`.
- **Capabilities.** If `discover` doesn't list an operation, this installation
  can't do it. Don't work around it.
- **Quality.** Nothing verifies voice similarity or pronunciation automatically.
  Ask the user to listen to the audition (`voice results`) before saving.
