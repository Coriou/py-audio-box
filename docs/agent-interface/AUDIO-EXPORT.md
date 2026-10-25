# Managed PCM24 WAV master — source candidate

`POST /v1/operations/audio_export`, CLI `audio export`, and MCP
`speech_audio_export` accept exactly:

```json
{"asset_id":"managed-asset-id","rate":48000,"channels":2,"loudness":null}
```

All fields are required. Rates: 24000, 44100, 48000, 96000; channels: 1 or 2;
loudness: null or finite -70 through -5 LUFS. Loudness uses single-pass loudnorm
and does not certify delivery loudness. Inputs are owned managed integer PCM WAV,
at most 60 seconds / 24 MiB. There are no public paths, filenames, URLs, executable
selectors, voice substitutions or automatic provisioning.

Supply one `Idempotency-Key` and a credential with `production-write`. The compact
receipt uses existing job get/list/events/wait/cancel/retry and artifact/asset reads.
The same key and canonical payload return the original job after reopen, even
when the converter is unavailable. Changed payload conflicts. A new export
creates a separate durable PCM24 WAV; original bytes remain unchanged.

After separately authorized converter enrollment/qualification, a real journey is:

1. `asset upload short-24k-mono.wav`, or select a managed generated speech asset.
2. `audio export --request '{"asset_id":"ID","rate":48000,"channels":2,"loudness":null}' --idempotency-key master-1`.
3. `job wait JOB_ID`; retrieve `artifact get ARTIFACT_ID` and its output asset ID.
4. `asset content OUTPUT_ID --output master.wav`; CLI verifies the full SHA-256.
5. Check actual PCM24/rate/channels/duration, source integrity, and replay the key
   after Core reopen. `artifact_get.audio_export` carries source ID/hash, settings
   and converter registration; output asset metadata carries measured format/hash.

This journey has **only inert ASGI/SQLite/fake-process source evidence** in r366.
No real conversion has run. Existing installed Kokoro/Pocket families do not
admit this operation. Discovery performs metadata reads only and omits unavailable
export; operation detail returns `unavailable_converter`. No discovery-time probe,
FFmpeg version command, installation or download occurs.

The trusted integration seams are `AudioExports` on the owner and
`RuntimeSupervisor(audio_converters=..., converter_image_digest=...,
export_reference_roots=...)` on the existing worker. They require a distinct
`audio-export-*` profile with no speech compatibility entries. They are not new
installed enrollment APIs. Existing installed entrypoints pass no converter
registration, and the owner explicitly refuses installed-family export.

The registration binds an immutable worker image digest, exact executable
path/hash, reported version/build and explicit runtime closure paths from that
same image. The image digest binds dynamic libraries; an arbitrary executable
hash is insufficient. Supervisor independently checks registration against its
trusted mapping and image identity, opens the read-only managed source, and
passes its descriptor into the existing supervised child. FFmpeg has a closed
argv and writes only attempt `output.wav`. Core verifies complete classic or
extensible PCM24 RIFF independently, including duration bounds. Existing
Authority, cancellation, descendant reap, admission, ACK and publication fences
remain responsible for lifecycle truth.

Parent prerequisites: select/capture a real converter artifact and image digest,
seal/register runtime closure using the deployment's trusted image/registration
mechanism, enroll an explicitly reviewed installed family, then qualify actual
conversion and process containment. No real digest is supplied by this candidate.
Linux/amd64, Intel Mac Docker Desktop, OrbStack, CUDA and native Metal are UNRUN
and unqualified. Release **NO-GO**; legacy **CLOSED**.
