# Managed PCM24 WAV master — source candidate

The source distribution exposes the fresh CPU export installer as
`speech-box docker-export install --context orbstack --request REQUEST --handle HANDLE`,
then `speech-box docker-export up|status|stop --handle HANDLE`. It selects the
export family explicitly; `speech-box docker` keeps the original family.
The export client requires Python 3.11 and the dependencies in
`service/requirements.lock` in its operator environment (no ML packages on the
operator Mac). Supply the reviewed request, immutable images, model bundle and
private credential first. These commands never acquire or qualify them.

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

New exports of every size require **publication-v2** on an unenrolled local
store or the explicitly checked fresh installed export family: coherent SQLite `user_version=2`, P1 revision 2 and the exact 36 MiB
publication CHECK. Fresh local roots use v2. Existing v1 roots retain their
format and reads; new export/retry admission returns HTTP 409
`publication_format_migration_required`, before allocation. Retained-key replay
still precedes this gate. Speech and each prepare file stay at 1 MiB.

The private stopped-local operator transition is
`python -m service.publication_format migrate-v1-to-v2 --root ABSOLUTE_CORE_ROOT --expect-v1`.
It supports only clean, same-UID, unenrolled local roots with no dispatch history,
pending publication, live/uncertain owner, native/installed/enrollment/recovery
state or unknown namespace. It validates a bounded WAL-aware private copy,
holds the maintenance lease/control and retained owner locks, and rebuilds only
`publication_work` in one transaction while preserving typed rows and rowids.
Exit 0 means migrated or validated already-v2; 2 means refusal; 3 means failure
with confirmed rollback; 4 means outcome unknown and explicit reinspection is
required. There is no public migration API, implicit upgrade or installed writer
authority. This source commission did not migrate an actual root.

Old installed/native families stay v1 and refuse v2 before their
Core writer. Their converter gates and media/accounting limits remain unchanged.
The separately selected fresh Kokoro export family uses checked v2.
Validated local v2 and explicitly checked new-family reconciliation use a separate 160 MiB media-name budget;
SQL/JSON stays at 16 MiB, with 2048 entries and the existing finite deadline.

After separately authorized converter enrollment/qualification, a real journey is:

1. `asset upload short-24k-mono.wav`, or select a managed generated speech asset.
2. `audio export --request '{"asset_id":"ID","rate":48000,"channels":2,"loudness":null}' --idempotency-key master-1`.
3. `job wait JOB_ID`; retrieve `artifact get ARTIFACT_ID` and its output asset ID.
4. `asset content OUTPUT_ID --output master.wav`; CLI verifies the full SHA-256.
5. Check actual PCM24/rate/channels/duration, source integrity, and replay the key
   after Core reopen. `artifact_get.audio_export` carries source ID/hash, settings
   and converter registration; output asset metadata carries measured format/hash.

This journey has **only inert ASGI/SQLite/fake-process source evidence** in
r366 and [r373](reviews/H2-PUBLICATION-V2-r373-sol.md), including a 1,152,044-byte
master on fresh and explicitly migrated disposable local roots.
No real conversion has run. Existing installed Kokoro/Pocket families do not
admit this operation. Discovery performs metadata reads only and omits unavailable
export; operation detail returns `unavailable_converter`. No discovery-time probe,
FFmpeg version command, installation or download occurs.

The trusted integration seams are `AudioExports` on the owner and
`RuntimeSupervisor(audio_converters=..., converter_image_digest=...,
export_reference_roots=...)` on the existing worker. They require a distinct
`audio-export-*` profile with no speech compatibility entries. They are not new
installed enrollment APIs. Old installed entrypoints pass no converter registration and remain unavailable.
The new private Kokoro daemon family constructs both owner and worker mappings
from its checked immutable configuration; caller maps are refused.

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


The fresh daemon family is explicitly selected through
`service.daemon_client.export_main`, using the existing private install request
and exclusive handle workflow: install, up, status, stop, then same-handle up.
It requires new named volumes, configuration version 2,
`docker-kokoro-export-split-v2`, exact helper/control/worker image IDs, the
closed converter/profile/principals configuration, and independently supplied
qualification facts. There is no usable distributed converter pin or readiness
receipt. The old installer remains v1; a disk marker cannot select this family.

The new control entrypoint consumes a private notification only after inspected
seal/image/config/profile/principal/boot/start/assets and registration brackets.
It repeats control-readable pre-Core and post-acquisition checks, reconciles,
and uses the existing single Dispatcher/worker lane. Same-key replay precedes
readiness checks after checked reopen; new work and retries require readiness.
The RO assets subpath and dropped descriptor helper retain private 0700/0600
ownership. New-family publication explicitly creates 0600 output copies even
when a factory caller did not establish the PID1 umask.

r392 adds an INERT fresh installed owner journey with no copied history rows,
including 1,152,044-byte output, public HTTP/CLI/MCP, verified CLI and range
downloads, same-owner-family reopen/replay without live readiness and generated
Kokoro asset export. Fake Docker/UID/process/reap and PCM assertions are source
fixtures, never installed or converter qualification. Actual FFmpeg, pins,
Linux/Desktop/OrbStack, CUDA and Metal remain UNRUN/unqualified. Independent
Sol and final Astra review remain parent-owned; release NO-GO, legacy CLOSED.
# Converter registration size

Converter registrations are bounded to 16 KiB of canonical UTF-8 JSON. The
resolved Debian FFmpeg runtime has 214 library paths (about 10.5 KiB), so a 4 KiB
registration cannot describe its complete closure. This does not change the
48,000-byte configuration/context limits, the 256-path schema bound, immutable
file checks, executable hash, finished-image binding or qualification requirements.
