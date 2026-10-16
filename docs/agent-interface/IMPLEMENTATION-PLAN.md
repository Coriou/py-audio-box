# Speech workspace implementation plan

2026-10-15 · **Canonical v1 delivery plan.** Current execution status and accepted gates are recorded in [EXECUTION-LEDGER.md](EXECUTION-LEDGER.md). This document supersedes earlier delivery sequences and mandatory-feature lists in [PLAN.md](PLAN.md) and the research reports. [PRODUCT.md](../../PRODUCT.md) records product direction; [IMPLEMENTATION-START.md](IMPLEMENTATION-START.md) is the agent entry point. The [fresh product review](review-fresh-product-astra.md) establishes GO to plan, not runtime or release acceptance. Inspected source baseline: `ae67cd6a572e7105b7d41365aac0be4059d79584`; recheck the implementation checkout before each lot.

The user confirmed the audience and capabilities below. Architecture, library choices, limits and sequencing are **working engineering defaults**, justified here and changeable through a bounded decision record. This plan authorizes no model download, paid resource, cloud provisioning, deployment or application change by itself.

At plan authoring, all phase artifacts and exit checks below were future implementation work. The document itself never proves completion: only the execution ledger and recorded integrated evidence establish accepted gates.

## 1. Product outcome and boundaries

Give agents reusable voices and recoverable speech work with little context, supervision or wasted compute. People and agents share one private, self-hosted workspace through HTTP, a small CLI, MCP and a complete web speech-production studio. Docker installation serves one person or a trusted team locally or on a private server.

V1 includes microphone/file/remote-source acquisition; presets, cloning and designed voices; reusable voice versions; synthesis, jobs/results; scripts, speakers, alternatives and selected takes; imported recordings/music/SFX; a multitrack timeline; preview, history and render/export. The early service milestone is useful but **does not constitute v1**.

Five release journeys govern implementation:

1. Install a declared profile, discover readiness, connect an agent, generate and retrieve first speech without reading source.
2. Record/upload/import, audition and select a reference, prepare a clone, explicitly request a test, then reuse the voice by ID.
3. Describe a voice, generate candidate designs, audition one and save a reusable identity with its origin and version.
4. Create a multi-speaker script, generate/select takes, arrange tracks, listen, export and reopen the same production.
5. Alternate human/agent edits, fix one line, recover failure and retain other takes, timing and completed results.

Preserve Qwen defaults, existing voice IDs/aliases, authored text, source bytes, provenance and current CLI/Redis workflows. CPU alternatives are explicit capabilities, never silent identity/engine substitutions. Existing internal transcription, QA and alignment remain where current workflows use them.

**Outside v1:** public STT/transcription, contextual dictation, TxCode integration, native dictation clients, realtime duplex, remote GPU orchestration, multi-tenant SaaS, model training, arbitrary executable workflows, MIDI/video/plugins and a mastering suite. TxCode speech-input experiments are independent. Optional completion delivery, portable production bundles, additional codecs/stems/captions, pronunciation dictionaries and formal review locks cannot become release prerequisites by appearing in research. Operational backup/restore remains mandatory.

## 2. Shared contracts and architecture

### Domain and transport boundary

Use a Python domain layer with FastAPI/Pydantic HTTP validation, SQLite metadata and immutable managed files. Use React/TypeScript/Vite for the compiled studio; keep editing state independent of any timeline component. A small Python CLI and stdio MCP adapter call HTTP. Pin maintained dependency/SDK versions during Phase 0; no inferred current protocol version from old research. These defaults fit the existing Python engines and avoid separate product state in each client.

Proposed new locations are `service/` (domain/store/API), `worker/` (supervision/adapters), `studio/`, `clients/` (CLI/MCP), `contracts/` (schemas/fixtures), and `tests/service/`, `tests/studio/`, `tests/release/`. Phase 0 records the final paths before agents divide ownership. Existing `apps/` commands remain available.

| Resource | Canonical meaning and invariants |
| --- | --- |
| Asset | Immutable original or derivative, hash, measured media properties, source/provenance; managed ID rather than arbitrary server path. |
| Voice / version | Stable identity and legacy aliases; immutable reference/preset/design lineage, exact engine compatibility and settings; archive without breaking old work. |
| Project / revision | Exact script blocks with stable IDs, speaker roles, voice assignments, selected takes, tracks/clips and export settings. One revision precondition covers each atomic edit. |
| Take | Immutable submitted text/settings, voice/engine assets, generation identity and audio; may be stale against current text without becoming unusable. |
| Job / attempt | Durable requested operation and frozen inputs; owned execution generation, deadline, stages, errors and results. A retry is traceable and never silently changes inputs. |
| Artifact / render | Immutable deliverable and manifest tied to input/revision hashes. Draft edits cannot change a running render. |

Version `/v1` operations and persisted schemas independently. Use typed discovery, voice/acquisition, generation, project-edit, render, job and artifact operations; do not expose arbitrary argv, filesystem access or generic JSON command execution. CLI/MCP are small task adapters, not a second business layer. Agent discovery returns cached support/install/configuration/readiness/verification states with timestamps and corrective actions; no inference import, weight load, network lookup or model download.

Require expected revision for edits. A stale request returns `409 revision_conflict`, current revision and bounded changed IDs; no silent merge/overwrite. Accept atomic semantic edits and bulk block generation/status; default reads are summaries, paginated lists or scoped blocks/ranges. Keep authored text exact; store any explicitly selected synthesis transformation separately. Pronunciation infrastructure is unnecessary for the initial contract.

Submission returns a durable job ID and compact receipt. Scope idempotency to principal and operation; same key/canonical request returns the original job, changed input returns `409 idempotency_conflict`. Persist the input fingerprint and returned IDs while the job record exists; an expired/deleted key is explicitly no longer deduplicated. Content reuse is a separate explicit policy; intentional alternatives use new requests. Unsupported controls fail before admission with stable codes, field/action and bounded detail. HTTP problem responses and CLI/MCP errors share these semantics.

### State and execution

One control owner writes SQLite on local persistent storage: WAL, FULL synchronization, foreign keys, short transactions, bounded busy handling and versioned migrations. Workers never open the database. This is a new small executor for the service; `lib/jobqueue.py` already has useful leases/reclaim/owner checks, but the legacy Redis runner remains independent. No Redis dispatch or bidirectional registry synchronization for new service jobs.

State path: `queued → running → succeeded|failed`; cancellation uses `cancel_requested → cancelled` only after stop/reap confirmation. Completion may win the cancellation race. Stages such as loading, preparing and rendering are progress, not separate terminal outcomes. A disconnected client does not cancel durable work. On restart, fence old attempts, reconcile publication and establish owner death before retry; report interrupted work with an explicit retry when safe automatic continuation is unavailable. Default heavy-operation attempts: one; automatic retries require a classified case and an explicit bounded policy. Exactly-once physical inference is not promised.

Workers write only attempt-owned staging. Control verifies required bytes/hashes and media, adopts immutable files on the managed filesystem, then commits current-generation references and success. Filesystem adoption and DB commit are **not jointly atomic**: crash recovery reconciles orphan files and incomplete attempts; success may never reference missing required media. Late generations cannot publish. Deleting staging follows confirmed worker reap; unreapable owners quarantine the lane. Preserve Pocket's parent-owned private `0700` reference namespace and Qwen's serialization/stop/reap boundaries.

The constrained-CPU default starts with one host-wide expensive-work slot, one resident backend configuration and FIFO admission. Inference, large decode/render and provisioning compete under the same conservative budget; lightweight reads/uploads/playback remain available. Reuse compatible resident weights across jobs, key conditioning by exact reference identity, and unload only while idle according to a configured policy. Extend concurrency or affinity only after measurements. Queue, upload, decoded duration, text, variants, disk and execution ceilings are enforced profile configuration, frozen with boundary fixtures in Phase 0—not guesses about universal hardware capacity.

### Packaging, provisioning and access

Baseline container topology: a lightweight control/UI image, the selected worker image, and a narrow fetch/provision helper invoked through typed operations. The optional native Metal worker is a separately qualified topology described below. Start with private Unix-socket IPC on a dedicated Docker volume and a `network_mode: none` worker, read-only assets and writable staging. The helper has bounded network access; the worker has neither credentials nor DB access. Verify socket ownership/reconnect/cancel on each supported daemon. If a target cannot support this arrangement, qualify a restricted internal transport explicitly; never retain a false networkless claim.

The baseline container install needs no source checkout, writable source mount, host ML installation, Redis credential or Docker socket. An optional packaged native Metal worker must bundle/manage its runtime instead of making manual host ML setup a hidden prerequisite. Serve compiled UI. Explicit provisioning stages downloads, verifies pinned assets, then atomically activates a bundle; interrupted/download-disk failures preserve the previous ready bundle. Provisioning identity and runtime package identity are recorded separately. Connecting, listing and ordinary synthesis must not provision. A versioned new `speech-economy-v1` preset may request one take and no optional QA/alignment; legacy defaults remain unchanged.

Loopback is the default bind; private-server deployment uses documented HTTPS. Expiring single-use setup establishes the operator; browser sessions use maintained session libraries, HttpOnly/SameSite cookies, Secure under HTTPS and CSRF/Host/Origin checks. Named revocable agent credentials have read, production-write or admin scope; protect media/range downloads too. Keep secrets out of URLs, default logs and tool results. Baseline MCP uses stdio plus a configured HTTP credential; remote MCP is optional and separately qualified.

Remote acquisition initially supports public HTTPS direct-media URLs and a named supported hosted-media importer backed by existing `voice-split` behavior; Phase 0 freezes the supported source list and fixtures. Reject arbitrary sites/credentials, cap redirects/bytes/time/decoded duration, validate every destination against private/link-local/metadata addresses and DNS rebinding, and isolate decoding. Any administrative private-source exception is installation configuration, never a per-request bypass. No remote pickle or untrusted prompt deserialization.

Keep originals, references, voice/project history, selected takes and outputs until explicit deletion. Default automatic cleanup covers only verified unreferenced transient files and rebuildable derivatives, with a grace period and live-job/reference checks. Disk pressure pauses admission with a recovery action. Archive and explicit dependency-aware deletion are separate; a UI action cannot silently erase shared production material.

### CPU baseline and accelerator support

The primary everyday targets are this constrained Intel Mac and the smaller remote Grok Bot machines. CPU-only operation is a first-class experience, not merely a fallback: cheap idle/discovery, bounded RAM/disk/threads, one expensive job initially, explicit provisioning and useful preset/clone workflows on declared profiles. Inventory each actual target's CPU/ISA, RAM, VM/container limits and available disk before setting its ceilings; the remote machines' specifications are not inferred from their names. STT timings do not establish TTS performance.

Better hardware must retain the same API, voice identities, production features and durable-job semantics while benefiting from compatible acceleration. Phase 0 defines a transport-independent worker/device profile; Phase 2 implements and qualifies CPU plus NVIDIA/CUDA profiles and a macOS Metal qualification lane. GPU/Metal support is an explicit delivery objective, not excluded by the CPU-first ordering. Unsupported or untested combinations are reported honestly rather than advertised as working.

- **CPU Docker:** baseline install without host ML tools; qualify the Intel Mac/OrbStack profile and the actual smaller Linux server profile independently. Preserve responsiveness under memory pressure and concurrent agent-host activity; do not assume an entire host is dedicated to speech.
- **NVIDIA/CUDA:** selected GPU worker image and runtime/device wiring on Linux, with Windows/WSL2 separately qualified. A private server with a GPU is an in-scope installation; deferred remote GPU orchestration means automatic cloud rental/dispatch, not exclusion of GPU-equipped private servers.
- **macOS Metal:** native Apple acceleration is distinct from Linux-container GPU passthrough. Keep the control/UI/API Docker-based, but permit an optional packaged host-native worker/companion when the selected TTS backend needs macOS MPS/Metal. Qualify exact hardware, backend, model, operators, precision and lifecycle; no generic claim that PyTorch MPS or Docker Model Runner makes every TTS engine work. The basic CPU install must not acquire mandatory host ML dependencies. This worker uses the same typed asset/job/result protocol, scoped authenticated local transport, stop/reap and control-owned publication; it does not open the service DB or rely on Linux paths being macOS paths. Its host isolation/network properties must be stated separately from the networkless container profile.
- **Other accelerators:** preserve adapter/profile seams and qualify named implementations when available; do not substitute an untested generic GPU promise for working CUDA/Metal paths.

Expose requested/effective engine, model, device and precision in the resolved plan and artifact metadata. New-service `auto` selection may choose only installed, qualified, compatible profiles within operator limits; selecting a device must not silently change the voice, model, quality policy or trigger downloads. Explicit accelerator requests fail with an actionable error if unavailable; CPU fallback requires a declared caller/operator policy and remains visible. Legacy defaults remain unchanged.

The single expensive-work slot is a conservative CPU default, not a permanent ceiling for powerful machines. Keep memory/VRAM admission and device ownership explicit so qualified profiles can increase batch size, resident-model capacity or concurrency after measured stress/cancel/recovery tests. Higher-quality models are explicit provisioned choices. Validate CPU throttling, idle unload, GPU/Metal memory release, cold/warm latency and output quality; never transfer an STT tuning workaround to TTS without measurement.

Platform references checked 2026-10-15: [Docker GPU passthrough](https://docs.docker.com/desktop/features/gpu/), [Model Runner inference engines](https://docs.docker.com/ai/model-runner/inference-engines/), [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html). These establish platform mechanisms, not this product's TTS compatibility.

## 3. Studio interaction contract

Use an **Operate** workspace: projects and voices are primary; connection, engines, storage and diagnostics are supporting settings. In Phase 0/1 establish the actual visual direction and surface brief using the existing Impeccable process; this plan does not invent a brand or require a visual questionnaire before engineering begins.

The production view pairs exact script/speaker/take editing with a resizable multitrack timeline, shared selection and transport. Required tools: waveform playback, scrub/zoom, explicit placement/gaps, overlap, trim/split/move, clip gain/linear fades, track/clip mute, track solo, imported beds, keyboard timing fields, undo/redo, revision history, autosave and queued export. No musical grid, effects graph or advanced mastering is implied.

Use 48,000 integer ticks/second, nonnegative browser-safe values, end-exclusive ranges and round-half-up conversion. Preserve original bytes; create explicit 48 kHz derivatives only when needed. Freeze channel mapping, fade endpoint behavior, output length and overlap rules in `contracts/`, shared by preview and renderer. Preview tolerances are measured separately from exact offline sample tests.

Fixed placement is default. Script reorder does not rearrange an edited timeline. First assembly is an explicit operation with a gap value; ripple edits explicitly name track/range/anchor and affected IDs. Changing text marks only its dependent takes stale and starts zero inference. Generation adds candidates, never automatically replaces a newer selection. Adopting a longer take leaves following fixed clips in place and exposes overlaps; adopting an incompatible shorter take requires trim adjustment while retaining the old playable arrangement. Undo creates a new revision; session undo/redo survives refresh through recorded operations, while history can restore an older revision without erasing intervening edits. After another client's edit, an undo conflict offers scoped recovery rather than reverting that client silently.

Autosave shows Saving/Saved/Offline changes/Conflict. Retain unsent browser drafts without claiming server persistence; reconnect performs a revision check. Voice acquisition supports secure-context microphone permission, visible recording/stop, upload and supported URL input; audition originals/candidates, trim a reference, prepare separately from test synthesis. Distinguish preset, clone and designed origins and technical readiness from human audition. Do not impose an approval/lock workflow.

Render defaults to 48 kHz PCM24 WAV, explicit mono/stereo mapping and a strict no-hidden-normalization clipping policy: fail an over-range final mix with a corrective action. Missing media/placeholders fail; stale selected takes require explicit inclusion in the render request. Partial exports name their selection and omissions. Render asynchronously from a frozen revision; provide audio and provenance manifest. Preview uses the same gains/placement, flags potential clipping and does not silently duck/normalize. Additional delivery codecs need separate qualification.

Test keyboard-only editing, visible focus, textual alternatives to dragging, labelled controls, non-colour statuses and reduced motion. Desktop is the full editing target; narrow/mobile layouts support recording, audition, job review and legible project inspection without misleading unsupported precision editing. Virtualize long scripts, request decimated peaks/range media and avoid decoding a whole production into browser memory. Operational paths, model identifiers and lease/IPC vocabulary belong in expandable diagnostics, not ordinary user flow.

## 4. Preservation and qualification map

| Existing surface | Required integration/preservation |
| --- | --- |
| `lib/voices.py`; `voice-synth` registry commands | Non-destructive import with aliases, versions, hashes, provenance gaps, tones/profiles and portable WAV/JSON; preserve rename/archive/export semantics. |
| `voice-split`, `voice-clone`, `voice-register` | Extract callable stages; structured candidates/selection/preparation; preserve VAD/separation/internal ASR and current legacy orchestration. |
| `voice-synth`, `lib/tts.py`, Qwen adapter | Clone, CustomVoice, design, language, variants/seed, tones, chunking, profile controls, optional QA/ranking/alignment. Advanced options may remain an explicit typed advanced/legacy path, never disappear. |
| `voice-cpu`, backend adapters, `run-cpu`, provision script | Preserve asset hashes, offline execution, warm batches, rejection semantics, metadata, ownership and publication; add service voice identities. |
| `lib/fidelity.py`, `audio-export` | Reuse concatenation/export primitives; build a separately tested pure composition renderer. Experimental SFX mixing is reference material, not a production import with remote generation enabled. |
| `job-runner`, `job-watcher`, `lib/jobqueue.py` | Preserve legacy manifests/defaults/Redis/Vast behavior; import completed beatsheet packages with source IDs. Pending jobs remain with their runner. |

Phase 0 expands this map to every public command/control and assigns service, advanced compatibility or unchanged legacy coverage. No deletion through omission. Dry-run migration reports collisions, missing files/provenance and pending jobs; apply copies into a distinct data root with hash checks and an import ledger. Never give legacy and service writers one mutable registry.

| Capability | Adapter truth to preserve | Qualification required for an advertised profile |
| --- | --- | --- |
| Preset speech | Kokoro FR/EN presets; no clone/direction/seed. Qwen CustomVoice has separate named speakers. | Explicit provision, cold/warm speech, language/control validation, listening. |
| Reference preparation / clone | Qwen Base 0.6B/1.7B; direction rejected. Pocket uses explicit per-language assets/reference/seed, no preset/direction. | Import/prep, identity, reference costs, cold/warm clone, timeout/reap and listening. |
| Directed built-in voice | Qwen CustomVoice 1.7B; 0.6B rejects direction; feature flag remains authoritative. | Correct checkpoint/controls and actual directed-speech audition. |
| Designed voice | Qwen VoiceDesign 1.7B candidate generation followed by reusable Base clone conversion. | Model switch, candidate/save/reuse lineage, memory/resource use and audition. |

Qualify the Intel Mac Linux-container CPU and actual smaller Grok Bot Linux CPU targets first, then named CUDA and macOS Metal profiles through the explicit accelerator lane above. Windows/WSL2 NVIDIA and arm64 CPU rows require separate release tests. Apple GPU, AMD/other accelerators and unavailable hosts remain unqualified until demonstrated; upstream support is insufficient. CPU installation must work on a declared older Intel target. Full v1 must demonstrate cloning and design on at least one declared capable profile, not pretend every CPU host supports them. A required accelerator profile without hardware evidence is an explicit open release decision, not silently dropped or labelled passed, and not authority to rent GPUs.

## 5. Ordered implementation phases

Each work package is a small independently reviewable lot with the stated output. Its gate is **GO** only for that outcome and evidence tier; unresolved required behavior is **NO-GO** for dependants. A branch with unit tests is not integration proof. Partial parallel lanes may continue against frozen fixtures; release requires all core gates.

### Phase 0 — Freeze executable boundaries and retire editor risk

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 0A Contract and compatibility pack | Versioned resource/operation schemas, state transitions, composition rules, errors, fixture data, complete feature mapping, profile ceilings and migration policy. | First; one contract owner resolves changes. Start with a three-speaker/30-block fixture plus impulses/sines and a 100-block/200-clip scale fixture. |
| 0B Editor decision | Small browser fixture: overlap, trim/split, keyboard movement, reload, undo and take replacement against canonical data. Test Waveform Playlist first; record adapter/licence/maintenance cost. | Uses 0A; framework/components can be explored with fixture data while schemas finalize. If unsuitable, use a thin custom timeline with a qualified waveform component; no whole-product race. |
| 0C Integration skeleton and acceptance ledger | Proposed directories, pinned toolchains, cheap discovery/MCP catalogue fixture, release Compose skeleton and named evidence matrix. | Merge after 0A; component adapter after 0B. Set initial limits from fixtures; record provisional resource/token targets, never measured claims. |

**Tests and exit:** schema examples and rejection cases validate; text/ID/legacy-default snapshots survive; browser fixture demonstrates intended overlap/reload/undo rather than relying on the existing headless probe. Capture token/schema bytes using a named tokenizer/client fixture. GO when a single state contract, editor approach and packaging boundary are executable decisions. No real weights or model campaign required.

### Phase 1 — Packaged walking slice with all clients

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 1A Durable core | SQLite migrations, identities/revisions, idempotent jobs, attempt fences, artifact publication, fake worker and synthetic audio. | 0A/0C; merge store/domain before transports. |
| 1B HTTP + CLI + MCP | Shared errors, compact discovery/submit/wait/status/download and scoped project edits; named credentials, artifact authorization, replayable job events. | Develop against 0A fixtures; integrate against 1A, not independent mock business logic. |
| 1C Installed browser slice | Compiled studio in control image; synthetic script/takes/timeline save/reload/play/render; connection and readiness states. | 0B + 1A/1B; Docker/fake-worker smoke from fresh volumes. |

**Tests and exit:** new local `tests/service/` fault harness injects restart before/after file adoption/DB commit, duplicate submits, changed-key payload, cancel/load/render races, stale lease/result, disk full, missing media and conflicting edits. One authoritative result; no false cancellation or lost draft. Same fixture traverses HTTP, CLI, MCP and browser against the same service. Metadata remains ML-free while fake work blocks. GO requires a packaged first task and browser round trip, not just route/unit tests.

### Phase 2 — Real engine integration and safe installation

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 2A Adapters and registry migration | Dry-run/apply import; Qwen generation through pinned service inputs; CPU adapter; structured progress/errors; compatibility fixtures. | 1A/1B. Temporary typed subprocess bridges allowed; extract callable orchestration where residency/errors demand it. |
| 2B Provision/resource lifecycle | Selected CPU/CUDA worker images, staged verified provision/activation, transport owner/cancel/reconnect, resident reuse and idle unload; install readiness UI. Optional packaged native-worker transport for Metal shares the same contracts. | Parallel against 1A worker protocol; integrate before real synthesis acceptance. |
| 2C Bounded TTS qualification | Intel Mac and smaller Linux CPU profiles, NVIDIA/CUDA path and explicit macOS Metal backend/companion qualification; per-capability support and measured limits. | 2A/2B integrated artifacts; use available authorized hardware. Missing required accelerator evidence stays open for release scope; independent work continues. |

**Tests and exit:** extend `test_capabilities_and_metadata`, `test_tts_runtime_contract`, `test_qwen_worker_diagnostics`, `test_voice_cpu_batch`, `test_cpu_output_publication`, `test_clone_prompt_identity` and registry/registration security suites where contracts change. Verify service-owned real cold/warm jobs, model-load count, failed-owner replacement, no overlap with unreaped owner, offline inference and preserved references. Cover CPU-only startup, invalid GPU selection, visible fallback policy, per-device admission, Metal companion absence/disconnection and actual effective-device execution. Test every advertised profile's constraints, not only a mocked accelerator flag; listen separately. GO for each real-speech profile requires its own worker/release evidence. Missing design/accelerator hardware remains an explicit open gate for that claim, not a fake pass or a block on independent fixture work.

### Phase 3 — Complete voice acquisition and creation

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 3A Acquisition | Mic/file/URL journeys, staged safe ingest, original/candidate audition, explicit segment selection, provenance and recoverable drafts. | 1 clients + 2 storage/worker; UI can start against fake fixtures. |
| 3B Voice creation | Separate prepare/test; presets, clone tones/profiles and designed candidate/save/reuse; capability-filtered controls and archive/history. | 3A + 2 adapters; clone and design can be separate lots over identical voice-version fixtures. |
| 3C Shared agent/human journey | Compact acquisition/design/voice tools and CLI parity, structured next actions, legacy advanced-control coverage. | Integrate after 3A/3B, with real shared IDs. |

**Tests and exit:** malformed/oversized media, URL redirect/DNS restrictions, unsafe archive/pickle rejection, denied mic/HTTPS failure, interrupted import/prep and stale voice-version races. Extend `test_audio_selection`, `test_voice_clone_quality_gate`, `test_voice_registration_atomicity`, `test_clone_prompt_compatibility` and selected profile/control suites. In real browsers complete localhost and private-HTTPS acquisition; a person auditions a prepared/designed voice and an agent reuses that exact version. GO needs all three acquisition routes plus distinct preset/clone/design behavior on qualified profiles.

### Phase 4 — Script, takes and targeted revision

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 4A Production editing | Speakers, exact blocks, bulk generation, candidate/selected takes, stale markers and explicit first assembly; scoped API/CLI/MCP mutations. | 1 project contract + 2 real speech. Can proceed alongside Phase 3 against voice fixtures. |
| 4B Human/agent history | Autosave/offline drafts, conflict recovery, revision changes view, undo/redo and beatsheet/completed-package import. | 4A; UI and transport owners share conflict fixtures. |

**Tests and exit:** three-speaker/30-block workflow plus ten-line human editing task. Change one line: only its declared generation dependency is stale; generation is explicit, other audio/timing unchanged. Late take cannot select itself. Concurrent person/agent edits yield conflict, not overwrite; refresh/reopen preserves choices/history. Exercise partial batch failure, new intentional takes and retry without duplicate work. GO requires one alternating human/agent task on the integrated real service; fakes cover combinatorial races cheaply.

### Phase 5 — Complete timeline and trustworthy delivery

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 5A Full arrangement | All Section 3 editing controls, imported beds, fixed/ripple semantics, keyboard/accessibility and long-project behavior. | 4 shared state and 0B adapter; implement continuously from early timeline, not a new editor. |
| 5B Pure renderer/export | Frozen composition plan, explicit conversions, gain/fades/overlap, strict validation, PCM24 master and manifest. | 0A semantics + 1 job lifecycle; synthetic renderer work can precede 5A, integrate preview before gate. |
| 5C Finish production | Browser preview/export agreement, asynchronous render status/download and scoped agent assembly/render. | 5A/5B combined with 4 takes. |

**Tests and exit:** impulses/sines assert sample count, trim endpoints, offsets, gains including zero, fades, overlap/silence, channels and clipping policy; missing media never disappears silently. Preserve source hashes; edit draft during render and verify captured revision. Extend `test_fidelity`/`test_audio_export` appropriately. Real browser keyboard/drag, invalid replacement trim, undo/reload and 100-block/200-clip inspection; actual listen to cuts/overlaps and final export. GO means completing the full production journey, not a waveform display or passing mixer unit test.

### Phase 6 — Release and operational qualification

| Package | Output / interface | Dependency and integration order |
| --- | --- | --- |
| 6A Install/upgrade/restore | Pinned release artifacts, clean-install guide, supported profile matrix, backup/restore/migration/rollback procedure and diagnostic bundle. | All core phases; packaging has already been exercised since Phase 1. |
| 6B Product acceptance | Actual first/returning agent tasks, measured context/compute, security/accessibility/scale checks and human workflow/audio acceptance. | Integrated release candidates and declared clients/hardware. |
| 6C Optional completion | Registered target, signed outbox event, bounded retry/deduplication/delivery history and one available host/runner adapter. | Independent optional lane after 1 events; does not gate core release unless advertised. |

**Tests and exit:** follow install instructions on fresh supported hosts/volumes without source/dev tools; explicitly provision, run all applicable capability journeys, disconnect network for inference, restart, upgrade and restore into a fresh root. Security tests exercise scopes, CSRF/Origin, authorized range reads, remote acquisition and redacted logs. Optional delivery tests cover receiver outage/duplicate/reordered events and actual host resumption; notification failure cannot rerun synthesis. GO to release only with the acceptance ledger below; no unsupported profile or pending acoustic decision is labelled accepted.

## 6. Evidence, release and operations

Use five evidence tiers, named in every handoff: **E1** static/schema/offline unit; **E2** integrated fake-worker/process/container failure tests; **E3** real worker/release-image/hardware measurements; **E4** actual browser/agent-host workflows and accessibility; **E5** human listening and production acceptance. E1/E2 do not certify engine quality, installed hardware or UX; E3 does not certify voice identity/intelligibility. No numerical speed, memory or token improvement without a controlled comparison.

Existing regression entry point is `scripts/run-unit-tests.sh`: it builds `Dockerfile.tests`, then runs network-disabled pytest excluding integration/Redis markers. Select relevant tests during lots; run the complete offline suite at integration/release. Update image COPY/dependencies when new modules/tests are introduced so collection proves the actual code is tested. Legacy Redis integration remains a separate disposable Redis test when its paths change. **Do not use `scripts/run-e2e-test.sh` or remote/Vast scripts as a local product gate**; they have external side effects. Build a new local service/browser/release harness with synthetic media and isolated volumes. No low-value tests that merely mirror getters or prose changes.

Agent measurement uses frozen tasks/voices/settings/hardware and a named agent model/client/tokenizer: first connection versus returning voice, cloning/design, multiblock production, one-line correction and failure recovery. Count onboarding/tool schemas including host duplication, all requests/results/retries, calls, active model turns, model loads/inference starts, regenerated blocks and wall time. Initial guardrails: eager catalogue/onboarding ≤2,500 model-visible tokens, receipt ≤2 KiB, status ≤1 KiB, list page ≤20; adjust only from recorded task success/cost evidence. Known-voice speech should expose one submission and one terminal result to the model; CLI/runner performs bounded waiting. No inline audio/logs or required preflight on every utterance.

Measure control-plane idle/loaded responsiveness separately from inference. Record exact image/assets/hardware/ISA/VM, threads/device, queue/load/generation/export times, audio duration, failures, disk, model-load count and memory scope. Process-group sampled PSS/USS maxima are observations, not hard host/container peaks; preserve incomplete-sample counts. CPU STT results are not TTS qualification. Publish resource ceilings and measured limits, not invented benchmark promises.

Upgrade procedure: stop new admission, drain or explicitly interrupt/fence jobs, take a consistent DB plus referenced-file backup, retain prior image/schema manifest, run versioned migration, start and verify readiness/reopen/render. Model activation is independent. Restore verifies hashes and schema into a fresh root before reopening. Rollback uses the compatible prior data snapshot if a reverse migration is unsupported; changing image tags alone is insufficient. Test restore with voices, selected takes, projects and finished exports; caches/bundles may be reprovisioned explicitly. Preserve backup retention and keys according to operator policy, never print secrets.

Release checklist:

- All five journeys and complete studio pass on the integrated candidate; all feature-map entries have coverage or an explicit unchanged compatibility path.
- HTTP/CLI/MCP and browser share IDs/revisions/errors; cheap connect never loads/downloads; cancellation/publication/conflicts survive injected failure.
- Declared older-Intel Mac and smaller Linux CPU installations work under measured limits. Each advertised capability/profile has exact runtime receipts; clone/design work on at least one declared capable profile. CUDA and Metal objectives are separately accounted for as qualified or an explicit unresolved release-scope decision, never silently omitted; stronger hardware is not capped by the CPU scheduling policy.
- Clean install, explicit provision, offline inference, upgrade, fresh restore and access/media boundaries pass; redistribution/dependency/model terms resolved before distribution.
- Real agent success/cost evidence, browser/accessibility acceptance and human listening are recorded independently. Remaining limitations are named in release notes; optional features are not implied.

## 7. Decisions, risks and agent handoff

| Decision | Status / closure |
| --- | --- |
| Audience, Docker, HTTP/CLI/MCP, full studio, voice acquisition/creation | Confirmed. Change only on new user direction. |
| Own service + current backends; SQLite; FastAPI; React/Vite; local supervised worker | Working defaults. Phase 0 pins implementation; changes require concrete compatibility/maintenance evidence. |
| Waveform Playlist | Candidate. Phase 0 browser fixture selects it or thin custom timeline + qualified waveform component; canonical state is unaffected. |
| Voicebox fork / LocalAI | Not selected. Reopen only for a concrete savings/gap map; no mandatory prototype race. |
| IPC, profile ceilings, exact dependency versions | Phase 0 engineering freeze, then packaged qualification. A failed target row cannot be silently labelled supported. |
| Profile/hardware and agent hosts | Qualify available named targets; unavailable targets stay unqualified. No external spend implied. |
| Visual identity / name | Working repository name; establish Operate surface direction during early UI, resolve distribution naming later. |
| Completion adapters, extra delivery/portable formats | Optional separate lots with their own acceptance; durable retrieval always works. |
| STT/dictation/TxCode and remote orchestration | Deferred; no v1 prerequisites or implementation here. |

| Risk | Early detection / response / owner |
| --- | --- |
| Editor fights canonical state or degrades at scale | Phase 0 browser probe and Phase 5 scale task; component adapter fallback; studio owner. |
| CLI extraction loses voice/default semantics | Phase 0 feature ledger, Phase 2 frozen input/output/identity fixtures; preserve typed legacy bridge; backend owner. |
| Clone/design impractical on target | Phase 2/3 capability-specific measurements; declare another available qualified profile or release gap; runtime owner. |
| Restart/cancel publishes wrong output | Phase 1 fault matrix and Phase 2 real stop/reap; quarantine/fence rather than optimistic retry; service owner. |
| Downloads/dependencies/disks overwhelm install | Phase 1 packaging, Phase 2 failed provision and Phase 6 clean host; explicit profiles/quotas/restore; packaging owner. |
| Tiny schemas create costly task failure | Phase 1 client fixture and Phase 6 real-agent comparison; adjust tools from total successful task cost; contract owner. |
| Research grows into unbounded scope | Every PR names core/optional/deferred mapping; integration owner rejects unplanned prerequisites. |

A new dev agent starts with [IMPLEMENTATION-START.md](IMPLEMENTATION-START.md), then reads PRODUCT, this plan's shared contracts/boundaries and assigned phase, its package contracts/fixtures and relevant source/tests. [CURRENT-PLAN.md](CURRENT-PLAN.md) and historical reports are optional context for a specific question, not mandatory bootstrap reading. Check applicable AGENTS instructions, current Git status/HEAD and the accepted gate ledger. Proceed without re-asking for routine bounded local checks already authorized by the implementation assignment. Legacy remote/cloud scripts, private-audio access and heavy model campaigns require authorization within that assignment; do not infer it from this planning document. Use synthetic/licensed inputs by default, and do not assume earlier runtime receipts apply to new source.

Known current verification command: `scripts/run-unit-tests.sh` with its image-build cost described above. New service, browser, client-build and release-smoke commands do not exist yet; Phase 0/1 must add their pinned package scripts and document exact invocation in the handoff. Do not invent `npm test`, a service entrypoint or a release Compose filename before that package exists. Read existing shell commands before running them; filenames do not establish safe/local behavior.

Assign one integration owner and one owner for shared schemas/migrations; backend, studio, clients and packaging lots may proceed in isolated worktrees after contract fixtures merge. Keep PRs bounded to one package/subslice. Schema changes land first with consumer fixtures and migration compatibility; consumers integrate next, then one combined vertical acceptance run on the integrated commit. Never copy isolated trees over another checkout or equate a detached patch's tests with integrated proof. Each handoff names exact commit, files/interfaces, fixture version, commands/evidence tier, limitations and the next dependency. Keep credentials and raw operational receipts out of shared reports.

Use this concise package template:

```text
Package / owner / base commit:
Outcome and core-or-optional scope:
Allowed files; preserved defaults/data; excluded side effects:
Required predecessor gates and contract/fixture versions:
Deliverables and interfaces changed:
Meaningful tests + evidence tier; runtime costs/authorization:
Integration order; migration/rollback; unresolved limitations:
GO/NO-GO with exact evidence and next owner:
```

Example Phase 1 exit: `[ ]` same service exercised by four clients; `[ ]` duplicate/conflict/restart/cancel/publication fixtures pass; `[ ]` browser synthetic overlap/edit/reload/render succeeds; `[ ]` compiled images run from clean volumes; `[ ]` discovery imports/downloads zero models; `[ ]` receipts/token measurements recorded; `[ ]` legacy regressions pass; `[ ]` integrated commit and E1/E2/E4 fixture evidence named. Any unchecked required item is NO-GO for claiming that milestone. It is not a reason to expand into unrelated research or to run unapproved heavy work.
