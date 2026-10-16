# Product journeys, speech studio and install research

Planning only, 15 October 2026. No application changes, image builds, model downloads,
inference, browser microphone access or secret inspection were performed. No
`AGENTS.md` was found in the checkout or its ancestor directories. The existing
`.impeccable` directory yielded no readable context files. The parent owns the
Impeccable planning workflow.

Confirmed scope: one person or a trusted team, self-hosted locally or on a private
server; HTTP API plus a small CLI and MCP access for any agent; a full speech
production studio with scripts and timeline editing. These are v1 requirements.
The staged delivery below sequences that scope; it does not defer the studio out
of v1.

## Product recommendation

Make this a shared speech workspace: people choose and approve voices, people and
agents prepare productions, and supervised workers generate only the requested
audio. The same voice, script, segment, timeline and export IDs must work in the
UI and agent interfaces. An agent should return a link to an editable production,
not leave its work in an unrelated folder of WAV files.

Keep two prominent entry points: **Create a production** and **Manage voices**.
Quick speech remains a short path inside the workspace. Avoid turning the first
screen into a queue monitor, model catalogue or infrastructure dashboard.

Suggested information architecture:

| Place | Primary human task | What belongs here |
|---|---|---|
| Productions | Write, arrange, listen and export | Recent projects; script editor; timeline; takes; export history |
| Voices | Choose or add a reusable identity | Searchable library; previews; record/upload/link; reference review; versions |
| Activity | Understand active or failed work | Queued/running/completed jobs; cancel/retry; links to affected segments |
| Connect an agent | Give an agent access | Endpoint; credential creation; HTTP/CLI/MCP snippets; a test connection |
| Settings | Maintain the installation | Available engines; model provisioning; runtime policy; storage; backup; upgrade |

The shared player, selected voice and current production should survive navigation.
Technical engine details live in voice details/settings, except where they explain
a real limitation. Show **Reference needs review**, **Ready**, **Unavailable on
this server**, and **Archived** in ordinary language; keep model diagnostics one
step away.

## First run and first useful result

1. An administrator starts a published, pinned Compose release. The install opens
   a local address and reports whether data storage, CPU/GPU detection and the
   selected model bundle are ready. A running HTTP process alone is not readiness.
2. Offer **Try a ready-made voice** or **Add a voice**. If an engine must be
   installed, state its capability, download size, disk requirement and access
   condition before the explicit download action. Start with one selected engine;
   do not provision every family to show a welcome screen.
3. Play a cached example before the user waits for inference. Label it **Example**;
   it is not a new sample generated on this machine.
4. Create a production with a short editable script. Select a voice, generate one
   take, listen, and export. Then offer **Connect your agent** using that voice and
   production as the example. Do not require agent setup before human evaluation.
5. Connection testing performs metadata discovery first. A separate explicit
   **Generate a test** demonstrates execution; an authenticated ping does not
   imply inference or download authorization.

For a trusted team, retain separate credential identity and job attribution even
without workspaces, billing, roles or tenant isolation. An unattended first-run
administrator can provision through CLI/API and then use the same UI state.

## Voice lifecycle and meaningful distinctions

**Recording** captures a source; **reference** is the selected speech used for
conditioning; **clone preparation** builds engine-bound derived material; **preview**
is synthesized output. Do not call all four a voice sample, and do not describe
inference-time conditioning as training.

The stable voice identity holds a display name, ID, language/capabilities and
approved version. Each immutable version records its engine, source/reference
hashes, transcript, preparation parameters and derived material. An agent's job
pins the version, not a mutable friendly name. Changing a display name preserves
ID. Replacing a reference creates a new version and does not alter past renders.
This requires a service-level identity layer: current CLI rename moves the slug
directory (`lib/voices.py`, `rename`), so slug must not become the permanent API ID.

An **Add voice** flow has three sources, followed by the same review steps:

- **Record:** ask for browser microphone permission only after clicking Record;
  show input level, elapsed time, stop/re-record and local playback. Supply a
  suggested reading passage, while accepting natural speech. Recording takes
  place on the browser's device, including when the server is remote. Never make
  a remote agent silently access the server's microphone. Agents can create a
  short-lived recording handoff link and resume when the person finishes.
- **Upload:** accept a local recording using a bounded upload, then retain its
  immutable original and decoded derivative. A local API client can upload bytes;
  it must not hand the server an arbitrary filesystem path.
- **Link:** fetch supported remote media in a separate bounded import operation.
  Show the source and optional time range, then the same preview/reference review.
  Restrict fetch size, time, redirects and address destinations; resolve and check
  redirects too. A failed remote fetch must preserve the voice draft and offer
  upload. Broad website support and unattended authenticated browsing are later.

After capture/import, listen and select the reference region. For an already
clean recording, avoid mandatory Demucs separation and ASR. Offer transcription
only when needed by the chosen engine, accept an exact transcript supplied by
the user, and make transcript correction possible. Preserve submitted text;
automatic cleanup must not silently rewrite it.

Show useful defects with concrete fixes: no speech, multiple speakers, clipped
audio, background music, reference too short for the engine. Distinguish a
measured defect from a heuristic warning. Engine-specific reference limits come
from capabilities; a universal recording duration is not a supported guarantee.

Prepare the clone as a job, generate a short test only on request, and let the
person compare **Original reference** with **Generated preview** using the same
player. Technical readiness and **Reviewed by you** are separate fields. A
successful WAV, ASR score or cache hash cannot approve resemblance/naturalness.
Store the user's approval against the exact preview and voice version.

Voices also include presets and designed voices. Label these origins. Show only
controls the selected engine supports: Kokoro presets, Pocket references and
Qwen clone/built-in/design modes are different capabilities. Do not imply that
every clone accepts emotional instructions; current Qwen clone delivery comes
from its reference, while built-in voices can have different instruction support.

Archive voices without breaking completed projects. Removing a voice version used
by a production should explain the dependency and retain existing renders; full
deletion needs an explicit storage/dependency policy. Portable exports contain
WAV/JSON, matching the current safe import restriction; engine caches are rebuilt
locally rather than accepted as external pickle objects.

## Speech production studio in v1

Start with a script that remains text, not a collection of generation forms.
Paragraph/line segments have stable IDs and a voice/version assignment. A person
can paste a script, assign speakers, insert pauses, generate selected lines or the
whole script, listen and revise. Agents create the same document and segments.
Generated takes retain exact submitted text and render settings.

The timeline is an actual editable arrangement in v1: waveform clips, reorder,
move, non-destructive trim/split, explicit silence, clip gain, short fades,
overlapping speech tracks when intentional, zoom, playhead, undo/redo and export.
Script and timeline show the same selected segment. Use integer sample/frame or
well-defined rational time positions; repeated edits must not accumulate floating
point drift. Preserve source take bytes; editing builds an arrangement.

The minimum meaningful workflow is: draft ten alternating-speaker lines; generate
them; reorder two; trim one; extend a pause; regenerate one changed line; compare
the two takes; retain the chosen take; export the arrangement. Regeneration must
not reset timing edits or replace the other nine lines. A previously edited trim
can become invalid when a new take is shorter: show **Trim needs adjustment** and
retain the old playable arrangement until resolved.

Separate script edits, regeneration and arrangement edits. Changing text marks
the affected render stale, displays the delta and offers generation. It does not
trigger a paid/heavy render on every keystroke. A generated result arrives as a
candidate take; late completion cannot overwrite a newer script revision or the
person's selected take. Jobs pin script/voice versions and publish once.

Assembly validates formats and makes any conversion explicit. Export starts from
a frozen arrangement revision and records clip identities, ordering, offsets,
trims, gains, fades, output format and rendered artifact hash. Offer WAV plus a
small set of explicit delivery presets, not an unbounded codec control panel.
An agent can construct and export this same arrangement without driving the UI.

Save drafts automatically with visible **Saved**/**Saving**/**Offline changes**
state. Use revision-aware writes and resolve a simultaneous agent/person edit
without last-writer-wins loss. Close/reopen must preserve the script, selected
takes, arrangement and undo boundary policy. The UI should make an agent's latest
changes reviewable in the project.

Music-bed support remains a material scope choice: a simple uploaded background
audio clip and gain/fades fit the clip domain, but effects, MIDI, plugins,
multitrack mixing automation, mastering and video editing are later. Word-level
alignment, karaoke highlighting and phoneme timing must not be promised from ASR
timestamps. Real-time streaming playback is valuable later; complete durable
artifacts and correct editing come first.

## Quiet states, errors and resource visibility

An empty voice library says **Add a voice** and **Try a ready-made voice**. An empty
production offers paste/type/import script and a brief starter script. No jobs
means **Nothing running**, not an installation error. A sleeping worker says
**Ready to wake when needed**; an unprovisioned engine says **Download required**.

Keep progress truthful: waiting, loading engine, preparing reference, generating,
assembling, exporting. Show elapsed time until an ETA has measured support.
Queue placement is not synthesis progress. A cancelled job publishes no partial
final audio; completed prior takes remain available. Provide a retry action only
when it can preserve the frozen input or explicitly make a new revision.

Useful errors include **Microphone permission blocked → Upload a recording**,
**This server needs HTTPS to record → Open secure address / Upload**, **Engine not
installed → Install**, **Reference unavailable → Restore / Select another version**,
**Not enough memory → Retry with another configured engine / Ask administrator**,
and **Export interrupted → Retry export**. Keep short messages in place beside the
affected object, with diagnostic IDs/details available separately.

Browser recording requires a secure context and user permission; localhost is
treated as a secure context, while an ordinary HTTP private-server IP is not.
This makes HTTPS part of the remote-install journey, not a late visual defect.
[MDN getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

Use visible keyboard focus, keyboard-accessible clip operations, textual timing
fields as an alternative to dragging, labelled playback/record controls, and
non-colour status labels. Test mobile recording and voice review even if desktop
is the primary timeline workspace. Respect reduced motion; waveforms communicate
audio, not decorative perpetual animation.

## Actual install/runtime audit

The checkout is an ML/audio toolbox, not a ready-to-install web product:

- `docker-compose.yml:1-24` runs an interactive `pab` service, mounts the repository
  into `/app`, and mounts writable work/cache paths. It defines no UI/API port or
  lightweight control plane and defaults to an external Redis address.
- `Dockerfile.cpu:1-18` is the stronger bounded packaging base: pinned Python base
  digest, backend-specific hashed dependency locks, narrowly copied application
  code, separate provisioning and a deliberate memory allocator policy. Its
  default command is CLI help, not a resident service.
- `run-cpu:23-38` forwards an existing HF token only for provision, mounts assets
  read-only for synthesis and launches non-provision commands with network disabled.
  Preserve this separation; an HTTP service wrapping that CLI cannot merely claim
  equivalent isolation while sharing an unrestricted network namespace.
- `scripts/provision-tts.py:16-74` refuses nonempty bundle destinations, validates
  Kokoro asset pins and resolves Pocket dependencies to immutable revisions before
  writing a manifest. New product downloads need staging and atomic activation;
  an interrupted partial directory is not a ready bundle.
- `apps/voice-cpu/README.md` describes a supervised warm batch worker for one
  explicit backend/language/voice. It is reusable groundwork, not a general
  daemon or dynamic engine/voice scheduler. Timeouts discard/reap the worker;
  parent-owned final publication and identity checks must survive service work.
- `docker-compose.watcher.yml:1-44` and watcher documentation describe Redis/Vast,
  object storage, SSH/rclone mounts and cloud instance lifecycle. Keep this as an
  explicit advanced remote-worker option. A local installation should not need
  cloud credentials, broad home-directory mounts, Redis or a cloud bill.
- `docker-compose.gpu.yml:34-59` still makes Maxwell/cu124 and speedup claims that
  conflict with README's SM7+ scope/caveats. `SETUP.local.md` also contains an old
  GTX980 path. Rewrite the support matrix from verified release artifacts and
  runtime probes; those comments are not reliable compatibility evidence.

## Topology and lifecycle recommendation

Recommend a small always-on control/UI service with persistent project/job state,
one supervised inference worker by default, and an explicit provisioning/import
path with network permission. Exact local queue/IPC selection belongs to the
architecture plan. Avoid requiring Redis solely to install a single-host app;
the existing Redis/cloud bridge can remain an adapter.

| Topology | Advantage | Cost / appropriate use |
|---|---|---|
| One image, API and supervised child | Fewest operational pieces | Couples large ML dependencies to API; all share container network/memory boundary; useful first development slice |
| Small control image plus selected worker image | UI stays responsive; engine isolation; smaller updates | One additional lifecycle boundary; recommended packaged local/private-server target |
| Control server plus remote GPU worker | Fits CPU-only laptop/private server; worker can sleep | Model/artifact transfer, authentication, failure recovery and possible cloud spend; optional profile |

Serve the compiled UI from the control process/static server; do not ship a frontend
development server. Published version/digest deployment must not need a repo clone
or a writable source mount. No Docker socket belongs in the ordinary UI container.
Do not add a Docker-socket-based wake manager casually: sleeping a model-owning
child inside its worker is a simpler first idle policy.

Expose configured **Keep ready**, **Sleep after inactivity**, and optional
**Prepare now**. Start with one resident engine/voice configuration and one active
inference job. Maintain a durable queue while asleep; serialize wake; report
loading state once. An idle timer cannot terminate an active render or a pending
publication. Metadata, listing voices and browsing projects must not load torch,
probe speaker weights, start a GPU or download a model.

Persistence should distinguish irreplaceable originals/references, voice metadata
and versions, projects/arrangements, jobs and exports from reproducible engine
bundles and derived caches. Named volumes or documented bind mounts survive
container replacement; image layers are not the data backup.
[Docker volumes](https://docs.docker.com/engine/storage/volumes/).
Support export/backup plus a tested restore on a fresh data root. A cache cleanup
must not delete references, active model versions or production outputs.

Upgrade flow: show release and data-schema compatibility; retain previous image
digest; drain/stop new scheduling; take a consistent data backup; run versioned
migration; start new image; verify readiness and reopening/render/export. Rollback
must account for schema changes; replacing an image tag alone may be insufficient.
Model upgrades are separate explicit activations, never implicit behind a stable
voice ID. Existing renders remain reproducible to the degree their engine and
revision can be retained; deterministic bytes must not be guaranteed universally.

For Linux NVIDIA, follow the current toolkit installation/runtime setup and verify
device access in the actual worker image.
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Compose supports selection of specific GPU device IDs; reserve one configured
device rather than all GPUs by default.
[Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).
For Docker Desktop, the documented Linux-container NVIDIA passthrough path is
Windows/WSL2; do not advertise that CUDA path as macOS GPU support.
[Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/).
An arm64 CPU release needs its own locked dependency and runtime validation;
"CPU works everywhere" is too broad for an x86_64-only tested artifact.

## Proposed resource budgets and how to prove them

These are initial engineering targets, not measured service promises:

| Resource | Proposed starting budget / policy |
|---|---|
| Idle control service | No imported ML runtime/model; aim at ≤256 MiB container memory after ordinary browsing; no inference work at idle |
| Local CPU starter | Start sizing at 2–4 CPU threads and a 4 GiB worker limit for the measured Kokoro profile; verify actual container headroom, host/VM overhead and largest accepted input |
| Qwen/Pocket/GPU | No validated universal RAM/VRAM minimum from this audit; publish per image/checkpoint/device measurements before claiming a tier |
| Concurrency | One inference job per configured worker initially; queue bounded and configurable; no parallel preview explosion |
| Browser | Load production editor on demand; use decimated waveform peaks and range media requests; do not load all full-length PCM into JS memory |
| UI delivery | Aim ≤300 KiB compressed entry bundle, with timeline code/media loaded on demand; verify actual bundle and a long production |
| Preview | Reuse a hash-keyed approved example; default one short take only when requested; no automatic variants, ASR or separation |
| Agent context | Aim <2,000 tokens total essential MCP catalogue; routine results under roughly 1 KiB with IDs/status/links; list filters/pagination; detailed provenance on demand; no base64 audio |
| Storage | Enforce measured import/text/duration/job limits; preflight staging plus final artifact space; retention explicit; no invented fixed model-size estimate |

Current local evidence is narrower: `docs/TTS_MEMORY.md` reports a frozen Kokoro
experiment with observed process-group PSS maxima of 852.7 MiB French and 873.9 MiB
English under a four-CPU/4 GiB Linux x86_64 limit. It reports lower RAM with slower
batch walls, not a simultaneous speed gain. These are sampled process-group
observations, not container peaks, guarantees for arbitrary input, or evidence
for Qwen/Pocket. The control-plane targets above need new measurements.

Every published tier needs cold startup/load, first result, warm RTF, whole-job
wall time, peak container memory, settled idle memory, GPU memory when relevant,
image/model disk use and idle CPU measured on a named environment. Repeat after
a long script and several voice/engine changes to expose retention. Keep
naturalness/identity listening and exact text-fidelity review separate from
successful execution and ASR statistics.

## Delivery stages within v1 and acceptance gates

1. **Shared service foundation:** immutable voice versions, durable jobs/artifacts,
   scripts/segments and arrangements; HTTP plus thin CLI/MCP; pinned worker and
   provision boundary. Gate: metadata queries never load/download models;
   concurrent duplicate submissions publish once; restart during work cannot
   corrupt a project or publish timed-out output.
2. **Voice library and first-run vertical slice:** record/upload/link, reference
   review, prepare, preview, approval, connect agent. Gate: complete recording on
   localhost and HTTPS private server; denial/offline/download failures preserve
   drafts; a person and agent address the same approved version.
3. **Production studio and handoff:** script, takes, actual timeline editing,
   revision-aware autosave, regeneration, assembly/export. Gate: the ten-line
   editing scenario above, browser refresh, simultaneous human/agent revision
   conflict, undo/redo, keyboard editing and exact frozen-export provenance.
4. **Packaged install and recovery:** CPU profile and one explicitly qualified GPU
   path; clean restore, upgrade and idle/wake. Gate: fresh host installs without a
   repo clone; inference can work with provisioned assets and no network; restore
   reopens and exports a production; measured budgets and acoustic review signed
   off for the advertised runtime profile.

V1 later increments can add scheduling sophistication, extra engines/languages,
remote cloud profiles and streaming; the first v1 is complete only after the
studio stage. Multi-tenant SaaS, training/fine-tuning, plugin effects, full DAW/video
editing and automatic promotion of voices/models are outside this confirmed scope.

Remaining material decisions for independent review: exact limits of music/audio
bed support; whether text edits should ripple timeline placement by default;
retention of originals/exports and irreversible deletion; local-only vs HTTPS
private-server release qualification first; GPU hardware for the first certified
profile; and the acceptable warm-memory versus first-result-latency tradeoff.
