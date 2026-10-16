# 0B round 1 — independent Sol-low review

**NO-GO for candidate `a7b7435fa1819a100f33c5f5c08e9d0dd8b2a530`**, base `9c50c320098c6c2dc9f2fabe0b92e1602deb5c13`, branch `t3/lot0b-browser-editor`, reviewed in `/Users/ben/.txcode/worktrees/py-audio-box/t3-b7ef8a9f` on 2026-10-16. Two concrete corrections below; no source repairs performed. This is a bounded 0B decision, not integrated phase or final Astra acceptance.

## Required corrections

### R1 — P1: accepted timing edit permanently blanks the fixture through unbounded ruler allocation

Location: `studio/src/main.tsx:403-413` (especially 406-407), timing form at 570 onward; `studio/src/adapter.ts` accepts schema-valid safe-integer placement.

Actual native T3 browser reproduction: open the 30 fixture, enter `9007199254700000` into `start`, click **Apply clip edit**. The request succeeds and localStorage's latest project contains that start. React then unmounts the application: `document.body.innerText === ""`. Task-owned Vite reports **`RangeError: Invalid array length > App src/main.tsx:406:23`**. This value and its clip end are browser-safe integers. The ruler allocates one element per 4,800 ticks across the entire duration, despite clips being windowed. Lesser large placements can consume substantial resources before the array-length limit. Preview's separate 60-second bound does not protect rendering. Because the accepted state is persisted, reopening replays the same accepted state into the same failing render; the in-page Reset/Undo recovery is gone.

Bounded correction: bound ruler/coordinate rendering independently of project duration (or visibly reject unsupported fixture extents before saving, without changing canonical timing semantics), and preserve usable recovery for existing unsupported persisted states. Verify a large valid placement cannot freeze/blank the page or poison reopen, while ordinary 100-block scrolling and undo still work. This is an exposed editor input/resource failure, not a demand for a production-length audio benchmark.

### R2 — P2: malformed persisted history is silently presented as successfully saved initial state on startup

Location: `studio/src/main.tsx:76-84`; compare the explicit recovery message in `load()` at 190 onward.

Actual browser reproduction: set this fixture's disposable storage key `editor-0b:1.3.0:30` to `{broken`, then reload the page. Observed revision **1**, status **“Saved locally · no service”**, while the storage value remains `{broken`. The catch suppresses the failure and shows initial content as if it were the saved project. The adapter correctly rejects malformed JSON, but the startup UI hides that rejection. Subsequent edits read the still-broken history and fail. No source or original media is corrupted, but reload/failure-state evidence does not support this misleading successful-persistence presentation.

Bounded correction: surface a startup history/storage error and explicit recovery while retaining the invalid bytes until deliberate Reset; do not label the fallback saved. Verify malformed JSON and replay-invalid snapshots on full page reload, plus successful normal reload/undo. This is fixture failure handling, not Phase 4 offline-draft implementation.

## Independent checks and actual browser evidence

Read PRODUCT, IMPLEMENTATION-START, orchestration protocol, canonical plan Section 3/shared boundaries/Phase 0B, CONTRACTS 1.3.0, EDITOR-0B including all caveats, source/tests/media/probe/styles and the 43-file change inventory/evidence packet. Inspected dependency pins, lock identity and retained licenses. Candidate worktree was clean before and after tests. `git diff --exit-code base candidate -- contracts PRODUCT.md` and canonical plan/start/protocol/ledger paths was empty. Schema, both JSON fixtures and lock SHA-256 agree with the packet. No backend/default/ledger changes occur in this candidate.

Executed with existing Node **25.2.1**, npm **11.6.2**, installed pinned dependencies and existing oracle venv:

- `CONTRACT_PYTHON="$PWD/studio/.test-venv/bin/python" npm test --prefix studio`: **6/6 pass**, including unmodified Python project/schema/edit-semantics oracle over edited histories.
- `npm run build --prefix studio`: TypeScript and Vite **pass**; JS **613.97 kB / 140.49 kB gzip**, known chunk warning. No install, lock mutation, model/app suite or remote e2e script.
- `npm run dev --prefix studio`: task-owned loopback Vite at 127.0.0.1:5178; stopped with Ctrl-C, process exit 130. Generated ignored build/cache artifacts only.

Native T3 `preview_status` found no tab; `preview_open` with loopback succeeded after an invalid `about:blank` URL attempt. Corrected unsupported `exact=true` locator syntax; those failed clicks did not count as behavior evidence. No alternate browser or standalone Playwright.

Actual clicks/keyboard/selects plus DOM and persisted-state reads demonstrated:

- Reset revision 1; focus c0 and ArrowRight → revision 2, c0 start **480**, c1 unchanged **4800**. Reload current, Undo and Redo → revisions 3/4; long replacement → revision 5, c0 out **9600**, same start and fixed neighbour, overlap exposed.
- Short replacement without adjustment refused at revision 5 with `short_take_requires_explicit_trim`; explicit adjustment succeeded at revision 6. Midpoint split succeeded at revision 7. Simulated writer advanced stored revision to 8; stale Undo left UI revision 7 and stored revision 8, reporting `revision_conflict`. Reload recovered.
- Gain **0** and fade-in **100** persisted and showed silent routing; faded split refused atomically at unchanged revision 2. Speech mute/bed solo controls persisted their routing values. Synthetic preview reached **device 48000 Hz**. No listening judgment or playback-tolerance measurement.
- 100-block/200-clip fixture initially rendered **47 clips / 94 waveform canvases / 12 script rows**. Scrolling reached **b99 and bed99**, with **25** mounted clips at the far end. WaveSurfer effect cleanup destroys each instance and cancels late creation; this supports bounded fixture DOM lifetime, not a heap-leak or production-memory claim.
- At 1280×800 and emulated 390×844, document width equalled viewport width; narrow inspector width was 390. Native labelled controls, pressed routing states, numeric alternatives and keyboard clip movement exist. Tab reached b0 with a solid focus outline. Candidate desktop/mobile images were visually inspected as supporting evidence. Desktop lower inspector controls require ordinary page scroll. This is basic keyboard/layout evidence, not a full screen-reader/WCAG or physical-touch audit.

These are **E1 tests/build/static** and **E4 browser/client evidence** under the canonical plan. EDITOR-0B's “E2 browser” label should be corrected in the handoff; these are not E2 service integration tests. No E3 runtime/hardware or E5 acoustic acceptance is established.

## Library decision, semantics and bounded restrictions

The custom timeline + WaveSurfer decision is reasonable at 0B. Independently ran retained Playlist probe: **31 component tracks versus 2 canonical tracks**, context **44100 Hz**, and `await window.probeZero()` returned **1** for gain zero. Device sample rate is not source/canonical-tick corruption. Installed manifests confirm Playlist **4.3.3 MIT**, WaveSurfer **8.0.2 BSD-3-Clause**; retained license notices are present. [Current official Playlist repository](https://github.com/naomiaro/waveform-playlist) describes a different modular React/multiple-clips model with optional playout adapters. Legacy probe findings do not condemn those packages. Modern docs-only comparison is disclosed and sufficient to permit this narrow fallback choice; another full rival implementation is unnecessary. [Official WaveSurfer docs](https://wavesurfer.xyz/docs/) support its waveform-only role and predecoded-peaks approach; production media delivery remains later work.

Adapter snapshots and requests live outside WaveSurfer; schema validation, revision checks, immutable take/asset/version comparisons and explicit supported operations preserve the ordinary tested paths. Python oracle checks are meaningful but `edit_semantics` is principally noninterference/identity validation, not an independent implementation of every split/replacement equation. LocalStorage CAS is explicitly nontransactional and must not become the service store. History grows by complete snapshots and replays on reads; no unbounded-history performance qualification is claimed. `save()` writes browser stores before setting React state, so storage-error claims must be read narrowly: it retains the prior in-memory project, not a durable queue of the attempted edit.

Fade equations match the canonical trim-relative endpoint formula; gain zero, mute/solo, additive overlap and clipping rejection are correctly represented in the inspected synthetic mixer. Four cached tiny PCM sources and a 60-second mix limit bound that preview. Missing/unsupported media rejects; no real-media/channel/resampling/export qualification is inferred.

The documented split restrictions are safe bounded 0B restrictions, not additional blockers: faded split rejects instead of inventing fade redistribution; split-block replacement rejects instead of deleting/reinterpreting clips. Ordinary split and pre-split take replacement are executable. The actual CONTRACTS prose says **“two new contiguous ranges”**, not two fresh IDs; the request supplies one new ID, so retaining the left clip ID is not demonstrably a two-ID violation. Shared owner should record this interpretation and resolve broader fade/multi-clip replacement semantics before those later features are admitted. Do not treat this fixture as their implementation or require full Phase 5 now.

Retained audit reports show zero production findings and two moderate dev findings through Playlist→uuid. The probe is dev-only and excluded from the production entry; no demonstrated affected UUID buffer operation is exposed by this synthetic probe. These findings and the fixture/schema-heavy bundle warning do not independently block this lot. No auto-fix performed.

## Handoff and preservation

Return R1/R2 to the implementer for bounded correction and independent recheck. Preserve the library decision and working ordinary paths. Full service CAS, pointer drag, general split/replacement, real peaks/range media, preview/export agreement, acoustic quality and integrated/final Astra review remain separate gates.

Only this report was written in the root checkout. Candidate source/contracts/lock/ledger remained clean; root research and existing untracked STT artifacts were left untouched. Disposable browser fixture was reset after diagnostics. No commit, push, PR, deployment, private audio, models, cloud or subagents.
