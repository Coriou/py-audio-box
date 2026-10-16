# Existing speech studios and browser editor components

Audit date: **15 October 2026**. Primary vendor documentation and upstream source
only. This is a source/documentation audit, with no accounts, installations,
inference, microphone access, paid actions or authenticated UI testing. It does
not establish acoustic quality, runtime performance, accessibility conformance,
released-package compatibility or reuse acceptance. Sources were accessed on the
audit date; their individual publication dates vary. Mutable `main` source links
describe that upstream tree, not a frozen release.

Scope: challenge [PLAN.md](PLAN.md)'s full speech studio v1 and identify interaction
patterns and reusable editor pieces. Whole-product Voicebox/VoiceStudio adoption
is handled in the parent's build-versus-adopt audit. No framework or visual style
is selected here. Impeccable's Operate criteria guide the interaction critique;
the parent's existing session context is retained.

## Recommendation

Do not build an entire timeline engine before a short component qualification.
**Waveform Playlist's current core/engine is the strongest component candidate**
for multi-clip speech editing; **wavesurfer.js is a strong audition/reference
waveform candidate**, with a narrower role. Keep a backend-owned project contract
and FFmpeg render path regardless of browser component. Avoid adopting a browser
library's mutable state as the service database schema.

The commercial products already cover much of the proposed human experience.
Descript now explicitly offers API, CLI and MCP with editable project handoff;
ElevenLabs has Studio APIs and an in-product editing agent. Therefore **a studio
plus agent access is not a new differentiator**. Our defensible proposition is a
self-hosted workspace with explicit engine/resource control, compact typed agent
operations, preserved authored text, durable private exports and shared human-agent
revision semantics. Those remain hypotheses to prove, not competitive claims of
superior quality or cost. [Descript developer overview](https://help.descript.com/developers),
[ElevenCreative Studio](https://elevenlabs.io/docs/eleven-creative/products/studio).

## Strong product references

### ElevenLabs Studio: narration units and generation-aware editing

Studio documents paragraphs with assigned voices/models, chapters, timeline media,
record/import, pronunciation rules, captions, comments and exports. A text/voice
change invalidates generated status. Delivery guidance can come from a recorded
performance in Actor Mode; this differs from choosing a voice identity. A play
action can also generate missing narration. Adopt the clear generation states,
but keep **Play existing audio** and **Generate** distinct in our resource-conscious
product. [Studio overview](https://elevenlabs.io/docs/eleven-creative/products/studio).

Paragraph generation history supports audition, individual download and restoring
an older result; deletion is permanent. Paragraph locking prevents accidental
changes after generation. These are stronger references than a bare undo stack:
we should keep take history, selected take and a content lock as separate concepts.
A lock is still not proof of editorial approval.
[Generation history](https://elevenlabs.io/docs/help-center/product/studio/studio/what-is-generation-history-in-studio),
[Paragraph locks](https://elevenlabs.io/docs/help-center/product/studio/studio/what-does-the-lock-icon-mean-in-studio).

Its TTS API exposes preceding/following text and request IDs for continuity when
replacing a middle clip, and versioned pronunciation-dictionary locators. A seed is
best effort, not a deterministic-output guarantee. This is a meaningful gap in
our plan: independent per-block rendering and crossfades alone do not preserve
prosody. Record the context window/neighbor identities and advertise supported
conditioning capabilities per engine. Do not pretend our current engines inherit
these provider controls. [TTS timing/continuity request contract](https://elevenlabs.io/docs/api-reference/text-to-speech/stream-with-timestamps).

Studio API endpoints cover project/chapter conversion and audio snapshots; each
chapter conversion creates a downloadable snapshot. This demonstrates persistent
project/audio identity, but does not prove API equivalence with every timeline,
comment or video operation. The Studio guide itself excludes SFX from project API
streaming, an example of meaningful UI/API disparity.
[Chapter snapshots](https://elevenlabs.io/docs/api-reference/studio/get-chapter-snapshots),
[Studio API project contract](https://elevenlabs.io/docs/api-reference/studio/get-project/).

An important adjacent reference is **legacy Dubbing Studio**, currently in
maintenance mode. It distinguishes fixed-duration generation from natural-duration
generation, warns of speed/fit problems, marks stale clips and exports AAF, clip
packages, captions and CSV. Borrow the explicit duration-policy distinction;
do not treat that legacy product as the current strategic implementation target.
Our v1 should preserve placement and report overflow, rather than silently force
speech into an old slot. [Dubbing Studio](https://elevenlabs.io/docs/eleven-creative/products/dubbing/dubbing-studio).

Studio offers review links with timestamped comments, but the inspected docs do
not establish a formal immutable revision-approval workflow. Its documented WAV
output can also be transcoded from lossy source depending on plan. For our exports,
show the actual source/render precision instead of implying that a `.wav` suffix
proves lossless provenance. [Studio sharing and export](https://elevenlabs.io/docs/eleven-creative/products/studio).

### Descript: text editing as media editing, with substantial automation

Descript's core interaction links words to recorded media: deleting/cut-pasting
text removes/moves the corresponding audio non-destructively. It separately
supports transcript correction that does not change the audio. **Our script editor
must distinguish these modes**: editing future narration, correcting an existing
transcript, and cutting existing audio have different effects. A single undifferentiated
text field is unsafe for this job.
[Edit like a doc](https://help.descript.com/hc/en-us/articles/10164808475149-Inline-notes),
[Correct transcript](https://help.descript.com/hc/en-us/articles/23054692507661-Unable-to-toggle-capitalization-or-punctuation-on-Windows).

Regenerate has a deliberate preview/restore/retry/accept workflow. The smoothing
operation benefits from original audio on both sides; docs limit it to English,
250 characters and particular boundaries. Changing recorded words requires an
authorized custom AI speaker and does not support stock voices. Thus **regenerate
a TTS block**, **heal an edit**, and **replace recorded speech** must be separate
capabilities. Our v1 can deliver the first without promising the other two.
[Smooth a section](https://help.descript.com/hc/en-us/articles/17676714027533-Use-Regenerate-to-fix-or-smooth-out-any-jumpy-abrupt-or-awkward-audio),
[Change recorded speech](https://help.descript.com/script-editing/change-meaning).

Its timeline has move/trim/range/blade/slip tools, including changing the source
portion inside a fixed-duration clip. It automatically uses microfades at cuts,
but excludes those fades from timeline exports. Adopt precision operations and
explicit boundary processing; do not infer preview/render/DAW equivalence from
having a waveform. [Timeline tools](https://help.descript.com/timeline/timeline-tools),
[Microfades](https://help.descript.com/timeline/automatic-microfades).

The current developer documentation covers HTTP, CLI, MCP, async job receipts,
uploads/URL imports and completion callbacks. The editing surface described is
Underlord-directed automation, followed by human review in the editor. This
supports a real agent-to-editor handoff, **not evidence that each manual edit is
available as a compact deterministic semantic mutation**. Our proposed typed
clip/block operations remain a valuable design choice for minimizing hidden LLM
work, if implemented and measured. [Developer documentation](https://help.descript.com/developers).

MCP supports project search/organization, composition targeting, edit jobs,
cancellation, publication and transcript export. Its documented limitations:
rendered media download through MCP requires publishing a web link; job history
lasts 30 days; scope is one Drive; YouTube import URLs are unsupported. This
resolves apparently conflicting marketing language about downloads. Our private
artifact download must not require publication. Do not describe Descript as
having no API or agent access.
[MCP overview and limitations](https://help.descript.com/api-and-mcp/mcp).

Descript also exports structured timelines preserving media/edits/source timecode
to other editors, while documenting unsupported elements such as sequences and
avatars. This is a better interoperability reference than "export WAV" alone.
We should include a portable project manifest and stems in v1; native AAF/XML
support can follow a named target-app qualification, rather than claiming general
DAW round-trip fidelity. [Timeline exports](https://help.descript.com/export-and-share/timeline-exports).

Its extensive shortcut reference includes playback/navigation and mode-sensitive
single-letter commands. This is evidence of keyboard features, not screen-reader
conformance. Our shortcuts must not fire while typing a script or editing a
numeric time field. [Keyboard shortcuts](https://help.descript.com/descript-tour/keyboard-shortcuts).

### WellSaid: the strongest simple script/takes/pronunciation reference

The current Studio starts with voice audition and sections, imports scripts with
paragraph/sentence/no-split choices, and exports common audio/caption formats.
That supplies a simpler entry path than exposing a complex timeline immediately.
Use script-first progressive disclosure while retaining the full timeline.
[Getting started](https://help.wellsaid.io/hc/en-us/articles/39187892094099-Getting-Started-with-WellSaid-Studio).

Take history restores both text and audio. Combining sections uses the active
take in top-to-bottom order with an explicit pause choice; the documented combine
operation has limits and sample-rate conversion behavior. Our selected take must
retain its script/settings snapshot, and restoring old audio must not silently
masquerade as audio for today's text.
[Takes](https://help.wellsaid.io/hc/en-us/articles/45678417886483-Creating-and-Managing-Takes),
[Restore section text](https://help.wellsaid.io/hc/en-us/articles/39047377715219-Repopulating-a-Section-s-Script),
[Combine sections](https://help.wellsaid.io/hc/en-us/articles/39404298753299-Combining-Sections).

Its replacement feature preserves the written script while controlling what is
spoken, with preview, alternatives and per-occurrence disable. Libraries can be
personal/shared; replacements can affect other projects. This is a direct reason
to add a **versioned pronunciation/spoken-text layer** to our domain, with an
affected-block preview before changing project defaults. Do not copy uncontrolled
global mutation. The vendor documents script/language restrictions; our capability
matrix must likewise avoid assuming universal phoneme support.
[Replacements](https://help.wellsaid.io/hc/en-us/articles/39187426777363-Guide-to-Replacements).

The API has replacement-library CRUD, respelling support and library IDs passed
to TTS. That establishes pronunciation automation, not full Studio section/take/
timeline parity. SRT/VTT/TXT caption download is separately documented in Studio.
No general-purpose multitrack timeline or formal approval gate was established
from the inspected WellSaid sources; that is an evidence limit, not a categorical
claim of absence. [Replacement API](https://docs.wellsaidlabs.com/reference/using-replacement-libraries),
[Captions](https://help.wellsaid.io/hc/en-us/articles/40107697876499-Downloading-Caption-Files).

### Podcastle is now Async: recording and generative pickups

The vendor renamed Podcastle to **Async in January 2026**; older Podcastle pages
and comparison tables can mislead. Current official material documents transcript
editing of multitrack audio, separate contributor tracks, clone-based added speech,
record/upload clone acquisition and specific-line regeneration. It is a useful
reference for moving between captured audio and generated pickups.
[Official rename](https://async.com/blog/podcastle-is-now-async/),
[Text Mode](https://async.com/tools/text-based-editing),
[Voice cloning](https://async.com/voice-cloning).

Its Voice API documents TTS/STT/streaming/cloning and pronunciation controls.
These docs do not establish external CRUD for the editor's composition, timeline,
take history or review states. Treat the shared voice model as engine parity,
not whole-editor API parity. No keyboard/a11y guarantee or exact fixed/ripple
replacement rule was established in the inspected public pages.
[Voice API](https://docs.async.com/).

## Reuse candidates: actual boundary and limits

| Candidate | Primary fit | License observed | Recommendation |
|---|---|---|---|
| wavesurfer.js | Single-reference selection, take audition, waveform/player | BSD-3-Clause | Reuse candidate for those surfaces; not a complete editor |
| wavesurfer-multitrack | Synchronized rows, position/cue/fade controls | BSD-3-Clause; README says paid support | Secondary prototype candidate; track/clip model mismatch to investigate |
| Waveform Playlist core/engine + UI adapter | Multiple clips per track, structural edits/history, playback adapters | MIT at upstream root | First component qualification for the studio |
| Remotion Player/Timeline/Editor Starter | Video-oriented composition and editor shell | Custom Remotion terms; paid timeline terms | Defer for speech-only v1 unless whole-product adoption makes it relevant |
| Native FFmpeg in worker | Canonical audio ingest/assembly/export | LGPL/GPL depends on exact build | Reuse existing dependency, qualify exact composition semantics |

The license names above come from the linked upstream license files below; they
are not an approval of our eventual distribution. Pin and inventory transitive
dependencies, retained notices, copied files and actual shipped binaries during
candidate qualification. No dependency was installed in this audit.

### wavesurfer.js and its multitrack plugin

wavesurfer.js provides waveform playback plus Regions, Record, Timeline, Minimap
and Envelope plugins. This makes it a strong candidate for reference trimming and
preview. Its official FAQ explicitly disclaims being an audio-processing editor;
large browser-decoded audio can exhaust memory, while long/streamed playback
needs precomputed peaks/duration. Marker dragging and envelopes are not durable
edit history or server rendering. Use our asset IDs and peak files, add accessible
controls, and leave the composition contract outside it.
[Official plugins](https://wavesurfer.xyz/),
[FAQ and limits](https://wavesurfer.xyz/docs/),
[BSD license](https://raw.githubusercontent.com/katspaugh/wavesurfer.js/main/LICENSE).

The multitrack source exposes an array of track objects with URL/peaks, start
position, source cues, fades and volume, and creates a wavesurfer/media instance
per row. This is **not the same documented model as many clips inside one track**.
No general split/merge/history/project serialization contract was established
from its public README/API shape. One row per speech clip is possible, but could
be a poor fit for hundreds of clips and a few speaker tracks. Prototype before
choosing it. The README distinguishes its support model from wavesurfer core;
license remains BSD-3-Clause.
[Actual multitrack source](https://raw.githubusercontent.com/katspaugh/wavesurfer-multitrack/main/src/multitrack.ts),
[README](https://raw.githubusercontent.com/katspaugh/wavesurfer-multitrack/main/README.md),
[License](https://github.com/katspaugh/wavesurfer-multitrack/blob/main/LICENSE).

### Waveform Playlist: much more than its older examples suggest

The current upstream describes React/TypeScript packages, a framework-independent
engine and an additional Lit Web Components layer; it is not just the historical
single-package demo. The root documents multi-clip editing, recording, WAV export,
annotations, precomputed peaks, keyboard actions and optional playback adapters.
This makes rejecting it based on old examples premature. It also has substantial
DAW/MIDI/effects scope we do not need; evaluate a minimal subset.
[Current upstream README](https://github.com/naomiaro/waveform-playlist),
[MIT license](https://raw.githubusercontent.com/naomiaro/waveform-playlist/main/LICENSE.md).

The engine documents integer-sample editing, move/trim/split constraints,
headless operation, change events and snapshot history grouped into transactions.
Its example explicitly uses sampleRate 48000, which aligns with our proposed
48,000-tick normalized derivatives. Musical ticks/PPQN are a separate domain;
do not equate them with our sample ticks. This substantially overlaps the hardest
browser editing mechanics in PLAN.md.
[Engine API](https://raw.githubusercontent.com/naomiaro/waveform-playlist/main/packages/engine/README.md),
[Clip time helpers](https://raw.githubusercontent.com/naomiaro/waveform-playlist/main/packages/core/src/clipTimeHelpers.ts).

The browser package distinguishes a multi-track/Tone-backed provider from a
single-track media-element provider; its lightweight no-AudioBuffer claim belongs
to that latter path. It does **not** establish that long multitrack sessions avoid
decoded-buffer memory. Its entry-point separation can help bundle size, while the
Tone export path is browser/offline rendering and cannot alone establish parity
with our FFmpeg worker.
[Browser package](https://raw.githubusercontent.com/naomiaro/waveform-playlist/main/packages/browser/README.md).

Qualification must resolve the following before reuse is approved:

- Freeze one commit/release and matching packages: current `main` READMEs discuss
  multiple major migrations; a cached package manifest alone cannot prove their
  published-version compatibility. Pinning an arbitrary newest package is unsafe.
- Test a few speaker tracks with many clips. Verify that collision constraints
  permit our intentional cross-track overlap and do not silently change placement.
- Map every gesture to a validated service mutation/revision. Browser undo is
  local interaction history, not persistent server history or cross-client undo.
  Reject stale commits and restore authoritative state without audible surprises.
- Keep AudioBuffers, DOM objects and playback state out of serialized projects.
  Preserve stable clip IDs, source offsets, end-exclusive ranges and integer timing.
- Measure long-session memory, decode concurrency, waveform virtualization and
  seek behavior. A supplied-peak visualization does not itself prevent PCM decode.
- Test keyboard-only operations and screen-reader names/order. Documented shortcuts
  are useful evidence, not a complete accessibility assessment.

### Remotion and FFmpeg

Remotion has a paid customizable timeline with multitrack drag/drop, selection,
zoom, keyboard shortcuts and Player integration. Its published license depends
on entity/use and includes restrictions; the Timeline has additional terms.
Do not label it MIT or assume that free eligibility permits redistribution as
our own editor. The Editor Starter is relevant to a future video product, but
adds a different rendering stack to this Python/FFmpeg speech scope.
[Timeline](https://www.remotion.dev/docs/timeline),
[Editor Starter](https://www.remotion.dev/docs/editor-starter),
[Remotion license](https://raw.githubusercontent.com/remotion-dev/remotion/main/LICENSE.md).

FFmpeg already fits our server artifact workflow. Its filters cover trim,
timestamp reset, mix and crossfade; `atrim` does not reset timestamps, and `amix`
normalizes by default. Therefore an adapter must specify ordering, time origin,
gain/normalization, end behavior and fade curves instead of accepting defaults.
Use the same canonical arrangement for browser preview and server render, with
separate conformance tests. A browser engine and FFmpeg need not produce
bit-identical DSP; audible/timing tolerances require explicit acceptance.
[FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html).

The shipped FFmpeg build's licensing depends on enabled components: upstream
documents LGPL with optional GPL parts. Record the actual release build/config
and corresponding notices/source obligations; the project name alone does not
identify the binary's license. [FFmpeg licensing](https://ffmpeg.org/legal.html).

## Required amendments to the v1 interaction/domain plan

These recommendations challenge the current plan; they are not an authorized
implementation change or a replacement for the parent synthesis.

| Decision | V1 addition or precision | Reason / acceptance scenario |
|---|---|---|
| Adopt | Authored text, spoken-text rules and transcript/captions as separate layers | Correct spelling in imported transcript without generating; change how an acronym is spoken without rewriting approved script |
| Adopt | Versioned pronunciation dictionaries with scope/precedence and affected-block preview | Update one brand rule; regenerate affected unlocked blocks only; old takes keep old dictionary snapshot; unsupported phonemes fail visibly |
| Adapt | Selected-take history with original text/settings, compare/accept/revert | Audition an old take while current text differs; adoption explicitly resolves the mismatch rather than silently changing text |
| Adopt | Content lock and revision-bound review state | Agent cannot alter a locked approved block without an explicit operation; any relevant edit invalidates review; successful generation alone never approves |
| Adapt | Speaker role separate from voice identity/version and per-block delivery | Replace narrator voice across a project with an impact count; preserve character labels; do not overwrite recorded-role metadata |
| Adapt | Context-aware render metadata/capabilities | Replace a middle sentence and assess prosody with neighbors; unsupported context conditioning is declared; never claim crossfade alone fixes coherence |
| Adopt | Distinct natural-duration, placement and ripple policies | A longer new take reports collision/overflow; fixed placement does not mean automatic time-stretch; ripple is a deliberate grouped edit |
| Adapt | Explicit playback and generation, plus budgeted render stale | Pressing Play starts no model/download; selected stale rendering has bounded job/character cost and preserves completed takes |
| Reuse candidate | Waveform Playlist core/engine and minimal view adapter | Pass clip ID/timing/history/headless synchronization/memory/a11y gates before selecting framework or writing a replacement engine |
| Reuse candidate | wavesurfer.js for reference and audition | Show precomputed peaks, playback, region selection and numeric trim alternative without forcing full editor dependencies |
| Adapt | Portable v1 production handoff | Export mix, rendered per-track stems, script, original asset/take IDs and a versioned timeline manifest; clearly distinguish stems from a native editable DAW session |
| Adopt | Explicit caption evidence | V1 can export authored segment-level captions derived from arrangement timing; word-level captions require alignment capability, confidence and correction UI; no invented word timings |
| Adopt | Semantic API parity for all required editing actions | Human opens an agent production, edits trims, agent sees the same revisions; CLI/MCP can perform required mutations without an internal editing LLM |

These text-layer distinctions do not require Descript-style word deletion from
recorded media in v1. Generated-script editing and ordinary clip editing can
ship first; transcript-driven cuts require a separately qualified alignment path.
After a clip is split or trimmed inside a sentence, captions must follow an
actual text-range mapping or be reviewed; exporting the complete original block
text for truncated audio would be misleading.

Review comments should be anchored to a segment plus revision/source range, with
an optional derived playhead time. Pure timestamps become misleading after ripple
or regeneration. For v1, lightweight notes/resolution and a review marker can serve
the trusted team; external anonymous links, mentions/email and complex approval
roles can wait. The schema should avoid losing review provenance when the UI grows.

Defer recorded-word healing, guaranteed duration-matched resynthesis, automatic
voice-quality rejection loops, performance-guidance transfer, native AAF/XML
round-trip, visual video/caption styling, music generation and full DAW effects.
They can be valuable later; none is implied by the confirmed speech studio scope.
Preserve imported supporting audio with trim/gain/fades and all core script/timeline
editing in v1. This recommendation does not shrink v1 to voice management.

## Bounded qualification before committing to an editor foundation

Use one fixture production: three speaker roles, ten alternating narration blocks,
several takes, one imported stereo audio bed, two intentional overlaps and a
trimmed clip reused twice. Run the same fixture in the leading whole-product
candidate and a minimal Waveform Playlist integration, then compare results.

The decision gate should cover:

1. **Round-trip semantics:** API create → browser edit → API read/export preserves
   IDs, integer positions, source offsets, gains/fades and explicit overlap policy.
2. **Regeneration:** change one line and one pronunciation rule; unaffected takes
   remain byte-identical; older playable revision remains; selected new take
   cannot silently ripple or inherit an invalid trim.
3. **Review:** locked content refuses accidental mutation; take approval is tied
   to the actual text/voice/settings revision; obsolete comments remain traceable.
4. **Playback/render:** playhead, silence, trim, overlaps and fade behavior match
   the frozen export within specified timing/loudness tolerances, including seeks
   near clip boundaries and context loss/resume.
5. **Resource use:** metadata-only interaction loads no model; measure cold editor
   bundle, browser memory for a long production, peak decode concurrency and
   repeated project-close cleanup. No unmeasured low-memory label.
6. **Input/a11y:** keyboard-only select/move/trim/split/undo/export, text-field
   shortcut safety, labelled time alternatives, focus restoration and screen-reader
   usability. Include recording-denied/no-device/import-failed states.
7. **Portability:** reopen a saved project on a fresh service without browser
   localStorage; export stems/manifest; compare an advertised native-editor handoff
   only if a particular interchange format is actually implemented.

Outcome of this audit: **GO for a component/whole-product comparison prototype;
NO-GO for declaring the present studio specification complete or any component
production-ready.** The newly exposed gaps are pronunciation/text layers,
regeneration context and duration policy, take adoption/review semantics,
interoperability and browser/server audio parity. None requires an AI-heavy
internal co-editor in v1.
This does not revoke the existing PLAN.md GO for contract/prototype work; that
document already states that it is not a frozen full implementation specification.
