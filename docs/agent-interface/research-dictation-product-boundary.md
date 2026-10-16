# STT service versus desktop dictation product

Research date: **15 October 2026**. Official current vendor docs/upstream source,
plus bounded read-only inspection of this checkout and host. No accounts,
installations, private recordings, microphone capture or inference were used in
this lane. Mutable upstream `main` is evidence of source intent, not proof of a
particular packaged release. The separate evaluation owner handles authorized
model/runtime measurements; no performance result is claimed here.

User direction now includes French, English and mixed speech, CPU operation on
the Intel Mac, and optional GPU execution on other qualified machines. The shared
speech production studio remains v1. A separate OS-wide dictation product has not
been approved for implementation.

## Concrete judgments

1. **Put bounded, reusable STT in py-audio-box.** Uploaded audio → supervised job →
   unedited ASR text/segments/artifacts is directly useful to reference preparation,
   imported production audio and agents receiving voice messages.
2. **A Docker backend alone cannot deliver a Wispr-like native experience.**
   Global shortcuts, microphone selection, focused-app context, paste/insertion
   and OS permissions belong on the user's desktop. A browser recorder is useful
   inside the studio, but does not create a system-wide input method.
3. **Try an existing native client before building a sibling.** OpenWhispr has an
   especially relevant self-hosted transcription mode; Handy is a simpler local
   dictation benchmark. Qualify them on this Intel Mac before accepting a new
   signing/updating/permission/support burden.
4. **Separate STT from rewriting and actions.** Default to the unedited ASR result,
   with explicit vocabulary hints/replacements. Optional local text cleanup is a
   separate transformation. No hidden cloud LLM, cloud fallback, paid account,
   screenshot or automatic command execution is required for basic transcription.
5. **A desktop adapter can benefit agents without expanding the core.** It captures
   a deliberate push-to-talk utterance and sends the resulting artifact/text to
   an explicitly selected agent session. The audio service does not become an
   always-listening desktop-control agent.

## What Wispr Flow actually adds

Wispr documents cloud processing of dictated audio. Its privacy, cloud-storage,
training and local-history controls are separate. Privacy Mode/zero retention
must not be presented as local/offline speech recognition.
[Security overview, updated 13 October](https://docs.wisprflow.ai/articles/3467817258-security-and-compliance-faq),
[Data-sharing/storage controls](https://docs.wisprflow.ai/articles/9609615338-private-cloud-sync-and-data-sharing-preferences-in-wispr-flow).

Context Awareness, on by default, reads the active app and nearby text, identifies
browser websites, applies app-category style and uses visible names/code context.
The documented context fields can include selection/cursor surroundings, screen
text, screenshot and conversation information. Platform coverage differs.
Its controls and exclusions are more complex than a "smart STT model" checkbox;
turning off model training is not equivalent to disabling all capture/transmission.
For our design, context acquisition should be independently opt-in and visible.
[Context Awareness, updated 1 September](https://docs.wisprflow.ai/articles/4678293671-feature-context-awareness).

Command Mode is another capability: hold a separate shortcut to edit text, draft
or search. It can send current textbox/selection context; it is off by default
and plan-restricted. Ordinary dictation and command execution have different
intent and should not share a trigger that guesses what the speaker meant.
[Current Command Mode guide, updated 3 October](https://docs.wisprflow.ai/articles/4816967992-how-to-use-command-mode).

Its dictionaries also distinguish vocabulary, explicit corrections and snippet
expansion. This is useful separation to adopt without mandatory LLM cleanup.
These are stored user preferences rather than a reason to continuously collect
all edited application text.
[Dictionary/snippet import](https://docs.wisprflow.ai/articles/8955301725-How-Do-I-Bulk-Import-Dictionary-Items-and-Snippets).

The competitive problem is therefore several products combined: low-friction
capture, ASR, correction/personalization, contextual rewrite, insertion and optional
actions. Success at file transcription proves only one of these layers. Native
usability and the quality of French/mixed dictation need separate evaluation.

## Existing clients worth qualifying

### OpenWhispr — strongest existing adapter candidate

Current upstream offers local/cloud dictation, global hotkeys, audio imports,
optional voice-agent context and public automation interfaces. It has separate
Intel and Apple Silicon Mac downloads. The README explicitly excludes ONNX-based
live speaker identity/fingerprinting on Intel Mac and describes a keyword-search
fallback there; do not transfer Apple Silicon feature claims to this host.
The desktop code is MIT; model and dependency licenses remain separate.
[Canonical upstream](https://github.com/OpenWhispr/openwhispr),
[MIT license](https://raw.githubusercontent.com/OpenWhispr/openwhispr/main/LICENSE),
[Requirements](https://docs.openwhispr.com/platform/system-requirements).

Its self-hosted ASR mode sends multipart audio to `/audio/transcriptions` and
expects JSON with a `text` field. Language/model/dictionary prompt may accompany
the recording. The documented path is batch HTTP, not streaming. A small
compatibility adapter to our canonical transcription jobs is plausible.
Importantly, the current example says the self-hosted panel does **not** supply a
Bearer key. Do not weaken core API authentication for compatibility: use an
explicit loopback-only native shim that holds a scoped service credential, or
verify/add authenticated client support before remote/private-server use.
[Actual self-hosted shim contract](https://raw.githubusercontent.com/OpenWhispr/openwhispr/main/examples/custom-asr-shim/README.md).

There is a significant default to inspect: dictation cleanup is **on by default**
and uses a language model configured separately from ASR and the voice agent.
Options include cloud, local and self-hosted. Local ASR therefore does not alone
establish an entirely local pipeline. A candidate test must explicitly disable
cleanup or bind it to a qualified local endpoint, disable optional cloud sync/
screenshots, and inspect the resulting configuration/data paths.
[Cleanup behavior and controls](https://docs.openwhispr.com/help/dictation/cleanup),
[Processing modes](https://docs.openwhispr.com/guides/cloud-vs-local).

The local CLI talks to the desktop's loopback bridge and can work offline;
automatic backend selection can otherwise fall back to the remote API when
configured. Force local mode for a private evaluation. Its existing CLI/MCP
surfaces are useful references, but are not proof that every local desktop feature
has the same remote capability or no-account execution path.
[CLI local/cloud selection](https://docs.openwhispr.com/cli/backends),
[MCP](https://docs.openwhispr.com/integrations/mcp).

### Handy — simpler local-first interaction reference

Handy is MIT, cross-platform local dictation with hold/toggle shortcuts, VAD,
model choice, clipboard insertion, history and beta CLI controls. It documents
real clipboard/focus failures and Linux/Wayland setup limits. That is evidence
that native insertion is its own engineering task. Use it as the simple baseline
before adding a notes platform or new desktop shell. Its source also supports
post-processing providers; evaluate plain transcription with post-processing
disabled, rather than assuming every feature advertised under "offline" has the
same data path. Source license does not grant its brand assets.
[Handy upstream/limitations](https://github.com/cjpais/Handy),
[MIT license](https://raw.githubusercontent.com/cjpais/Handy/main/LICENSE),
[Post-processing settings source](https://raw.githubusercontent.com/cjpais/Handy/main/src-tauri/src/settings.rs).

No documented remote-ASR adapter equivalent to OpenWhispr's self-hosted mode was
established in this bounded review. Handy's local recognition is still useful on
its own; adapting it to a shared backend would require a source/API assessment.

### VoiceInk — good native UX reference, Intel fit unverified

The GPLv3 upstream describes app/URL modes, context-aware enhancement, personal
dictionary, shortcuts and local recognition; paid packaged releases fund
convenience/support. The current README says macOS 15+, while its official website
says **Apple Silicon only**, macOS 14.4+. Therefore it is not a verified install
choice for this Intel Mac. Resolve the packaged release/build identity before
recommending it. The website distinguishes optional cloud enhancement of text
from local speech transcription. The README currently says PRs are not accepted,
which matters if adaptation requires upstream cooperation.
[Canonical source](https://github.com/Beingpax/VoiceInk),
[GPLv3 license](https://raw.githubusercontent.com/Beingpax/VoiceInk/main/LICENSE),
[Official requirements/privacy FAQ](https://tryvoiceink.com/).

### Whispering and Talon — useful references with different boundaries

Current Whispering lives in EpicenterHQ/epicenter. Its README explicitly calls
the current voice-note workflow a **development milestone, not a release**. It
separates transcription and text-transformation connections, can use a chosen
OpenAI-compatible endpoint without a hosted account, and preserves files/history.
The current monorepo is AGPL-3.0-or-later; older MIT descriptions should not be
applied to today's tree. It is a useful transcript/provenance reference, not a
verified ready-to-install Flow replacement from this audit.
[Whispering README](https://raw.githubusercontent.com/EpicenterHQ/epicenter/main/apps/whispering/README.md),
[Current repository status](https://github.com/EpicenterHQ/epicenter),
[License index](https://raw.githubusercontent.com/EpicenterHQ/epicenter/main/LICENSE).

Talon demonstrates app-specific command grammars and voice-driven computer use,
not merely speech-to-text. Its distributed runtime and licenses must be evaluated
separately from the MIT community command scripts. It is not selected as our open
core or as evidence that desktop automation belongs inside the audio service.
[Talon](https://talonvoice.com/),
[Licenses/sources](https://talonvoice.com/docs/LICENSES.txt),
[Community-script license](https://raw.githubusercontent.com/talonhub/community/main/LICENSE).

## Responsibility boundary

| Responsibility | Shared Docker service | Studio browser | Optional desktop client |
|---|---|---|---|
| Audio file/asset ingest, decode limits and provenance | Owns | Uploads | Uploads captured file/bytes |
| STT model provisioning/load/inference/timeout | Owns when selected backend | Requests jobs | Calls service, or uses its own explicitly selected local model |
| Device microphone choice/capture | No user-device ownership | Browser permission and device capture | Native audio APIs and OS permission |
| Global push-to-talk shortcut | No | In-page controls only | Owns OS registration and conflict handling |
| Frontmost app, selected text, cursor context | No implicit capture | Knows only this page's own state | Optional bounded native context capture |
| Screenshot/system-audio capture | No implicit capture | Separate browser/platform permissions | Separate native capabilities/permissions; off initially |
| Insert into another desktop app | No | User copy/paste or explicit integration | Accessibility/automation/clipboard strategy |
| User dictionary/profile | Versioned service resource if shared | Edits/reviews | Local app rules; sends only selected bounded hints |
| LLM cleanup/rewrite | Separate optional job/capability | Explicit preview/diff | Optional explicit local/remote provider, independent of ASR |
| Act on a spoken instruction | Returns text/artifact | Deliberate submit to agent | Separate command intent and explicit selected agent session |
| Raw result/history/retention | Durable project artifacts by policy | Review/delete | Device capture/draft retention by policy |

macOS grants microphone, Accessibility and input-monitoring permissions separately.
The exact set depends on implementation; input monitoring is not universally
required merely because a hotkey exists. OpenWhispr documents automatic paste
as needing Accessibility, with manual clipboard fallback when denied. Screenshot
and system-audio permissions are separate from ordinary mic dictation.
[Apple input monitoring](https://support.apple.com/en-ie/guide/mac-help/mchl4cedafb6/mac),
[OpenWhispr macOS permissions](https://docs.openwhispr.com/platform/macos).

Browser capture requires user permission and a secure context (localhost or HTTPS).
The captured microphone belongs to the browser's machine, not to a remote Docker
server. A PWA or desktop webview is not automatically allowed to read other
applications' text or register portable global shortcuts.
[MDN microphone contract](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

Docker Desktop's general FAQ says it does not provide direct USB passthrough;
USB/IP is an additional device-specific setup, not a simple portable microphone
installation. Even forwarding audio into Linux would not provide macOS app
focus, insertion or Accessibility integration. Native Linux device forwarding
also does not create an OS-independent desktop dictation product.
[Docker Desktop FAQ](https://docs.docker.com/desktop/troubleshoot-and-support/faqs/general/),
[USB/IP limitations](https://docs.docker.com/desktop/features/usbip/).

## This machine and compute scope

This lane verified `uname -sm`, `sw_vers` and the CPU brand without running models:
Darwin x86_64, macOS 15.7.4, Intel Core i7-8700B at 3.20 GHz. The parent separately
reported six physical/twelve logical cores, 64 GiB host RAM and a Docker Linux VM
with 12 vCPU/about 15.7 GiB RAM. **Host RAM is not the container's available budget.**
These are session-specific observations, not generic product requirements.

For this Mac, qualify the Linux/amd64 CPU image and actual Docker VM limits.
Do not advertise CUDA/Metal/CoreML just because the backend is containerized.
Docker documents Linux-container NVIDIA passthrough in Desktop as the
Windows/WSL2 path; Linux NVIDIA workers need their own driver/toolkit/image
qualification. Native macOS engines may use different CPU libraries/accelerators,
but that does not prove performance on this Intel machine.
[Docker GPU boundary](https://docs.docker.com/desktop/features/gpu/).

The checkout already has faster-whisper infrastructure in `lib/tts.py`:
`build_whisper_model`, `transcribe_segment` and `transcribe_ref`; synthesis QA has
another cached ASR path with word timestamps in `apps/voice-synth/voice-synth.py`.
There is no standalone STT/dictation app under `apps/` from this inspection.
Reuse and extract this capability rather than add a second audio/job/storage
system, but do not expose the current reference helper's lossy language mapping
or process logging as the canonical transcript contract. Preserve raw language
codes, model/revision, exact unedited ASR output and available segment evidence.

An STT worker should load independently from Qwen/Kokoro/Pocket. On a CPU machine,
start with bounded sequential inference and separate STT/TTS budgets so a long
render cannot unexpectedly make interactive dictation wait indefinitely. Decide
priority/admission after measurements; do not load every engine permanently.
FR/EN/mixed recognition, load time, warm latency and memory must be tested at the
same selected precision/model/image and CPU quota. A "local" model can still be
too slow for acceptable dictation even when batch transcription succeeds.

## Proposed scope: shared core now, optional adapter next

Core STT belongs beside synthesis as a typed audio job: an immutable asset,
language mode, optional bounded vocabulary hints, requested timestamp detail and
known engine/profile. Results include unedited ASR text, segments/word timing when
actually available, language evidence, provenance, timing and artifact IDs. Keep
the normal receipt short; the full transcript is an artifact/detail resource.
"Raw" means unedited ASR output, not a guarantee of verbatim accuracy. ASR timing
is not automatically forced alignment or reliable word-editing ground truth.

A compatibility endpoint for a selected existing client can accept the documented
multipart shape and return `text`, while the canonical async jobs remain the
authority. Bound synchronous wait/file size/duration and return truthful timeout
errors. No silent cloud retry. Preserve enough request identity to avoid duplicate
work when a desktop client retries. A local unauthenticated shim must remain
loopback-only; it is not the service's public security policy.

For inexpensive useful context, stage the capabilities:

1. **Default:** unedited local ASR; explicit language/auto choice; explicit user
   vocabulary; no app reading, screenshot, LLM, auto-send or continuous listening.
2. **Deterministic assistance:** versioned corrections/snippets and optional
   app-profile choice. Preserve the pre-correction text and show applied rules.
   Vocabulary hints and output replacement are distinct; neither guarantees the
   recognizer heard the intended word.
3. **Explicit context:** selected text or a small cursor neighborhood captured by
   the desktop after opt-in. Show which context and destination are active; bound
   length and clear it after the operation. Do not scan the whole desktop by default.
4. **Optional local rewrite:** separately provisioned small text model, explicit
   action/profile, raw/final diff and separate latency/memory budget. A local LLM
   is additional compute, not free smartness. External providers require explicit
   user configuration and a visible destination; basic STT must work without them.

Context and transcript contents are untrusted data, not instructions to the server
or connected agent. Dictation insertion must not press Enter or execute terminal
commands by default. Capture the intended target before recording; if focus moves,
offer copy/confirm rather than paste into an unrelated app. Clipboard restore is
best effort and must not overwrite content copied by the user during processing.

## Agent value without desktop bloat

An agent can transcribe an incoming voice memo, attach it to a production,
draft a response and synthesize a chosen voice using the same jobs/artifacts.
An optional native push-to-talk adapter can supply a transcript to the selected
agent session; sending and executing it are separate choices. Voicebox's Hermes
provider is a concrete official precedent for local TTS+STT agent integration,
including voice messages/push-to-talk.
[Hermes Voicebox provider](https://github.com/jamiepine/hermes-voicebox).

Expose `transcribe(asset_id, ...)` rather than dozens of OS-specific MCP tools.
The desktop can expose its own narrow local capture handoff separately. Agents do
not need screen images, entire transcript history or frontmost-app metadata for
ordinary audio transcription. Avoid an internal conversational LLM when the caller
already has one: the audio service provides grounded text and artifacts.

## Decision gates and remaining human choices

**GO:** design core STT as another supervised audio capability; evaluate existing
native clients and a thin adapter. **NO-GO:** promise Docker-only system dictation,
Intel performance, Flow-level contextual quality, or start a new sibling product
before those evaluations. These judgments do not delay the shared speech studio.

The next small comparison should use one client with cleanup disabled, local
CPU STT and optional qualified GPU STT. Measure press-to-listen, stop-to-text,
model cold/warm times, CPU/container memory and insertion success in a browser,
native editor and terminal. Include French names/numbers, English technical terms,
mixed sentences, silence, cancellations, focus changes, clipboard races,
permission denial, server unavailable and network-disabled operation after
provisioning. Keep recognition fidelity, polished writing and desktop reliability
as separate acceptance axes.

Human decisions still needed: is the immediate objective dictating into arbitrary
apps or sending voice to one agent/studio; should a local native engine remain an
optional backend if Docker is slower; is deterministic correction enough first
or is explicit local LLM rewriting required; and what latency/memory tradeoff is
acceptable for short FR/EN/mixed utterances. Those choices should precede any
sibling product commitment or broad context-capture default.
