# Current position and delivery plan

Implementation handoff is now written: use [IMPLEMENTATION-START.md](IMPLEMENTATION-START.md) and the canonical [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md). This recap remains background; its earlier sequencing is superseded where different. Implementation has not started and no delivery phase is accepted yet.

2026-10-15. Consolidates the product research, existing-solutions audit, contextual-dictation clarification and hands-on debugging. This is the current planning entry point; individual reports retain their historical evidence and independent recommendations.

Latest user scope decision: STT/contextual dictation is deferred to a future upgrade; the user will test it in TxCode first. The [fresh global Astra review](review-fresh-product-astra.md) is complete: GO to write the thorough product plan now, with no prototype/model campaign or human questionnaire prerequisite. The earlier consistency review applies to the previous scope, not a release approval.

## Where we stand

We have enough evidence to start a bounded contract/prototype phase. We do not yet have a new service, a studio implementation, a selected reuse strategy or a release-ready installation. Existing voice tooling remains the working foundation; its defaults and workflows have not been replaced.

The confirmed product serves one person or a trusted team, locally or on a private server. It provides HTTP, a small CLI, MCP and a complete human speech-production studio with scripts and timeline editing. Docker is the installation path. Agents should accomplish normal work without reading the repository or learning backend internals.

V1 focuses on voice management/creation, TTS and speech production. STT, contextual dictation and a possible native dictation sibling are deferred. The user will evaluate speech input in TxCode first; that work does not block this release. Existing clone-preparation/QA/alignment internals remain available where current features use them; this defers a new public STT product surface, not existing capabilities.

## What the exploration established

**The main product gap is coordination and usability.** The existing engines and audio operations are valuable. They need coherent voice identities, compact discovery, durable jobs, reliable cancellation/results, shared projects and an accessible studio. Simply wrapping command lines in HTTP would preserve too much operational complexity.

**Reuse is plausible but not yet decided.** The source audit shortlisted Voicebox adaptation against a focused service plus a maintained editor component, initially Waveform Playlist. LocalAI is an independent serving choice beneath either. Headless component and export probes exposed concrete integration issues, including overlap/undo behavior and missing-media/normalization semantics. These findings inform a reuse decision, not automatic adoption or rejection. Voicebox has not been selected or made a dependency. Fresh Astra recommends our existing backend plus selectively reused editor components; Voicebox stays inspiration/conditional reuse. A whole-product prototype is justified only by a credible paper savings map, not required by default. No whole studio has been qualified in a real browser/audio workflow.

**Agent efficiency needs measurement.** Compact schemas, bounded responses, scoped project/artifact retrieval, bulk operations and completion subscriptions are the proposed mechanisms. Token savings and effortless onboarding have not been measured in a shipped workflow. A completion event is not a universal guarantee that an agent host wakes its model.

**This Intel Mac is a viable CPU development target.** The hands-on failures exposed setup defects: CT2/OpenMP placement under OrbStack, inadequate pull supervision/deadlines, an unsupported CLI flag with a misleading zero exit code, and an upstream native-built binary executing unsupported AVX-512 instructions. Explicit worker placement worked repeatedly for CT2; an explicit AVX2-targeted whisper.cpp build worked. The lowest-level CT2 scheduling cause and historical zero-output pull remain unresolved. See [debugging evidence](evaluation/stt/DEBUGGING.md).

**Speed does not establish useful dictation.** Small CT2 ran the fixed clips near real time but missed a French proper name. Turbo matched that one French sentence at roughly twice audio duration. Tiny whisper.cpp was fast but poor on the French sample. No winner follows from two clips, and natural mixed-language/contextual accuracy remains untested.

## Recommended product structure

One shared domain service owns voices, assets, productions, jobs and immutable results. HTTP is canonical; studio, CLI and MCP are clients of the same contracts. A lightweight control service stays responsive while supervised inference workers do expensive work. Model downloads/provisioning are explicit. Discovery never loads weights or downloads models.

Durable job state and versioned project edits prevent duplicate work, lost results and human/agent overwrites. Start with one host-wide heavy-inference slot and explicit model/device/resource profiles. Preserve the existing CLI/Redis workflows separately during non-destructive migration.

For a future STT upgrade, clients supply bounded work context for dictation and own microphone permissions, recording, caret position and safe insertion. The service may combine recognizer hints and constrained terminology resolution; optional rewriting is separately requested. Preserve meaning, language switching, numbers and negation, expose ambiguity, and never silently downgrade or auto-send.

## Ordered next work

### 1. Close implementation choices with small, comparable evidence

The fresh review recommends writing the thorough plan now. The following bounded engineering lanes belong inside its delivery milestones, not as prerequisites to writing it. STT is not a v1 lane:

- **Reuse decision:** determine whether any whole-studio adaptation is warranted; Voicebox is a research candidate only. Qualify useful editor components against a shared synthetic production. Compare maintenance cost, canonical-state integration, editing quality and deployment complexity before selecting dependencies. Evaluate LocalAI only where it addresses a concrete serving gap.
- **TTS qualification:** preserve existing voice/default behavior and qualify install, provisioning, synthesis quality, latency and resource use on selected CPU/GPU profiles. Reuse the debugging lessons without making STT evaluation a dependency.
- **Contract prototype:** specify versioned resources, capability states, typed operations/errors, job lifecycle, revision checks and completion semantics. Exercise compact HTTP/CLI/MCP fixtures and a fake-worker restart/cancel/publication slice; measure actual agent task/context cost.

Exit: select build-versus-adapt, identify candidate TTS deployment profiles with explicit limits, and have executable contract fixtures. Do not broaden into a full application while these choices remain unresolved. No additional product questionnaire blocks this phase.

### 2. Ship the first working end-to-end service slice

Docker install → cheap discovery → usable voice → speech job → optional bounded wait/event → audio artifact, through HTTP, CLI and MCP. Verify restart, duplicate submission, cancellation, late-worker fencing and low disk with real workers. Keep legacy behavior intact.

Exit: a fresh agent succeeds without exploring source code; work/results survive restart; failed or canceled work cannot masquerade as success. This is an engineering milestone, not the whole v1 release.

### 3. Complete the shared production studio

Voice library and mic/file/remote acquisition; scripts and speakers; takes and targeted regeneration; imported audio; multiple tracks, trim/split/move, overlap, gaps, gain/fades; playback/waveforms; undo/history; asynchronous render/export. Agents operate on the same revisions and selected takes. Preview and export follow one composition contract with explicit missing-media and clipping policies.

Exit: a person and an agent can alternately edit a real speech production without lost edits, accidental retiming or unnecessary regeneration. The complete studio is required for v1.

### 4. Complete release qualification

STT/TxCode contextual-dictation experiments proceed separately at the user's initiative. Reading issue #8 did not modify TxCode or adopt its hosted-provider proposal.

Qualify clean install, model provisioning, upgrades, backup/restore, access controls, resource limits, agent completion integration, UI accessibility, rendered/listened audio and actual agent/studio task success. Publish a tested CPU/ISA/OS/GPU matrix with clear unsupported profiles. Resolve redistribution/model/dependency packaging before release.

Exit: practical user acceptance and reproducible installation/runtime evidence across the declared support matrix. A planning review is not release approval.

## Decisions deliberately left open

Whether whole-studio adaptation warrants further work, which editor components to reuse, TTS hardware profiles, name and visual identity. STT model selection, contextual resolution and any sibling dictation product are future decisions outside v1. These are bounded decisions with a place in the sequence, not reasons for more unstructured research.

The immediate next milestone is the **thorough product and delivery plan**, with requirements distinguished from proposed engineering choices and qualification stages. A release-shaped walking slice, including early UI and compact agent contracts, follows during implementation. Broad market research is sufficient to proceed; further research should answer a specific observed gap.
