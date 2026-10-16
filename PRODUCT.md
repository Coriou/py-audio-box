# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Agents are the primary automation users. People and trusted teams install the product locally or on a private server and use a complete speech production studio, including scripts and timeline editing. This first-release audience was confirmed by the user on 2026-10-15.

## Product Purpose

Give agents a voice. Make discovering capabilities, managing voices, creating voices, generating speech, assembling productions, and obtaining results straightforward and economical in context, tokens, and compute.

## Operating Context

The first release serves one person or a trusted team on self-hosted infrastructure. Agents connect through HTTP and a small CLI, with MCP integration. People use a web studio. These interface and deployment priorities were confirmed by the user on 2026-10-15.

## Capabilities and Constraints

The existing project provides Docker-based audio tooling, a named voice registry, reference-based cloning, built-in and designed Qwen voices, speech generation, optional CPU engines, audio utilities, and Redis-based job automation. The new product is being planned; its service API and web studio are not implemented.

The requested product includes microphone, local-file, and remote-source voice acquisition; clean queue management and capability discovery; efficient optional completion notifications for agents; scripts and timeline editing. Exact editing depth, supported deployment hardware, framework, API contracts, authentication design, and wakeup integrations remain planning decisions.

## Brand Commitments

Proposed motto from the user: "give your agents a voice". Product name and visual identity remain open.

## Evidence on Hand

Repository implementation and README files, docs/TTS_MEMORY.md, docs/TTS_PERFORMANCE.md, and the existing backend contracts describe current behavior. Repository evidence does not establish new API, UI, installation, hardware, or acoustic acceptance.

## Product Principles

- Make a first successful agent task possible without reading the repository.
- Keep discovery and routine responses compact; expose detail when requested.
- Give people and agents coherent access to the same voice and production work.
- Make long-running work manageable and results recoverable without frequent polling.
- Preserve explicit control over compute and voice behavior.

## Deferred Speech Input and Portability Research

Latest scope decision (2026-10-15): STT and contextual dictation are deferred from v1. The user will first test them in TxCode. They are future upgrade candidates, not release requirements or prerequisites for this product. Keep the research and general CPU/Docker lessons without implementing new public STT-specific contracts in v1. Preserve existing transcription/QA/alignment internals wherever current cloning or production features depend on them; deferral does not remove existing features.

Earlier on 2026-10-15 the user requested dedicated STT research, with French, English and mixed-language speech leading evaluation. CPU-only older Intel Macs are first-class targets, and larger models/GPU evaluations are authorized where hardware is available. Easy Docker setup for agents across supported machines is a core requirement, alongside broad qualified GPU support. “Any box” is an installation goal that needs an explicit tested support matrix, not a proven universal claim.

Research recommends bounded shared-service STT for agent voice messages and media, with raw recognition and derived corrections kept separate. A Wispr-like system-wide dictation client/sibling remains a separate product decision; existing native clients should be evaluated before building one. No sibling application, native permissions integration or new default model has been selected.

For that future upgrade, contextual dictation is a first-class speech-input goal: preserve French/English and technical intent while reducing user correction effort. Clients such as TxCode supply bounded work context and own recording and safe draft insertion. Faithful media transcription remains a separate intent; no dictation sibling or hosted provider is selected.

## CPU and Accelerator Priority

The primary everyday environments are the constrained Intel Mac and smaller remote Grok Bot machines. Efficient CPU-only operation is a first-class requirement. Better machines should benefit from qualified GPU/CUDA and macOS Metal acceleration, suitable model choices and measured larger resource/concurrency profiles through the same interfaces. Do not hardcode CPU-only limits or silently change voice/model identity. Actual remote CPU/RAM/ISA specifications need inventory; no universal accelerator compatibility is established. Docker remains the baseline install; Metal may require an optional packaged native worker while the control/UI remains Docker-based.
