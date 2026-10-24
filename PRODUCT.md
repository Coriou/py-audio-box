# Product

<!-- impeccable:product-schema 1 -->

## Platform

Self-hosted headless audio service: HTTP API, small CLI and MCP. A human web client is deferred.

## Binding release direction — 2026-10-24

This user decision supersedes the 2026-10-15 requirement to ship a complete web studio. The current release has three priorities:

1. Ship useful existing fixes and audio capabilities behind a complete public HTTP API, with thin CLI/MCP coverage for agents and a future human web client.
2. Make CPU, NVIDIA/CUDA and Apple Silicon native Metal work on explicitly qualified profiles, with hardware-aware compatible backend selection and measured performance.
3. Make voice catalogue, acquisition, cloning, preparation, explicit audition, reuse and management coherent and dependable.

No web UI ships in this release. Studio, browser accessibility, microphone recording UI and visual timeline editing are deferred, not release gates. Public STT, dictation and TxCode speech input remain deferred. Preserve internal transcription/QA/alignment used by existing audio workflows.

## Users and purpose

Give agents a voice. One person or a trusted team installs locally or on a private server, discovers capabilities cheaply, manages reusable voices, generates and assembles audio, and retrieves durable results. Future human clients use the same public API, identifiers, revisions, permissions and jobs; they must not need a privileged second backend.

## Existing product and remaining work

The repository contains Qwen preset/design/clone tooling, explicit CPU alternatives, voice registration, audio export and Redis job workflows. It also contains a synthetic HTTP/CLI/MCP skeleton, contracts, and partially accepted service persistence, supervision and recovery source. The current public API is not the complete durable speech service; source presence is not installed or acoustic qualification. Preserve accepted fixes, voice IDs/aliases, authored text, provenance, defaults and legacy workflows while integrating the service.

The release includes headless scripts, takes, stitch/assembly and existing audio operations. Deferring the visual editor does not delete these capabilities. The canonical implementation plan maps existing functions and planned operations to finite implementation packages, including explicit deferred editor-only contracts.

## Hardware and installation

CPU on constrained Intel Macs and smaller Linux servers is first-class. NVIDIA/CUDA and Apple Silicon native Metal are also required release families, not optional roadmap items. Qualify exact hardware/runtime/model combinations; unsupported combinations remain explicit. Docker is the baseline control-plane install. Where Docker cannot expose native Metal, ship a packaged macOS arm64 worker with managed runtime, authenticated local transport and the same job semantics. Apple Silicon CPU Docker alone does not satisfy Metal acceptance.

Auto selection chooses among compatible, available, provisioned profiles, preferring measured qualified performance within memory and concurrency limits. Users can override engine/model/device/profile. Never silently substitute a voice identity, model or incompatible engine; explain an unavailable selection and offer explicit alternatives. Provisioning is deliberate, verified and resumable. Discovery never downloads or loads models. No best-possible performance claim without measurement. Unavailable hardware blocks actual release qualification, not independent source implementation. No cloud rental, paid provisioning or new costs are implied.

## Product principles

- A first successful agent task must not require reading source.
- Keep discovery, receipts and errors compact; fetch detail and audio explicitly.
- Preserve immutable voice versions, exact inputs and recoverable job/artifact history.
- Support bounded waiting, cancellation and durable events without model-driven polling loops.
- Enforce trusted-team authentication, storage integrity, ownership and restart safety.
- Measure constrained-host usability and stronger-hardware throughput separately; retain explicit model and voice semantics.

## Authority and evidence

[IMPLEMENTATION-PLAN.md](docs/agent-interface/IMPLEMENTATION-PLAN.md) is the current delivery plan; [IMPLEMENTATION-START.md](docs/agent-interface/IMPLEMENTATION-START.md) is the entry point. Historical reviews and the execution ledger retain their original evidence and scopes. Their older studio requirements and diagnostic-first sequencing are superseded by this release decision, not retrospectively rewritten as passes. No release, hardware, native-crash diagnosis or operational acceptance is established by this document.

Proposed motto: “give your agents a voice”. Product name and visual identity remain open and do not block headless implementation.
