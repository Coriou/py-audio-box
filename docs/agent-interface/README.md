# Agent interface planning packet

Date: 2026-10-15. Planning only; no service API or studio has been implemented.

**Implementation agents:** start with [IMPLEMENTATION-START.md](IMPLEMENTATION-START.md), then the shared sections and assigned phase of [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md). The latter is the canonical delivery plan, authored by Astra and independently read for scope and integration consistency. Its phase gates supersede older delivery sequences and research feature lists. No phase is complete merely because the plan exists.

[Orchestration protocol](ORCHESTRATION-PROTOCOL.md): Sol-low implementation, independent Sol-low review per lot, orchestrator-owned integration and final Astra review/corrections. [Execution ledger](EXECUTION-LEDGER.md) starts with launch preparation and Phase 0 pending; application implementation has not started.

Latest scope: STT/contextual dictation is deferred from v1; the user will test it in TxCode first. The [fresh global Astra review](review-fresh-product-astra.md) is complete: GO to write the thorough product plan now. Recommended baseline is our existing backend plus selected studio components; no mandatory Voicebox fork/prototype. Earlier STT reviews remain research evidence, not current release requirements. Voicebox is not adopted or a selected dependency.

Optional research navigation: [recap](CURRENT-PLAN.md), [earlier engineering proposal](PLAN.md), [STT and portability research](STT-AND-PORTABILITY.md), [existing-solutions audit](AUDIT.md) and [candidate acceptance catalogue](ACCEPTANCE.md). Read only the evidence relevant to the assigned decision; this is not an onboarding checklist.
[../../PRODUCT.md](../../PRODUCT.md) records the user's confirmed direction.

Earlier research verdict: [Astra STT/portability review](review-stt-portability-astra.md) — GO for bounded core STT and contract/prototype comparison; NO-GO for CPU starter promotion, universal hardware claims, product adoption or a dictation sibling.
Hands-on follow-up: [failure diagnosis and local CPU profile](evaluation/stt/DEBUGGING.md), with separate throughput, portability and accuracy evidence.
Prior adoption verdict: [Astra existing-solutions review](review-solutions-astra.md).
Earlier architectural verdict: [Astra R2](review-astra-r2.md).
First review: [Astra R1 and concrete dispositions](review-astra-r1.md).

Research evidence:

- [Existing source and feature audit](research-feature-audit.md): reusable operations, queue/worker behavior, migration constraints, gaps and release gates.
- [Agent contracts and current protocol research](research-agent-contract.md): HTTP/CLI/MCP, production edits, durable jobs, context cost and completion integrations, with primary sources.
- [Product, studio and installation research](research-product-install.md): full speech editor, voice acquisition, first run, deployment, resource targets and qualification.

Research reports retain their independent recommendations. IMPLEMENTATION-PLAN.md resolves current scope, working defaults and delivery gates; earlier recommendations are supporting evidence. Proposed budgets are not measurements, and a planning review does not establish runtime, visual or acoustic acceptance.

Existing-solutions audit (2026-10-15):

- [Whole-product adaptation and build-versus-adopt](audit-build-versus-adopt.md).
- [Self-hosted servers, models and agent clients](audit-self-hosted-solutions.md).
- [Commercial voice APIs and agent integrations](audit-commercial-agent-solutions.md).
- [Studios and reusable editor components](audit-studios-editor-components.md).

The audit reopens implementation selection while preserving confirmed product requirements. Independent reports contain competing scope recommendations; AUDIT.md and PLAN.md record their disposition. The initial audit was source-based; the bounded probes and partial CPU measurements below add evidence without qualifying a product or runtime.

STT/deeper comparison research (2026-10-15):

- [CPU runtimes, model tiers and accuracy evaluation](research-stt-runtimes.md).
- [Agent STT contracts, context and provider hooks](research-stt-agent-contract.md).
- [Native dictation versus shared service](research-dictation-product-boundary.md).
- [Docker/platform qualification and installation design](research-docker-portability.md).
- [Serving and whole-product integration comparison](research-serving-adoption-comparison.md).
- [Published editor headless probe](evaluation/editor/README.md), [Voicebox export-function probe](evaluation/voicebox/README.md), [Mac Docker socket smoke](evaluation/ipc/README.md), and [host inventory](evaluation/host.json).
- [Final public-data CPU evaluation](evaluation/stt/RESULTS.md): two timeouts/partial EN-only output; whisper.cpp blocked before inference. No efficient CPU starter is qualified.

The expanded research preserves a Docker-first shared service and full studio; a native dictation sibling remains unselected. CPU/model support and acceleration are qualified per profile rather than inferred from an upstream hardware list.

Contextual dictation clarification (2026-10-15):

- [Contextual recognition, terminology resolution and TxCode client contract](research-contextual-dictation.md).
- [Astra contextual-dictation review](review-contextual-dictation-astra.md): GO for contract/prototype; quality and CPU latency remain unqualified.
