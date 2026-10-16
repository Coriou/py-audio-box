# Lot 0A — executable contracts and compatibility pack

Status: ready for Sol-low implementation after launch baseline preparation. This is a Phase 0 lot, not a server/UI implementation. Follow [the orchestration protocol](../ORCHESTRATION-PROTOCOL.md) and [Phase 0 of the canonical plan](../IMPLEMENTATION-PLAN.md).

## Outcome

Give service, studio, CLI/MCP and worker agents one versioned, executable vocabulary and shared fixtures. Another agent must be able to build 0B/0C against the result without inventing state, timing, errors or legacy feature coverage.

## Ownership

Implementer owns new `contracts/**`, `tests/contracts/**`, `docs/agent-interface/phase-0/**`, and minimal contract-test collection/dependency changes to `Dockerfile.tests`. A dedicated lightweight test launcher/requirements file may be added if necessary for reproducible checks. Do not edit existing application code, the orchestrator's ledger/ticket, legacy defaults, production Compose or research receipts. Avoid broad dependency/lockfile churn. New tests must not break the existing offline test image's collection.

## Deliverables

1. Versioned machine-readable resource and operation schemas for capabilities/profiles, assets, voice/version, project/revision/block/speaker/track/clip/take, jobs/attempts, artifacts, bounded lists and stable errors. Define intended HTTP methods/routes and CLI/MCP operation correspondence sufficiently for consumers; no arbitrary argv or opaque command bus. Schemas/fixtures are the interface, not a duplicate application implementation.
2. Normative invariants and executable checks/negative fixtures where structural schema cannot express cross-field behavior: expected revisions, idempotency fingerprints, job transitions/cancel races/fencing, immutable identity/publication, 48k integer ticks, ranges/trims/fades, fixed versus explicit ripple, stale/candidate/selected takes and explicit rendering policies.
3. A deterministic synthetic three-speaker/30-block production, and 100-block/200-clip scale fixture with imported-bed/overlap/take-replacement cases. Include tiny generated impulse/sine media or a reproducible generator; no private audio/model weights. Include expected results/invalid cases usable by the browser and renderer lanes.
4. Complete public command/control compatibility ledger from current source: each existing capability maps to service, typed advanced compatibility or unchanged legacy coverage, with rationale and preservation tests/fixtures. Retain Qwen modes/defaults, IDs/aliases, references/provenance, CPU adapters, internal ASR/QA/alignment and existing exports; no silent omission or invention of support.
5. CPU/CUDA/Metal profile and worker/control boundaries, readiness/support states, explicit requested/effective device/model/precision and visible fallback policy. Finite configuration ceilings and synthetic example values are provisional until real qualification. Do not claim remote Grok Bot specs, Metal support or STT timings as TTS evidence.
6. Non-destructive migration policy into a distinct service root: dry-run, collisions, legacy aliases/source hashes, missing provenance, pending jobs and rollback boundaries. No migration execution on real data.
7. Minimal meaningful contract test harness, documented exact commands and results. Keep canonical names/versions and consumers' import/validation paths explicit. Record engineering decisions succinctly; do not restart broad research or require choosing a product name.

## Preservation and exclusions

No public STT/contextual-dictation APIs, TxCode changes, Voicebox fork/dependency, new serving platform, real database/executor, inference, model downloads, cloud/remote actions or private data. Hardware discovery fixtures must not imply measured support. Existing CLI/Redis workflows remain untouched. Do not run the legacy remote `scripts/run-e2e-test.sh`.

## Verification and handoff

Validate examples and purposeful rejection cases with a standards-based schema validator where appropriate; do not implement a generic homegrown JSON Schema engine. Test semantic failures that structural schemas cannot catch. Demonstrate IDs/text/default preservation and fixture determinism. Run the focused contract suite in a reproducible isolated environment; verify the existing unit image collects the new tests and run relevant existing regression checks if test-image dependencies change. No real models are needed.

Return a bounded committed candidate: base/candidate SHAs, paths/interfaces, schema/fixture version, exact checks and outcomes, known gaps and instructions for 0B/0C. Do not modify integration branch or mark the phase complete. A separate Sol-low reviewer must examine actual code/diff and fixtures before integration.
