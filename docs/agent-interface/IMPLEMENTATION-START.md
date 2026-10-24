# Start here: headless implementation

The 2026-10-24 user decision is binding: ship existing fixes and full HTTP/small CLI/MCP control, working qualified CPU/CUDA/Apple Silicon native Metal, and clean voice cloning/management. **No web UI ships now.** Studio/browser accessibility/visual timeline and public STT/dictation are deferred. Preserve existing headless scripts, stitch/audio operations, internal QA/alignment, Qwen defaults, voice identities and legacy workflows.

Read [PRODUCT.md](../../PRODUCT.md), [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) sections 1–4 and your assigned H package, then current source/tests for that package. Read only relevant ledger/checkpoint entries; r146/r147 explain inherited evidence but their studio/diagnostic-first sequence is superseded. Historical reviews are not rewritten. [ORCHESTRATION-PROTOCOL.md](ORCHESTRATION-PROTOCOL.md) supplies role discipline; the new plan governs current scope and Astra milestone cadence.

## Current checkpoint and next work

Root inspected at `9b39f6061abebd6fdfd90885ee25d4aa75cdb28c`; recheck before edits. The accepted source integration at ec78661f is PARTIAL/validation-pending. CLOSED81 candidate86c7c744 is source accepted, not root adopted; r155 private preparation was in r156 review at entry; the parent has since recorded bounded source/inert GO, with no build/runtime authority. C/T2 is separate. Historical 81 remain UNRUN; native cause remains unknown. No operations are known running from the handoff, without a live probe. All resources/lane remain HELD and untracked STT must remain untouched.

**NEXT: H0 inventory plus durable immutable voice catalogue slice**, using existing SQLite/core and voice semantics, with the plan's explicit allowlist. This can proceed without the 81 campaign or native diagnostic completion. Do not auto-build, launch, adopt diagnostic branches, create private preparations or add instrumentation. Production recovery/security/ownership risks remain release gates through the explicit replacement matrix.

## Execute one bounded unit

- Record exact checkout/SHA, existing changes, literal owned files and shared contract owner. Preserve unrelated data/history/resources.
- Use Sol-low implementation and separate Sol-low review; parent integrates. Astra reviews material integrated milestones only. Four slots permit parent, two implementers and one rotating reviewer; do not spawn work unless commissioned.
- Reuse service/worker/client source. The current public API and clients are synthetic reads, not complete durable speech. Contract declarations and Compose filenames are not implementation evidence.
- Run only meaningful assigned checks under the authority actually given. Docs-only work needs content verification, not tests. `scripts/run-unit-tests.sh` builds Docker; remote/e2e scripts are not routine local gates.
- Keep source/synthetic, installed runtime/hardware, real agent and human listening evidence distinct. Missing hardware allows source progress but blocks release qualification; CPU, CUDA and native Metal are all required.
- Report changed behavior/files, exact checks/outcomes, carried versus new evidence, remaining blockers and next owner. Do not claim release completion or native causality from source review.

The plan itself grants no runtime, Docker, cloud-spend, cleanup or HELD-lane authority. Future web work uses the same public API; it adds no current release gate.
