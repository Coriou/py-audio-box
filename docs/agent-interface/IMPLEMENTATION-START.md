# Start here: implementation agents

Use [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) as the canonical implementation plan. The research archive explains decisions; it is not a prerequisite reading list or an additional set of release requirements. Latest user instructions take precedence over the plan.

Execution follows [ORCHESTRATION-PROTOCOL.md](ORCHESTRATION-PROTOCOL.md): orchestrator → Sol-low implementer → separate Sol-low adversarial reviewer → integration verification → final Astra review/corrections. Read the rules for your role and ticket; the orchestrator maintains [EXECUTION-LEDGER.md](EXECUTION-LEDGER.md).

## Read only what the assignment needs

1. Read repository instructions and [PRODUCT.md](../../PRODUCT.md), then the plan's scope, shared contracts, execution guidance and your assigned phase/work package.
2. Inspect the current source, dependencies and test entry points for that package. The plan describes future files as well as existing ones; confirm which exist before running commands.
3. Read a research report only when the plan cites it for a decision you actually need. Do not restart broad product/model research.

V1 is the shared voice/TTS production service and complete studio, through HTTP, CLI and MCP, installed with Docker. Public STT/contextual dictation and TxCode integration are deferred. Preserve existing internal reference-processing/QA/alignment capabilities, Qwen defaults, voice identities and legacy workflows. Voicebox is not an adopted dependency or a mandatory fork. CPU-only operation on constrained Macs/small servers is primary; use the plan's hardware-profile contract to preserve CUDA/Metal paths and avoid hardcoding CPU limits into every installation.

## Execute one bounded package

- Record the baseline, branch/worktree, existing uncommitted work, owned files, dependencies and interface owner. Preserve unrelated work and research evidence.
- Use the plan's engineering defaults. Change a shared contract deliberately with its fixtures and affected clients; do not independently invent competing schemas or stores.
- Build the smallest complete slice, including relevant failure states. Parallel work must have non-overlapping ownership or an explicit integration owner.
- Run the package's meaningful fast checks first, then its required integration/browser/runtime checks. `scripts/run-unit-tests.sh` is the existing isolated unit entry point. Inspect commands before execution: the legacy `scripts/run-e2e-test.sh` performs remote operations and is not a routine local acceptance gate.
- Verify the integrated state, not just an isolated patch. Model hardware, browser, acoustic and release evidence are distinct; fake-worker success does not satisfy a real-audio gate.
- Continue only when required exit conditions are evidenced. Missing hardware leaves that profile unqualified; it does not justify broad platform claims or blocking unrelated work.

## Hand off enough evidence to continue cheaply

Report package/phase, resulting behavior, changed paths and contract revisions, exact checks and outcomes, evidence locations, unresolved failures/limitations, and the next unblocked package. Distinguish implemented, tested and accepted. Never claim a phase complete because code exists or unit tests alone pass.

This packet authorizes no work by itself; carry out the implementation assignment actually given. No application code was implemented while preparing this plan.
