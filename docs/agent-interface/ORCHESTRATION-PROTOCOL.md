# Implementation orchestration protocol

2026-10-16. Execution companion to [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md). The orchestrator coordinates; **Sol low implements, a separate Sol low adversarially reviews each implementer, and Astra reviews the integrated result and corrections at the end**. This document is a procedure, not evidence that implementation or phase exits have passed.

## Roles and authority

- **Orchestrator (parent):** owns decomposition, shared-contract decisions, file ownership, scheduling, integration, evidence ledger and phase admission. Sends self-contained bounded briefs, resolves disagreements from source/tests, and reports progress and unresolved limits. Does not let parallel agents create competing APIs or silently broaden scope.
- **Implementer:** `gpt-6.1-sol`, reasoning `low`, one named work package/subslice. Implements and verifies its assignment, preserves defaults/data and supplies a reviewable commit plus evidence. Does not approve its own implementation.
- **Adversarial reviewer:** a separate `gpt-6.1-sol`, reasoning `low`, assigned to that implementer's lot. Fresh context, no shared implementation ownership, read-only source/diff inspection and permitted isolated tests. Never repairs the candidate while reviewing it. The implementer's summary is supporting evidence, not the review's conclusion.
- **Final adviser/reviewer:** `gpt-6-astra`, using the configured reasoning setting unless separately specified. Reviews the integrated candidate against the complete product journeys, contracts, phase gates, preservation requirements and evidence. May make bounded corrections on an assigned branch or return work to implementers; corrections still need independent review and final integrated verification.

Resolve model IDs/settings from the live orchestration catalogue at launch and record the actual selection. Never silently replace the requested models/effort. Prefer native same-provider delegation when it supports the selection; use app-owned delegated tasks for worktree isolation or other required orchestration. Never create top-level conversations as a substitute for child work. Retain task IDs and use a new delegated task/request ID for each app-owned review round.

## Before the first implementation lot

1. Inspect current HEAD, status, repository instructions and ownership. Preserve unrelated tracked/untracked work; capture a recoverable inventory rather than cleaning/resetting the checkout.
2. Curate and version the product direction, canonical plan, short entry point, this protocol and required supporting references as a planning baseline. Do not blindly add model downloads, compiled builds, private audio, credentials or raw operational receipts. Preserve excluded research locally; make needed non-sensitive evidence accessible to reviewers.
3. Record the baseline commit, contract/fixture versions, plan revision and initial gate state in [EXECUTION-LEDGER.md](EXECUTION-LEDGER.md). No implementation phase starts marked complete.
4. Create an integration branch and task-owned worktrees from the recorded baseline. App-owned isolated tasks start from committed parent state: **uncommitted planning files are not copied**. Verify required files exist in each checkout before launching implementation; do not copy an entire dirty tree.
5. Assign one owner for shared schemas/migrations and one integration owner. Phase 0A establishes the contract pack; Phase 0B's editor work and Phase 0C's skeleton use its recorded fixture version. Minimum predecessor gates must exist before dependent lots begin.

This preparation is a mechanical launch prerequisite, not a new product research phase. When this protocol was authored, the checkout had untracked product/research documents. The execution ledger records subsequent baseline preparation; verify its committed handoff before isolated implementation starts.

## One lot, one observable outcome

The orchestrator creates a ticket with:

```text
ID / phase / outcome / owner:
Model + effort / task ID / worktree / branch / base SHA:
Required predecessor gates + contract/fixture versions:
Allowed files; shared-file owner; preserved behavior/data:
Required behavior, failure cases and explicit non-goals:
Deliverables; tests/commands; evidence tiers and resource bounds:
Integration dependency/order; migration/rollback implications:
Definition of done; reviewer; next unblocked lot:
```

Split a package if its outcome, ownership or test surface cannot be reviewed coherently. Agents read the short entry point, shared contracts and assigned phase/source; the archive is on demand. Include unresolved findings and exact revisions when continuing a lot, rather than forwarding an entire conversation.

Working state path:

`ready → implementing → awaiting_review → changes_requested|review_passed → integrated → integration_verified`

`blocked` records a concrete missing dependency/evidence and the work that can continue independently. A reviewer GO is only local candidate approval; it is not an integrated phase GO. Stale reviews never approve a changed candidate.

## Implementation and independent review cycle

1. **Implement:** the Sol works only within the ticket's scope. It runs meaningful targeted checks and produces a bounded commit, changed-file list, behavior/contract delta, exact commands/results, evidence tiers and remaining limitations. The orchestrator verifies the candidate is actually present and the diff contains no unrelated changes.
2. **Assign adversarial review:** give a separate Sol the requirement, preservation rules, base/candidate SHAs, file scope, contract/fixture versions, test access and known limitations. The reviewer independently reads actual source and the complete diff. Inspect callers and failure paths, not only added tests.
3. **Challenge:** check feature/default preservation, unsupported controls, identity/cache correctness, concurrency and cancellation/publication, input/media/access boundaries, resource use, API/CLI/MCP/UI consistency and the relevant phase-specific failures. Verify tests exercise behavior rather than mirror implementation. Rerun the necessary checks in a disposable context when useful; do not duplicate expensive suites without a reason.
4. **Report:** return `GO` or `NO-GO` for the exact candidate, with evidence inspected/run. Each blocking finding needs a location, concrete failure/contract violation, reproduction or reasoned failure path, and a bounded fix/verification expectation. Label uncertainty. Suggestions/style preferences are non-blocking unless they violate an agreed requirement. Missing material evidence can block a claim even without a proven code defect.
5. **Correct:** the implementer fixes accepted findings on a new commit and supplies a finding-by-finding response. The same independent reviewer rechecks the affected behavior and final diff; a replacement reviewer receives the original brief, prior findings, responses and unresolved objections. No self-certification or unreviewed last-minute edits.
6. **Bound the loop:** after two unsuccessful correction rounds on the same issue, the orchestrator narrows the lot, identifies the unresolved premise and obtains a targeted Astra decision when architectural guidance is needed. Do not repeat identical tests or escalate model effort silently. There is no round limit that converts a known defect into GO.

A reviewer has no defect quota and may return GO after a substantive inspection. An implementer/reviewer disagreement is resolved by the orchestrator with evidence, not by finding another reviewer who will approve it. A change in reviewer cannot discard earlier unresolved findings.

## Parallelism and constrained-host scheduling

Default to at most two independent implementation lots at once, with reviews scheduled within available agent slots. Each implementer receives its own independent reviewer; reviewers can run sequentially. With four total native slots, the parent plus two implementers and one active reviewer fits; do not assume two implementer/reviewer pairs can all run simultaneously.

Contracts/migrations and overlapping files serialize through their owner. Consumers may use frozen fixtures in parallel but cannot change the shared contract independently. Integrate contract changes first, consumers second, then run the combined slice. The orchestrator resolves any merge conflict deliberately and reviews changed semantics.

One expensive local resource lane serializes model runs, heavy builds and large container/browser tests as needed on the constrained host. Record task-owned processes/containers, deadlines and cleanup evidence; never stop unrelated workloads to manufacture a benchmark. CPU/ISA/RAM/VM limits are measured. GPU/CUDA/Metal qualification uses the same profile/gate discipline; unavailable hardware remains unqualified. Synthetic/fake-worker tests enable ordinary implementation without new models or paid resources.

## Integration and phase exit

The orchestrator accepts a reviewed commit into the integration branch using a recoverable Git operation, records source and resulting SHAs, and runs the combined checks required for that slice. If integration changes behavior or contracts, obtain adversarial review of that new delta. Resolve conflicts through the owning implementer where appropriate; never replace a checkout wholesale.

For a phase GO, require all of:

- All required lots independently reviewed and integrated, with findings closed or explicitly non-blocking and justified.
- The phase's exit behaviors demonstrated on the integrated candidate, including required failure scenarios and legacy regressions.
- Exact contract, source/image/asset/profile identities and actual commands/results recorded; implementation, tests and acceptance are distinguished.
- Required E1–E5 evidence from the plan present at its proper scope. Units/fakes do not prove installed hardware, browser usability or acoustic quality.
- No broken dependent client, undisclosed default/identity change, unqualified hardware claim or required evidence gap hidden as success.

The ledger records the phase decision and newly unblocked work. No repeated human confirmation is required for routine bounded checks already covered by the implementation assignment. Escalate genuinely unresolved product decisions, unavailable necessary resources or actions outside authorization; do not ask the user to choose routine libraries already assigned a working default.

## Final Astra review and corrections

When required phases are integrated, freeze a candidate SHA and give a **fresh Astra context**: product scope, canonical contracts/plan, complete change range, phase/lot review index, integration tests, release artifacts, hardware matrix and unresolved limitations. Astra inspects the real code and cross-phase behavior; it does not merely endorse Sol summaries.

Astra returns a global GO/NO-GO with specific findings. Bounded fixes may be authored by Astra or assigned to the appropriate Sol. Every code correction, including Astra's, receives independent Sol-low review; the orchestrator integrates it and reruns affected plus cross-cutting checks. Astra then reviews the final corrected candidate and gives the closing verdict tied to its SHA. Earlier reviews do not cover later edits.

Final readiness requires the complete v1 journeys, not only the headless slice. Required CPU/GPU/Metal claims, visual/accessibility and human listening acceptance need their own evidence. An overall local code GO is not permission to claim unavailable hardware or publish/deploy. Preserve explicit remaining limitations in the release decision.

## Hard boundaries and reporting

Preserve existing Qwen defaults, voices and legacy workflows. Public STT/TxCode integration stays deferred. Voicebox is not an assumed dependency. No silent model/device fallback, download on discovery, arbitrary path/command exposure, forced unlock, destructive cleanup or hidden remote infrastructure operation.

Use synthetic/licensed fixtures by default. The existing isolated unit launcher is `scripts/run-unit-tests.sh`; inspect its current implementation and build cost. The legacy remote `scripts/run-e2e-test.sh` is excluded from routine gates. No cloud rental, remote mutations, private-audio access or broad model campaign is inferred from the protocol. Verify actual source before executing any named script.

User updates name the current phase/lot, demonstrated progress, material uncertainty and next check; do not dump internal logs. Final handoffs include the integrated commit, completed gates, actual tests/evidence, open limitations and next step. Store sanitized manifests/hashes and bounded reports, not secrets or unnecessary raw receipts.
