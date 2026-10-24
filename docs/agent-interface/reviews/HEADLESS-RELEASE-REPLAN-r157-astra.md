# Headless release replan — r157 Astra

2026-10-24. **GO for the revised headless source implementation plan and H0 as NEXT. NO-GO for release or operational continuation.** The user's new binding scope replaces the complete-studio requirement. No user confirmation is needed for this scope update. This report is a design/source assessment, not independent runtime acceptance.

## Changes and authority

Exactly four files are owned by this assignment:

1. `PRODUCT.md`: three priorities, no shipping web UI, required CPU/CUDA/native Metal, coherent voice management, explicit auto-selection/provisioning and evidence boundaries.
2. `docs/agent-interface/IMPLEMENTATION-PLAN.md`: replaces the old studio delivery sequence with H0–H5, concrete reusable source inventory, HTTP/CLI/MCP coverage, finite allowlists/exits, hardware and installed safety gates, proportional diagnostic replacement and two feature lanes.
3. `docs/agent-interface/IMPLEMENTATION-START.md`: removes stale studio requirements and directs new agents to H0, current integration status and bounded authority.
4. This report, `docs/agent-interface/reviews/HEADLESS-RELEASE-REPLAN-r157-astra.md`.

No ledger, historical review, application code or other file was edited. No commit, delegation, Docker, tests/checker execution, model/runtime work, lane action, cleanup or new private bundle. Existing untracked STT and all retained resources/history remain untouched. Content verification only; no prose tests.

## Read basis and source limits

Inspected root HEAD `9b39f6061abebd6fdfd90885ee25d4aa75cdb28c`; no tracked changes at entry. Read the canonical documents, relevant execution-ledger checkpoint sequence through r155/r156 commission, r146/r147 accounting and r155 preparation status. Source inspection covered the HTTP/client skeleton, Compose topology, core/store method boundaries, contract operations and existing application/backend inventory. This is a bounded source inventory, not a hundred-report re-audit, engine execution or exhaustive legacy-control audit. H0 owns the finite command-level mapping.

Concrete correction: `service/api.py` is explicitly a synthetic fixture with only four public read operations and no auth/mutation/DB/ML service. CLI/MCP share its read-only HTTP transport. Broader JSON contracts and existing core job/events/reconciliation methods are reusable work, not proof of full public API implementation. `service/compose.release.yml` is still a fake-worker fixture. The revised plan no longer claims the service does not exist at all, nor that it already ships.

The accepted source checkpoint was root-adopted at ec78661f after r149/r150; preserve its accepted fixes and bounded evidence. CLOSED81 source86c7c744/docs c6354dd0 is source/inert accepted, not root adopted; private prep155 was in review156 at entry. Concurrent parent ledger updates now record r156 bounded source/private-closure/retained-inert GO, without build/runtime authority. Separate C/T2 source1d741cb is not implied integrated. No operations are known running from the handoff; no live operational probe was made. Historical 17 PASS/018 FAIL/81 UNRUN, separate017 and accepted extended018 remain distinct. Native cause remains UNKNOWN; lane/resources remain HELD.

## Product and engineering decisions

Full HTTP control and thin CLI/MCP parity are required for the shipped feature set: voice/media, clone/design/generate, headless scripts/takes/stitch/audio export, durable jobs/events/wait/cancel/retry/artifacts, discovery/provisioning and scoped administration. Existing commands/defaults/IDs/provenance stay compatible. Editor-only planned contracts are explicitly classified during H0; removal of a web gate is not permission to drop existing audio capabilities. Future humans use this same public API.

CPU constrained-host usability, CUDA, and Apple Silicon packaged native Metal are all required. Docker control plus managed native worker handles the Metal boundary; native DB ownership is excluded. Compatible automatic selection respects exact voice/model/engine constraints and overrides, explicit provisioning and measured resource/concurrency profiles. Missing hardware is an actual qualification blocker with finite work remaining; no paid provisioning and no optional-roadmap downgrade. No best-possible performance promise.

The blanket 81 diagnostic campaign is deferred from the critical path. Production deadline/ownership/reap/publication, unexpected signals, recovery durability, reader lifetime, auth and media security remain required. Section 6 maps these to installed product tests, preserves historical failures and requires an independently reviewed risk-equivalence mapping before release. Harness-dependent evidence cannot be reused while skipping the controls needed for that claim. No automatic81 build or extra instrumentation.

Native V1 setup/signal failures, V2 UTF8 pre-hook failure and V3 missing SHM capture remain separate unclosed evidence. Linux DB qualification does not clear Darwin failures. The selected Docker-control/native-worker boundary determines which production tests apply; any failing native path actually shipped must be resolved and qualified. Diagnostic causal investigation can be deferred when it is outside that topology, with the production safety property retained.

## Precise next unit and parallel work

**H0 now:** map existing commands/contracts once and implement the durable immutable voice catalogue slice in the current store: metadata registration against managed assets, list/get versions, aliases/archive and deterministic conflict/dependency handling. Exact prospective allowlist and meaningful restart/identity/race/migration tests are in plan section 5. No inference, ingest, recovery/guard edits or operational activation in that unit. It produces usable product source without waiting for81 or native diagnosis.

After reviewed H0 fixtures, Lane A implements voice/media/production domain and Lane B implements HTTP/auth/CLI/MCP. Each has Sol-low implementation plus separate Sol-low review; parent + two implementers + rotating reviewer uses four slots. Shared schemas belong to the parent. Reuse slots for engine/native profiles and packaging/persistence as the first lanes finish; sequence protocol/packaging overlap. Astra returns at the integrated durable headless journey and three-family release candidate, not after every unchanged preparation.

Remaining unknowns are bounded: command coverage, concrete available hardware/model compatibility, C/T2 integration and installed fault evidence, performance/listening and any later operational use of the source-reviewed r155/r156 preparation. None changes the user's scope or warrants another research ladder.

## Content verification and handoff

Read back the rewritten documents and searched studio/browser/optional/81/Metal/NEXT language: web references are deferrals, historical context or future-client boundaries; all three hardware families remain mandatory. Verified H0 is the single next unit, full clients use one service, and H3 synthetic packaging can proceed before final H4 real-engine acceptance. `git diff --check` reports no whitespace errors. Git changed-path inspection is limited to the four authorized files; untracked STT is preserved. No test result, clean install, benchmark, listening acceptance, native diagnosis or release completion is asserted.

Parent can commission H0 and maintain its ledger separately. This replan grants no runtime/build/HELD-resource authority, adds no operational approval to r155/r156, and does not adopt86c7c744. Final HEAD advanced concurrently to `d45c23eb3ddca3c65bab95d8c13b3e6b867377a1` through parent documentation commits f031a0a/d45c23e (r156 acceptance and r158 feature-inventory handoff). Those ledger/report additions were preserved, not authored here. Final status lists only the three canonical edits and this new report, alongside 136 pre-existing untracked STT entries.
