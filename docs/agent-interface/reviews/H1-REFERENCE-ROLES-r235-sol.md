# H1 reference roles — r235 Sol implementation

2026-10-24. Contract correction implemented for independent Sol review; no self-adoption or real-engine activation. The integrated synthetic domain remains compatible. Real Pocket/Qwen cloning still requires the successor migration and separately authorized actual qualification.

## Pins and scope

Entry checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-5a862749` was clean at `7d5ffe703a1397b314b58a7d73311f42d29e7659`. Created clean own worktree `/private/tmp/pab-r235-sol`, branch `impl/reference-roles-r235`, from that commit. Source commit: **`bf6fdf85d1fe60a50e0078ed724376758f02c75c`**. Report is a separate subsequent commit. Root advanced independently to `9b6f327e4a9aa192e8ba31cc02c8af68a0645b81` during this work; no adoption/rebase was performed. Root had pre-existing untracked STT evidence, left untouched. Sole root write is the copy of this report.

Read r188 blocker 2/assignment 3, PRODUCT, canonical implementation-plan preservation/freeze requirements and related voice-plan provisions, r180 consumer inventory, actual schema/oracle/scenarios/generator and production/worker/publication consumers. Bounded memory search returned no relevant entry and supplied no implementation premise. No applicable AGENTS.md was found.

Exact source changes:

- `contracts/v1/schema.json`, `contracts/v1/operations.json`, `contracts/validate.py`, `contracts/scenarios.py`, `contracts/generate.py`.
- New canonical fixtures `contracts/fixtures/v1/transcript-voice_prepare-roles.json` and `contracts/fixtures/v1/transcript-voice_design-roles.json`.
- New `tests/contracts/test_reference_roles.py`.

No service/core/worker or existing consumer edits. All pre-existing canonical fixture bytes remain unchanged, including operation-requests, production projects, voice versions and four existing transcripts. No frozen plans or historical reports were rewritten.

## Contract and compatibility

The explicit `reference-roles-v1` definition is staged through optional `worker_plan.reference_roles` and matching `voice_candidate.reference_roles`. It does not silently switch the running synthetic producer. `operations.json` records the staged contract and migration requirement for prepare/design. Public request/receipt definitions, exact identity/settings, worker envelopes, attempt fencing, publication ownership and immutable save semantics retain their existing authority.

The control-owned plan freezes reference kind, reference output ID, audition output ID and exact audition text, in addition to the existing original input, segment, reference transcript, reuse configuration/settings and identity. These fields participate in the plan hash. Neither output ID may be an input ID; reference ID must equal the frozen reuse configuration.

Ordinary clone prepare requires distinct `prepared_reference` and `generated_audition` roles and exactly two published outputs. Each role carries the complete typed asset (full SHA-256, frames, rate, channels, format/subtype and provenance). The prepared role also carries the exact full authorized original asset, selected half-open ticks and its associated reference transcript. Its provenance binds original ID/hash/kind, and duration equals the selected range. Candidate/version transcript, reference IDs, configuration and provenance must agree. Audition text is separately bound to the frozen audition text. Distinct reference/audition IDs and hashes reject the second-generation substitution. The oracle does not prove an audio transformation or speech recognition from metadata; the trusted worker and existing byte-verifying publisher still own those facts.

Design explicitly uses `generated_reference`, with the actual spoken request text, identical generated audition asset and exact existing design lineage. This deliberate generated-reference rule cannot be applied to ordinary prepare. Generated auditions require positive frames and generated or synthetic provenance. All reference/audition metadata must match the actual typed manifest outputs; both hashes must be verified, both outputs belong to the planned attempt, no extra output/caller asset is accepted.

Absent role definitions are now restricted to the exact historical `cpu-synthetic`/`cuda-synthetic`/`metal-synthetic` profile namespaces, and prepare/design publication additionally requires synthetic output provenance. Their same-asset/reference-text semantics remain fixture compatibility only, explicitly unsuitable for real cloning. This is a trusted-fixture restriction, not cryptographic engine attestation. Existing synthetic service fixtures continue to pass. Existing stored plan hashes are unchanged; a historical non-synthetic prepare/design plan without roles would now refuse. Existing saved immutable versions are not rewritten. Older validators reject new role-bearing plans/candidates, so producer deployment must be coordinated with this validator.

## Canonical generation and consumers

`contracts.generate` still generates every operation request and the four existing complete exchanges, and now additionally generates both versioned role transcripts. Scenario construction detaches wire objects, publishes both prepare outputs, exposes asset_get metadata for each and saves/retrieves the version referencing prepared audio. Design retains a single intentionally generated reusable output. Both new exchanges validate against the same operation catalogue, then feed their saved immutable version into a complete speech_generate publication whose only reference dependency is the saved reference, never the clone audition.

Rechecked the r180 inventory: exact all-operation loops in test_contracts and test_voice_store, production API's all-request/exchange consumer, scenario users in corrections/frozen/output boundaries, and actual generator/determinism. Existing consumers need no migration because the old fixtures remain explicit synthetic compatibility cases. New unittest consumers cover both added fixtures, all their exchanges and subsequent saved-version generation. No other fixture consumer was changed or bypassed.

## Executed checks

All checks used `/private/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and owning-worktree imports. No pytest collection, conftest or guard executed. Existing pytest-style functions were imported and directly called, expanding their declared parameter matrices and supplying fresh production fixtures.

- `python -m contracts.generate`: complete actual generation succeeded; only the two named new fixture files were added. Existing bytes remained identical.
- `/private/tmp/r235-contract-wrapper.py`: **144 direct function cases PASS**, covering all functions and parameter cases in test_contracts, test_corrections, test_frozen_boundary and test_output_boundary, including actual generator subprocess/determinism. Log `/private/tmp/r235-contract.log`.
- `python -m unittest discover -s tests/contracts -p test_reference_roles.py -v`: **7 methods PASS**. Includes valid non-synthetic prepare proposal, Pocket no-transcript conditioning, Qwen/reference reuse, design lineage/spoken text, exact input transfer authorization, and negative role conflation/transcripts/original/segment/provenance/media metadata/missing role/missing or foreign output/configuration/settings/stale generation/hash/schema cases.
- `python -m unittest discover -s tests/service -p 'test_production*.py' -v`: **13 methods PASS**; `/private/tmp/r235-production.log`.
- Voice-store discovery: **20 methods PASS**; `/private/tmp/r235-voice-store.log`.
- Production client discovery: **1 method PASS** (complete shipped-operation/reference journey).
- `git diff --check`: PASS. Own worktree clean after source commit.

Total: **185 final executed cases/methods**, with mutation loops inside the seven new methods. The broader 144-case direct wrapper preceded the final versioned-only audition provenance check and Pocket test addition; final seven-method unittest reran after those changes. Existing production behavior was unchanged by those final additions.

No Docker, application network/listener, model/ML/hardware execution, provisioning, HELD operation, delegation or self-adoption. No installed/runtime/acoustic/release acceptance is claimed.

## Successor interface and remaining gate

Parent should commission independent Sol review of source commit first. Do not activate real prepare against the absent-role compatibility path.

The separately authorized domain/worker successor must update `service/production.py` to allocate/freeze separate reference/audition output IDs and exact audition text rather than overwriting the reusable reference with an audition ID. Carry the managed original and selected segment through the existing owner-bound input publication; workers must actually prepare that authorized segment and generate the separately labelled audition. Return matching full role metadata and two verified outputs through the existing result manifest. Save still selects only a committed candidate and references the prepared output; original/segment/transcript association remains in its durable manifest and frozen job context, while saved version/reference provenance and transcript remain exact. Restart/reuse must retain those relations. For design, generated reference must use actual spoken text and design lineage.

Expected separately transferred files remain r188's `service/production.py`, `service/execution_context.py`, `worker/synthetic.py`, `worker/isolated_synthetic.py`, new Pocket adapter and tests, existing backend forwarding if needed. `service/core.py` must be assessed for accepting/verifying both output files and retaining their ownership/publication atomically; this contract lot does not authorize editing it or assume its current one-output runtime can already execute the new path. Any worker/runtime manifest limits and reconciliation dependencies must be named before migration. Existing synthetic producer remains on the compatibility fixture path until that cohesive migration is reviewed. Real Pocket/Qwen speech correctness, clone similarity/intelligibility and restart/cancellation qualification remain parent-owned actual gates.
