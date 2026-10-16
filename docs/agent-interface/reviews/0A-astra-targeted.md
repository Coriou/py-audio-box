# Lot 0A — targeted Astra correction contract

2026-10-16. **NO-GO for freezing candidate `aa9a61a183545df65c96b5ccc5df071aa464c714` as the complete executable handoff. R2-1, R2-2 and R2-3 are upheld.** Proceed with one coherent, bounded contract correction in the existing owner lane, followed by independent Sol-low review. This is targeted advice, not final product, integration or Phase 0 approval. No objection is waived.

## Scope and evidence

Canonical planning/reviews were read in `/Users/ben/Projects/py-audio-box`: PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, IMPLEMENTATION-PLAN scope/shared contracts/Phase 0, lot 0A, and both Sol reports. Candidate inspected read-only in `/Users/ben/.txcode/worktrees/py-audio-box/t3-032fdaf1`; HEAD matched the SHA above and its worktree was clean. Read candidate CONTRACTS, RESPONSE-0A-r1, schema/catalogue, validator, generator and correction tests; checked actual Qwen, Kokoro and Pocket selector/reference expectations in source. No applicable AGENTS.md was found in the checked ancestry or relevant subdirectories.

Evidence here is E1 source/schema and small offline counterexamples only. Used existing `/tmp/pab-0a-contracts-venv/bin/python` with `PYTHONDONTWRITEBYTECODE=1`, candidate cwd, and `tests/contracts` on `sys.path`; imported `execution` and `contracts.validate`. No inference, models, private audio, network, image build or heavy suite. The reported 118 selected passing tests belong to the independent R2 report; they were not rerun or independently recertified here. Existing untracked root research was left untouched. Only this report was written.

The original F1/F3/F4/F5/F6 corrections are substantive and consistent with inspected source. Keep their closure at the scope evidenced by R2; new cross-object cases do not invalidate those narrowly stated closures. F2 remains incomplete. Passing local invariants cannot prove relations between independently valid objects.

## Findings independently assessed

1. **R2-1 upheld, P1.** `voice_version` has no executable preset selector; `frozen_plan` only checks speech version membership and engine/model/bundle. `contracts/generate.py:23` creates preset versions with no selector. `lib/tts.py:965–988` needs a Qwen CustomVoice `speaker`; `lib/tts_backends/kokoro.py:36–46` requires an explicit available `preset`. Opaque service IDs are not either selector. The existing `execution()` preset plan passed `worker_message` without one. A downstream worker cannot derive it honestly.
2. **R2-2 upheld, P1.** `contracts/validate.py:152–162` binds the plan hash but compares only the message's requested identity to request identity. Request fallback is not compared. `publication` at lines 115–123 has no stored resolution to compare with artifact identity. Reproduced acceptance of request `fallback=forbid` with explicit CUDA changed to CPU under message `allow_cpu`; also acceptance of changing both requested/effective artifact models together. Reproduced independent message `profile_id` substitution after deep-copying its identity away from the plan. This is authority drift, not merely missing schema enums.
3. **R2-3 upheld, P2.** `worker_message` checks only caller-listed input IDs against `allowed_assets`. `frozen_plan` checks reference metadata existence. Reproduced a clone version requiring `impulse` with both message input IDs and trusted allowlist empty. Required bytes are neither guaranteed available nor authorized by this relation.

Closely related holes to close in the same boundary, rather than another patch round:

- **Engine-specific settings:** the schema requires integer `seed` plus Qwen sampling controls for every plan. Actual Kokoro rejects any non-null seed and only accepts explicit fr/en language; Pocket accepts a seed/reference but rejects presets/instructions. An admitted Kokoro plan cannot faithfully pass this common settings object to its backend. A generic dictionary or silent filtering is not a fix.
- **Authority duplicates:** changed worker `deadline` and artifact `project_revision=999` both passed the current helpers. Bind these existing claims to the control-owned context. The deadline is an operational limit, not a worker preference. Render revision is output provenance, not free metadata.
- **Origin versus execution:** a saved designed voice needs a reusable realization, not an instruction to generate another random identity. Historical design provenance and the model/configuration used to reuse that voice are distinct. Reject unsupported combinations explicitly instead of letting arbitrary engine/origin combinations pass.

## Implementer brief: one frozen execution boundary

Allowed correction remains contracts, contract tests and phase-0 consumer documentation under lot 0A ownership. Do not implement a service, scheduler, database, engine adapter, browser, new migration executor or provisioning workflow. Preserve legacy source/defaults and the accepted byte protocol. Update schema/protocol/fixture versions and all affected examples deliberately; remove contradictory old version statements in CONTRACTS. These unintegrated fixtures have no deployed migration obligation, but consumers need one unambiguous revision.

The deliverable is a small set of closed variants and four explicit relations: **voice configuration → resolved execution plan → trusted job/attempt context → verified published result**. Extend the existing oracle, rather than build another contract system.

### 1. Freeze a typed reusable voice configuration

Add one closed discriminated configuration to each immutable voice version. Keep `origin` as lineage; make the discriminator identify the actual execution mode. Use a finite supported combination table, with these minimum meanings:

- Qwen preset: CustomVoice mode, exact backend `speaker`, compatible pinned model/bundle. Selector must be nonempty and belong to the control-supplied selector catalogue for that bundle; a string shape alone proves no membership.
- Kokoro preset: exact `preset`, compatible bundle and explicit supported language. Bind preset/language compatibility to the bundle's declared selectors. No references, instruction or seed pretending to be supported.
- Qwen reference clone: explicit selected managed reference(s), exact transcript and conditioning mode where relevant, compatible Base checkpoint/bundle. The small initial branch can accept exactly one prepared reference; reject unsupported multi-reference combinations instead of picking the first. Never transfer a remote/untrusted pickle as configuration.
- Pocket clone: one selected authorized WAV reference and matching bundle language, no preset/instruction.
- Designed voice reuse: retain design origin/description and source lineage, but freeze a concrete reusable reference realization and its compatible clone configuration. Record the design-generation identity separately when it differs from the reuse identity; no implicit model switch during speech. Do not re-run VoiceDesign to reuse an identity. If the candidate chooses another existing backend-supported reusable representation, document its exact input mapping and test it without models before admitting it.

This does not require implementing every legacy advanced option. Unsupported engine/mode/origin combinations fail before dispatch and remain explicitly unchanged legacy where appropriate. Existing discovered capabilities must not imply that every engine supports every mode. Synthetic selector catalogues are acceptable E1 inputs and must be labeled synthetic, never measured or installed.

Use a single authoritative location for reference selection/text. If legacy fields remain duplicated, enforce equality with the typed configuration at every containing validation entry point. A speech plan carries the exact version snapshot returned by version-get, not a fresh independent configuration overriding it. Freeze selector/configuration bytes into the plan digest; a selector change requires a new immutable version and a different plan.

Make effective synthesis settings a closed engine/mode variant too. Qwen retains explicit supported language/seed/sampling/instruction values; Kokoro has its supported language controls and no non-null seed or Qwen sampling controls; Pocket has its supported language/seed controls. Define omitted/default values at control resolution time, then serialize explicit effective values. Do not ignore submitted unsupported fields. Preserve authored text and distinguish any selected transformation. If `settings_sha256` claims to hash these settings, define one canonical preimage and verify it; do not leave a caller's arbitrary digest as authority over a different settings object.

### 2. Freeze control-approved resolution and execution policy

Keep the original request unchanged. Add a required resolved execution section inside the hashed plan, containing the approved requested/effective identity, fallback policy and actual reason, selected profile snapshot/version identity, and effective supported settings. The profile snapshot must bind the execution-relevant device/topology, exact backend/model/bundle/precision compatibility and configured limits; a mutable `profile_id` lookup alone is insufficient. A compact immutable snapshot is adequate; no hardware discovery implementation is requested.

Define these checks in the oracle with trusted inputs supplied by control:

- Resolved requested identity equals the original request identity for operations that have one. Effective engine/model/bundle and selected voice configuration cannot silently change. Explicit precision cannot change; effective device/precision cannot remain `auto`.
- Request `forbid` is never widened by worker or operator. Request `allow_cpu` is permission, not an obligation: control/operator may restrict it further. An actual fallback requires permission from both applicable policies, CPU destination and a nonempty reason; it preserves voice/model/bundle/settings. For prepare/design, whose current requests have no fallback field, freeze `forbid` as the explicit default, or add the same typed field deliberately; absence must never mean worker discretion.
- `auto` resolution is recorded once against an installed, qualified compatible profile within operator limits. CPU, CUDA and Metal use the same relation; no universal CPU fallback and no assumption of Metal support. Separate structural plan validity from admission: synthetic unqualified profile fixtures can be structurally valid, but an admission oracle must refuse them. Mock positive admission evidence is E1 test data, not real qualification.
- Bind the profile's immutable configuration, not volatile discovery timestamps, to execution identity. Later busy/disconnect/quarantine states can stop execution; they cannot authorize a different profile under the old hash. No model downloads occur through resolution.
- Render freezes its composition/revision/policy and execution profile without claiming that a single TTS model generated mixed imported media. Use an operation-specific render execution variant; retain original per-take synthesis identities as source provenance. Prepare/design similarly use their actual operation inputs, not arbitrary unrelated project settings.

`input_sha256` is the canonical hash of the entire validated frozen plan including this resolved section, selected voice configuration, settings and input asset metadata/hashes. Keep request idempotency fingerprint separate: repeated caller requests must resolve to the existing job, not silently re-resolve after profile changes.

Control stores this plan/hash before dispatch. Pass the **stored** plan (or an immutable context containing it) alongside the authoritative job into message and publication validation. Verify its hash against job.input_sha256. A received plan with a freshly calculated hash is not authority to update the job. Compare any duplicated message resolution/settings/profile claims exactly to the stored plan, including progress/result paths. Keep the existing owner, generation, staging and reap checks.

Reconnect and same-input retry preserve the same frozen plan/hash. Owner/staging may change only through the existing replacement protocol. A changed resolution/profile/configuration needs a newly resolved job with explicit lineage/authorization, never replacement under the old input hash. Freeze/check attempt deadline against trusted job/attempt context; an authorized retry deadline can be new attempt context without altering acoustic input identity. Worker-supplied deadline changes reject.

### 3. Derive required assets, then authorize them

Provide one pure `required_asset_ids(plan)` relation, shared by freeze/message checks and transport instructions. Compute it from executable fields, not the caller's list and not every historical reference in an attached project:

- Prepare: selected request source asset (with its frozen trim/transcript).
- Speech preset: no input media unless an explicitly supported configuration requires it.
- Speech clone or saved design reuse: exactly the configuration's selected reference dependencies.
- Voice design: no existing input media for text-only design; generated audition/reference files are outputs.
- Render: unique assets of every clip in the frozen composition, including muted clips under the existing missing-media contract. Imported beds and take-backed clips count equally. Unselected takes and old voice conditioning references are provenance, not render byte dependencies.

Require duplicate-free `message.input_asset_ids` to equal this derived set; every required ID must resolve to unique, matching frozen metadata and be included in the trusted authorized set. The authorized set may be larger, but that grants no extra input to this job. Prefer exact dependency metadata for non-render plans; retained extra historical metadata must be explicitly non-authorizing. Duplicate IDs/conflicting snapshots reject, including conflicting copies between project and plan when they refer to the same resource.

The allowlist comes from control's authenticated access decision, never from the worker or plan. Content transfer additionally enforces job/attempt-scoped access and current authorization, verifies bytes against frozen SHA256, and refuses unauthorized IDs even if their metadata is present. Asset presence/hash is not access permission. Bundle model files remain separately provisioned pinned resources, not arbitrary media-path inputs. Actual byte-transfer/race enforcement remains later runtime work; E1 must still reject the set mismatch now.

### 4. Compare outputs to the same trusted context

Publication accepts the stored plan/context, verified current-attempt result manifest and control-verified bytes, in addition to existing ownership fencing. Artifact resolution/profile must equal the frozen approved execution section; for render, revision must equal the frozen project revision. Define the neutral revision value for non-project operations instead of accepting any number. Voice version/configuration provenance must be recoverable from the immutable input manifest; duplicate explicit claims must match it.

Define the canonical manifest preimage for `manifest_sha256` and compare artifact output IDs/hash metadata to the verified current-attempt manifest. The existing `present_hashes` argument is explicitly trusted verification evidence; preserve that boundary, but do not treat any globally present managed asset as this attempt's result. Result IDs and artifact IDs must agree under a documented operation-specific output mapping. Input IDs and generated output IDs have different roles. Do not require inference or media decoding in the E1 oracle.

## Required positive/negative regression matrix

Use deep-copied or JSON-round-tripped wire objects for mutations. Current `execution()` shares nested identity objects with its plan: mutating a message identity can accidentally mutate the plan and produce `input_mismatch`, concealing the missing comparison. Each negative must first prove its baseline passes, change only the claimed field/relation, and assert the intended semantic rejection. A schema error or TypeError is not evidence of an ownership/policy fix.

| Boundary | Positive case | Negative cases required |
| --- | --- | --- |
| Preset chain | Two different selectors in one bundle: version-get → frozen plan → message → publication; exact selectors survive and hashes differ | Missing/empty/unknown selector; selector from another bundle; Qwen speaker used as Kokoro preset; changed selector under same version/hash |
| Engine settings | Qwen explicit supported controls; Kokoro fr/en compatible preset without seed; Pocket reference/seed | Mandatory/added Kokoro seed, unsupported instruction/sampling, incompatible language, Pocket preset, mismatched settings digest |
| Clone/design reuse | Prepare source → typed reusable version → speech; design candidate → save → reuse exact realized reference | Missing/ambiguous references, wrong conditioning mode/checkpoint, design reroll substituted for reuse, changed realization under same version |
| Policy resolution | Exact CPU; compatible CUDA/Metal; auto→compatible profile; expressly permitted CPU fallback with reason | Caller forbid + worker allow, operator forbid + caller allow, missing fallback reason, precision change, effective auto, simultaneous requested/effective model or bundle substitution |
| Profile binding | Immutable selected profile/config retained through reconnect/retry | Unknown/wrong-device profile; profile-only change; same ID changed limits/topology/compatibility; unqualified/absent profile admitted as ready |
| Frozen authority | Serialized plan roundtrip keeps hash; exact retry/new owner after valid reap keeps input identity | Rehashed received plan against unchanged trusted job; changed text/settings/voice/profile/assets; changed deadline; later project draft replacing frozen snapshot |
| Prepare assets | Exact selected source authorized | Omitted source list, absent metadata, present but unauthorized source, wrong hash |
| Speech assets | Preset empty dependency set; clone/design references fully authorized | Empty caller list hides reference; only one of multiple required dependencies authorized; duplicate/conflicting IDs; unrelated authorized asset added as job input |
| Render assets | Bed and take clips authorized, including muted clips; unused history needs no read grant | Omitted/unauthorized bed or muted clip; conflicting project/plan asset metadata; unauthorized historical asset transfer |
| Publication | Current verified result wins allowed cancellation race with exact plan identity/revision/manifest | Both models changed together, profile/device/precision changed, wrong revision, wrong result set/manifest, globally present unrelated asset substituted, late/wrong owner/staging/generation |
| Preservation | Existing F1/F3/F4/F5/F6 positive and negative fixtures remain valid at updated revision | Regressions in byte roundtrip, intrinsic nested resolution, immutable edits, ripple exact maximum/overflow, fencing/reap |

Add one complete synthetic transcript per admitted heavy operation (prepare, design, speech, render), using catalogue fields only. The transcript must obtain all needed values from preceding typed responses or explicit trusted control fixtures; no magic ID-to-selector convention, hidden DB lookup, mutable defaults or manually invented result linkage. Exact repeat/failure inputs should be small enough for another reviewer to inspect.

## Closure and next handoff

Implementer supplies one reviewable candidate SHA, bounded diff, updated version/fixture declaration, finding-by-finding response, the engine/mode support table and four transcripts. Run focused contract tests and deterministic regeneration in an isolated writable test context, not the read-only review checkout. Preserve exact legacy hashes/defaults. Run affected selected legacy checks once on the final candidate; no heavy model suite or repeated full image builds absent a relevant change. Identify the exact source used for any reused image.

Independent Sol-low reviewer must inspect actual schema/validator/backend mapping and the final diff, rerun the original R2 counterexamples using new valid envelopes, and independently mutate detached objects across the matrix. They should not accept renamed fields or tests that only call a new helper without reaching worker/publication entry points. Report separate closures for R2-1/2/3 and the adjacent settings/deadline/revision/manifest relations. Tests cannot establish true hardware qualification or actual synthesis identity; they establish honest, buildable boundaries.

**Closure condition:** no admitted plan requires guessing a backend selector/default, widening fallback, changing a frozen identity/profile, or reading an input outside the derived authorized dependency set; no result can claim a different frozen execution/revision through publication. All prior required findings remain closed on the exact new SHA. A local reviewer GO then permits orchestrator integration and affected integrated checks; it does not itself approve 0B/0C integration, Phase 0 exit or the product.

## Agent context and settled byte transport

Keep F1 closed at E1. Base64 JSON with authenticated explicit ranges is a workable bounded transport; no binary-HTTP redesign is required. The concrete context hazard is exposing a maximum 16 MiB byte response as MCP text: base64 alone is 22,369,624 characters, before JSON. Small download ranges reduce one response but do not make an entire audio file economical to route through the model's conversation.

Bounded follow-up for the existing client lane: ordinary job/artifact/status replies remain metadata-only; explicit byte operations remain opt-in. CLI/host-side adapters transfer and decode bytes outside model context and return a compact managed result/receipt. A direct text-only MCP client must use a conservative declared range/payload cap and direct users to that existing CLI transfer path when host-side resource handling is unavailable. Upload bytes likewise should be supplied by the host adapter, not authored as millions of base64 characters by an agent. Measure the chosen client's payload/token behavior at the already planned 0C/1B gate; do not claim agent-host acceptance from E1 or require a new transfer platform in this correction. This hazard does not reopen byte correctness or block this targeted contract repair.

CPU remains primary; qualified CUDA and optional native Metal remain supported profile paths. No public STT, Voicebox, private assets, cloud access, runtime expansion, commit, ledger edit, push/deploy/PR or subagent launch is authorized by this report.
