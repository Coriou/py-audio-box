# H1 reference-role contract independent review — r237 Sol

2026-10-24. **Bounded GO for the staged reference-roles-v1 contract. No blocking finding. No real-engine activation or release approval.** Domain/synthetic migration and separately authorized Pocket/Qwen actual qualification remain later work; no additional broad gate is introduced.

## Exact candidate and scope

Reviewed `/private/tmp/pab-r235-sol`, branch `impl/reference-roles-r235`, base `7d5ffe703a1397b314b58a7d73311f42d29e7659`, source `bf6fdf85d1fe60a50e0078ed724376758f02c75c`, final documentation HEAD `3879e3e2a630b2020219c4f882ea2670a50461f9`. Entry and final candidate status were clean. Source delta is exactly eight files: schema, operations, validator, scenarios, generator, the two new role transcripts, and test_reference_roles.py. The ninth base-to-HEAD path is the separate r235 report. No service/worker source or old fixture delta.

Read PRODUCT, IMPLEMENTATION-PLAN voice preservation/freeze provisions, r188 blocker 2/assignment 3, r235, r180 consumer inventory, actual schema/oracle/scenarios/generator, production/save/reuse, synthetic producer, Core publication and relevant consumers/tests. No applicable AGENTS.md found. Bounded memory search returned no relevant hit and supplied no premise.

## Contract assessment

- **Authority and membership:** `contracts/validate.py:143-198` retains attempt/generation/transport owner/staging/input hash and artifact/manifest binding. Exact output cardinality is two for prepared reference and one for design; output IDs cannot overlap required inputs, and every output needs its verified hash. `:201-223` requires full role asset equality with the actual typed output, so a role cannot merely name an unrelated asset. Both distinct prepare roles exhaust the two-output set. Input metadata remains exact and owner-authorized through the existing worker-message/input-content path.
- **Ordinary preparation:** `:207-221` binds reference/audition IDs to the frozen plan, reference version/configuration/transcript/provenance to the reference asset, and audition text to the frozen audition text. Prepared reference embeds the exact authorized original and requested segment. Positive frames and `frames * 48000 == segment_ticks * sample_rate` enforce segment duration without conflating source and output rates. Both IDs and hashes must differ from the audition. Original source ID/hash/provenance kind are checked. This is a coherent metadata contract for an actual prepared segment, not evidence that audio was prepared or that a transcript was recognized.
- **Design:** `:222-223` deliberately requires generated reference and audition to be the same asset with the same actual spoken text and exact design lineage. Existing `:189-196` checks preserve reuse identity, description, source candidate and generation identity. This exception cannot be applied to ordinary preparation.
- **Freeze and compatibility:** optional closed versioned definitions in `contracts/v1/schema.json:4055` onward are confined to prepare/design plans and candidates. `contracts/validate.py:393-409` checks role kind, reuse configuration, disjoint input IDs and operation-specific separation. `plan_hash` hashes the complete validated plan (`:413-415`), including role IDs/text, existing segment/transcript/settings and identity. Existing immutable save semantics are unchanged. Old validators reject role-bearing plans; coordinated producer/validator migration is required, as r235 states. Absent roles require one of the three explicit historical synthetic profile IDs (`:332-334`), and publication additionally requires synthetic output provenance (`:184-188`). This is a trusted control namespace, not an engine attestation. It does not authorize a real worker to masquerade under a synthetic profile.
- **Consumer coherence:** `contracts/scenarios.py:54-108` creates detached complete wire exchanges with both prepare assets, asset_get metadata and a saved version referencing the prepared output. `tests/contracts/test_reference_roles.py:25-38` independently exercises each saved version through complete speech generation and checks exact reference dependency/metadata/transcript; prepare never reuses its audition. Design intentionally does. Generator adds only the two named fixtures (`contracts/generate.py:81-82`); all old fixtures remain byte-compatible. Existing all-operation and running synthetic service consumers pass.

Trusted workers must actually perform the authorized transformation and speak the declared text; metadata/hash consistency cannot prove those acoustic facts. Core must verify the real files and atomically publish both outputs. Current `service/production.py:23` still allocates a conflated audition/reference ID; `worker/synthetic.py:18-33` still produces the historical candidate, and `service/core.py:497,521,537,565-566` retains one-output assumptions. These are explicit successor dependencies, not a defect in an optional staged contract. r235 names them accurately and makes no silent real-cloning claim.

## Independent execution

All checks used `/private/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, isolated unittest/direct function calls, and candidate imports. No pytest collection/conftest/guard ran.

- Inspected `/private/tmp/r235-contract-wrapper.py`: it imports all four named contract modules, expands declared parameter matrices, and provides fresh production fixtures. **144 direct cases PASS on final HEAD**, closing r235's stated timing limitation (its 144 run preceded the final small change).
- Final reference-role unittest: **7 methods PASS**, including its mutation loops and saved-version-to-speech checks.
- Production unittest discovery: **13 methods PASS**.
- Voice-store unittest discovery: **20 methods PASS**.
- Production client unittest discovery: **1 method PASS**.

Total: **185 final executed cases/methods**, not 185 individually enumerated mutation assertions. Initial wrapper launch lacked PYTHONPATH and failed at import before running cases; corrected launch with `PYTHONPATH=.` passed. The existing determinism function invokes the actual generator in the candidate and verified the complete fixture filename set and bytes unchanged. Additionally ran actual `contracts.generate` in an owned temporary contracts copy at `/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/pab-r237-generator-qj9a4l28`; exact complete fixture set/bytes matched. Candidate remained clean. `git diff --check` passed.

Supplemental direct checks passed: a non-synthetic profile without roles refuses with `reference_roles_required`; changing frozen audition text changes the plan hash; base schema rejects the new role-bearing plan. Source inspection confirms the same closed old-schema behavior for candidates.

No Docker, application network/listener, ML/model/hardware, provisioning, HELD operation, source edits, commit, delegation or unrelated cleanup. Sole root write is this report; the authorized temporary generator copy is retained.

## Handoff

Accept this staged contract lot. Keep actual production on its explicit synthetic compatibility path until the cohesive domain/worker/publication migration supplies and retains separate prepared reference and audition, authorized original/segment/transcript lineage, exact settings/model identity, both verified files and restart/reuse relations. Qualify actual Pocket/Qwen clone speech separately later. This review supplies contract/source and isolated synthetic evidence only.
