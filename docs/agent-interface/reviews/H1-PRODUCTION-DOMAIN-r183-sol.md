# H1 trusted production domain r183

Status: IMPLEMENTED FOR INDEPENDENT REVIEW. No self-adoption, installed activation, release GO or real engine/cloning-quality claim.

## Exact pins and authority

Sole successor checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-f525ccdd`, branch `t3/trusted-production-domain`. Clean entry HEAD `18b735ddd771989f3f93a041de6534c2fa1c78cd`. Source commit `8dd7e91fb682dba9b102f70e7ed9fec69184bed6`. Predecessor r182 was read and not resumed. User explicitly extended the worker allowlist; no missing contract dependency was found. Read PRODUCT.md, IMPLEMENTATION-PLAN H0/H1/H2/H4, r182, Core, voice store, ingest, persistence-facing publication flow, contract validation and prepare/speech transcripts, synthetic worker paths and both conftests.

## Files and trusted interface

Only source files changed: `service/production.py`, `service/core.py`, `worker/synthetic.py`, `worker/isolated_synthetic.py`, `tests/service/test_production.py`, `tests/worker/test_production_synthetic.py`. No persistence schema, contracts, HTTP, clients, dispatcher, legacy command or engine changes.

`Production(core)` exposes `prepare(principal,key,context)`, `results(principal,job_id)`, `save(principal,key,job_id,candidate_id,voice_id,expected_revision=...)`, and `generate(principal,key,context)`. Contexts are trusted operator resolution snapshots, not public payloads. H2 must supply authenticated principals and a separately reviewed adapter. Existing `ingest_local` supplies the reference upload.

Prepare replaces asset metadata with verified managed metadata and derives the future audition ID from principal/key/request. Core admits prepare and clone speech using existing jobs, idempotency, attempt, deadline, cancellation and publication machinery. Input IDs and owned exact asset metadata are checked consistently at submission, result validation and the publication barrier. No second scheduler or success-row shortcut exists.

Both synthetic worker implementations emit exact prepare candidate configuration/settings/reference text/identity from the frozen plan, with the planned audition asset and explicit synthetic provenance. Outputs remain fixed 4,800-frame 48 kHz mono PCM16 silence. Existing contract publication validation rejects mismatched candidates, attempts, assets and frozen inputs. Synthetic identity/provenance is fixture evidence only, never real Qwen/Pocket inference or cloning quality.

Results require the owning principal, successful prepare job, committed publication, matching artifact/manifest identity and retrievable published audio. Save selects only a published candidate ID, derives a stable immutable version ID and delegates revision/idempotency/hash checks to VoiceStore.register. Generation resolves the principal's saved immutable version and managed references; the contract freezes exact text, settings and engine/model/bundle identity.

## Checks

Executed only the prescribed interpreter and isolated unittest discovery (no pytest collection/conftest execution):

- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_production.py -v`: 4 tests PASS.
- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/worker -p test_production_synthetic.py -v`: 1 test PASS.
- `git diff --check`: PASS before source commit.

Service tests use real temporary Core SQLite/files and own historical-local synthetic subprocesses: upload -> prepare -> publication -> explicit candidate save -> restart -> clone generation; exact authored text freezes; durable duplicate prepare/save; principal rejection; unpublished/missing candidate; revision conflict; idempotency conflict; invalid settings hash; missing audition file; stale manifest/artifact binding; legacy preset speech. Worker test validates candidate schema and exact settings/configuration plus caller-mutation isolation. One added assertion initially used a nonexistent schema name (`candidate`); corrected to existing `voice_candidate`, then all checks passed.

## Evidence limits and next owner

Linux isolated runtime was NOT invoked on Mac. Its shared proposal builder is directly tested and its module imported; authenticated isolated transport/runtime execution remains unqualified here. Existing broad core/worker pytest tests were not collected or run. No full suite, maintenance guard, Docker, network, model/ML, hardware, HELD operation, delegation, adoption or historical cleanup occurred. Only normal own temporary fixtures were cleaned.

This implements the commissioned reference-to-immutable-voice-to-generation slice, not the whole H1 plan's remote acquisition, design, script/take, stitch/export catalogue. No HTTP/client/dispatcher activation. Next owner: independent Sol review of the exact source commit, followed by parent integration/interface acceptance. Report is committed separately and copied verbatim as the sole root deliverable.
