# Shared Core historical retry replay repair — r339 Sol

2026-10-25. Implementation candidate for r338 F1/P2; pending fresh independent review. No self-adoption. Historical retry rows now replay through the shared submission entry point after reopen, with durable source binding. Installed release remains NO-GO; Metal UNQUALIFIED.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-7d6f09f5`, branch `t3/historical-retry-replay`. Clean starting parent `9f3c44f7b1a530602a6eb59c75e1b470b5c5505e`, containing adopted legacy reconciliation. Cherry-picked requested unmerged r335 `cf53ee2aaa2ac4113ed803fe8df45ffcb0c815bf` as `24659ea` (full ID obtainable from this commit's parent). Read full root-relative r338, r335, r334 including delivery 1; inspected both Core duplicate checks, Dispatcher production_retry, ordinary production lookup and internal synthetic submit fallback. No AGENTS.md found in worktree/adjacent discovery.

No root-repository writes, merge, push, delegation, network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup. Tiny fixture-owned filesystem/SQLite, in-process ASGI and fabricated process observations only; no heavy work or 6 GiB waiver. Memory registry search had no relevant hit; technical conclusions use current source/reports.

Parent `lib/tts.py`, `contracts/v1/legacy-controls.json` and `tests/contracts/test_legacy_controls.py` have no diff from starting parent. Legacy reconciliation remains CLOSED; the older r335/r334 historical ledger prose is not current authority and was not rewritten.

## Repair

`Core.execution_duplicate` performs the principal/operation/key lookup and checks SQL `retry_of` for ordinary as well as retry submissions. A retry may match either the new source-bearing digest or the historical request digest only when durable lineage equals the requested source. Serialized principal/operation/key and optional lineage must agree. The source SQL row must have the same principal/operation; its serialized identity must agree. Submitted, stored retry and source contexts must have identical canonical encodings; stored retry and source input hashes must equal the canonical plan hash. Missing optional historical job lineage is accepted via SQL lineage, never inferred from a request hash.

Both optimistic and transactional duplicate checks call this same method. New allocations retain the r335 source-bearing fingerprint; no historical fingerprint, job bytes, context bytes, schema or source lineage is rewritten. Ordinary/retry collisions, other-source collisions, changed requests, canonical numeric context differences, missing/changed lineage, mismatched stored context and altered source input hashes refuse without allocation.

Public `production_retry` now uses this shared check rather than bypassing Core with a lineage-only query. The ordinary public duplicate lookup also uses it, preventing an old retry request digest from replaying as an ordinary submission. The internal synthetic submit duplicate fallback likewise joins the exact fixed context and requested source. Ordinary request duplicates retain request-fingerprint semantics before mutable authority resolution.

Fresh retry admission, readiness, unreaped/unknown/acknowledgement/publication/operator checks, frozen source validation, native refusal, CAS, cursors, wait and authentication fences are unchanged. No native lifecycle enabled.

## Regression evidence and exact final receipt

Read `tests/conftest.py` and `tests/service/conftest.py`; direct unittest does not invoke their maintenance/listener/subprocess fixtures. Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B -`: CPython **3.14.7**, executable SHA256 `b08b055365df6dcd3fff0b805b688428f0f6a01839303f22c0716780155c480d`. No Python 3.11 or pytest runner.

Final meaningful selection executed **once after final source/test changes: 28 PASS, 9.601 s, exit 0**. No development test execution or repeated final selection. `git diff --check` PASS.

Exact runner body (stdin to that interpreter, from own worktree):

```python
import sys, unittest, tempfile, json
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path('tests/service').resolve()),
               str(Path('tests/clients').resolve()), str(Path('tests/contracts').resolve())]
names = ('test_public_job_control', 'test_production_clients', 'test_production_api',
 'test_voice_store.VoiceStoreTests.test_serialized_revision_race_and_alias_collision_rollback',
 'test_voice_store.VoiceStoreTests.test_archive_preserves_old_references_and_reserves_names',
 'test_voice_store.VoiceStoreTests.test_contract_fixtures',
 'test_historical_retry', 'test_production_admission',
 'test_reference_roles.ReferenceRolesTests.test_generated_fixtures_and_all_wire_consumers')
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
import test_core, test_contracts
from service.core import Core
for name in ('test_persisted_idempotency_conflict_scope',
             'test_retry_canonical_numeric_context_rejected_unchanged',
             'test_retry_checks_prior_stored_plan_hash'):
    def run(name=name):
        with tempfile.TemporaryDirectory() as tmp:
            core = Core(Path(tmp)/'core', historical_local=True)
            try:
                context = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
                getattr(test_core, name)(core, context)
            finally: core.close()
    suite.addTest(unittest.FunctionTestCase(run, description=name))
for name in ('test_schema_and_all_fixtures',
             'test_operation_requests_jobs_artifacts_and_error_mapping',
             'test_binary_fixture_boundary'):
    suite.addTest(unittest.FunctionTestCase(getattr(test_contracts, name)))
r = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not r.wasSuccessful())
```

This retains original 15 public HTTP/CLI/MCP/new-retry/voice journeys; adds three historical regression cases, three inert shared-Core follow-ups, three admission tests and four indirect contract checks. The three new cases reconstruct the exact pre-change request fingerprint plus serialized job without optional retry_of, retain SQL lineage, close/reopen Core, assert exact replay and unchanged full jobs/events SQL row tuples, and check collisions/refusals without allocation. New source-bearing digest behavior is retained. Transaction race coverage hides the row from only the first optimistic lookup, exercising the real second lookup inside the transaction for exact replay and conflicting lineage; it is a deterministic visibility simulation, not concurrent-owner/process evidence.

## Final source pins and limits

```text
546719cabacf65de2aeb7064022846418022ffc38be3e7a1296ddae8f0bc69ad  service/core.py
c93d9f02eac62d7f5b6bec0994a15e5b6df12c91b20b7d458555b619f65d3ce1  service/dispatcher.py
d5cf6879e628e338eb2966c088d4bd526a812469defcd730e393d9603aff5173  tests/service/test_historical_retry.py
fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528  lib/tts.py
716654429e5c77db6dd2ea0a6c5e62c75c811ae19682154168d57cd86783e947  contracts/v1/legacy-controls.json
b9a05f0ab8f4e514c36f91ea9112bdb98bfb07f24b19743b1b82b207a56b5a94  tests/contracts/test_legacy_controls.py
```

All receipts are SOURCE/inert. Reconstructed historical bytes establish compatibility at the shared API, not an installed upgrade qualification. No actual runtime cancellation/reap, hardware, acoustic, installed restart/trust renewal or release acceptance. CPU/CUDA/Metal qualification remains open, Metal UNQUALIFIED and release NO-GO unchanged. Fresh independent review and parent integration remain required.
