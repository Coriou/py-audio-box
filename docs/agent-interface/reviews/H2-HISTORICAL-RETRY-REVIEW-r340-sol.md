# Independent adversarial public-control / historical-retry review — r340 Sol

2026-10-25. **Bounded GO for the reviewed r335+r339 delivery-1 SOURCE slice. No blocking finding established; r338 F1/P2 is closed at SOURCE/inert level.** Parent integration/adoption remains parent-owned. This report performs no adoption, merge or release acceptance. Installed release remains **NO-GO**, Metal **UNQUALIFIED**.

## Authority and verified pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-7d6f09f5`, branch `t3/historical-retry-replay`, HEAD `7fd0aa0c1c637d3e4bfd7d2c1b09c8c3cd27603a`; parent/cherry-pick resolution `24659ea47ee6c30e9d60634038d5a73f2675f3c2`; supplied base and verified merge-base `9f3c44f7b1a530602a6eb59c75e1b470b5c5505e`. Candidate status was empty before and after execution; `git diff --check` passed. Base-to-HEAD scope: 15 files, 1010 insertions/36 deletions. No applicable AGENTS.md found in candidate discovery or checked ancestors.

Read full r338 (including exact historical-row reproduction), r339, r335, r334 including delivery 1, and PRODUCT. Inspected actual changes and affected Core, Dispatcher, Production, selection, auth, VoiceStore, HTTP/CLI/MCP, catalogue/schema/fixture generator consumers and selected tests. Memory skill supplied scope/evidence procedure only; technical conclusions use current candidate source and independent execution.

Sole lasting write: this root report. No candidate edits, commits, merge, push, delegation, operations, listener/network, Docker, install/download, actual child execution, ML, crash, HELD/STT work or operational cleanup. Only the authorized tiny fixture-owned filesystem/SQLite, in-process ASGI and fabricated process observations were executed. No heavy work or 6 GiB gate waiver.

Verified SHA256:

```text
546719cabacf65de2aeb7064022846418022ffc38be3e7a1296ddae8f0bc69ad  service/core.py
c93d9f02eac62d7f5b6bec0994a15e5b6df12c91b20b7d458555b619f65d3ce1  service/dispatcher.py
d5cf6879e628e338eb2966c088d4bd526a812469defcd730e393d9603aff5173  tests/service/test_historical_retry.py
fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528  lib/tts.py
716654429e5c77db6dd2ea0a6c5e62c75c811ae19682154168d57cd86783e947  contracts/v1/legacy-controls.json
b9a05f0ab8f4e514c36f91ea9112bdb98bfb07f24b19743b1b82b207a56b5a94  tests/contracts/test_legacy_controls.py
```

Those three legacy files have no base-to-HEAD diff. Parent legacy reconciliation remains CLOSED; historical r334/r335 RED/mismatch prose is superseded task context and does not reopen that lane.

## Historical retry and adversarial assessment

The pre-r335 implementation independently inspected at the supplied base fingerprints `plan['request']` alone. r338's exact historical representation is therefore legitimate: request-only fingerprint, SQL `retry_of` retained, optional serialized job `retry_of` absent. The selected historical test reconstructs precisely this representation, closes/reopens Core, replays through `submit_execution`, and compares complete jobs/events SQL tuples. It now returns the same historical job without rewriting rows or allocating another execution.

`Core.execution_duplicate` scopes SQL lookup by principal/operation/key, requires exact SQL lineage, checks serialized principal/operation/key and optional lineage, and permits only the source-bearing fingerprint or legacy request fingerprint. For retry replay it additionally requires an owned same-operation source with matching serialized source identity, identical canonical encodings of submitted/stored/source contexts, and retry/source input hashes equal to the validated canonical plan hash. The fallback does not infer lineage from a request digest. The source-bearing and historical digest cases both undergo these checks.

Canonical encoding uses finite, sorted, compact JSON rather than Python dictionary equality, preserving integer/float/boolean distinctions and exact strings. Although the final stored-request comparison uses dictionary equality, the submitted request fingerprint and full canonical-context equality already bind its primitive representation. There is no request-only escape around context/hash checks for retry rows. Ordinary submission with a retry key fails the SQL-lineage comparison, including for a historical request-only digest; another source fails even when frozen contexts are identical. Changed request, canonical numeric context, missing/changed lineage, altered stored context and altered source hash refuse without allocation in the selected regressions.

Both optimistic lookup and the second lookup inside Core's `BEGIN IMMEDIATE` transaction use the same method. The deterministic race regression hides only the first lookup and exercises the real transactional lookup for replay and collisions. This establishes branch consistency; it is not multi-process or competing-owner qualification. Core's existing one-owner persistence model is unchanged.

Public `production_retry`, ordinary production duplicate lookup and the internal synthetic fallback now reuse the check. The fallback uses the actual fixed synthetic fixture request/context and requested source, replacing the former unconditional principal/key return. Ordinary request duplicates deliberately remain request-based and return before resolution/ranking/readiness/admission; this preserves the previously selected job/artifact when current ranking or authority changes. A retry duplicate requires the original frozen context, also before mutable readiness/resolution. Replaying an existing execution requires no new current admission because it creates no execution. Native runtime/frozen-native refusal still precedes public retry lookup, consistently with the explicitly unavailable lifecycle.

Fresh public retries retain terminal/confirmed-reap checks, readiness, unknown/unacknowledged dispatch refusal, publication-history refusal, operator reconciliation, current resolver/selection checks for execution/identity/profile/settings, original dispatch settings, and Core's frozen plan/media/input-hash and transactional admission/reservation fences. Current selection cannot replace the submitted historical context: changed authority refuses, while a duplicate returns its previously committed job. No native history, boot/trust, cap or success-only rule was weakened. No new scheduler or publication path was added.

## Whole delivery-1 surface

Jobs/events retain principal ownership, ordered Core sequences and exact referenced-row validation for cursors. Principal/kind/job binding plus canonical positive ASCII sequence rejects malformed, foreign, wrong-job and gap cursors. Pages are append-ordered, not snapshots of mutable job states; full pages may lead to an empty final page. Empty event tails retain the cursor. Existing 12 KiB production response bounds remain enforced.

Wait performs detached owner reads and asynchronous sleeps outside ownership, uses a monotonic bounded 0–5 second deadline, and returns events/terminal state/deadline with a resumable cursor. Disconnect introduces no job mutation. The selected pending-wait/cancel test exercises owner release; no socket-level disconnect or hard real-time guarantee is established.

Authentication requires read scope globally and production-write additionally for mutations. Named read routes reject duplicate queries, path overrides and extra fields. CLI and MCP use the shared finite catalogue and canonical HTTP contracts. Profile detail has canonical and compatibility routes plus CLI/MCP parity. Configured profile fixtures establish no hardware discovery qualification.

Alias/archive adapters reuse VoiceStore transactions, ownership, CAS and idempotency. Canonical/archived names remain reserved; historical immutable versions and generated artifacts survive archive/reopen. The selected journey covers HTTP/CLI/MCP IDs, cancellation/reap using fabricated observations, retry, artifact retrieval and reopened durable records. Native retry is absent from advertised capabilities and reports unavailable lifecycle; invoking it refuses without new jobs.

Closed request/response schemas, optional job lineage, catalogue coverage, indirect request/exchange/reference-role/binary consumers and complete seeded generator filename-to-byte identity pass. Public raw upload bytes remain separate from fixture-only binary decoding. No legacy-contract regeneration was executed.

## Independent exact execution receipt

Read `tests/conftest.py` and `tests/service/conftest.py` before execution. Direct unittest avoids their autouse maintenance harness and listener/subprocess fixtures. Verified `/private/tmp/pab-0c-venv/bin/python`: **CPython 3.14.7**, executable SHA256 `b08b055365df6dcd3fff0b805b688428f0f6a01839303f22c0716780155c480d`. No Python 3.11 or pytest runner.

**Exact r339 28-test selection independently executed once: 28 PASS, 9.477 seconds, exit 0.** No test rerun, extra probe or broad suite. Invocation from the pinned candidate: `/private/tmp/pab-0c-venv/bin/python -I -B -`, with this exact stdin body:

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

## Remaining parent-owned limits

Parent may assess integration of this bounded SOURCE candidate while retaining its adopted legacy reconciliation. This independent report performs no source adoption. Fresh integrated changes require their own assessment.

All execution evidence is SOURCE/inert. Reconstructed historical rows establish shared-API compatibility, not installed upgrade/restart qualification. Current-authority/ranking changes and unknown-history fences were source-inspected rather than exhaustively fault-injected; the transactional race test simulates visibility rather than actual concurrency. No actual child cancellation/reap, native failure/reboot/trust renewal, hardware selection qualification, deployment, resource/performance, acoustic or human listening evidence was obtained. CPU/CUDA/Metal and installed release gates remain open; **Metal UNQUALIFIED, release NO-GO unchanged**. No CLOSED81, retained resource or operational authority is inferred.
