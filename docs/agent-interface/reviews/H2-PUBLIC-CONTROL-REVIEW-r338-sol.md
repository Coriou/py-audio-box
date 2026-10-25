# Independent adversarial public-control SOURCE review — r338 Sol

2026-10-25. **NO-GO for adopting r335 as-is: one P2 shared-Core historical retry compatibility regression.** The independent exact final selection is **15/15 PASS**. Public job/voice adapters otherwise support the bounded delivery-1 SOURCE slice inspected here. Parent must repair historical duplicate semantics without weakening source binding, obtain focused independent regression evidence, and preserve its already-adopted r336/r337 legacy reconciliation when integrating. No self-adoption or release acceptance.

## Authority, pins and scope

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-c7e09191`, branch `t3/public-job-voice-control`; HEAD `cf53ee2aaa2ac4113ed803fe8df45ffcb0c815bf`; supplied base and verified merge-base `b9f2b6b53e5efb7c8919ecd4e97cffcb732fd5a9`. Candidate `git status --short` empty before and after execution; `git diff --check` PASS. Actual base-to-HEAD diff: 13 files, 798 insertions/21 deletions. No applicable AGENTS.md found by candidate discovery or checked ancestors.

Read full r335, full r334 (including delivery 1), PRODUCT, relevant canonical implementation-plan obligations; inspected actual changes, shared Core/Dispatcher/Production/admission/selection, authentication, VoiceStore, HTTP/CLI/MCP, schemas/catalogue/generator and indirect fixture consumers. Memory skill supplied evidence-boundary procedure only; conclusions use candidate source and independent execution. Sole lasting write is this root report. No candidate edits, commit, merge, push, delegation, listener/network, Docker, install/download, actual child execution, ML, crash, HELD/STT work or operational cleanup. Temporary fixture-owned filesystem/SQLite and fabricated process observations only; no heavy run or 6 GiB waiver.

Parent r336/r337 reconciliation is accepted task context, not re-reviewed here. Candidate's older legacy ledger is expected. This review does not call the root digest RED, reopen reconciliation, or authorize replacing the parent's ledger with the candidate copy.

## Finding F1 — P2: old retry keys cease to replay in shared Core

**Location:** `service/core.py:368–384` (both duplicate checks). Prior source computes every job fingerprint from `plan['request']`. Candidate changes retry fingerprints to `{request: plan['request'], retry_of: source}` but accepts only the new digest on duplicate lookup. Existing retry rows retain the old digest and durable `jobs.retry_of`; there is no version-aware compatibility path. The new optional public `job.retry_of` field is likewise absent from historical serialized jobs, as expected.

**Reproduction, no worker or operational data:** create a temporary historical-local Core using `transcript-speech_generate.json` context; submit source key `source`; cancel its never-started job and reconcile; submit retry key `retry` with that source. Restore that retry row to the exact pre-change representation: fingerprint `oracle.fingerprint('p', operation, context['plan']['request'])`, serialized job without the newly introduced `retry_of` property, retaining the existing SQL `retry_of` foreign key. Close and reopen Core. Repeat `submit_execution('p','retry',context,retry_of=source_id)` unchanged. Observed **`ContractError: idempotency_conflict`**, with two jobs retained and no new allocation. The pre-change formula was independently checked with `git show BASE:service/core.py`; fixture mutation reconstructs historical bytes, not corruption of a live owner or a claim of runtime upgrade testing.

The public `production_retry` path at `service/dispatcher.py:241–244` separately looks up SQL lineage and bypasses Core for duplicates, so the passing public replay journey does not cover or cure this shared API regression. Ordinary non-retry keys retain their previous fingerprint. New retry fingerprints correctly distinguish source identity; simply reverting the change would lose that protection. A compatibility fix should inspect durable lineage as well as the stored fingerprint, replay only the exact legacy source/request/principal/operation/key, reject ordinary/retry and other-source collisions, and leave historical rows unchanged. Both outer and transactional duplicate checks need consistent treatment. Add an explicit pre-change-row/reopen regression through the shared entry point; retain new-source/ordinary-key collision checks. This is a bounded historical idempotency failure, not evidence of duplicate execution or data loss.

## Other reviewed behavior

- Jobs/events use existing Core sequence rows and ownership. Cursor decode requires canonical positive ASCII integer within the safe range and exact principal/kind/job binding; owner verifies the exact referenced owned row, rejecting holes, foreign rows and wrong-job tokens. Pages use ordered `seq > cursor`, preserving append order; a full last page can legitimately lead to an empty page. Event empty tails preserve the input cursor. No snapshot claim about mutable job states. The existing 12 KiB production budget is enforced before return; smaller job pages are required when full detail pages exceed it.
- Wait reads detached pages through the owner then sleeps asynchronously outside it. Deadline is monotonic and bounded to integer 0–5 seconds; terminal/events/deadline return job, events and resumable cursor. Owner/transport overhead can exceed the requested wait duration; no real-time guarantee. Disconnect path has no cancellation mutation. Independent tests cover zero deadline, disconnect callback and concurrent cancel/reap while a waiter is pending; they do not prove real socket disconnect behavior.
- HTTP requires read scope globally, and mutations resolve production-write in addition. New reads use closed named query/path schemas, reject repeated query keys and path override fields. CLI and MCP route the same finite named operations through HTTP; no client business store, scheduler or generic public command bus was introduced. Direct `production_call` is an internal borrowed-owner adapter; bounded wait is implemented in the asynchronous HTTP path.
- New non-native retries bind owned source and supplied input hash, reject native runtime/frozen native context before allocation, and check source terminal/reaped status, readiness, unknown/unacknowledged dispatches, publication history and operator reconciliation before a fresh allocation. Current resolver/selection and exact execution/identity/profile/settings comparisons precede submission of the original context; Core checks frozen inputs/media/hash and current admission in the existing transaction. Public replay lookup precedes mutable authority/capacity checks and binds durable source lineage. Native capability omission and `unavailable_lifecycle` detail are tested; no native lifecycle weakening was found. Unknown/unacknowledged and changed-current-authority branches were source-inspected, not exhaustively fault-injected.
- Alias/archive adapters invoke existing VoiceStore transaction, ownership, revision CAS and idempotency semantics. Canonical names and archived names remain reserved; aliases cannot equal their canonical voice ID or collide with another voice. Archive retains immutable versions and historical generation references. Independent HTTP/CLI/MCP journey verifies alias replay, changed key payload, stale revision, conflict, archive and archived-version reuse across reopen. Configured profile detail has canonical and compatibility HTTP routes plus CLI/MCP parity; the test patches metadata and establishes no hardware discovery qualification.
- Catalogue/schema/request fixtures remain finite and closed. Response change for archive is the actual canonical voice record. Added optional job lineage preserves validation of old job fixtures. Generator check compares the complete seeded fixture filename-to-byte map, including hand-authored preserved files. Existing request/exchange/reference-role and binary-fixture consumers pass; raw-byte public upload remains distinct from fixture base64 decoding.

## Independent execution receipt

All executions from the pinned candidate, `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit `sys.path` entries only. Verified **CPython 3.14.7**, executable SHA256 `b08b055365df6dcd3fff0b805b688428f0f6a01839303f22c0716780155c480d`. Read `tests/conftest.py` and `tests/service/conftest.py`; direct unittest never invokes their maintenance/listener/subprocess fixtures. No Python 3.11 or pytest runner.

**Exact r335 final selection executed once, now, without retry: 15 PASS, 8.471 s, exit 0.** Runner:

```python
import sys, unittest
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path('tests/service').resolve()),
               str(Path('tests/clients').resolve())]
names = ('test_public_job_control', 'test_production_clients', 'test_production_api',
 'test_voice_store.VoiceStoreTests.test_serialized_revision_race_and_alias_collision_rollback',
 'test_voice_store.VoiceStoreTests.test_archive_preserves_old_references_and_reserves_names',
 'test_voice_store.VoiceStoreTests.test_contract_fixtures')
r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
 unittest.defaultTestLoader.loadTestsFromName(n) for n in names))
raise SystemExit(not r.wasSuccessful())
```

This includes corrected determinism setup and final query validation together. It does not rewrite r335's historical 14-pass/one-setup-failure receipt or its focused subsequent passes.

Narrow shared-Core/admission follow-up: **5 unittest cases PASS, 0.538 s, exit 0**. One wrapper directly calls three inspected inert existing functions, each with its own tiny Core/context: `test_persisted_idempotency_conflict_scope`, `test_retry_canonical_numeric_context_rejected_unchanged`, `test_retry_checks_prior_stored_plan_hash`. One adversarial case reconstructs/reopens the historical retry above and asserts the reproduced failure; raw observation `LEGACY_RETRY_REPLAY: idempotency_conflict`. Three `test_production_admission` cases pass (atomic duplicate/authority, reservation rollback/no allocation, missing/foreign/altered references). Thus a green probe means successful defect reproduction, not successful compatibility.

Indirect contract follow-up: **4 PASS, 0.640 s, exit 0**: direct FunctionTestCase wrappers for `test_contracts.test_schema_and_all_fixtures`, `test_operation_requests_jobs_artifacts_and_error_mapping`, `test_binary_fixture_boundary`, and `test_reference_roles.ReferenceRolesTests.test_generated_fixtures_and_all_wire_consumers`. These exercise affected catalogue/generator consumers without running legacy-ledger reconciliation or unrelated suites. No repeated final selection or broad suite.

## Parent-owned exit and limits

Repair F1 and independently verify legacy exact replay plus changed-source/ordinary collision refusal, preserving unchanged historical records and the already-passing new public journeys. Integration must retain parent r336/r337 ledger changes. No candidate repair or adoption was performed here.

All results are SOURCE/inert, including actual in-process ASGI, owner/Core/SQLite, tiny WAV files, synthetic outputs and fabricated reap/TLS/process observations where fixtures use them. No actual runtime cancellation/reap, installed upgrade/restart/trust renewal, packaging/topology, resource performance, audio quality or human listening acceptance. Native retry remains unavailable. CPU/CUDA/Metal qualification and installed three-family release gates remain open; **Metal UNQUALIFIED, release NO-GO unchanged**.
