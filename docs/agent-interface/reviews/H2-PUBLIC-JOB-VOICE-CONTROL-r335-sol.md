# Public job control and basic voice management — r335 Sol

2026-10-25. Implementation candidate for r334 delivery 1; **pending fresh independent Sol review, no self-adoption or release acceptance**. Public HTTP/CLI/MCP now expose principal-scoped job pages, cursor events, bounded wait, explicit non-native retry, alias replacement, archive and configured profile detail. Installed three-family release remains NO-GO; Metal **UNQUALIFIED** and actual hardware work unrun.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-c7e09191`, branch `t3/public-job-voice-control`, clean starting commit `b9f2b6b53e5efb7c8919ecd4e97cffcb732fd5a9`. Read full r334 milestone, PRODUCT, relevant implementation-plan coverage, Core submission/events/jobs, Dispatcher/admission/selection, VoiceStore and public adapters/contracts. No applicable AGENTS.md found in worktree or checked ancestors. Memory used for evidence-boundary conventions only.

Only source/tests and this report changed in this worktree. No root-repository writes, merge, push, delegation, installation, download, network/listener, Docker, runtime child, ML, crash work, HELD/STT work or operational cleanup. Disposable small filesystem/SQLite and fabricated runtime observations only. No heavy run or 6 GiB gate waiver. Temporary test trees are fixture-owned; this is not an operational cleanup policy.

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, verified **CPython 3.14.7**, binary SHA256 `b08b055365df6dcd3fff0b805b688428f0f6a01839303f22c0716780155c480d`. No Python 3.11 or pytest invocation. Read both `tests/conftest.py` and `tests/service/conftest.py`; direct unittest avoids their listener/subprocess fixture and maintenance harness.

## Behavior and finite surface

- `jobs_list`: `GET /v1/jobs?cursor=&limit=20`, CLI `jobs list`, MCP `speech_jobs_list`.
- `job_events`: `GET /v1/jobs/{job_id}/events?cursor=&limit=20`, CLI `job events ID`, MCP `speech_job_events`.
- `job_wait`: `GET /v1/jobs/{job_id}/wait?cursor=&wait_seconds=5`, CLI `job wait ID`, MCP `speech_job_wait`. Integer wait bound 0–5 seconds. Returns when events exist, the job is terminal, or the deadline expires; response carries job/events, resumable `next_cursor` and `deadline_reached`. Each read releases the owner; asynchronous sleeping occurs outside it. Disconnect/cancellation of the waiter has no job mutation. Calls still incur ordinary owner/transport overhead; this is not a hard real-time guarantee.
- `job_retry`: existing named POST route, CLI `job retry ID --input-sha256 HASH --idempotency-key KEY`, MCP `speech_job_retry`. Requires owned source and supplied original input hash. A new job retains exact serialized context, input hash and durable/public `retry_of` linkage. Failed/cancelled, fully reaped, acknowledged history only (or never-started cancelled history); unresolved dispatch/publication histories refuse before submission. Current resolver/selection, exact execution/identity/profile/settings, Core input validation and current admission remain required. Duplicate lookup binds source ID before mutable authority/capacity checks. Core retry fingerprints now include source linkage, so an ordinary submission key or another retry source cannot silently replay a different job. Historical rows are not rewritten.
- Native runtime/history retry refuses `unsupported_control` before allocation, is absent from capability operations and reports `unavailable_lifecycle` in operation detail. No native history rule/cap/trust check was weakened. Thin CLI/MCP retain the named contract and propagate the refusal. Non-native retry still depends on the configured resolver and current admission; installing a new retry lifecycle is outside this change.
- `voice_aliases` and `voice_archive`: existing VoiceStore CAS/idempotency, no replacement business state. POST named routes; CLI `voice aliases|archive --request JSON --idempotency-key KEY`; MCP `speech_voice_aliases|speech_voice_archive`. Return canonical voice metadata. Archive preserves immutable version reads and generation by historical version ID. Alias replacement reserves canonical/archived names and checks stale revision/conflicts.
- `profile_get`: `GET /v1/profiles/{profile_id}`, CLI `profile get ID`, MCP `speech_profile_get`; the authenticated `/skeleton/profiles/{profile_id}` path remains a thin compatibility alias.

Jobs/events reuse Core's global monotonic row sequences. Cursor syntax includes principal/kind/job binding plus canonical positive sequence; validation additionally requires the exact referenced owned row. Foreign, malformed, noncanonical, wrong-job and nonexistent/gap positions refuse. Binding is a deterministic scope discriminator, not a secret bearer token: authentication/ownership always decide access. Pages are append-stable, not frozen snapshots of mutable job state. Full pages may return a cursor followed by an empty final page. Event cursors remain resumable on an empty tail. Existing 12 KiB response budget applies; use smaller job pages if a full page exceeds it.

Finite canonical catalogue and reachable schemas include five added operations; request fixtures cover every catalogue operation. Existing HTTP transport and MCP named allowlist consume the shared production list, with no new client state or generic command bus. No scheduler or private DB. Generator output filename-plus-byte equality includes hand-authored files preserved beside generated fixtures. Legacy controls/digest ledger was deliberately untouched under the separately commissioned lane; its older coverage table is not rewritten here.

## Validation receipts

All observations are SOURCE/inert: actual ASGI app, owner thread, Core/SQLite, managed tiny WAV bytes and fabricated process/reap receipts; no process containment, TLS deployment, installed execution or acoustic claim.

Direct runner pattern (explicit root and test paths; no conftest loading):

```python
import sys, unittest
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path('tests/service').resolve()),
               str(Path('tests/clients').resolve())]
names = (
    'test_public_job_control', 'test_production_clients', 'test_production_api',
    'test_voice_store.VoiceStoreTests.test_serialized_revision_race_and_alias_collision_rollback',
    'test_voice_store.VoiceStoreTests.test_archive_preserves_old_references_and_reserves_names',
    'test_voice_store.VoiceStoreTests.test_contract_fixtures',
)
r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(n) for n in names))
raise SystemExit(not r.wasSuccessful())
```

Executed via `/private/tmp/pab-0c-venv/bin/python -I -B -` from this worktree.

- Development selection: 6 tests, 5 PASS / 1 ERROR. Retry test omitted the already-required `input_sha256`; corrected test and CLI, and enforced hash equality. Focused retry rerun: 1 PASS.
- Final meaningful selection above, run once: **15 tests, 14 PASS / 1 FAIL**, 9.206 s. Failure was the new determinism test's empty-directory setup: generator intentionally preserves the hand-authored `voice-catalogue.json` rather than generating it. Corrected setup seeds the entire existing fixture filename/byte map before generation. Focused failed-test rerun: **1 PASS**, 0.747 s. No unchanged broad repeat.
- Subsequent static inspection found path/query key override ambiguity. Rejected repeated path fields; focused `test_public_job_control.PublicJobControlTests.test_http_query_contract`: **1 PASS**, 0.170 s. No further implementation change after this receipt.
- `git diff --check`: PASS.

Coverage: preparation → disconnected submit return → list/events/wait → cancellation/reap → retry → successful result; preparation/save/generation/artifact retrieval through HTTP/CLI/MCP; alias replay/conflict/stale revision/changed key and archive, immutable archived version reuse; owner reopen and the same artifact/version/voice/events/retry records through public adapters. Cross-principal reads/mutations deny; malformed/gap/wrong-job cursors deny; concurrent append preserves page order; deadline/disconnect waits leave work live; pending wait releases owner for cancellation. Native refusal advertises unavailable and adds no jobs; admission refusal adds no jobs. Shared contract request and exchange consumers are checked. Profile parity uses configured metadata supplied by an inert patch, with no hardware discovery claim.

## Limits and parent handoff

Fresh independent review must assess this candidate before adoption. Actual Linux/native runtime retry/cancel/reap, restart/trust renewal, hardware selection, deployment, resource/performance and human listening remain unrun. The retry receipt here exercises cancelled synthetic work, not actual failed Linux execution. Unknown/unacknowledged histories have explicit source fences; this receipt is not exhaustive fault-injection clearance. Source changes do not qualify CPU/CUDA/Metal or close installed release rows.

No voice exchange, acquisition, scripts/takes, topology, UI, STT or legacy provenance reconciliation was added. `lib/tts.py` SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; `contracts/v1/legacy-controls.json` remains `bcd6036992223e50a20599901733f707167a54a9db152f808b85b43602176834`. Their separate mismatch lane remains parent-owned.

## Candidate file SHA256 pins

```text
a61409633db3140490edb50e57868061e71887e0f5dd8691136fabc1b898750c  clients/cli.py
cae31795a6e66b10b654e2db34167dfb1b96ebcc1f975a3cbedce1621141d11d  contracts/fixtures/v1/operation-requests.json
09da669fd2794ec1a40d88d2439ecadf508cc57b33b204fc9ec82fc56fb678af  contracts/v1/operations.json
ad56a9601bb003307fda2a063af10f998cb967b0ad7f9f9aaff902fdd7024830  contracts/v1/schema.json
f6aa53b915a4c4278527b8b705b4eb1ecc9c12da6d8c91885456fd81104a5d61  service/api.py
f5891530e1518c8d338f5954bba9002f7f66bcacd9581402fe3dfa833ab91391  service/catalogue.py
81cbae6983280775f5bfb43ad4210eeec2e3f27f1af487e8db1ba99c68c4eb5c  service/core.py
853e2b7e23233049179ca99395c80443ac571bdc3435bf9789025d2441a534c1  service/dispatcher.py
0976a1172cbf22a66d4f2d68f3c9b93479d90270c1f7211a2d8c1a21977ad8ee  service/domain.py
fcb58e0e300d91d4fdc008a634c3f35b468ec3beb58399968732dbf1a4e64707  service/job_control.py
057d6f5d9104f47d1aabb78398bc7bb3e09e86e6d6704a8aba2a80f4eb87e088  tests/clients/test_production_clients.py
20a47f6f0a369bdf456fedae24c96646e408b11a8d104441542b225d03575ab6  tests/service/test_public_job_control.py
```
