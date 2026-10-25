# Independent native retained restart review — r310 Sol

2026-10-25. **Bounded NO-GO for adoption of r309 retained-history SOURCE.** Five independently reproduced acceptance gaps remain, including negative durable resource charges. The final authorized existing-test selection passed **40/40 in 21.509s**, once, exit 0; those tests do not cover these gaps. No self-adoption or candidate correction.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-6ebb26e1`, branch `t3/native-owner-restart`. Verified clean at entry and after all execution, HEAD `8f875ae686da6daf0f30570c7066b28404182d65`.

Exact chain:

- Base `0375aae7437536156b2aa4f1a5bfeb04cb7f53da`.
- Initial SOURCE `9409178dfb4ee4fec25f679a3f4db82145e9f4a1`.
- Final SOURCE `ee682f633ab81f091d3d5f2d2082cce8050f2793`.
- Report HEAD `8f875ae686da6daf0f30570c7066b28404182d65`; only the r309 report differs from final SOURCE.

Verified SHA256: native_activation.py `9c13dc03a6818d3d0590880f98431fd04a9fe3f2e7a074c6c676c5b1cdafa673`; core.py `13ef9dca789b82b68d3798af76fdfe3eed07c285b05654af28d11bef35a4a5ee`; native_package.py `879106b73fe379de93e5f069fc4c56648ac8cb70c2b3d5d396134eb37dbed152`; unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known legacy digest RED remains unmodified/unwaived; no digest regeneration/test.

Read PRODUCT.md, NATIVE-PACKAGE.md and relevant normative r277, r296/r303/r305/r306/r307/r308/r309. Inspected actual base-to-final diff, Core/Dispatcher/entrypoint, native preparation/runtime, Authority, detached schema/document helpers, admission/resource storage, publication intent and reconciliation consumers, tests and conftests. No applicable AGENTS.md found in candidate or checked ancestors. Memory supplied evidence-boundary conventions only; technical findings use this pinned candidate. Applied the memory evidence-gate skill's scope/receipt guidance; user authority explicitly permits these inert probes and finite tests.

Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or operational cleanup.

## Findings

### F1 — P1: malformed negative reservations pass the native proof and resource gate

`service/native_activation.py:164–166` checks only retained/category, configuration digest and `disk_bytes >= attempt_disk_bytes`. SQL DDL does not constrain the demand JSON. Reconciliation checks its keys but skips acknowledged dispatch allocation validation. ResourceStore.rows simply decodes it; reopening does not run reserve_tx's strict numeric/positive validation.

Independent mutation of an otherwise completed fixture: preserve configuration and vram_bytes, set slots, ram_bytes, pids, disk_bytes and attempt_disk_bytes all to -1. Reopen **ACCEPTED**, installed_checked=True, quarantined=False, resource_fault=False. This satisfies -1 >= -1. Existing ResourceStore._capacity/check_disk sums retained disk_bytes, so accepted negative charges reduce accounting for later admission. This is a retained-history correctness defect within the advertised scope, not a request for new hardware qualification.

Require closed exact primitive types, finite bounds and positive mandatory dimensions, and corroborate the retained demand against the frozen execution/profile reservation rules before clearing import closure. Do not re-reserve or repair history.

### F2 — P2: retry lineage is not checked

`service/native_activation.py:149–158` requires one attempt but never checks `stored['retry_of']`. `_installed_detached_rows` enforces foreign keys, which permit a self-reference. `_installed_detached_documents` does not validate retry lineage.

Independent mutation `UPDATE jobs SET retry_of=id` **ACCEPTED**, installed_checked=True, quarantined=False, extra ACKs=0. This impossible self-retry survives even though Core.submit_execution only allows retry of a failed/cancelled predecessor and the advertised native slice refuses retried histories. Require retry_of IS NULL for this finite slice; do not infer it from attempt count.

### F3 — P2: the persisted request fingerprint is not corroborated

The same native job loop and shared document validator bind id/principal/operation/key, but omit `jobs.fingerprint`.

Independent mutation `UPDATE jobs SET fingerprint='not-the-request-fingerprint'` **ACCEPTED**, installed_checked=True, quarantined=False, extra ACKs=0. Dispatcher._production_command and Core.submit_execution subsequently trust this column for duplicate-request identity. The original identical request will conflict despite startup declaring the history checked. Require the exact canonical fingerprint derived from principal, operation and the retained frozen plan request.

### F4 — P2: unknown command operations are accepted as discharged obligations

`service/native_activation.py:141` tests only the state column. The canonical SQL schema leaves op/state as arbitrary TEXT; shared detached document validation and reconciliation do not close this command namespace.

Independent insertion:

```sql
INSERT INTO dispatch_commands(dispatch_id,op,state)
SELECT seq,'unrecognized-operation','returned' FROM dispatches;
```

Reopen **ACCEPTED**, installed_checked=True, quarantined=False, extra ACKs=0. A returned label alone cannot establish the semantics of an unknown command. Validate the finite supported operation/state domain and its dispatch relationship before accepting completed history. Unknown obligations must refuse without deletion or ACK.

### F5 — P2: frozen joins use Python numeric equality instead of exact retained identity

`service/native_activation.py:153,157–158` uses dictionary/scalar equality for context, frozen job and outcome. Unlike canonical bytes, this treats integer 0 and float 0.0 as equal. JSON Schema integer validation accepts integral floats, and reinserting TEXT documents into detached SQLite does not inspect their nested primitive representation.

Independent mutation: decode only jobs.job, replace job['generation'] with float(job['generation']), update jobs.job; leave frozen Authority, terminal receipt, publication and dispatch outcome unchanged. Reopen **ACCEPTED**, installed_checked=True, quarantined=False, resource_fault=False. Thus the job no longer has the exact frozen representation or exact retained outcome, yet the equality checks approve it. This repeats the primitive-identity class addressed for enrollment by r305, now in new history joins.

Compare canonical bytes for immutable joins after explicit normalization of only the intended mutable completion fields. Add numeric type-substitution regressions at those joins; do not rewrite accepted history.

## Independent reproduction method

Two finite stdin probe invocations used `/private/tmp/pab-0c-venv/bin/python -I -B -`, asserted Python 3.14.7, and explicitly prepended the absolute candidate and candidate/tests/service paths. No probe code was written to disk. Each mutation used a separate NativeHistoryTests fixture:

```python
t = NativeHistoryTests(); t.setUp()
try:
    t.prepare()                 # actual tiny Core/publication, fake TLS/process/backend
    c = Core(t.root)
    try:
        # Apply exactly one mutation described above using c.db.
        ...
    finally:
        c.close()
    try:
        d = t.reopen()
        print(d.installed_checked, d.call('status')['quarantined'])
    finally:
        t.close()
finally:
    t.doCleanups()
```

First invocation: F2/F3/F4, each ACCEPTED with no extra ACK (session 33565, output chunk 06beda). Second invocation: F1/F5, each ACCEPTED, resource_fault=False (session 38887, output chunk 21e1a0). Both exited 0. These are successful reproductions of invalid acceptance, not passing correctness assertions. No rerun/tuning of any probe, actual child or ML execution. Temporary fixture-owned teardown only.

## Positive source assessment and limits

The final ordering correction is sound in the inspected path: Core acquires flock, opens its sole SQLite connection, invokes the exact-type NativeActivation private hook, then configures WAL/initializes schemas. Missing catalogue cannot be concealed by normal initialization. The selected regression confirms missing catalogue tables stay missing. Fresh/retained state mismatch catches competing initialization in the precheck/acquisition gap. Existing Core process ownership and lifetime flock remain authoritative.

Native runtime construction and SQLite validation stay on the Dispatcher control writer. Detached Authority enters synchronous authenticated collect; no Core/SQLite callbacks enter TLS, and no SQL transaction spans collect. Schema identity compares canonical DDL tokens including indices/triggers before interpreting table data. SQL rows/scalar bytes are checked against 2048/16 MiB before data materialization; schema reading itself is fetchall and aggregate counts scan the input, so these are retained-materialization limits, not an adversarial database I/O-time bound. The immutable dedicated-account/no hostile same-UID writer premise remains material; repeated snapshots do not provide hostile-writer/ABA isolation.

The reader correctly uses native terminal_receipt and current authenticated collect, exact retained receipt/proposal/output metadata, publication oracle, actual asset hashes/staging bytes, durable publication intent, exact spool membership and import intent/completion. Default preparation refusal remains intact. In-memory import closure is cleared only after the implemented proofs; F1–F5 show those proofs are incomplete. No new history repair/deletion/ACK, scheduler, boot learning or readiness promotion was found.

recovery.py, reconciliation.py and native_runtime.py are unchanged relative to base. Linux native-history refusal remains explicit; native receipt substitution regression passed. No broad Linux acceptance route was repurposed. The findings concern the new native acceptance policy, not deferred public lifespan/topology/onboarding/hardware work.

## Once-final finite test receipt

Inspected tests/conftest.py and tests/service/conftest.py first. Direct unittest avoids pytest maintenance autouse and socket/uvicorn fixtures. Inspected fixture seams: tiny filesystem/audio/ZIP/model metadata, injected certificate/TLS/process/reap/backend observations; archive check is in-process.

One final selection, candidate cwd, `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute candidate and candidate/tests/service sys.path entries. Printed and asserted interpreter **Python 3.14.7**, Clang 17.0.0, not target Python 3.11. defaultTestLoader.loadTestsFromNames, count asserted 40, TextTestRunner(verbosity=2, failfast=True), nonzero exit on failure.

Exact selection:

- tests.release.test_native_history.NativeHistoryTests — all 16.
- tests.release.test_native_activation.NativeActivationTests — all 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — all 10.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.service.test_installed_activation.ActivationTests.test_preowner_refusal_never_constructs_core.
- Same ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- Same ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core.
- Same ActivationTests.test_changed_post_acquire_second_hello_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.

Added preowner and second post-acquisition identity checks because shared Core startup ordering changed. **Ran 40 tests in 21.509s; OK; exit 0.** Session 41801, chunks 01fade/3e0530/9aaf9e preserve verbose receipts. No second final selection. `git diff --check base..HEAD` passed; candidate clean at exact report HEAD afterward.

Observed free disk 3,004,832 KiB, below 6 GiB. Only expressly authorized tiny inert work occurred. No network/listener, Docker, installation/download, actual child, ML/crash, HELD/STT or operational cleanup; heavy ACTUAL admission was not exercised or waived.

## Parent-owned gates

Correct F1–F5 in a new pinned candidate, add focused refusal regressions preserving evidence/no ACK, then obtain fresh independent source review before adoption. Existing 40 PASS does not waive these reproductions.

Public authenticated installed lifespan/credentials, Docker/native topology/mounts, fresh owned voice onboarding and stopped boot/trust renewal remain separate work and are not current blockers asserted by this review. No reviewed arm64 wheelhouse exists. Actual Darwin dependency/install/import/ABI, real TLS/topology/denied peers, process/FD/death/resource, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio/performance and release qualification remain parent-owned and unrun. **Native Metal remains UNAVAILABLE / UNQUALIFIED.** CPU and CUDA release rows remain mandatory; legacy lib/tts digest RED remains unwaived. This SOURCE/inert review supplies no operational, hardware or release acceptance.
