# r346 Sol — independent adversarial r344+r345 Lot A SOURCE review

2026-10-25. **NO-GO for complete Lot A SOURCE acceptance, adoption or operator use: one reproduced blocking interruption defect.** The ordinary fixed-template two-renewal journey passes independently, but a completed intent staging write cannot resume after a rename interruption. This recommendation confers no adoption or operational authorization. Lot B remains next after Lot A repair/review; public old-voice origin derivation is outside this review's acceptance scope. Legacy reconciliation stays CLOSED, Metal UNQUALIFIED, release NO-GO.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-532a7465`:

- Base: `374855445b0a4ec8ca57de483bf4339ee7a09236`.
- r344 cherry-pick / HEAD parent: `eb2ef8bf25874ebd306d15bb4f59301dea8c57c8`.
- Reviewed source/report HEAD: `08e03baed189939cfb0f099a74256d703a65beb2`.

All resolve exactly; HEAD^^ is the base and base ancestry succeeds. Candidate status was empty before review and after execution. No applicable AGENTS.md was found in candidate discovery or checked ancestors. Read the full normative r343 contract, r344 incomplete report, r345 completion report and r341/r342, and inspected the complete changed source/test/doc/manifest set and relevant ownership, history, runtime, registration, catalogue/archive and reconciliation consumers. Technical conclusions use this source and fresh execution. Memory supplied only review/evidence-boundary conventions.

Sole lasting write is this root report, mode 0600. No candidate edits, commits, merge, push, delegation, network/listener, Docker, install/download, actual OS child, ML, host crash, HELD/STT operation or operational cleanup. Tiny fixture files/SQLite, injected TLS/process/backend observations and a fixture rename fault only. Ordinary fixture teardown is unchanged. The 6 GiB heavy gate was neither exercised nor waived. Inspected `tests/conftest.py` and `tests/service/conftest.py` before direct unittest; no pytest listener/subprocess fixtures ran. Selected tests named actual_child exercise fake channels/handles/backend observations.

`git diff --check BASE HEAD` is **not clean**: r345 report line 259 has a new blank line at EOF. This is a nonblocking documentation nit, not the reason for NO-GO; r345's stated diff-check PASS does not match this committed diff.

## Blocking finding F1 — exact intent stage cannot resume (P1)

Locations: `service/native_renewal.py:90-112`, `:152-179`, `:304-324` and `:368`.

Trigger: an ordinary `begin` writes/fsyncs the complete canonical `001.intent.stage`, then rename to `001.intent` fails. Reissue the identical begin request with unchanged predecessor, successor inputs, request ID and stopped Core evidence.

Observed: `Refused native_renewal_partial_or_conflicting`; complete stage bytes are retained and the final intent remains absent. This is not a malformed or foreign stage. The first publisher completed its write and fsync before the injected rename failure. The same control namespace also prevents normal startup. There is no supported explicit request path to complete this phase without changing retained evidence.

Cause: `namespace(resume=True)` validates but omits all `.stage` documents from its returned values. Consequently `begin` finds no replay or pending final intent and calls `selected_source`. That calls `tip` with normal `resume=False`; namespace rejects the very stage which explicit maintenance was meant to resume, before `publish` can compare and rename it. The same control flow applies to a later epoch's intent stage; the fresh dynamic reproduction below targets epoch one only.

Impact: a supported fsync/rename interruption can strand admission before host retirement. This violates r343 section 3's exact-request resume rule and section 7's phase-boundary test requirement, and contradicts the operator documentation's unconditional exact-stage resume claim. Retaining/refusing arbitrary partial bytes is correct, but cannot substitute for resuming a complete owned stage.

Required repair: recognize and validate the exact staged intent under the existing maintenance ownership, bind its request/predecessor/successor/stopped fingerprint, and explicitly resume its publication. Preserve normal-startup staging refusal, changed-request/fork refusal and evidence retention. Do not delete the stage or relax namespace validation generally. Add first-epoch and later-epoch intent rename/fsync-boundary regressions, including changed payload/request refusal, then obtain fresh bounded review. No fix was made here.

Independent reproduction, executed once using an in-memory subclass of `RenewalCompletionTests` and only `Probe('test_intent_stage_resume')` (not inherited discovery):

```python
self.prepare()
paths, config, templates = self.successor_inputs(1)
args = dict(**paths, owner_root=str(self.root), request_id='1'*32)
real = r.os.rename

def rename(a, b):
    if str(b).endswith('.intent'):
        raise OSError('injected intent rename')
    return real(a, b)

with patch.object(r.os, 'rename', side_effect=rename):
    with self.assertRaises(OSError):
        r.begin(self.record, **args)
stage = self.root/'native-renewal'/'001.intent.stage'
before = stage.read_bytes()
self.assertEqual(r.document(stage)['request_id'], '1'*32)
# Actual identical retry raised Refused native_renewal_partial_or_conflicting.
r.begin(self.record, **args)
```

The executed probe caught and printed the exception, asserted unchanged stage bytes and absent final intent, and returned **1 test, 1.136 s, OK**. Its OK means the defect was reproduced, not that renewal resumed. No cleanup/repair workaround was attempted.

## Systematic source assessment

1. **Ownership and activation authority.** `native_maintenance` reserves the same Core registry, binds root identity and takes `control.lock`; existing SQLite opens read-only/query-only under a transaction. It does not construct a second Core, initialize schema or recover/schedule. Generic Core checks the renewal namespace before schema work; native activation selects the exact committed envelope, and stale epoch-zero inputs cannot bypass pending or committed renewal. Numbered commit publication is the sole successor activation point; no SQLite update is coupled to it.
2. **Stopped host retirement.** The pure extracted journal reader operates under existing `native.lock`, with exact root/token file membership, terminal ACKed rows, matching binding/snapshot/runtime/receipt membership, one owned-process terminal receipt and retained input/output validation. Unknown records/files, launch uncertainty and missing ACKs refuse. Retirement marker precedes export; any retirement-prefixed partial marker blocks Journal construction/save and supervisor dispatch/prepare/start. No automatic ACK, kill, reap inference, repair or deletion is added.
3. **Historical/current partition.** Closed `EpochEvidence` and `RetiredEvidence` select retired exports by exact binding membership/native tuple; unsealed current jobs retain live authenticated collect. No generic caller callback or failed-live fallback was introduced. Every global job joins one dispatch/reservation; shared spool membership is exact. Duplicate retired bindings refuse, and unmatched bindings must equal the current tuple. Existing strict schema, command order, immutable Authority/outcome/ACK, success publication/pair, negative no-output/no-publication, media/reference/import and final reconciliation checks remain active.
4. **Permanent evidence versus legitimate extension.** Per-binding seals cover terminal SQL rows, events/commands, immutable selected version rows and retained asset/spool/stage hashes. The whole stopped SQL fingerprint is a transaction bracket only. Catalogue aliases/cursors and new jobs can extend the database without changing old seals. `VoiceStore.version` reads immutable versions independently of archived voice resolution; catalogue archive visibility semantics are not replaced. Existing maintenance archive consumers remain unchanged and strict; the new reconciliation branch recognizes only the finite validated renewal namespace, not arbitrary descendants.
5. **Successor and brackets.** Successor enrollment pins explicit new boot/lane/configuration/trust and the unchanged runtime manifest. Provisioned host predecessor digests carry retired export order and charges; hello must bind that exact digest, inventory and explicit tuple with zero launches. Final hello and repeated immutable snapshots bracket commit. A provisioned successor root with an initialized journal cannot be constructed again to reset its incarnation/launch count. Boot reuse across the chain refuses. Existing later execution hardware freshness fences remain in force.
6. **Closed compatibility and custody.** Canonical comparisons preserve primitive types and allow only native configuration/lane/boot, new trust paths/pins and positive same-shape inventory timestamps. Package/model/settings/profile/budget/topology and owner/spool/staging remain fixed. New documents use bounded canonical reads, duplicate/nonfinite refusal and the shared no-follow single-link mode/UID reader. Publication uses exclusive deterministic stages, fsync, rename and parent fsync. Existing complete-file/enrollment/commit phase resumption tests pass, but do not cover F1's intent publication path.
7. **Independent capacity dimensions.** Control preallocation follows duplicate lookup and precedes job creation, checks history/registration count, global spool accounting, prospective SQL rows/bytes and host byte/journal estimates. Host independently carries predecessor records/registrations/bytes/journal charges, persists allocation registration snapshots and checks registration/prepare/save limits. Origin template slots are charged once; retired allocation and journal charges accumulate across exports. Renewal metadata is separately capped at 16 MiB, exports at 2 MiB, mappings at 64, and chain at eight epochs including zero. Capacity refusal retains evidence; no cap raise, eviction or new deployment bypass is implemented. The fresh selection covers two-renewal cumulative counters, capacity primitive refusals, spool refusal before queue/replay, host limits and ninth-epoch refusal. This is bounded source/fixture evidence, not exhaustive qualification of every numerical combination or real disk behavior.
8. **Commands/package/Lot B boundary.** Explicit private begin/retire/provision/commit parsing reaches the transaction implementation; distribution membership includes the new module and archive byte determinism passes. No public provisioning route was added. Origin mapping is a closed principal/origin/current digest format and chain relation only. Existing first-match owned-voice derivation is not accepted as Lot B renewal authority; public old-preset/clone reuse still requires the separately commissioned shared derivation and independent supervisor/control checks.

No additional blocking authority or capacity bypass was established in this bounded review. This statement does not negate F1 or qualify installed execution.

## Fresh execution receipts

Inspected `/tmp/r345_tests.py` before execution: it inserts this exact candidate root, loads explicit classes/methods and selects completion methods defined directly on `RenewalCompletionTests`, avoiding inherited broad duplication. Driver SHA-256: `f44e0c84d64a88ccd6d9df6b2eb79ee51d514035f38c9ab9aee18ca446b6ed20`.

Executed the coherent **current 110 selection exactly once** from candidate cwd:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys; sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-532a7465/tests/service"); exec(compile(open("/tmp/r345_tests.py").read(),"/tmp/r345_tests.py","exec"))'
```

Printed interpreter: **3.14.7 (main, Aug 5 2026, 10:29:49), Clang 17.0.0**. Receipt: **Ran 110 tests in 140.886 s — OK; exit 0**, no failures/errors/skips. The selected current inventory test was inspected at lines 442-449: it changes valid closed-wire `host.architecture` to `x86_64`, expects `native_activation_hello` and asserts zero launches. That strengthened body passed in this same coherent run.

The selection comprises all RenewalFiles/History, NativeTerminal/CancelledHistory, InstallOwnership and NativeActivation tests; directly defined RenewalCompletion methods; explicitly chosen negative/failed offline seal and cancelled/failed two-renewal methods; three managed-clone regressions, one native lifecycle regression, five supervisor regressions and two package regressions, exactly as enumerated by the inspected driver. No historical broad development run was repeated. r344/r345's historical failures remain historical failures, not fresh passing receipts.

The separate F1 probe used the same pinned interpreter with `-I -B -`, explicit candidate and candidate/tests/service paths, one explicit test instance and `TextTestRunner(verbosity=2)`. Combined fresh execution is 110 passing existing cases plus one successful defect-reproduction probe; it is **not 111 acceptance tests**.

## Verified source fingerprints

These independently match the r345 source pins (the report itself is excluded):

- `service/native_renewal.py`: `953ef68f0ba18616824da34ff3d4e0c7225b59face37a8e3465646473bcaafbf`
- `service/core.py`: `23167776bc92e758df78dae42bca73c139fad9a334eeb279416036804e0a237c`
- `service/native_activation.py`: `97ff1a9f46ae290ccf28048533b25e6e7a125f11d594eaed0f32bdfd43e50d01`
- `service/native_launch.py`: `50697bf883f592fd8e57d40927c10ecd051f1f0334ac72542682dc306fe60245`
- `service/native_package.py`: `9113b7b40bdeb6f70bdcec937b61951de0b6ad94beb61085e796343a5058bfa8`
- `service/reconciliation.py`: `3b2802f18126580771322e68b51e8ceee7ad1966f9187ede7e2c1edb9cd8a739`
- `worker/native_protocol.py`: `81a0454ccfa314d02129d681304f498df6840d64a0756bbba3851d5a9cae854d`
- `worker/native_supervisor.py`: `d9d71ee2073ad66dca74ed52da9497fa90f1771b6882798648ee0f64f149eced`
- `tests/release/test_native_renewal.py`: `64cf3a752644bd9452b90f31dd930b3023f55b42577b8b96e981fe77eb0c92f1`
- `native-dist-manifest.json`: `70c931cb7c7e3c14a42ef1324f587ebb0cc39b8422723db2e7ccecbfe8cf2b00`
- `docs/agent-interface/NATIVE-PACKAGE.md`: `55ed43fdffe9c9fd9c9f067ab5e87e3581e5d74362d8b3eb09c54d0e1034f5a4`

## Parent-owned remaining gates

Repair F1 and independently verify exact intent-stage resumption and refusal boundaries before complete Lot A acceptance. Parent retains adoption/integration authority. Lot B must remain the next dependent implementation/review lane: stable origin-template derivation on both sides and the canonical public preset/upload/prepare/results/save/generate renewal journey without per-voice edits. Fixed explicit-template tests do not satisfy that public exit.

Installed topology/mount/peer isolation/routing, real arm64 dependency/ABI packaging, actual authenticated installed TLS/process stop/retire/renew/reopen, retained artifacts, audio/listening/resource/performance and CPU/CUDA/Metal qualification remain separately authorized and UNRUN here. Unknown/unreaped/crashed recovery, lost host custody, upgrades, retention reclamation, automatic freshness/trust renewal, native retry, UI and STT are not added. **Lot A SOURCE NO-GO pending F1; legacy CLOSED; Metal UNQUALIFIED; release NO-GO.**
