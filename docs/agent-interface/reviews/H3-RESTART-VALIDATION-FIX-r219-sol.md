# H3 r219 Sol — restart validation remediation candidate

2026-10-24. **Implementation candidate for independent review; no self-adoption or operational GO.** Addresses R218-1/R218-2 through shared detached validation and a bounded retained predecessor DAG. Parent owns independent review and adoption.

## Pins and authority

Own fresh supplied checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-7733b4d1`, branch `t3/restart-history-validation`, clean entry `2ba7f1276a41bb29596048be57e27867042d7116`. Cherry-picked only requested source `fbd0aff770213024aff27c344ba0a2fa8db65d18`, producing `492c94cd92ff9e7f287de858081d0e31315c3f95`. Fix source/tests commit **`33fc3097e111700417c950017eb834d71b2bfe8f`**, direct parent that cherry-pick. No r217 documentation commit was cherry-picked.

Read complete root r198 normative design, r217 handoff and r218 review including reproductions; both conftests were inspected (no release conftest). No applicable AGENTS.md found. Bounded memory-registry search supplied no relevant premise. No delegation, self-adoption or source copying to root.

Explicit follow-up authority added `worker/runtime_journal.py` solely for pure shared validation extraction, and `service/reconciliation.py` solely for pure typed data/reference/owner extraction. Fix changes exactly those two files plus `service/recovery.py`, `service/host_install.py`, and `tests/service/test_installed_restart.py`. No registration writer, journal writer, lock implementation, recovery mutation, schema migration or activation change.

**Provenance correction:** `070ae98419bc1f4a4fd48c08e32076ccc805280b` IS an ancestor of the entry base and candidate (`git merge-base --is-ancestor` exits 0). The contrary r217 claim is incorrect, as r218 established. Historical reports remain unchanged; this new handoff corrects the claim.

## R218-1: detached complete validation

`Journal._load` now calls pure `validate_payload(value, maintenance)`. Its existing payload/maintenance/record/allocation/receipt/proposal validation is extracted without semantic changes: same closed shapes, record/UID/byte limits, receipt and proposal checks, and legacy errors. Envelope reading/checksum, maintenance reading, locks, save/recovery behavior and filesystem checks remain in their original callers.

`recovery._validate_snapshot_journal` is the original current snapshot journal validation loop extracted verbatim apart from indentation: Authority construction/binding, runtime/boot prefix, registered incarnation, dispatch coverage, receipt and proposal joins. Both current snapshot and detached history call it. Detached history additionally calls the journal loader's shared pure validator, so current-file validation cannot stand in for a different embedded payload.

`reconciliation.typed_references` is the existing typed contract/reference/catalogue/B1/B2/owner/dispatch/reservation phase extracted with explicit data, recovery and contract-document inputs. It returns the same reference/owner maps and closures consumed by the unchanged filesystem phase. Detached validation uses those exact checks and bounded diagnostic callbacks, rejecting any diagnostic reason/row. It does not invoke `_collect` or open an old root. The current scanner still obtains its original contract document and retains its filesystem/inode/ownership observation separately.

Detached source validation closes and bounds schema/data/rows/inventory, rejects pending maintenance/recovery, validates enrollment against the exact current enrolled public record and marker, validates every registration, invokes both shared journal validators and the shared typed phase, checks allocation coherence, and rederives all safety-relevant summaries. Existing jobs/reservations/dispatches are rederived alongside monitor/integrity/fault/publication/affected/registration/journal/inventory/version/epoch summaries. Historical byte accounting is bounded rather than claimed as a fresh filesystem measurement.

Schema inspection is confined to an authorizer-limited in-memory SQLite connection: no application database, path, attach, function, view, virtual table or retained trigger execution. Table/column coverage and catalogue schema identity are checked. Retained inventory claims are checked for closed fields, known classification/reason, source-derived owners/references, typed inode metadata and committed asset coverage/hash claims. These validate retained claims; they do not reobserve historical inodes or independently prove death.

Hashes remain byte bindings, never substitutes for the semantic validators. No new receipt/proof format, history repair or authority fallback is introduced.

## R218-2: complete finite predecessor closure

Both the full retained reader and prospective registration before retention/writer call `_installed_history_graph`. Nodes are bounded by the existing owner limit, edges by its square, depth by the owner limit, and topological elimination is iterative. Every predecessor resolves to its exact digest-keyed retained registration and validated provenance. Missing provenance, cycles, incomplete pre-insert closure and divergent v2 branches refuse. Existing independent v1 provenance roots remain supported, including the existing orphan-coverage test.

Each v2 closure contains the previously validated prefix, with coherent lane, same role/image/volume/container bindings, monotonic incarnation timestamps and a strict changed incarnation. Worker boot equality must agree with unchanged worker incarnation; process start ticks/PID namespaces cannot be reused for a changed process. Predecessor journal prefixes and schema are checked against their exact retained provenance payloads. This binds the existing registration digest links and retained provenance content/identities; no independent dead-process observation is asserted.

Graph checks run before registration writer or notification. Exact retries reuse original saved history bytes; historical rows/provenance/journal files are never rewritten by the implementation. The positive three-incarnation test preserves earlier provenance bytes and succeeds through retry and handoff. Legacy `host_registration_union` is byte-identical to entry base and cherry-pick.

## Verification

Meaningful red used this own candidate's new regression tests with the three affected source modules plus reconciliation temporarily restored to exact cherry-pick bytes in this checkout, restored in `finally`. The two-test run produced **7 expected assertion failures, exit 1, 8.999 seconds**: six malformed nested historical mutations and the two-node cycle were accepted by the original full retry. This reproduced the r218 defect rather than merely checking an internal helper. No production evidence was involved.

Green coverage includes the three exact review examples, separate false-schema and pending-maintenance cases, foreign boot, malformed proposal, hidden fault/monitor summaries, two-/three-node cycles, acyclic future prestate and replay, complete retry/handoff refusal, absence of writer/notification and unchanged DB rows, positive multiple clean restarts, nested job/context rejection through current and detached paths, and shared loader equivalence for valid and malformed record/binding/receipt/proposal/UID payloads.

Extraction verification passed on the final source: AST-identical journal block after `self.value/self.maintenance` parameter substitution; AST-identical snapshot journal loop; AST-identical typed reference phase with explicit document parameter; scanner filesystem phase byte-identical. Legacy union source-span equality and `git diff --check` passed. Intermediate implementation failures were corrected; none was waived.

Final pinned eight-module run: **175 tests PASS, exit 0, 65.265 seconds**, at source commit `33fc3097e111700417c950017eb834d71b2bfe8f` (original 169 plus six new tests, with multiple negative subcases). An earlier same-source run before correcting the test guard also passed 175; the pinned run above is the final evidence.

Exact isolated command (from the own checkout, with no pytest fixture activation):

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-7733b4d1","/Users/ben/.txcode/worktrees/py-audio-box/t3-7733b4d1/tests"];names=["tests.service.test_installed_restart","tests.service.test_installed_registration","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"];r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names));sys.exit(not r.wasSuccessful())'
```

## Evidence limits and handoff

Source/disposable-fixture evidence only. Python is 3.14.7, not Python 3.11 packaging evidence. Host/process/UID/hello/reap claims remain injected; no actual Docker/proc/mount/UID transition, runtime construction, network/listener, ML/model/hardware, pytest guard/full suite or HELD operation. No retained production evidence mutation; only ordinary test-owned fixture cleanup. In-memory schema validation supplies no actual installation or migration authority.

Finite original byte/row/owner/journal/UID budgets remain. No compaction, reset, acknowledgement manufacture, reservation release, repair or provenance rewrite. Root receives only an identical copy of this report; no source adoption or root commit. Independent review of the exact source pin and authorized extraction deltas is next. Actual supported Linux behavior, packaged launch/activation, Python 3.11 build/package qualification, speech/user journeys and release acceptance remain parent-owned and outstanding.
