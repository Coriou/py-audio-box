# H3 r218 Sol — independent installed-history source review

2026-10-24. **NO-GO for source adoption at this pin.** Two blocking validation defects remain in the new retained v2 history boundary: incomplete validation of the embedded predecessor snapshot/journal, and acceptance of cyclic/nonhistorical predecessor links. The permitted eight-module run independently passes all 169 tests; additional inert disposable-fixture reproductions demonstrate both defects through the complete registration retry, with actual provenance/enrollment validators enabled. This is not an actual restart or activation review.

## Pins, authority and provenance correction

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-8e33a0d6`, branch `t3/installed-history-restart-adapter`, clean HEAD before and after checks:

- Base `81cad73dc6f56d8d0fdc275ca66f3dc79be1b6e1`.
- Source/tests `fbd0aff770213024aff27c344ba0a2fa8db65d18`, direct parent the base.
- Documentation HEAD `142a7e74c6023bb928f3ca0d95415b2fc7451534`, direct parent the source commit.

Verified exactly five source/test files changed: `service/host_install.py` (+118/-28), `service/recovery.py` (+75), `tests/service/test_installed_host_policy.py` (+4/-1), `tests/service/test_installed_registration.py` (+8/-2), new `tests/service/test_installed_restart.py` (+248). Source-to-HEAD changes only r217 documentation. `git diff --check 81cad73..HEAD` passes.

The supplied/r217 statement that `070ae98419bc1f4a4fd48c08e32076ccc805280b` is NOT an ancestor is incorrect in this repository. Both `git merge-base --is-ancestor 070ae984... 81cad73...` and the HEAD check return 0. Merge `5c9db48fc8847eefc72abe6862fac6e84506a3e9` has parents `ae3bfe95a9b1394e11d61b683eb30c96c822adf9` and `070ae98419bc1f4a4fd48c08e32076ccc805280b`; base is its direct child. No replace refs; repository is not shallow. Independently verified byte equality at 070ae984 versus base for all four claimed carried files: `service/host_install.py`, `service/recovery.py`, `service/entrypoint.py`, `tests/release/test_headless_install.py`. Equality and ancestry are separate verified facts here. Correct r217's provenance statement; this documentation error is separate from the source blockers.

Read root normative r198 in full, including restart/failure rules, r213, r216 and r217; inspected changed source, snapshot/allocation/journal dependencies, startup/handoff and fixtures. No applicable ancestor/scoped AGENTS.md found. Bounded memory-registry search returned no relevant hit and supplied no premise. No skill, delegation, source edit, commit or adoption.

## R218-1 — blocking: embedded clean-history payload is not fully validated

`service/host_install.py:1029–1042` closes only the outer history/state shapes and checks caller-recomputable digests, then calls `service/recovery.py:2859–2899`. That function rederives four summaries and invokes `_allocation_coherent`, but does not apply the complete snapshot/journal semantics to the embedded retained payload. In particular, the loop at `2891–2897` does not close journal records/allocation objects, validate the allocation snapshot against its binding/Authority, validate the record boot/runtime prefix against retained registrations, or validate proposals. The existing current snapshot checks at `service/recovery.py:853–901` and journal loader at `worker/runtime_journal.py:56–98` perform materially stronger checks. Validating the *current* receipt file cannot validate a different embedded historical journal.

Likewise, closing `state` and comparing its projection hash does not validate the projection's schema/table set or pending recovery rows. Several asserted summaries are neither rederived nor checked. A digest recomputed from malformed embedded data establishes consistency of those bytes, not validity of the claimed clean prestate.

Independent reproduction used `ClosedRestart.setUp()`, `TerminalRestart.terminal_fixture(case)`, `case.restart()`, then `rid=case.f.run_registration()`. For each experiment, only a test-owned retained provenance file was changed using canonical `encode`; current SQLite, actual test journal and host observations stayed unchanged. After a journal change, recomputed:

```python
s['journal_digest'] = digest(dict(payload=hist['journal'], sha256=digest(hist['journal'])))
s['evidence_projection']['journal'] = s['journal_digest']
s['source_digest'] = digest(s['evidence_projection'])
```

Each of these independently returned the same successful `rid` from the full `case.f.run_registration()` retry:

1. Added `invented_owner_claim='accepted'` to the historical journal record.
2. Replaced historical `allocation.snapshot` with `['not an authority']` and historical record `boot` with `'f'*32`, retaining the old allocation binding. This is not a valid authority or registered journal incarnation.
3. Set historical `evidence_projection.schema=[]` and `evidence_projection.data.p1_recovery_plans=[['invented pending maintenance']]`, then recomputed the projection digest. The claimed clean predecessor now explicitly contains invalid schema/pending maintenance and still passes.

Observed outputs respectively: `unknown journal record field full retry accepted: True`, `invalid historical allocation snapshot and foreign boot full retry accepted: True`, `false schema/pending recovery full retry accepted: True`.

Required correction: validate the detached historical payload with complete closed, typed, bounded snapshot/source/journal rules; rederive every safety-relevant clean-state summary, enforce enrollment/schema/maintenance/ownership/allocation/boot/receipt/proposal bindings, and reject unknown fields throughout. Preserve the trusted-host evidence model: this is validation of retained claims, not independent proof of actual death. Add negatives through the real closed reader and full retry/handoff, including the accepted examples above. Do not reopen historical filesystem state as though it were current.

## R218-2 — blocking: predecessor closure permits cycles and future claims

`service/host_install.py:1033–1041` checks predecessor IDs against the embedded registrations and rejects a direct self-link only. `_installed_history_read` at `1080–1086` checks that referenced registration records exist somewhere in the combined retained set. It does not check graph acyclicity, strict predecessor ordering, or that the embedded prestate could precede the registration carrying it. Consequently presence is accepted as historical provenance.

Independent full-retry reproduction, starting from the same valid two-registration fixture and restoring the second provenance after the R218-1 experiment:

- Keep second v2 provenance B unchanged, linking to first registration A.
- Load original A from `case.raw`; change its kind to v2 and copy B's history into it.
- Replace A's embedded `state.registrations` with `[B.registration]`, its `history.predecessors` with `[B_id]`, and its projected `p1_registrations` with `[[B_id, encode(B.registration), B_id]]`.
- Recompute A's `state.source_digest` from its modified projection; write canonical A back to the test-owned original provenance filename.
- Full `case.f.run_registration()` returns B successfully: `two-node predecessor cycle full retry accepted: True`.

Thus A claims B as predecessor while B claims A. The history for A also uses a state containing the later B registration. Neither direct self-link rejection nor bounded file count catches this. The current code does not recursively traverse links, so this reproduction is a false-history acceptance, not a demonstrated infinite recursion. Byte/count limits do bound storage; they do not establish an acyclic historical order.

Required correction: check the complete retained graph with explicit finite node/edge/depth budgets, cycle rejection and coherent pre-insert/chronological ownership ordering. Bind each predecessor registration to its validated retained provenance and require a valid predecessor closure for the particular new incarnation, rather than mere membership in the final directory. Reject cycles/future-state replay before writer/notification. Keep exact retry bytes and immutable historical files; do not repair accepted evidence in place. Add two-node/multinode cycle, future-prestate and replay negatives.

## Corroborated behavior and bounded coverage

Legacy `host_registration_union` is exactly source-identical by AST source-span extraction at base and HEAD (SHA-256 `c56dc9968b46a06ff3676895e896fc000566c42ed740a15e448cca466eb12bd7`). Legacy ambiguity refusal remains exercised. No recovery writer, registration child writer, Dispatcher activation, package or lifecycle implementation changes occur in this delta.

The installed current projection is independently obtained through `InstalledObserver.current_owners` (`host_install.py:833–840`) and `_installed_current` (`1092–1103`), then checked for complete ID coverage, stable role/image/volume bindings, one lane and monotonic StartedAt by `installed_history_current` (`recovery.py:2827–2856`). Complete host inventory/policy/process observation and hello brackets remain around registration. Retained claims are not presented as fresh observations of old processes. Process/worker-boot reuse checks at `host_install.py:1156–1173` do not infer descendant death from StartedAt.

The live prestate path still takes a healthy private snapshot, stable journal, and conservative all-ACK clean boundary before supersession. `installed_clean_history` requires terminal/reaped jobs, retained reservations, returned commands, dispatch/journal coverage, matching receipt digests and sequential UIDs; the defects above concern the completeness of historical revalidation. Dirty/unknown/missing provenance/config/image/volume/boot/lane/process/UID and exhaustion cases have focused coverage. Retained unacknowledged terminal proposals remain outside this clean adapter; this is a documented conservative boundary, not permission to acknowledge or reset them.

Registration still retains provenance before the unchanged control-UID writer, checks exact pre/post rows and other source continuity, and keeps control exclusion across child/reap and brackets (`host_install.py:1188–1235`). Exact v2 retries reuse saved history bytes. Handoff reads all retained history and rechecks its identities before/after notification (`1242–1285`); this also makes the history-validation defects relevant to handoff. The integrated test at `tests/service/test_installed_restart.py:185–208` checks old row/provenance/journal bytes, new startup selection, handoff and retry.

Fixture correction is effective: `tests/service/test_installed_registration.py:173` explicitly exits the contexts previously entered through `TestCase.enterContext`; the closed fixture uses the real provenance/enrollment validators. My reproductions used that same closed fixture. The existing corruption negative at `test_installed_restart.py:241–245` only adds a top-level history field, so it does not cover the accepted nested payloads or graph cycle.

Stored input is bounded by per-file/aggregate 1 MiB provenance limits and finite authority/owner counts; `host_json` rejects duplicate keys, nonfinite numbers and parser recursion failure. Full retained snapshots consume the byte budget before nominal row capacity in some cases, with refusal rather than compaction. No unbounded recursive traversal was found in the new reader. These bounds do not cure R218-1/R218-2.

## Independent execution and limits

Read both existing conftests; no release conftest. Executed only isolated unittest loading, without pytest fixture activation, with this command from the exact clean candidate HEAD:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-8e33a0d6","/Users/ben/.txcode/worktrees/py-audio-box/t3-8e33a0d6/tests"]; names=["tests.service.test_installed_restart","tests.service.test_installed_registration","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"]; result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(not result.wasSuccessful())'
```

**169 tests PASS, exit 0, 51.355 seconds**, in one run including the added exhaustion test. Interpreter independently reports Python 3.14.7; this is not Python 3.11 package/build evidence. Extra reproductions used the same interpreter/isolation/absolute paths and only disposable fixture files with injected host/process/UID/hello/child behavior. No retained production evidence was modified. The successful reproduction outputs are validation failures, not acceptance receipts.

Source/injected evidence only. Terminal/PID1 reap/ACK claims in the fixtures are trusted injected source evidence, not actual descendant-death proof. No actual Docker/proc/mount/UID transition, network/listener, ML/model/hardware, pytest guard/full suite, HELD operation or unrelated cleanup. Only normal test-owned temporary cleanup occurred. Candidate remains clean and pinned. Sole persistent write is this root report; no source was copied, adopted or committed.

Parent owns remediation and a fresh independent review before adoption. Packaged ordinary launch, activation/post-Core integration, actual supported Linux permission/peer/mount/process/restart qualification, Python 3.11 packaging and audible speech/user journeys remain outstanding. No operational, installed, speech or release GO follows.
