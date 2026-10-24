# H3 r217 Sol — installed historical-incarnation adapter

2026-10-24. **Implementation complete for independent Sol source review; no self-adoption or operational GO.** This is the separate installed-only restart boundary unit requested after r213/r216, not packaging, activation or an actual stop/up receipt. The existing retained receipt format can prove the bounded clean predecessor used here: exact allocation binding, PID1 waitpid/ECHILD reap receipt, terminal Core job, retained reservation and matching journal/Core acknowledgement. No reap is inferred from a newer Docker StartedAt or container exit.

## Checkout and pins

Own supplied fresh checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-8e33a0d6`, branch `t3/installed-history-restart-adapter`. Entry HEAD `5c9db48fc8847eefc72abe6862fac6e84506a3e9`, clean tracked/untracked status. Fast-forwarded to current root `81cad73dc6f56d8d0fdc275ca66f3dc79be1b6e1`, clean before edits. Source/tests commit **`fbd0aff770213024aff27c344ba0a2fa8db65d18`**, direct parent that FF base. Source commit changes exactly five files, 453 insertions and 31 deletions:

- `service/recovery.py`
- `service/host_install.py`
- `tests/service/test_installed_restart.py` (new)
- `tests/service/test_installed_registration.py`
- `tests/service/test_installed_host_policy.py`

The current-root integration carries the relevant reviewed `070ae98419bc1f4a4fd48c08e32076ccc805280b` startup source byte-for-byte: `service/host_install.py`, `service/recovery.py`, `service/entrypoint.py`, `tests/release/test_headless_install.py` compared through `git show` at 070ae984 and the FF base. **070ae984 is not an ancestor of this checkout**; this is verified carried-source equality, not an ancestry claim. Root remained `81cad73dc6f56d8d0fdc275ca66f3dc79be1b6e1` during this work.

Read complete normative r198, r213 checkpoint and r216 review, including registration, immutable reuse, failed-child retention and restart boundaries; inspected actual observer/provenance/snapshot/registration/journal/UID allocation and startup-reader paths. No applicable AGENTS.md was found. Bounded memory-registry search supplied no relevant premise. No skill, delegation or self-adoption. This report is a separate documentation commit; its identical copy is the sole root write.

## Separate installed history and current projection

Legacy `host_registration_union` remains **exactly source-identical** to the FF base, checked by extracting its AST source span. Its same-ID/differing-StartedAt `host_ambiguous_incarnation` refusal is also exercised after successful installed restart. Legacy host/B2 capsule readers and all recovery writers remain unchanged.

`InstalledObserver.current_owners` directly inspects every retained/candidate container ID. `_installed_current` derives one complete current projection from those inspections, checks full ID coverage, unchanged image/role/volume bindings, finite budgets, one retained lane and monotonic incarnation timestamps. Caller-selected container records must equal that independently derived projection. Complete consumer enumeration, installed policy/configuration/volume checks and actual-current process observations still run through the existing observer and its brackets. Missing IDs and unknown consumers refuse; an old historical StartedAt is never passed off as a fresh inspection of that stopped incarnation.

`_installed_history_read` validates all protected provenance, including provenance-only interrupted registrations, canonical bytes, filename/digest, metadata/identity and total byte/entry budgets. Every DB registration needs its provenance. Historical v1 observations remain immutable retained claims and are content-validated against the same enrolled policy; they are not reobserved as old live processes. V2 predecessor links must resolve to exact retained registrations. Both registration and handoff use this complete reader, and retain the original read identities through later continuity checks.

## Clean predecessor and immutable linkage

Before replacing a same-ID historical incarnation, the adapter requires a healthy current private-copy snapshot and validated stable journal on the fixed receipt volume. It refuses active/nonterminal/unknown ownership, uncertain commands, outstanding executing/queued reservations, pending recovery, maintenance/lane changes, quarantine and exhausted history/UID budgets. Replaced provenance-only registrations without a completed DB row refuse. An exact current provenance-only interruption can still complete the original registration transaction; it is never treated as already successful.

`installed_clean_history` requires terminal jobs with confirmed reaped attempts, retained reservations, exact acknowledged Core dispatches, returned commands, complete dispatch/journal coverage, terminal acknowledged journal records, identical receipt digests/bindings and sequential unreused UIDs 20000 onward. It rederives the used summaries from retained source rows and checks allocation coherence plus source/journal digests. A new StartedAt cannot supply any missing receipt or ACK. This bounded clean-stop path conservatively requires all predecessor dispatch/journal ACKs complete; it does not add a restart reconciliation path for unacknowledged terminal proposals.

New installed provenance uses the closed `installed-registration-v2` shape only when history is superseded. Its `history` contains predecessor registration digests, the exact checked pre-insert snapshot and validated journal payload. Snapshot/source/journal binding and clean obligations are revalidated on reads; self-links and unknown history fields refuse. All historical provenance and DB rows stay intact. Existing v1 provenance and exact retries keep their original bytes. A new worker incarnation requires a different boot, and boot cannot change for an unchanged worker incarnation. Changed live process incarnations must have new start ticks and PID namespace identities. Old processes are compared as retained claims, never actually inspected after death.

Existing control exclusion spans snapshot/child/reap and post-insert checks. Fresh host observation and authenticated healthy hello still bracket insertion; the control-UID child remains the sole registration writer. Exact restart retries reuse the saved history bytes rather than rebuilding a different history from the later DB state. Provenance continuity fixes remain in place. Handoff now validates all retained history and derives current owners before checking the selected successful provenance and notifying; all historical identities are checked again after notification. The existing startup reader selects the new exact registration coherently without reading protected history from control.

## Tests and corrections

Read both existing conftests; no release conftest exists. Ran only `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit absolute checkout and checkout/tests paths inserted into `sys.path`, using unittest loading. Interpreter is Python 3.14.7, not Python 3.11 package evidence.

Final broad focused run: **168 tests PASS, exit 0, 48.576 seconds**, across:

- `tests.service.test_installed_restart`
- `tests.service.test_installed_registration`
- `tests.release.test_headless_install`
- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

After adding one lane/history exhaustion test and strengthening the integrated journal-byte preservation assertion, the final targeted run passed **2 tests, exit 0, 3.248 seconds**: `Restart.test_lane_and_history_exhaustion` and `ClosedRestart.test_real_provenance_new_startup_and_handoff`. This gives **169 distinct passing tests** across the two final runs, not a claimed single 169-test run. `git diff --check` passed before source commit. Source checkout was clean after that commit.

New coverage uses real temporary SQLite/provenance/journal files and injected host/hello/UID child behavior: first registration; retained terminal job and matching PID1 reap/ACK evidence; same IDs/new StartedAt/new process/new boot; second registration; exact old row/provenance/journal preservation; startup and handoff selecting the new registration; exact retries and provenance/insert/bracket interruptions; dirty old attempt, stale ACK, unknown owner, UID mismatch, missing provenance/container, unknown consumer, changed config/image/volume, reused boot/process, closed-history corruption, lane/history/UID exhaustion and legacy ambiguity refusal.

A new closed-history corruption negative exposed an existing test-fixture error: `.stop()` does not remove patches entered through `TestCase.enterContext`. The named closed-observer transaction was still bypassing enrollment/provenance content validators. The directly affected fixture now explicitly exits those contexts; its closed observer and the new integrated restart tests run with the actual validators active. The Docker fixture now reports a missing injected container as a bounded ValueError instead of leaking StopIteration. Intermediate failures were corrected; none was waived.

## Evidence limits and handoff

Source/injected evidence only. Test terminal/reap data are injected claims, not actual descendant-death evidence. The positive retained job uses a synthetic fixture context inside the disposable installed DB; it does not qualify production Kokoro execution, saved voice/artifact journeys or actual OS permissions. No actual Docker/proc/mount/UID transition, build/pull/exec, network/listener, model/ML/hardware, pytest guard/full suite, HELD operation, recovery writer, activation-gate or package change occurred. Only ordinary test-owned fixture cleanup occurred. No source was copied/cherry-picked into root.

Finite existing owner/provenance/registration/UID budgets remain; retaining full immutable prestate makes the provenance byte budget potentially bind before the nominal row budget. Exhaustion refuses, with no lane/journal reset, UID reuse/rotation, deletion or history compaction. Historical closure is protected host-generated evidence, not independent inspection of a dead incarnation. Host observations and source checks retain the existing trusted-administrator/bracket model; actual Linux behavior and privileged mutation timing are unqualified.

**Independent Sol review is next**, including this exact source pin, full original brief, immutable history/clean-state rules, conservative ACK boundary and fixture correction. Parent owns adoption. The packaged launcher follows independently; this unit adds no activation/release gate and claims no installed, speech, actual restart, operational or release acceptance.
