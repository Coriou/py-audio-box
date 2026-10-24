# H3 r216 Sol — independent round-two notification cleanup review

2026-10-24. **GO for bounded source adoption of the partial startup/handoff slice with the r215 correction. R214-1 is resolved.** No additional blocking source defect established. This is not full Lot B, installed activation, restart, packaging or release acceptance. Adoption remains parent-owned; this review performs none.

## Pins and authority

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/r215-notification-cleanup`, branch `fix/r215-notification-cleanup`, clean HEAD `070ae98419bc1f4a4fd48c08e32076ccc805280b` before and after checks. Verified direct-parent chain:

- Base `90aad5f619a51141bfe9787888387e79f205cf0a`.
- Carried source `a9d13617bb9f00ee45d325761fcf3c76b631fac8`.
- Correction source/tests `c8723da6b80f6ef63f536ec3c30e09bddf6e9d2b`.
- Documentation HEAD `070ae98419bc1f4a4fd48c08e32076ccc805280b`.

Original `c6d71b725052b6e772634d5981e8ef1574db3e0c` and carried source have identical stable patch ID `07b921216ae30dae5a3592fb7781906e759c0f33`: five files, 677 insertions. Correction changes only `service/host_install.py` and `tests/release/test_headless_install.py` (95 insertions, two deletions); source-to-HEAD changes only r215 documentation.

Read full normative r198, r213 checkpoint, r214 NO-GO and r215 correction. SHA-256 checks confirm candidate r198/r213/r214 are byte-identical to root copies. No applicable ancestor or scoped AGENTS.md found. Bounded memory registry search returned no relevant hit and supplied no premise. No skill or delegation used.

## R214-1 and teardown review

`service/host_install.py:1207–1235` replaces output context teardown with an explicit finally and captures `sys.exception()` separately at both close boundaries. Each explicit close is attempted once. On an active primary failure, pipe/output close failures add fixed text plus exception type, without raw exception content. On ordinary exceptions these are finite notes, not a stream or retry loop. Without a primary, close failure escapes; cleanup cannot turn a failed operation into delivery success.

At `1213–1220`, communicate failure remains the exact primary on successful finite reap. Kill failure contributes its existing type note. Failed reap retains the child and raises `_RegistrationChildUnreaped` with the communicate error as cause. At `1221–1235`, both failed closes preserve that sentinel and its cause; if only pipe close initially fails, subsequent output close failure annotates that pipe error. Popen failure still closes output, without accessing an unassigned process. Nonzero status, unexpected output and output inspection failures also pass through output teardown while preserving their primary failure. TemporaryFile creation failure has no acquired output to close. No notification retry, new child operation, timeout change or retained-state erasure was introduced. `sys.exception()` is available in the intended Python 3.11 baseline, though that package was not executed here.

The actual authority wrapper at `service/host_install.py:917–932` catches the preserved sentinel and retains duplicate lock and authority-directory descriptors before its stability finalizer and original descriptor closes. The combined regression at `tests/release/test_headless_install.py:283–309` checks the escaping sentinel, exact cause, both notes, child identity, two valid retained descriptors, and second-authority refusal after wrapper exit. Test-owned duplicate descriptors are closed only after assertions. Existing wrapper stability checks may independently refuse changed authority; the correction does not suppress them or move duplicate retention behind them. No new masking or unintended primary-exception loss established.

The other five regressions at `tests/release/test_headless_install.py:277–281` cover reaped primary plus pipe/output close, unreaped primary plus output close, and cleanup-only failures. I inspected the reported six-test old-source RED/new-source GREEN result; I independently ran the candidate GREEN suite, without editing/replaying old source.

## Carried boundary assessment

The r214 assessment otherwise carries forward, corroborated against unchanged source:

- `service/entrypoint.py:24–65`: closed canonical bounded public projection binds the exact registration and two processes; no protected receipt/key/seed/credential access. The provenance digest remains a label under the trusted host successful-registration channel, not proof available to control.
- `service/host_install.py:909–916,1159–1206`: successful retained registration precedes handoff; DB/provenance/live host bindings bracket fixed exec notification. Context, control identity, exec user, isolated import and helper action remain fixed. Return means delivery only.
- `service/entrypoint.py:75–163`: private descriptor-held WaitingControl publication/read, exclusive no-follow file creation, owner/mode/link/identity checks and fixed bytes after first read; stale reuse, malformed/partial content and substitution refuse. No repair or overwrite is added.
- `service/entrypoint.py:166–200`: PID1/private cgroup incarnation and authenticated hello bracket the pre-owner reader while control exclusion is held. No Core, listener or owner is constructed. Post-Core binding remains a caller obligation outside this slice.
- `service/recovery.py:3458–3546`: explicit SPLIT reader uses the stable WAL-aware private copy, copied SQLite read-only/query-only with trusted schema disabled, bounded reads and installation/configuration/registration/image/boot/process checks. B2/pending recovery refuses. Original SQLite and protected receipts/provenance are not opened by this reader; teardown failures refuse.

Legacy startup/activation guards and B2 writers remain unchanged. These are bracketed source checks under the declared trusted-channel model, not atomic protection against arbitrary privileged mutation. Historical-incarnation restart remains the authorized separate successor; its absence is not a new blocker for this partial source adoption.

## Independent checks and limits

Read both existing conftests before execution; no release conftest exists. Ran only `/private/tmp/pab-0c-venv/bin/python -I -B` (Python 3.14.7), inserting explicit absolute candidate and candidate/tests paths and loading these seven unittest modules:

- `tests.release.test_headless_install`
- `tests.service.test_installed_registration`
- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

**147 tests PASS, exit 0, 31.678 seconds.** Additional ephemeral inert checks reused the real disposable registration/projection fixture and injected Popen/output objects for (1) Popen failure plus output-close failure, (2) nonzero status plus output-close failure, and (3) output-fd validation failure plus output-close failure. All three preserved the primary and exact type-only note, with output close once and pipe close once only where a child existed. No persistent check file was created. `git diff --check 90aad5f HEAD` passed; candidate remained clean.

Evidence is source/injected, with real disposable files, private SQLite copies and lock APIs. No actual Docker exec/build/run/pull, Linux proc qualification, mount/UID transition, network/listener, model/ML/hardware, pytest guard/full suite or HELD operation occurred. No source edit, commit, delegation, adoption or unrelated cleanup. Only ordinary test-owned fixture cleanup and retained test-fd cleanup occurred. Sole persistent write is this root report.

Actual supported Linux permissions/peer/mount/process behavior, remote Docker exec descendant death (local client reap is insufficient), Python 3.11 package/build, ordinary packaged command, post-Core integration, restart and activation remain pending under the existing parent-owned scope. No new broad gate is imposed and no operational/release GO follows.
