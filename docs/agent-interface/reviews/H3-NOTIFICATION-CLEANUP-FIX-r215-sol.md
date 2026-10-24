# H3 r215 Sol — R214-1 notification cleanup correction

2026-10-24. Bounded implementation delivered for independent review; no self-adoption or activation approval. Installed activation remains closed. Historical-incarnation restart remains the separate next lot.

Fresh own checkout: `/Users/ben/.txcode/worktrees/py-audio-box/r215-notification-cleanup`, branch `fix/r215-notification-cleanup`. Root base `90aad5f619a51141bfe9787888387e79f205cf0a`; cherry-picked only checkpoint source `c6d71b725052b6e772634d5981e8ef1574db3e0c`, producing `a9d13617bb9f00ee45d325761fcf3c76b631fac8`. Correction source/tests commit `c8723da6b80f6ef63f536ec3c30e09bddf6e9d2b`. Report is a separate documentation commit, and the sole file copied to root. No source was adopted into root.

Read root r213, root r214 and full normative r198, including uncertain-child retention and restart policy. Read both existing conftests; no release conftest or applicable ancestor AGENTS.md was found. Bounded memory registry search had no relevant hits and supplied no premise. No delegation or skill was used.

`_installed_notify` now preserves the active primary exception through stdin and temporary output close, including `_RegistrationChildUnreaped`. Cleanup failures add bounded exception-type notes to that same exception. Its original communicate failure remains the sentinel's cause; a reaped communicate failure retains its object/diagnostics. With no prior failure, close errors escape, so cleanup cannot produce false success. Output close also runs after Popen failure and status/output validation failures. No notification retry, extra child operation, timeout change, retained-state erasure or broader launcher behavior was added.

The registration helper's local pipe-close pattern was inspected: its primary-error note handling is useful, but its output context and unreaped-path ordering are not a complete reusable solution for this boundary. It was left unchanged. This patch explicitly captures `sys.exception()` at each notification teardown boundary (available in Python 3.11), attempts each close once, and preserves the primary while attaching cleanup notes.

Six new source/injected regressions cover reaped communicate failure plus pipe close, reaped failure plus output close, unreaped failure plus output close, successful communication plus output close, successful communication plus both closes, and combined communicate/reap/pipe/output failures through real `installed_launch_authority`. The last asserts escaping sentinel/cause/notes, retained injected child plus duplicate lock and authority directory descriptors, valid retained fds, and refusal of a second launch-authority acquisition after the original wrapper exits. Only these test-owned retained descriptors are closed after assertions; production retained state is not cleared. Real disposable registration/SQLite/projection fixtures and protected authority/lock validators are reused, with privilege and external observations injected.

Red/green on own candidate: exact six new tests against the cherry-picked pre-fix source: 6 run, 4 failures, 1 error, exit 1, 1.916s. The combined case escaped output OSError rather than the sentinel. Candidate source was restored immediately after this bounded red check. Final candidate: **147 tests PASS, exit 0, 30.075s**, comprising the original 141 plus six new tests. An earlier candidate run also passed 147 in 31.962s; the final run followed the test double's context-manager compatibility adjustment for the exact red check. `git diff --check` passed.

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, Python 3.14.7. Explicit absolute candidate and candidate/tests paths were inserted into `sys.path`, then unittest loaded exactly these seven modules:

- `tests.release.test_headless_install`
- `tests.service.test_installed_registration`
- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

No pytest guard or full suite was invoked. No actual Docker exec/build/run/pull, runtime or Linux proc qualification, mount/UID transition, network/listener, model/ML/hardware, HELD operation, delegation, self-adoption or unrelated cleanup occurred. Only ordinary own fixture cleanup and close of own retained test fds after assertions occurred. These results are source/injected evidence, not installed Linux, Python 3.11 package/build, remote Docker exec reap, speech, activation, restart or release evidence. Independent review and any adoption remain parent-owned.
