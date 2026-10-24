# H3 r211 Sol — retained registration identity correction

2026-10-24. R210-1 implementation complete for independent review; adoption remains parent-owned. No installed activation, operational or release GO.

## Pins and authority

Fresh own worktree `/Users/ben/.txcode/worktrees/py-audio-box/r211-identity-fix`, branch `t3/r211-identity-fix`, created at source parent `ad0002aadd1d9a1fd9400afbcce25ad45ce14e95`. Cherry-picked only `4fd3d5af7469024eba9ed1dd97293985b58a1502`, yielding `dfd1222c0572dbbc36fb6db7be23601d842053e1`; the complete trees of original and cherry-pick compare identically. Fix source commit: `dd526fa950418336f7cbe0d802018716d89764ea`. Entry thread checkout was clean at `fc203c7822ae53c0ce1b1bdac867643afd0cfd1a`; it was not modified. Root observed at `27dae81580455a8b7ebd5754a3694a4eb7c9b6c4` during handoff preparation.

Read root r209, r210 and complete normative r198, including retained identity, interruption and owner-union requirements. Root r198 matches own base copy. No applicable AGENTS.md found; bounded memory search returned no relevant hit and supplied no premise. No delegation or self-adoption.

Only source correction files: `service/host_install.py` and `tests/service/test_installed_registration.py`. This report is committed separately and is the sole file copied to root, under `docs/agent-interface/reviews/`. No root source change or root commit.

## Correction

The bounded provenance reader returns canonical bytes and identity from the same validated open descriptor, after its existing metadata, stable-read and path corroboration checks. Enumeration retains that identity directly. Exact retry no longer overwrites any existing baseline. Prewriter and completion checks compare each reader's bytes/descriptor identity pair with the original pair.

New provenance retains the created descriptor's post-flush/fsync identity before close. The common bounded reader corroborates the path against that descriptor-derived baseline before writer invocation, then again at completion. No later independent path stat establishes a baseline.

No schema, authority, activation or protocol change. Existing provenance-before-write, interrupted-attempt retention, complete owner union, locks, budgets and postwriter checks remain in place.

## Verification

Inspected `tests/conftest.py` and `tests/service/conftest.py` before running tests. Used only `/private/tmp/pab-0c-venv/bin/python -I -B`, isolated unittest loading with explicit absolute own-worktree and own-worktree/tests paths. No pytest runner, conftest execution or guard invocation.

Red: all four final new regressions FAIL on own cherry-picked source, exit 1, 4 tests in 1.594 seconds. Each unexpectedly accepted a byte-identical replacement. Temporarily restored only own source file to HEAD for this check and restored the fixed bytes in a finally block; no shared checkout mutation.

Green: **116 tests PASS, exit 0, 19.481 seconds** on final source/test bytes: the inherited 112 plus four new tests across exactly `tests.service.test_installed_registration`, `tests.service.test_installed_host_policy`, `tests.service.test_host_install`, `tests.service.test_install_ownership`, `tests.service.test_split_init`, `tests.service.test_install_child`. `git diff --check` passed before source commit.

New regressions assert actual fixture inode changes with identical bytes: exact retry replacement during first prewrite hello; first prewrite observation; immediately after the validated reader returns and before the former independent baseline stat; and new-file replacement during authority-directory fsync before path corroboration. Each requires refusal before writer, no returned ID and unchanged registration rows. The read-return seam makes the formerly separate read/stat window deterministic without timing assumptions. Existing exact-retry positive, postwriter identical replacement, interruption and owner-union coverage continue to pass.

Reproduce green:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/r211-identity-fix","/Users/ben/.txcode/worktrees/py-audio-box/r211-identity-fix/tests"];s=unittest.defaultTestLoader.loadTestsFromNames(["tests.service.test_installed_registration","tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"]);r=unittest.TextTestRunner().run(s);sys.exit(not r.wasSuccessful())'
```

## Limits and next gate

Real temporary SQLite, private copies, files, descriptor identities, replacement and locks are exercised with injected host/UID/namespace observations. Narrow transaction regressions inject content validation as in the existing fixture; they establish identity continuity, not actual host qualification. No actual mount/unshare, UID transitions/chown, Docker, Linux proc qualification, network/listener, ML/model/hardware, pytest guard/full suite, maintenance/apply or HELD operations. Only normal own fixture cleanup occurred.

Independent review is next. Parent adoption, actual Linux ownership/peer/mount qualification and Lot B packaged launch/activation remain separate and unperformed.
