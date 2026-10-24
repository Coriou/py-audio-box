# H4 override validation fix — r175 implementation handoff

2026-10-24. Bounded source fix complete; independent review next. No self-adoption, runtime qualification or release approval.

## Provenance

Own fresh checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-75c7e0b9`; starting clean HEAD `f9d0639f9250fe1737a8f1d742d797ca55b030ea`, initially branch `t3/75c7e0b9`, now `t3/fix-profile-override-validation`. No applicable AGENTS.md found. Memory registry search had no relevant hit; no memory-derived implementation fact used.

Requested candidate `73673ef15f036e6ef83e2e4c9f3251bc367faa2e` and starting HEAD diverge at original base `6e982c828df7c9d4a25974fca5d198916995c44c`; fast-forward ancestry was unavailable. As authorized, cherry-picked only original source `d1491b18403bb54bd75eda6ba4be89dcf8eef236`, producing local source baseline `c89b22c20bd14fb4ff43f4bb2c385576b75e79cb`. No historical checkout or root source/branch was changed.

Read root reports `H4-HARDWARE-PROFILES-r172-sol.md` and `H4-HARDWARE-PROFILES-REVIEW-r174-sol.md`. Root r172 document commit: `f9d0639f9250fe1737a8f1d742d797ca55b030ea`; root r174 document commit: `4acb766a7ac298a3b8535259ac33b232c3dc761f`. Original r172 docs commit: `73673ef15f036e6ef83e2e4c9f3251bc367faa2e`.

Fix source commit: `05366aab73ff14931ed8ed0d72b301eecbf2e845`. Exactly `service/profiles.py` and `tests/service/test_profiles.py` changed. This report is committed separately after that source commit; its commit SHA is supplied in the final handoff. Only this report is copied to root, without a root commit.

## Change

`VoiceWorkRequest.__post_init__` validates every non-None device override before selection: tuple of length two, DeviceFamily instance, and exact nonnegative int index. Invalid shapes/families/indices raise ValueError; nothing is coerced. This prevents False/True from matching integer device indices 0/1 and returning selected_override.

Focused regressions reject False and True indices; malformed arity, list/scalar shapes, raw string/None families, negative, float, string and None indices. Valid integer 0 and 1 fixtures select the exact requested profile with ready/selected_override. Existing missing profile, unavailable device and conflicting profile/device checks still return override_unavailable without fallback.

Selection/ranking and conservative preferred-device unmeasured policy are unchanged. r174 did not establish that policy as a second defect.

## Verification and limits

Executed only the authorized isolated unittest commands with `PYTHONDONTWRITEBYTECODE=1`:

- `/private/tmp/pab-0c-venv/bin/python -m unittest tests.service.test_profiles -v`: 13 PASS.
- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/worker -p test_hardware.py -v`: 4 PASS.
- `git diff --check`: PASS before source commit.

These are pure source/fake-policy fixtures, including existing explicit fake-provider checks and fresh-process import/discovery restrictions. No actual device/provider/model was exercised or qualified. No pytest, guard runner, full suite, Docker, network, ML/model import, hardware probe, benchmark, HELD resource operation, cleanup or delegation was performed. Hardware tests use only inert fake providers. No root source or historical worktree was modified.

Independent review must assess this source commit before parent adoption. Caller integration, qualification/probe provenance and freshness, actual resource admission, real adapters/provisioning/packaging, and actual CPU/CUDA/native Apple Silicon Metal qualification remain parent-owned gates. This handoff conveys no runtime readiness or release GO.
