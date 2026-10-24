# H4A independent round 2 override review — r177

2026-10-24. **Bounded source GO: r174 P2 closed; no regression finding in the reviewed slice.** All 17 authorized focused tests pass. This is approval of the trusted inventory/profile selection foundation and its override correction only. Hardware integration and actual qualification remain required; no runtime or release GO is conveyed.

## Verified pins and scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-75c7e0b9`, branch `t3/fix-profile-override-validation`, clean HEAD before and after checks:

- Root base `f9d0639f9250fe1737a8f1d742d797ca55b030ea`.
- Cherry-picked baseline `c89b22c20bd14fb4ff43f4bb2c385576b75e79cb`, direct child of root base. Diff against original `d1491b18403bb54bd75eda6ba4be89dcf8eef236` is empty for all four carried files: `worker/hardware.py`, `service/profiles.py`, `tests/service/test_profiles.py`, `tests/worker/test_hardware.py`.
- Fix source `05366aab73ff14931ed8ed0d72b301eecbf2e845`, direct child of baseline. Exactly six validation lines in `service/profiles.py` and 23 test lines in `tests/service/test_profiles.py` added.
- Docs/HEAD `902e34100ecb081af36e23b40bbf4472f290ac5f`, direct child of fix source; only r175 report added.

Read root r172, r174 and r175 reports, all four candidate source/test files, and canonical IMPLEMENTATION-PLAN hardware requirements and H4. No applicable AGENTS.md found in candidate or report ancestry. Memory registry search had no relevant hit; no memory-derived fact used. This root report is the sole authored file; no source edits, commits, delegation or cleanup.

## Closure and carried review

`VoiceWorkRequest.__post_init__` now rejects every non-None override unless it is a tuple of length two containing a DeviceFamily and an exact nonnegative int index. The exact-type check excludes False/True before selection can compare them with device indices 0/1. Invalid input raises ValueError without coercion. This closes the r174 P2 at the declared internal dataclass seam.

New regressions reject both bool indices, malformed tuple arity, list/scalar shapes, raw string/None families, and negative/float/string/None indices. Valid integer 0 and 1 each resolve to their exact profile with ready/selected_override. Carried unavailable and conflicting overrides still refuse with override_unavailable and cannot use the explicit CPU fallback policy.

The remaining selection and hardware code is unchanged from the reviewed baseline. Re-read and focused tests retain exact engine/checkpoint/revision/conditioning matching, features/language/dtype/attention eligibility, installation/provisioning/qualification/probe gates, memory/concurrency limits, explicit identity-preserving fallback, comparable cold/warm workload ranking, deterministic explicit unmeasured policy, and native Darwin arm64 Metal topology restrictions. Discovery consumes trusted persisted records without ML discovery or implicit probing; explicit fake providers remain fail-closed.

The r174 observation about a preferred eligible family lacking comparable measurements remains the documented conservative refusal/integration choice, not a newly established defect. Preference and measurement ordering are untouched. Existing trusted-input, caller identity/settings binding, measurement-status disclosure and bounded-record-count limitations carry forward; this review adds no public-schema or broad integration gate.

## Checks and evidence limits

Executed in the pinned candidate with `PYTHONDONTWRITEBYTECODE=1` and `/private/tmp/pab-0c-venv/bin/python`:

- `-m unittest tests.service.test_profiles -v`: 13 PASS.
- `-m unittest discover -s tests/worker -p test_hardware.py -v`: 4 PASS, including the existing fresh-process import/discovery restriction test.
- `git diff --check c89b22c 05366aa`: PASS.
- Candidate `git status --short`: empty before and after checks; HEAD unchanged.

These are source/fake-policy fixtures, not actual device, model, provider or job execution. No additional fixture files were created. No pytest, guard runner, full suite, Docker, network, ML/model import, hardware probe, benchmark or HELD-resource operation was performed.

## Remaining parent-owned gates

Parent adoption and integration still require immutable authorized voice/settings binding, execution/result identity fencing, serialization and fallback/measurement disclosures, durable qualification/probe provenance with freshness/invalidation, and actual resource admission. Real adapters, provisioning/packaging and native Metal transport/lifecycle integration remain required. Actual install/generate/cancel/restart, constrained CPU responsiveness, bounded higher-concurrency/resource measurements and human listening on the required CPU, CUDA and native Apple Silicon Metal rows remain open. This bounded source GO neither qualifies those rows nor authorizes resource operations.
