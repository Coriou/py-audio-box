# H3 r208 Sol — independent round 2 observer bounds review

2026-10-24. **GO for bounded source adoption of the corrected installed HOST observer. R206-1 is closed; no blocking finding remains in this review.** Adoption is parent-owned. This decision grants no actual Linux, installed activation, speech or release acceptance. Registration writing and launch/activation remain outside this unit; their absence is not a new gate.

## Verified candidate and review basis

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/r207-observer-bounds`, branch `fix/r207-observer-bounds`, clean before and after checks. Verified the direct-parent chain:

- Base `86e3fc6e8c5de8bf7a9bbfa1a388a58763bb70e4`.
- Carried observer `bbe4066ef0eed544859b53fa7c985c50cf32e11a`, sole parent the base.
- Correction source `306089b20c795973f6c14ea0dc84374f43de5eea`, sole parent the carried observer.
- Documentation HEAD `d0fe258fb5bb40b5909bd7bab2d39b49bc7d3b48`, sole parent the correction.

Original observer `4f282794b2b81e62d16ac7bf705b19db8f9c3dd5` and the base-to-carried delta have identical stable patch ID `94e4e5a74479935789b22ca6ad49614639c5d4b7`. The carried patch changes exactly `service/host_install.py`, `service/install_policy.py`, `service/recovery.py`, `tests/service/test_host_install.py`, and `tests/service/test_installed_host_policy.py`. Correction changes only host_install and the installed-policy tests; documentation changes only the r207 report. No hidden source delta in the documentation commit.

Read root r198 normative design, r205 observer report, r206 NO-GO and r207 correction report. Inspected the correction and relevant carried observer/policy/validation paths against the prior assessment. No applicable ancestor/candidate AGENTS.md found. A bounded memory-registry search returned no relevant hit and supplied no premise. No delegation.

## R206-1 closure

References below are candidate file lines.

- `service/host_install.py:596–615`: `entry_count` starts at one for the root and increments on discovery of each child, before classification or queuing. Pending directories already consume allowance. Popping a directory does not release or recount it. Thus nested scans cannot reuse the budget reserved by another pending directory. The admitted tree is at most 4,096 entries, root included.
- Lines 610–615: the deadline is checked before every `next`, and after a yielded entry before append. At full capacity one additional yield is permitted solely as an overflow probe, then refusal occurs without append. An exhausted iterator can terminate successfully at exact capacity. Root scan with an infinite iterator stops at yield 4,096: root plus 4,095 admitted children and one refused child. Root discovery of two directories leaves 4,093 child admissions for the next scan; yield 4,094 refuses. No off-by-one admission or nested budget bypass found.
- Lines 599 and 605: expired deadline refuses before directory traversal and again before scanner open. Lines 607–616 keep the iterator in a context manager, including deadline, overflow and iterator-error exits. Only the bounded names list is sorted. No unbounded materialization remains.
- Lines 617–658 preserve traversal, per-file 512 MiB and aggregate 2 GiB limits, manifest/hash/extra-file checks, immediate file identity checks and final checks of every file and directory before the inventory digest returns. The extra probe does not enter those collections. The correction does not bypass or weaken final identity checks.

This is a cooperative filesystem budget: synchronous stat/open/scandir/next/read calls cannot be preempted by these checks. Sorting and later finite validation/rechecks are not a hard wall-clock watchdog. That existing limitation does not reopen the specific unbounded iterator collection defect or create a new actual-host requirement for this source decision.

## Preserved observer assessment

The correction production delta consists of enumeration bounds and a credential comment correction. Fixed protected authority, closed policy/configuration/enrollment binding and legacy selection remain intact: `service/install_policy.py:296–371`, `service/host_install.py:707–840`, and `service/host_install.py:843–916`. Exact configured resources, tmpfs, roles, users/caps, input mounts and inherited historical topology validation remain in place. The narrow seven-source mount proof at lines 685–704 and the complete-container/source/process brackets remain unchanged. Default legacy topology validation remains separate from the explicit installed-policy/configuration branch in `service/recovery.py:2831`.

Credential descriptor open/fstat is allowed for metadata corroboration (`host_install.py:555–568`); credential contents are never read or hashed by `_installed_sources` (lines 661–682). The revised comment accurately corrects r205's overstrong no-open wording. Configuration/model reads and hashes remain explicit and bounded. No environment/credential-content projection, automatic policy migration, fallback authority, registration writer or activation was introduced.

## Independent checks

Read both conftests before execution. Ran only the five authorized modules through `/private/tmp/pab-0c-venv/bin/python -I -B`, from the candidate cwd, inserting the explicit absolute candidate and candidate/tests paths into `sys.path`:

`tests.service.test_installed_host_policy`, `tests.service.test_host_install`, `tests.service.test_install_ownership`, `tests.service.test_split_init`, `tests.service.test_install_child`.

**85 unittest tests PASS, exit 0, 9.610 seconds.** This includes the four failguard tests at `tests/service/test_installed_host_policy.py:346–395`, covering oversized lazy iteration, already-expired deadline, nested pending-directory reservation and expiration after one yield. Historical injected subcases are nested coverage, not additional unittest totals. The r207 RED-before evidence is implementer-reported; this review independently confirms GREEN and inspects the prior faulty expression, without modifying or executing a historical checkout.

Additional independent inert stdin probe, same isolated interpreter and explicit paths: injected finite directory iterators, stat identities and monotonic time; no actual model tree or large directory created. With empty synthetic bundle configuration, 4,094 and 4,095 child directories both returned successfully (4,095 and 4,096 total entries). All 4,095/4,096 scanners respectively opened and closed; source-stat counts were 8,190/8,192, confirming traversal plus final identity rechecks for every directory. At 4,096 children, refusal was `installed_source_budget` after exactly 4,096 yields, with the sole opened scanner closed. This directly checks exact-capacity acceptance, overflow refusal and context cleanup. Injected empty-bundle input is a private function boundary probe, not an accepted production enrollment.

`git diff --check 86e3fc6..HEAD` passed. Candidate tracked/untracked status remained clean. No source edits or commits.

## Limits and handoff

Only source inspection, authorized unittest fixtures and inert injected checks were performed. No actual Docker/proc qualification, privileged mount/UID operation, network/listener, ML/hardware, pytest runner/guard/full suite, HELD operation or unrelated cleanup. Only ordinary test-owned fixture cleanup occurred. Sole persistent write: `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-HOST-BOUNDS-REVIEW-r208-sol.md`.

Parent may adopt the exact carried observer plus correction source pins above. Protected provisioning, complete retained-registration acquisition, authenticated dropped-UID hello, protected provenance, control-UID registration writing and installed launch/activation remain parent-owned under the existing policy. Actual Linux permissions/setgid, mount equivalence, peer/process/namespace behavior, shutdown/reap/restart and CPU Kokoro speech still require separately authorized qualification. None is claimed here.
