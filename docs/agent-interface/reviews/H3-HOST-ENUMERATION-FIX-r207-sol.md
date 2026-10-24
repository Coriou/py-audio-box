# H3 r207 Sol — R206-1 model enumeration bounds

2026-10-24. Implementation complete; independent review is next and source adoption remains parent-owned. No operational or actual qualification GO.

## Exact pins and scope

Fresh own worktree `/Users/ben/.txcode/worktrees/py-audio-box/r207-observer-bounds`, branch `fix/r207-observer-bounds`, created from current root `/Users/ben/Projects/py-audio-box` HEAD `86e3fc6e8c5de8bf7a9bbfa1a388a58763bb70e4`. Cherry-picked only source `4f282794b2b81e62d16ac7bf705b19db8f9c3dd5`, producing `bbe4066ef0eed544859b53fa7c985c50cf32e11a`. Clean candidate before edits. Correction source commit `306089b20c795973f6c14ea0dc84374f43de5eea`, direct parent that cherry-pick. Source delta contains only `service/host_install.py` and `tests/service/test_installed_host_policy.py`. This report is committed separately; its identical copy is the sole root write. No source adoption.

Read root H3-INSTALLED-HOST-r205-sol.md, H3-INSTALLED-HOST-REVIEW-r206-sol.md (the actual filename of the requested r206 review), and full normative H3-INSTALL-OWNERSHIP-r198-astra.md. No applicable AGENTS.md found; bounded memory registry search returned no relevant hit and supplied no premise. No delegation.

## Correction

`_model_tree` checks the shared deadline before directory descriptor traversal, again before opening scandir, before every iterator advance, and after each yielded entry before append. It collects names into a bounded list and sorts only that list. One global discovery counter includes the root and every child immediately, including pending directories; nested scans cannot reuse another pending directory's reserved allowance. At most 4,096 tree entries are admitted, including the root. An overflow probe may consume one additional iterator entry but never append it; exhaustion immediately refuses. No full materialization of an oversized iterator.

Existing per-file 512 MiB, aggregate 2 GiB, manifest bounds, traversal structure, ownership/mode/no-follow checks, hashes, final file/directory identity rechecks and closed policy/legacy behavior remain unchanged. No broad refactor or depth/traversal relaxation.

Credential wording correction: read-only descriptor open and fstat are allowed for metadata corroboration. Credential contents are never read or hashed. Corrected the misleading source comment; this report supersedes any r205 wording claiming no open. No credential value appears here.

## Red/green verification

Read tests/conftest.py and tests/service/conftest.py. On the own candidate before the production edit, ran the four new ModelEnumerationBounds tests with the isolated interpreter and explicit absolute candidate/test paths. All four failed by hitting their iterator consumption failguards, exit 1 (4 tests, 0.045 seconds). No actual large/infinite directory was created.

After the fix ran exactly:

```
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/r207-observer-bounds","/Users/ben/.txcode/worktrees/py-audio-box/r207-observer-bounds/tests"];s=unittest.defaultTestLoader.loadTestsFromNames(["tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"]);r=unittest.TextTestRunner().run(s);sys.exit(not r.wasSuccessful())'
```

**85 tests PASS, exit 0, 9.855 seconds**: existing 81 plus four enumeration regressions. Historical injected subcases remain nested coverage, not additional unittest totals. `git diff --check` passed.

New failguard-backed lazy iterator assertions: oversized scan stops after 4,096 yielded names (root plus 4,095 admitted children); already-expired deadline opens no scanner and consumes zero entries; root discovers two pending directories and the next scan stops after 4,094 yields (only 4,093 admitted, respecting both pending reservations); deadline expiration during scanning stops after one yield. Assertions check exact consumption/opened directories rather than merely eventual ValueError. Tests inject metadata, directory iteration and monotonic time; they are source evidence, not actual host qualification.

## Limits and handoff

No actual Docker/proc qualification, privileged mount/UID operation, network/listener, ML/hardware, pytest runner/guard/full suite, HELD operation, delegation or self-adoption. Only normal test-owned fixture cleanup. Synchronous filesystem calls themselves are not preempted; checks bound cooperative enumeration between calls. No claim of actual Linux permission, mount, process, speech or release acceptance.

Independent review of the exact correction pin is next. Protected provisioning, retained-registration acquisition, dropped-UID hello, provenance, original registration writer, installed launch/activation and separately authorized actual qualification remain parent-owned and closed under the existing policy.
