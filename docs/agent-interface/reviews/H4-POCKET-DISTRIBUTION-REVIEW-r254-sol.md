# H4 Pocket distribution — independent r254 correction review

2026-10-24. **Bounded SOURCE GO for the r253 distribution correction; R252-1 CLOSED.** The real explicit source archive now carries the helper invoked by the supervisor and the two previously omitted Pocket engine modules. Independent release verification: **31 PASS, 3.150 s, exit0**. No additional blocking source defect found in this correction. Parent owns source adoption. This does not qualify actual installed execution, installation activation/dependencies, cross-UID/kernel enforcement, PID1/helper lifecycle or audio.

## Pins and review boundary

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution`, branch `t3/pocket-distribution-r253`, clean HEAD **5a604ace4d0c780e902e55adb94ee979accb409e**, based on guard-integrated **fccc616f2b17447c1f9692c54a32ab6abdfb72ad**. Repair **1979f60bef60b5463a0ca9c5b5affb0d111406f7** was integrated as **8331d411380d1ae337f73ab99f18a968125d2a70**; r251 report as **038cdb60fc69ae67c67212bcd392b74881a267d8**. Correction source **f6b776c39c7247f3891f0bfc171322c822caf224** changes exactly `headless-dist-manifest.json` and `tests/release/test_release_package.py` (89 insertions); HEAD adds the r253 report.

Read FULL r248 normative, r251 implementation, r252 independent review and r253 correction report. Inspected both correction files, archive producer, actual supervisor helper invocation, helper source, Pocket wrapper/execution binding, child imports and backend/package initializer dependency paths. No applicable AGENTS.md found in candidate/report trees or checked ancestors. Bounded memory registry query had no relevant hit and supplied no premise.

Independently compared all six repair files at historical source commit 1979f60 against integrated 8331d41 and final candidate HEAD: **byte-identical**, with empty Git content diffs and a separate direct `git show` bytes comparison. Files: `service/admission.py`, `service/recovery.py`, `tests/service/test_pocket_lifecycle.py`, `tests/worker/test_runtime_conditioning.py`, `worker/runtime_conditioning.py`, `worker/runtime_supervisor.py`. Integration history introduces no repair-source drift. The r252 82 + 87 + 3 PASS results remain prior review evidence, not rerun results here; no changed repair source or new failure justified repeating them.

## R252-1 closure and dependency assessment

`headless-dist-manifest.json:90–91,96` adds exactly `worker/engines/pocket.py`, `worker/engines/pocket_execution.py` and `worker/runtime_conditioning.py`. The manifest has **96 unique files**. `service/package_install.py:207–225` still reads only this explicit set, sorts it, validates source paths, fixes ZIP timestamps/modes and adds one generated launcher: **97 ZIP entries**. No glob, archive-builder change, installer/runtime change, dependency/lock change or activation change occurs in correction f6b776c.

The actual mandatory call remains `worker/runtime_supervisor.py:301–305`: `['python', '-m', 'worker.runtime_conditioning', stage, str(row['uid'])]`, with UID10001, allocated child GID, empty supplementary groups, closed FDs and DEVNULL streams. The added file supplies this module. Its imports are standard-library `os`, `stat`, `sys`; import alone performs no housekeeping. Existing helper behavior and lifecycle ordering are unchanged from r252.

`tests/release/test_release_package.py:266–348` builds through the real archive producer, checks exact membership/count, compares all three added modules' ZIP and extracted bytes to source, and imports from extraction in a separate `-I -B` interpreter whose cwd is outside the checkout. The archived supervisor AST is checked for the literal `-m worker.runtime_conditioning` target; `find_spec` and imported local-module origins must lie under extraction. This confirms helper invocation **module resolution**, without executing helper main, credential switching or removal. It does not claim a real supervisor-spawned helper run.

Independently inspected the test's dependency algorithm rather than relying on the report: its recursive AST walk starts at the three additions plus supervisor/child, visits function-local imports, resolves relative import levels, and adds existing local module prefixes/package initializers. Actual Pocket execution imports `.pocket_execution`, `worker.transport`, lazy `lib.tts_backends.pocket_tts`, and lazy `lib.tts_backends.contract`; backend dependencies include `.assets` and `.contract`. Those files and initializers are explicitly present. Child's lazy Pocket import is included. No missing current local Python dependency was identified. The walk also follows initializer branches; it does not establish arbitrary computed/plugin import completeness or availability of external runtime dependencies.

The extraction probe forbids imports of torch, torchaudio, numpy, soundfile, pocket_tts, kokoro, transformers, huggingface_hub and onnxruntime and passed. The current lazy backend path was inspected, not executed. Existing deterministic test at `tests/release/test_release_package.py:246–265` builds two real archives and compares their digests, checks every listed file's archived bytes, rejects cache/tests/Git entries, and executes extracted launcher help/import probes. Exact membership in the new test plus the unchanged explicit allowlist preserves exclusions: no credentials, model assets, retained runtime evidence or unrelated checkout files are added. The three added sources contain implementation code, not private runtime data. Existing redaction and injected transport tests also pass; this is bounded source/exclusion verification, not a general secret audit.

## Independent command and result

Inspected `tests/conftest.py` and `tests/service/conftest.py` before execution. The former installs an autouse pytest maintenance guard; the latter defines an HTTP listener fixture. Neither was collected or executed. No root/release-specific conftest exists. Ran once from candidate cwd:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B - <<'PY'
import sys, unittest
sys.path[:0] = [
    '/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution',
    '/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution/tests/service',
]
print(sys.version)
r = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromName('tests.release.test_release_package'))
sys.exit(not r.wasSuccessful())
PY
```

Python **3.14.7**, **31 tests PASS / 3.150 s / exit0**. This includes archive extraction/dependency checks, deterministic archives, CLI publication/redaction, topology and injected readiness/transport cases. Inspected listener/server/Docker seams: they are inert replacements; no real listener or Docker operation ran. Python3.11 runtime behavior was not tested. `git diff --check fccc616 HEAD` PASS; candidate remained clean and at the pinned HEAD.

Observed `/private/tmp` available **3,796,936 KiB**, below the 6-GiB actual gate. Only small inert fixtures and ordinary owned temporary-fixture teardown ran. No model/ML execution, Docker, network/download, listener, process-crash exercise, operational cleanup, HELD operation, delegation, source edit or commit. Only lasting write is this root report.

Parent may adopt the reviewed source separately. Actual cross-UID/kernel permissions, installed caps/mounts, authenticated transport, PID1 engine/helper reap, install activation/dependencies, resources and audio/listening remain unqualified. Nonempty interrupted conditioning snapshots and mid-flight restart remain **UNKNOWN** under r248; this correction supplies distribution files and does not expand that acceptance boundary.
