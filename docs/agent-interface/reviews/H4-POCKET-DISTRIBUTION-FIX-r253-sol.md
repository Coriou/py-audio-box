# H4 Pocket distribution omission — r253 Sol implementation

2026-10-24. **R252-1 corrected in source; parent fresh independent review/adoption remains.** The explicit headless archive now includes all three required Pocket modules. Real archive/extraction and isolated dependency tests pass. This is inert source-distribution evidence, not actual shipping, installer activation, installed execution, cross-UID/kernel or audio acceptance.

## Pins and scope

Fresh owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution`, branch `t3/pocket-distribution-r253`, based on guard-integrated root **fccc616f2b17447c1f9692c54a32ab6abdfb72ad**. Root/historical candidate were left unchanged. No guard commits were repeated.

Applied only requested historical repair **1979f60bef60b5463a0ca9c5b5affb0d111406f7** as **8331d411380d1ae337f73ab99f18a968125d2a70** and requested report **bc91810dd466e86c26138c9f5082bcb9e54d2233** as **038cdb60fc69ae67c67212bcd392b74881a267d8**, in that order. Historical source `/Users/ben/.txcode/worktrees/py-audio-box/t3-3133b78f` was only read. Distribution correction source commit **f6b776c39c7247f3891f0bfc171322c822caf224** changes only `headless-dist-manifest.json` and `tests/release/test_release_package.py`. This report is a separate subsequent commit.

Read FULL r251, root r252 review and r248 normative contract. No applicable AGENTS.md found in the worktree or checked ancestors. Bounded memory registry search had no relevant hit and supplied no premise. Prior r252 82 + 87 + 3 passing tests are prior review evidence only, not rerun or claimed as this lane's results.

## Correction and dependency coverage

Added exactly `worker/runtime_conditioning.py`, `worker/engines/pocket.py` and `worker/engines/pocket_execution.py` to the existing explicit file allowlist. No broad glob, archive-builder change, dependency/lock change, installer configuration/activation, helper/recovery/lifecycle change or test support file was needed.

The new release test calls real `service.package_install.archive` into a small temporary ZIP and extracts it. It checks exact archive membership against the manifest plus generated launcher, computing counts from the manifest rather than freezing the old 93-file count. Current computed values: **96 manifest files, 97 ZIP entries**. All three added files have exact source bytes in both ZIP and extracted tree. Existing deterministic archive test also compares every manifest file's ZIP bytes with source and checks exclusions; allowlist and secret/exclusion boundaries remain unchanged.

An AST dependency walk starts at all three new modules plus actual runtime supervisor and child. It follows local absolute/relative imports, including imports inside functions and package initializers, recursively against repository source and requires every discovered local module in the manifest. This includes Pocket's lazy `lib.tts_backends.pocket_tts` path and its local dependencies. No other missing required local source module was found in that closure. This source check covers explicit Python imports, not arbitrary future computed/plugin imports or external package availability.

A separate `-I -B` interpreter, cwd outside the checkout and with only the extracted package explicitly added to `sys.path`, imports the archived supervisor, child and all three new modules. It verifies local imported modules originate under the extracted package. It examines the archived supervisor AST for its literal `-m worker.runtime_conditioning` invocation and resolves/imports that exact helper from the archive. A meta-path guard forbids torch, torchaudio, numpy, soundfile, pocket_tts, kokoro, transformers, huggingface_hub and onnxruntime imports. It does not execute helper `__main__`, `remove_empty`, generation or backend loading. Thus invocation dependency resolution is covered without actual housekeeping or ML execution.

## Verification and limits

Inspected `tests/conftest.py` and `tests/service/conftest.py` before execution; no root or release-specific conftest exists. Used explicit unittest loading, not pytest collection or its autouse maintenance guard/listener fixture. Command from owned worktree:

```python
# PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -
import sys, unittest
sys.path[:0] = [
    '/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution',
    '/Users/ben/.txcode/worktrees/py-audio-box/r253-distribution/tests/service',
]
r = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.loadTestsFromName('tests.release.test_release_package'))
sys.exit(not r.wasSuccessful())
```

Final result **31 PASS, 3.276 s, exit0**. Earlier run before adding extracted-byte assertions: 31 PASS, 3.166 s. Final `git diff --check` PASS. Existing suite's Docker/HTTP/server observations use inert seams; no listener was opened. Only archive/help/import interpreter probes and small ordinary temporary fixtures ran. Observed `/private/tmp` available **3,801,116 KiB**, below 6-GiB actual gate. No Docker, model/download/network, process crash, operational cleanup, HELD operation, delegation, PR or merge. Normal owned temporary-fixture teardown only.

No new concrete helper/recovery/lifecycle defect was found. Parent must independently review the concrete correction before adoption. Installed configuration remains Kokoro-only; real mounts/caps/UID enforcement, PID1 helper lifecycle, installation activation/dependencies, runtime resources and audio remain separate gates. R248's nonempty interrupted snapshots and mid-flight restart remain conservative UNKNOWN.
