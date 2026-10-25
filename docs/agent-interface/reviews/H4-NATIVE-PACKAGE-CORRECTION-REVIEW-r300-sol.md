# Independent corrected partial native package review — r300 Sol

2026-10-25. **NO-GO for parent adoption of this partial SOURCE candidate: F1 (P2) below remains a metadata-consistency blocker.** The original r298 active dependency graph and unchecked launcher inflation paths are corrected. Fresh final tests: **22 package + 26 regression = 48 PASS**, each selection exactly once. A small independent malformed wheel still passes contradictory singleton metadata. F2 records a bounded ZIP accounting discrepancy; it is not an unbounded-allocation claim. Full Lot B, installed activation, actual native Metal and release remain CLOSED / UNQUALIFIED; known legacy digest RED remains unwaived.

## Authority and exact pins

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b`, branch `t3/r298-native-package-fixes`, clean at entry and after execution. Resolving full docs HEAD `e34e1a049360d5c0726184f9f2ddf4972cacb9f4`; parent `1580c6aea48c18796fd4f50636b18074ebfac551`; original adopted source `c60f07c6b94b8ce4cc3f8aedc95107f6f5249b75`; adopted report `ddd7b2dce6237a4beea700691dfb1ab425665061`; correction SOURCE `2bc64a01f9053f192a555f6f955d4248ce339cda`. Correction-to-docs diff contains only r299. Base-to-HEAD `git diff --check` PASS.

Verified SHA256:

- `service/native_package.py`: `5b436d3332abe032376718e7d8ffee6054e7b12b2fe57b04591a9d6d9e8c4583`.
- `tests/release/test_native_package.py`: `bcfe4ae1ba74e6a7a0dec2561aa942e035b14fece1462274054cd24a24835ab8`.
- `docs/agent-interface/NATIVE-PACKAGE.md`: `24a1fd9aa9eda2fe7aefd04bb8020905fafb5054d52c2e3d0d6711520c5613e2`.
- `native-dist-manifest.json`: `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`.
- `headless-dist-manifest.json`: `ae379794d324b19f3dd76063e78d1c92aef41b07536e3f9787cf608935b1477f`.
- unchanged `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read full r277 (including full B/C), r297/r298/r299, complete package module, manifest/docs and package tests, correction diff, both conftests, and relevant actual install path walker, NativeRuntime construction and TLS/pin implementation. No applicable AGENTS.md found in candidate discovery or checked ancestors. Bounded memory registry search had no relevant hit; no memory technical facts used. No delegation, source edits, commits or self-acceptance. Sole lasting write is this report in the root checkout.

## Findings and inert reproductions

**F1 — P2, `service/native_package.py:204–212`: duplicate singleton METADATA fields silently select the first value.** `BytesParser` permits repeated headers, while `info['Name']`, `info['Version']` and `info['Requires-Python']` read only the first. Thus contradictory selected artifact identity/Python constraints can be accepted as metadata-verified even though all declared constraints have not been checked. This is independent of ABI, actual installation provenance or hostile writers.

One inert probe invocation used `NativePackageTests.setUp`, its tiny metadata-only wheel and `rewrite_metadata`, then actual `runtime_manifest` via the existing `validate` fixture (only platform identity and installed empty edge graphs injected). Artifact hashing, selected wheel parsing and source archive verification ran. Appending each following header sequence to the fixture's existing `Name: packaging` / `Version: 1` independently produced acceptance:

```
Requires-Python: >=0
Requires-Python: >=99
```

Accepted manifest digest `285bc0d19662974384cfe44ea1fae41ce9ff34265948c98af94fa59db639759d`.

```
Name: contradictory
```

Accepted digest `3389e03e9f18ba9766782bf1d0c80a9a75540db0dbe32bf16311d396f059542f`.

```
Version: NaN
```

Accepted digest `b6c0c17fc9a785d866105e14a5f59196d64f61cebc2e80d8a81a71137a31fe1b`. This last case is an ignored duplicate textual header, not acceptance of a NaN declared distribution version or nonfinite JSON.

Required bounded correction: reject duplicate singleton identity/compatibility headers before consuming them; require exactly one Name/Version and at most one Requires-Python, validate the selected values, and retain legitimate repeated Requires-Dist/Tag fields. Add tiny negative fixtures for contradictory duplicates. No giant metadata framework or actual install is needed.

**F2 — P3, `service/native_package.py:169–176`, wheel constructor at `:199`: EOCD counts are compared only to each other, not actual central records.** In the same probe invocation, the exact generated source ZIP was copied to a tiny fixture and both little-endian EOCD count fields (offsets +8 and +10) changed to zero, leaving offsets, central bytes and member bytes intact. Actual `verify_archive` accepted. Therefore declared count is not authenticated against parsed count. For wheels, there is no later actual entry-count cap; the 32768 ceiling can be bypassed by false counts. Source verification still enforces exact actual membership.

The 4 MiB central-directory and whole-file ceilings remain effective, so original unbounded constructor allocation is fixed. Do not describe this as the original unbounded F2 recurring. A small bounded central-record walk before construction could check actual count/extent against EOCD and the finite ceiling; alternatively accurately narrow the declared-count contract, but that would not enforce the requested actual wheel entry limit. No large archive was constructed. Independent shifted-offset, unequal disk-count and trailing-byte fixtures all refused `native_package_zip_bounds`.

Additional probe observations, not defects: 513 requirement strings and an over-4096-character requirement refused. A valid marker `python_version < "NaN"` evaluated inactive and produced no edge: marker operands are strings, not JSON nonfinite numbers. A blank line followed by apparent Requires-Dist text was accepted as an empty graph; that text is the RFC-style description body, not a hidden header, so this is not evidence of a dependency bypass. JSON duplicate/nonfinite refusal is covered by the passing package method. Finite canonical document input bounds malformed types; helper type errors fail closed through CLI redaction, with no success receipt. No claim of universal malformed-input reason-code normalization.

## Corrected and preserved behavior

`active_edges` at `:136–154` is shared by installed metadata and selected wheels; caps 512 strings/4096 characters, parses before marker evaluation with no extras selected, refuses active URL/extras/absent/incompatible edges, normalizes names/specifiers, and returns sets. Runtime manifest `:245–249` requires exact equality to each installed graph, not merely separate satisfiability. Valid inactive markers are ignored; differently expressed equivalent constraints may conservatively refuse. Existing wheel filename/version/tag and Requires-Python checks remain, subject to F1's duplicate ambiguity.

ZIP preflight reads at most 65557 tail bytes before ZipFile; classic single-disk extent/comment/trailing/size checks enforce source 64 MiB, wheels 4 GiB and central 4 MiB. ZIP64 records create a nonmatching classic central extent and sentinel offsets refuse. Declared entry ceilings are 512/32768, subject to F2. Metadata declared/output limit is 65536 bytes. `zip_read` uses ZipExtFile limit+1 reads. Launcher exact declared size is checked before open; ordinary members match local size before read. No unchecked launcher full inflation remains. Bounded rejection of a valid comment containing an EOCD signature is conservative availability behavior, not acceptance authority.

Private documents/trust retain no-follow component walking, fixed owner/mode/single-link/stable identity and private immediate roots. Executable/artifact hashes and exact current Python build/platform checks remain. Subsequent SSL/path reopening relies on the documented immutable operator inputs and no hostile same-UID writer premise; this is not FD-pinned loading or hostile-code containment. `serve_checked` consumes the checked snapshot, rejects duplicate registration bindings and uses actual supervisor classes. `enrolled_runtime` retains all five configured identities, exact boot hello equality, quarantine/import-admission refusal and real TLS pin enforcement; it does not learn boot or grant installed readiness. Fake connector tests exercise actual framing/dispatch without networking.

Archive allowlist/exact membership, secret/lock exclusion, deterministic bytes, extracted package verification and launcher help PASS. This is source dependency membership evidence, not a complete installed import/ABI experiment. No dependency acquisition or actual ABI claim is accepted. Lot A production modules, Linux policies, Core publication and runtime ownership fences were not changed by the correction.

## Independent executions and limits

Before execution, inspected `tests/conftest.py` (maintenance autouse guard) and `tests/service/conftest.py` (HTTP socket/uvicorn fixture). Used direct unittest loading, no pytest collection or either fixture activation. Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B`, `PYTHONDONTWRITEBYTECODE=1`, candidate cwd, explicit sys.path prefix:

```
/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b
/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b/tests/service
```

One interpreter invocation, sequential selections with `TextTestRunner(verbosity=2, failfast=True)`, exit nonzero on failure:

1. `tests.release.test_native_package`: **22 PASS / 2.050s**.
2. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`, `tests.service.test_installed_host_policy.InstalledHost`, `tests.worker.test_native_protocol`: **26 PASS / 1.351s**.

Exit 0. Neither selection rerun. Separate small probe invocation exit 0, observations above; not additional unittest methods. Implementation r299 results are historical, not counted. No broader Lot A run or legacy digest run/regeneration/waiver.

Available storage observed 3,195,816 KiB, below 6 GiB. Only tiny inert ZIP/fake wheels and existing fake model/TLS/process fixtures used. No actual socket/listener, process launch, ML, hardware, Docker, network/download/install, crash, operational cleanup, HELD action or delegation. Disposable fixture teardown only. Actual-work headroom gate unchanged.

## Parent-owned disposition

Correct F1 and resolve the bounded F2 accounting contract, then independently review the exact successor before partial SOURCE adoption. Installed native enrollment/history/control/mount policy and supported empty-catalogue onboarding remain separately required for full B; their explicit absence is not a defect against this partial scope. Real artifact acquisition/install provenance/import/ABI, TLS bridge/firewall, Darwin process/FD/resource/death/crash observations, Metal probe/load/generate/cancel/restart, clone lifetime/audio/performance and CPU/CUDA release qualification remain separate mandatory gates. Actual native Metal remains UNAVAILABLE / UNQUALIFIED. Full B and release are not accepted; legacy digest RED remains unwaived.
