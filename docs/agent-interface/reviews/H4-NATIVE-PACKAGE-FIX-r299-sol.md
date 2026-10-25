# r299 — partial native package F1/F2 correction

2026-10-25. SOURCE correction delivered, UNMERGED and pending fresh parent Sol
independent review. No self-acceptance. Full Lot B, installed activation, actual
native Metal and release remain CLOSED / UNQUALIFIED. Known legacy digest RED
remains unwaived, unregenerated and unexecuted.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b`, branch
`t3/r298-native-package-fixes`, clean entry HEAD
`1580c6aea48c18796fd4f50636b18074ebfac551`: fresh base
`df0fafe2fe6a06925c46be4ae57a8c98af131ab4` plus review documentation.
Cherry-picked requested source `dba2d33033c25f081a709c557fbbafa56fd49af8`
as `c60f07c`, then report `bf188fd99745fdc9335c1d3e82cb9cdfd584cd04`
as `ddd7b2d`. Historical t3-0d06d538 was not changed.
Correction SOURCE commit: **`2bc64a01f9053f192a555f6f955d4248ce339cda`**.
This report is a separate successor commit.

Read full r298 review, r297 report and r277 contract including Lot B/C. Inspected
both conftests before tests; no applicable AGENTS.md found in worktree discovery.
Bounded memory registry search had no relevant hit; no memory technical facts used.
Only correction writes: `service/native_package.py`,
`tests/release/test_native_package.py`, `docs/agent-interface/NATIVE-PACKAGE.md`;
plus this report. No install/enrollment/Core/Lot A changes.

SHA256 at correction SOURCE:

- native_package.py: `5b436d3332abe032376718e7d8ffee6054e7b12b2fe57b04591a9d6d9e8c4583`
- test_native_package.py: `bcfe4ae1ba74e6a7a0dec2561aa942e035b14fece1462274054cd24a24835ab8`
- NATIVE-PACKAGE.md: `24a1fd9aa9eda2fe7aefd04bb8020905fafb5054d52c2e3d0d6711520c5613e2`
- unchanged native-dist-manifest.json: `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`
- unchanged headless-dist-manifest.json: `ae379794d324b19f3dd76063e78d1c92aef41b07536e3f9787cf608935b1477f`
- unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`
- r277: `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`

## Corrections and concrete limits

F1: both installed and selected-wheel Requires-Dist use the same active_edges
validator: at most 512 requirement strings of at most 4096 characters each;
parse valid markers before evaluating with no selected extras. Valid inactive
requirements carry no edge; malformed markers refuse. Active URL, extra, absent
distribution and incompatible exact-version edges refuse. Installed and artifact
canonical active edge sets (normalized name, canonical specifier string) must
agree per distribution. Both must close against the exact declared installed set;
merely both being independently satisfiable is insufficient. Differently expressed
but logically equivalent constraints may conservatively refuse. Duplicate active
edges collapse as sets. Requires-Python and wheel identity/tag checks remain.
Tests exercise absent/incompatible/URL/extras/bad-marker edges, valid inactive
markers, canonical agreement and contradictory but independently closed graphs.
No download or URL dereference occurs.

F2: preflight reads only the bounded ZIP end region before ZipFile construction.
Classic single-disk source ZIP ceiling: 64 MiB and 512 declared entries; wheel:
4 GiB and 32768 declared entries; central directory ceiling: 4 MiB. Directory
extent must meet the end record exactly; ZIP64 and trailing data are unsupported.
Even a lying declared count cannot cause unbounded constructor input: central
bytes and whole archive bytes are finite. This is a small package-specific bound,
not a general archive security framework or hostile same-UID defense.
Launcher declared size must equal generated LAUNCHER length before member open;
ordinary source members retain exact local-size checks. All member reads now use
ZipExtFile bounded reads (limit plus one), never full ZipFile.read inflation.
WHEEL/METADATA retain 65536-byte declared/output limits. Tiny launcher and metadata
oversize cases assert the bounded read helper is never entered for those members;
central accounting tests assert constructor never entered. Exact source-byte and
membership checks, secret exclusion and deterministic archive bytes pass.

Operator immutable input/private-owner premise remains. Metadata verification
proves neither selected wheel installation, installed binary identity/ABI, source
authenticity, upstream no-extra-process behavior nor actual readiness.

## Executions

All project test interpreters: `/private/tmp/pab-0c-venv/bin/python -I -B`.
Explicit sys.path prefix in every invocation:

```
/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b
/Users/ben/.txcode/worktrees/py-audio-box/t3-7618924b/tests/service
```

Direct `unittest.defaultTestLoader.loadTestsFromNames`,
`TextTestRunner(verbosity=2, failfast=True)`; nonzero exit on failure.
No pytest collection, maintenance autouse guard or HTTP listener fixture.

Development invocations retained honestly:

1. Package run stopped after 5 methods / 0.751s, exit 1: existing negative fake
   distribution lacked metadata Name newly required for graph mapping. Updated
   fake record; no production refusal bypass.
2. Package run stopped after 7 methods / 0.877s, exit 1: new purported inactive
   URL fixture lacked required whitespace before marker semicolon. Corrected
   fixture to valid requirement syntax; malformed input correctly refused.
3. Final package source run, exactly once after corrections: **22 PASS / 2.933s,
   exit 0**, `tests.release.test_native_package` (original 15 plus seven new).
4. Final specified regression selection, exactly once: **26 PASS / 1.420s,
   exit 0**:
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`,
   `tests.service.test_installed_host_policy.InstalledHost`,
   `tests.worker.test_native_protocol`.

Final passing total 48 methods; development counts overlap and are not added.
`git diff --check` PASS. No broader Lot A96 run or claim.
Observed free space 3,190,964 KiB, below 6 GiB gate: only tiny inert ZIP/fake wheel,
model metadata and existing fake TLS/process fixtures. No actual socket/listener,
model process, ML, hardware, Docker, network/download/install, crash experiment,
operational cleanup, HELD operation or delegation. Only disposable fixture teardown.
Actual-work headroom gates unchanged.

Parent must perform fresh independent Sol review against these exact pins before
adoption. Remaining B enrollment/history/control packaging/onboarding and actual
artifact acquisition/install/import/ABI, native TLS/firewall/process/FD/resource,
Metal probe/load/generate/cancel/restart and clone lifetime qualification remain
unrun. CPU/CUDA release rows remain separate mandatory gates. Source evidence does
not close native Metal or full B; legacy digest RED remains unwaived.
