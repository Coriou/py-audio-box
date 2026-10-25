# r301 — native metadata singletons and EOCD count correction

2026-10-25. Partial SOURCE correction delivered UNMERGED, pending fresh parent
Sol independent review. No self-acceptance. Full Lot B, installed activation,
actual native Metal and release remain CLOSED / UNQUALIFIED. Known legacy digest
RED remains unwaived, unregenerated and unexecuted.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-6a2f2baf`, branch
`t3/r301-native-metadata-fixes`. Clean entry HEAD
`b3ff933dce1ba976dece7648f1b88c77c380e627`, root
`df0fafe2fe6a06925c46be4ae57a8c98af131ab4` plus review documentation.
Requested original package commits were UNMERGED at entry and cherry-picked:

- `dba2d33033c25f081a709c557fbbafa56fd49af8` → `8c5641727ecb773ed81824d3b7280cff43e70702`.
- `bf188fd99745fdc9335c1d3e82cb9cdfd584cd04` → `54af674c53ad93bd21eed5f881c29663ab7a1294`.
- `2bc64a01f9053f192a555f6f955d4248ce339cda` → `936b958fc3f610fa2bc8e210e680ab1f2c8f539f`.
- `e34e1a049360d5c0726184f9f2ddf4972cacb9f4` → `a322ee7825bf431d1fdadfdf9074074dd4a6dab7`.

Correction SOURCE: **`46c464152c107c34177bbc6f0607f659a19e57ba`**.
This report is a separate successor commit. Historical worktree
`t3-7618924b` was not written; final read-only check found it clean at
`e34e1a049360d5c0726184f9f2ddf4972cacb9f4`.

Read full r300/r299/r298/r297 and r277 including B/C. Inspected complete package
module/tests and both conftests before execution. No applicable AGENTS.md found
in worktree/ancestor discovery. Bounded memory registry search returned no
relevant hit; no memory technical facts used. No delegation.
Correction writes only `service/native_package.py`,
`tests/release/test_native_package.py`, necessary clarification in
`docs/agent-interface/NATIVE-PACKAGE.md`, plus this report. Original package
manifest writes are the explicitly requested cherry-picks.

SHA256 at correction SOURCE:

- `service/native_package.py`: `b1691db603941f232772969dc351442734df0e3688ecfae6e3252c3458c43258`.
- `tests/release/test_native_package.py`: `b597d9aef295870cf9c17eb7e98b4e2cd1d45ff5c63fa31321272381ea8a090b`.
- `docs/agent-interface/NATIVE-PACKAGE.md`: `fce447ccb3fdba60cb87d6412ef1fe727c35bf06ccc36ab32f5ed367b57afcff`.
- unchanged `native-dist-manifest.json`: `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`.
- unchanged `headless-dist-manifest.json`: `ae379794d324b19f3dd76063e78d1c92aef41b07536e3f9787cf608935b1477f`.
- unchanged `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
- r277: `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.

## Concrete correction

F1 P2: one small case-insensitive singleton helper uses email get_all before
consuming values. METADATA requires exactly one nonblank Name and Version,
at most one nonblank Requires-Python and Metadata-Version. Identical and
contradictory repeats both refuse, including mixed case and folded values.
A present Requires-Python must parse and admit the current Python. Installed
metadata uses the same helper for identity/constraint checks both while mapping
the installed distribution set and reading its graphs; version comes from the
validated metadata rather than a separate first-value accessor.

WHEEL Wheel-Version and Root-Is-Purelib also reject any duplicate. They remain
optional under this preflight's existing minimal-header contract; present values
must be `1.0` and `true`/`false`, respectively. This is not a complete wheel-format
validator. Repeated Tag and Requires-Dist remain allowed. Existing finite active
edge policy, exact wheel/installed graph equality and inactive valid markers
remain intact. No new metadata architecture or installation machinery.

Five new unittest methods cover wheel Name/Version/Requires-Python duplicates
across case, folded/nonfolded, identical/contradictory cases before active-edge
acceptance; missing Name/Version; supported/unsupported single Python constraint;
WHEEL singleton duplicates/invalid values; repeated tags and active/inactive
requirements; installed singleton consistency; and ZIP accounting below.
Existing fake installed metadata records now use actual email Messages with
Name and Version, matching the interface the production metadata parser uses.

F2 P3: ZIP preflight returns the declared EOCD count. Immediately after ZipFile
construction, both source and wheel paths compare actual infolist length with
that count and the respective 512/32768 ceiling, before member reads. The tiny
96-entry wheel and source ZIP each refuse patched equal disk/total declarations
of 0, 1, actual minus one and actual plus one; a zip_read spy proves no member
read occurs. Unmodified actual declarations pass. The wheel's 96 entries are
metadata plus empty inert members, not a large resource experiment.

Preconstructor whole-file/4 MiB central byte caps, classic single-disk/no-ZIP64
extent policy, bounded limit-plus-one member reads, exact source membership and
launcher prelength refusal remain. The constructor still parses a byte-bounded
central directory before the actual count comparison. This repairs bounded
accounting; it does not claim a preconstructor actual-record walk or a newly
unbounded allocation vulnerability.

## Execution receipt

One test interpreter invocation, no development test runs or reruns:
`PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -`.
Explicit sys.path prefix:

```
/Users/ben/.txcode/worktrees/py-audio-box/t3-6a2f2baf
/Users/ben/.txcode/worktrees/py-audio-box/t3-6a2f2baf/tests/service
```

Sequential `unittest.defaultTestLoader.loadTestsFromNames`, each with
`TextTestRunner(verbosity=2, failfast=True)`, exiting nonzero on failure:

1. `tests.release.test_native_package`: **27 PASS / 2.833s** (existing 22 plus five).
2. **26 PASS / 1.394s**:
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`,
   `tests.service.test_installed_host_policy.InstalledHost`,
   `tests.worker.test_native_protocol`.

Exit 0; fresh total **53 PASS**, each selection exactly once in the same isolated
interpreter. No pytest collection, maintenance autouse guard or HTTP listener
fixture activation. `git diff --check` PASS. No broader Lot A96 run or legacy
digest test/regeneration/waiver.

Observed available storage 3,146,408 KiB, below 6 GiB. Only tiny inert ZIP/fake
wheel/model metadata and existing fake TLS/process fixtures. No actual socket,
listener, native/model process, ML, hardware, Docker, network/download/install,
crash experiment, operational cleanup, HELD action or delegation. Disposable
fixture teardown only; actual-work headroom gate unchanged.

## Parent-owned remaining gates

Fresh independent Sol review of the exact correction SOURCE is required before
partial adoption. This implementation does not accept itself or complete B.
Native durable enrollment/startup/completed history, Docker control/mount/topology
policy and supported empty-catalogue onboarding remain separate B obligations.
Actual artifact acquisition/install provenance/import/ABI, TLS bridge/firewall,
Darwin process/FD/resource/death/crash observations, Metal probe/load/generate/
cancel/restart, audio/performance and clone lifetime remain unrun. CPU and CUDA
release qualification remain separate mandatory rows. Metadata/source evidence
cannot qualify installed binaries or native Metal. Legacy digest RED is unwaived.
