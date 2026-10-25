# Independent corrected partial native package review — r302 Sol

2026-10-25. **Bounded GO for parent consideration/adoption of this partial SOURCE package at the exact pins below.** No remaining material finding in the reviewed finite package scope. r298 graph/launcher findings and r300 singleton/count findings are corrected. Fresh independent final execution: **27 package + 26 regression = 53 PASS**, each selection once. This does not accept full Lot B, installed native activation, dependency ABI or actual Metal. Those remain CLOSED / UNQUALIFIED; known legacy digest RED remains unwaived. Parent adoption is separate.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-6a2f2baf`, branch `t3/r301-native-metadata-fixes`, clean at entry and after execution. Verified docs HEAD `90b1bf8e17270cebaab30643b65daa7ce87d3f60`; parent `b3ff933dce1ba976dece7648f1b88c77c380e627`; original package `8c5641727ecb773ed81824d3b7280cff43e70702`, report `54af674c53ad93bd21eed5f881c29663ab7a1294`; graph/ZIP correction `936b958fc3f610fa2bc8e210e680ab1f2c8f539f`, report `a322ee7825bf431d1fdadfdf9074074dd4a6dab7`; metadata/count SOURCE `46c464152c107c34177bbc6f0607f659a19e57ba`. The resolving commit chain matches the request. Base-to-HEAD `git diff --check` PASS.

Verified SHA256:

- `service/native_package.py`: `b1691db603941f232772969dc351442734df0e3688ecfae6e3252c3458c43258`.
- `tests/release/test_native_package.py`: `b597d9aef295870cf9c17eb7e98b4e2cd1d45ff5c63fa31321272381ea8a090b`.
- `docs/agent-interface/NATIVE-PACKAGE.md`: `fce447ccb3fdba60cb87d6412ef1fe727c35bf06ccc36ab32f5ed367b57afcff`.
- `native-dist-manifest.json`: `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`.
- `headless-dist-manifest.json`: `ae379794d324b19f3dd76063e78d1c92aef41b07536e3f9787cf608935b1477f`.
- unchanged `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read r277 including full B/C and r297–r301, complete package code/tests/docs/manifests, correction diff and both conftests. Inspected actual no-follow path walker, NativeRuntime initialization/hello/RPC and TLS context/pin/frame consumers. No applicable AGENTS.md found in repository discovery or checked ancestors. Bounded memory registry search returned no relevant hit; no memory technical facts used. No delegation, candidate edit, commit or self-adoption. Sole lasting write is this report in the root checkout.

## Disposition of material findings

**r300 F1 P2 closed.** `service/native_package.py:108–130` uses case-insensitive `get_all`, requiring exactly one nonblank Name/Version and at most one nonblank Requires-Python/Metadata-Version. Identical and contradictory duplicates refuse. Present Python constraints parse and admit the current interpreter. Installed mapping and graph inspection both use this helper (`:133–158`), with the validated Version rather than a separate first-value accessor. Selected wheels use it at `:239`; WHEEL singleton checks at `:240–243` retain the explicitly optional minimal-header contract, validating present Wheel-Version/Root-Is-Purelib. Repeated Tag/Requires-Dist remain allowed. This is bounded identity/compatibility validation, not a universal metadata-format validator.

**r300 F2 P3 closed as bounded accounting.** Preflight `:182–202` returns declared EOCD count under whole-file/central-byte limits. `zip_count` at `:205–210` compares actual infolist length with declaration and actual ceiling. Wheel invokes it at `:232`, source at `:339`, before member reads. The five added methods include both paths with equal disk/total declarations patched to 0, 1, actual−1 and actual+1; zip_read spies establish refusal before member reads, and original counts pass. The constructor still parses the byte-bounded central directory before actual count comparison. No preconstructor actual-record walk is claimed.

**r298 F1/F2 remain closed.** Shared `active_edges` at `:161–179` parses finite requirements and validates active absent/incompatible/URL/extra edges; inactive valid markers carry no edges. Runtime `:285–286` requires exact canonical installed/wheel active graph equality. ZIP source/wheel ceilings are 64 MiB/4 GiB, central bytes 4 MiB, entries 512/32768. Launcher exact declared size is checked at `:351` before bounded reading; all package member reads use limit-plus-one `zip_read`. No unchecked full launcher inflation remains.

Additional independent probes ran once, separately from unittest counts. Using the existing tiny NativePackageTests fixture and actual wheel/runtime validation, identical mixed-case Metadata-Version duplicates, contradictory Metadata-Version duplicates, blank Metadata-Version, blank Requires-Python and invalid `Requires-Python: nonsense` all refused (four `native_package_metadata_singleton`, one `native_package_python_constraint`). Direct actual metadata_identity calls refused blank Name and Version. A tiny synthetic infolist containing 32769 references with matching declaration refused `native_package_zip_bounds`, exercising the actual wheel-count ceiling without constructing a large archive. **Eight expected refusals, exit 0.** No counterexample accepted. Existing final tests supply the mixed-case/folded Name/Version/Python duplicate, installed-consistency and real ZIP count reproductions; these were not rerun as probes.

## Preserved package boundaries

Private config/trust reads retain no-follow component walking, bounded stable single-link owner/mode checks and private immediate roots. Canonical documents reject duplicate/nonfinite JSON. TLS subsequently reopens checked paths under the explicit immutable operator-input/no-hostile-same-UID premise; this is not FD-pinned loading or hostile-code containment. `preflight` at `:355` checks executable/build/platform/hash, artifact hashes, exact source archive, model mapping and actual bundle verifier. `serve_checked` at `:423` consumes the checked snapshot, rejects duplicate registration bindings and hands off to actual supervisor classes. No new source changes touch Lot A runtime ownership/publication modules or Linux policy.

`enrolled_runtime` at `:396` constructs actual NativeRuntime, checks every configured native identity including boot against healthy hello, and refuses quarantine/closed import admission. Actual TLS1.3/CERT_REQUIRED and frame peer pins remain in `worker/native_protocol.py:168–194`; NativeRuntime retains numeric endpoint and local-import UID checks. Fake connector evidence exercises real framing/dispatch without a network socket. This private consumer is not durable installed enrollment or readiness.

Exact archive membership/bytes, secret and lock exclusion, deterministic archive bytes, extracted verification and launcher help pass. Metadata-only fake wheels and injected Darwin identity remain nonqualifying. Prepared/metadata-verified output explicitly keeps ABI unrun, qualification unqualified and speech unavailable. Hashes/tags/graphs do not prove selected wheel installation provenance, installed binary identity, ABI or upstream no-extra-process behavior.

## Independent execution receipt

Inspected `tests/conftest.py` maintenance autouse guard and `tests/service/conftest.py` HTTP socket/uvicorn fixture before execution. Direct unittest loading avoided pytest collection and fixture activation. One final interpreter invocation:

```
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -
```

Candidate cwd; explicit sys.path prefix was candidate root followed by candidate `tests/service`. Sequential `unittest.defaultTestLoader.loadTestsFromNames`, each `TextTestRunner(verbosity=2, failfast=True)`, exit nonzero on failure:

1. `tests.release.test_native_package`: **27 PASS / 2.667s**.
2. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`, `tests.service.test_installed_host_policy.InstalledHost`, `tests.worker.test_native_protocol`: **26 PASS / 1.387s**.

Exit 0. Neither selection rerun; no development test run in this review. Separate eight-probe invocation exit 0, not additional unittest methods. No broad Lot A run or legacy digest test/regeneration/waiver. Candidate remained clean.

Available space observed 3,147,232 KiB, below 6 GiB gate. Only tiny inert ZIP/fake wheel/model metadata and existing fake TLS/process fixtures, plus a small in-memory count fixture. No actual socket/listener, native/model process launch, ML/hardware, Docker, network/download/install, crash experiment, operational cleanup, HELD action or delegation. Disposable fixture teardown only. Actual-work headroom gates were neither exercised nor weakened.

## Parent-owned remaining gates

Parent may separately adopt the exact partial source; this review performs no merge or acceptance of full B. Native durable enrollment, owner/startup/completed-history integration, Docker control/mount/topology policy and supported empty-catalogue Qwen onboarding remain required B work. Real artifact acquisition/install provenance/import/ABI, bridge/firewall/TLS peers, Darwin process/FD/resource/death/crash observations, Metal probe/load/generate/cancel/restart, clone lifetime, audio/performance and CPU/CUDA release rows remain mandatory and unqualified here. Actual native Metal stays UNAVAILABLE / UNQUALIFIED. Known legacy digest RED remains unwaived.
