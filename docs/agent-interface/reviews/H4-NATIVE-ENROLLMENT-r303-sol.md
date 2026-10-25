# Native durable enrollment SOURCE preparation — r303 Sol

2026-10-25. Implementation candidate for fresh independent review; **no self-adoption and no full Lot B completion**. Durable private enrollment and exact startup-input reopen are implemented. Installed owner activation/history remains CLOSED. Metal remains UNAVAILABLE / UNQUALIFIED.

## Authority and exact pins

Isolated cwd `/Users/ben/.txcode/worktrees/py-audio-box/t3-b05cd780`, branch `t3/native-enrollment-startup`. Entry/base HEAD `a2e07de7ce91d7503adf2d61398b22cb2f48ea3c` (reviewed partial package integration), clean at entry. SOURCE commit `8a402214afca47528af8670198f52ab18c8114ff`. This report is committed separately afterward; final handoff supplies report HEAD. No root checkout writes, merge, push or delegation.

Read PRODUCT.md, NATIVE-PACKAGE.md, r277 native contract including remaining B/C, r296 validation supplement, r302 review; inspected host_install/entrypoint/recovery startup and native_package/native_runtime/context/Dispatcher seams. No repository AGENTS.md found or file in checked ancestors. Bounded memory registry search supplied evidence-boundary conventions only; source decisions use this checkout.

SOURCE SHA256:

- `service/native_package.py`: `cea1ed22f3515ce02a56842baa3d7e3318f7ec19ed52e3c4c840de6ddd3dbda3`.
- `tests/release/test_native_enrollment.py`: `81c0cefc0ef9c519244085c7c5feb86d8d0268752f2d96a80480932f321eaee5`.
- `docs/agent-interface/NATIVE-PACKAGE.md`: `9322ead10586ecbc8e09b00dc043a757ca724192c18837256cc6600ae04078fd`.
- Unmodified legacy `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; known legacy-controls digest RED remains unwaived, not regenerated or tested here.

## Finite implementation

Existing native_package module now has explicit private `enroll_native` and `native_startup_inputs` functions. No new public registration endpoint, generic runtime policy, scheduler, job database, readiness projection or installed command. Existing named source manifests already include this module; no new runtime file needs archive adoption.

Enrollment accepts an explicit record destination, existing owner directory, control configuration and canonical template document. It instantiates actual NativeQwenContexts from bounded closed registration rows, checks duplicate principal/profile/version keys, native tuple, execution/model/immutable voice metadata, frozen plan/admission and current native hardware snapshot, and reuses finite_budget. Actual model bytes remain the native supervisor's responsibility. Template inventory provenance is operator-supplied; fixture readiness metadata establishes no product qualification.

Write-once record binds owner path/device/inode/UID; literal configuration/template paths and canonical hashes; each CA/client certificate/client key byte hash; exact deployment/configuration/lane/boot/runtime-manifest tuple. Existing enrolled_runtime supplies real NativeRuntime construction, TLS pin and exact healthy hello checking, including configured boot and retained-import admission. Repeated snapshots bracket hello. Record uses O_EXCL/O_NOFOLLOW 0600, file fsync and parent fsync. Existing destination refuses even on an identical retry. Failed/partial writes remain blocking evidence; no repair or cleanup algorithm.

Startup preparation privately reopens the canonical durable record, requires exact expected binding before TLS I/O, reconstructs actual runtime and contexts, then repeats input/record checks. It returns runtime, contexts, profiles and budget only. Changed/missing/malformed/quarantined state refuses; boot is never learned from hello. The trusted immutable operator-input/no hostile same-UID writer premise remains. The bracket detects observed changes; it is not a filesystem transaction or protection against hostile same-UID ABA replacement.

This deliberately stops before Dispatcher/Core acquisition. Neither function reads or blesses completed history, changes Linux enrollment, or grants persistent activation. No Core object, SQLite connection or callback crosses to TLS. Existing owner-thread persistence, publication/import-intent uncertainty fences and Linux installed readers are untouched. Preparation must not be mistaken for an installed activation credential.

## Execution receipt

Inspected tests/conftest.py maintenance autouse guard and tests/service/conftest.py socket/uvicorn fixtures before direct unittest loading. Interpreter `/private/tmp/pab-0c-venv/bin/python`, observed full version `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`; **not target Python 3.11**. Every test invocation used PYTHONDONTWRITEBYTECODE=1 and `-I -B`, explicit source import path, direct unittest, verbosity 2 and failfast. No pytest collection.

Development selection: initial six enrollment tests, **6 PASS / 0.586s**, exit 0. Then added bracket mutation and uncertain-spool cases; no source edits after final execution, only documentation.

Final single invocation: sys.path prefix `['.','tests/service']`, defaultTestLoader.loadTestsFromNames with this exact allowlist:

1. `tests.release.test_native_enrollment`: 8 tests.
2. `tests.release.test_native_package`: 27 tests.
3. `tests.service.test_installed_host_policy.InstalledHost`: 20 tests.
4. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`: 1 test.

**56 PASS / 4.833s, exit 0**. Tool output is the receipt; no external log file was created. `git diff --check` passed before SOURCE commit. No broad Lot A or timing/crash repeat. Archive help runs in-process with runpy, not an actual child.

New tests exercise actual NativeRuntime codec/handler fake TLS and NativeQwenContexts: durable exact reopen and context resolution; write-once refusal; missing/malformed/duplicate-key retained record without repair; changed key/control/template/owner incarnation before TLS; wrong hello boot and quarantine without updating record; changed native registration and duplicate row refusal; record mode/symlink/hardlink refusal; changed trust during hello at enrollment and reopen; uncertain retained spool evidence blocks reopen without deleting it. Existing package and Linux host-policy selections pass. These fixtures are seeded/inert, never clean installed onboarding.

Disk observed approximately 3 GiB at entry and 3,103,372 KiB available after SOURCE commit, below 6 GiB actual-work admission. Only tiny inert ZIP/fake wheel/model/certificate and injected runtime fixtures used. No Docker/network/download/install/listener/actual child or ML process/crash experiment; no operational cleanup, HELD action or STT changes. Test-owned temporary fixture teardown only. Actual-work disk gate is unchanged.

## Remaining parent-owned gates

- Fresh independent source review and explicit parent adoption of these exact commits. This report supplies no GO for installation or full B.
- Native owner-thread startup precheck/post-acquisition binding under existing Core exclusion; distinct native durable installation schema and completed-history reader validating native receipts and publication/import-intent obligations. Keep Linux native-history refusal intact until that exists.
- Installed entrypoint/lifespan/credential and activation status wiring. `native_startup_inputs` alone must never enable installed Dispatcher construction or fake readiness. Existing API/CLI/MCP remain canonical, no web UI or public STT added.
- Stopped/reconciled boot and trust renewal with old-boot history support. Current immutable record intentionally refuses changed boot/config/trust, even for an otherwise clean new supervisor boot.
- Docker-control mount/topology/owner incarnation policy, bridge/firewall configuration and actual authenticated TLS reachability/denied peers.
- Supported explicit owned Qwen preset registration or narrow managed clone import for an empty catalogue; no DB seeding as product onboarding.
- Reviewed actual Darwin arm64 interpreter/wheelhouse/artifact acquisition, installed provenance/import/ABI, model mapping and native resources. No reviewed wheelhouse exists here.
- Actual Apple Silicon probe/offline load/generate/cancel/restart, process/FD/death evidence, clone conditioning lifetime, audio/performance; separate CPU and CUDA release qualification rows.

The bounded slice is durable source enrollment plus startup preparation, not full installed native control integration. Actual Metal remains unavailable/unqualified; legacy digest RED remains unwaived.
