# Independent partial native package review — r298 Sol

2026-10-25. **NO-GO for adoption of this partial r297 SOURCE candidate pending F1/F2 below.** The independently executed final 15 package methods and 26 relevant regression methods PASS. Two small inert reproductions establish gaps in selected-wheel dependency validation and bounded archive reading. Neither finding alleges installed readiness, arbitrary-code containment, or a defect merely because the rest of Lot B is absent. Full B, installed native activation, actual Metal and release remain CLOSED / UNQUALIFIED; known legacy digest RED remains unwaived.

## Authority and exact candidate

Review candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-0d06d538`, clean at entry and after checks. SOURCE `dba2d33033c25f081a709c557fbbafa56fd49af8`; docs HEAD `bf188fd99745fdc9335c1d3e82cb9cdfd584cd04`. The prompt's base `f0fafe2fe6a06925c46be4ae57a8c98af131ab4` does not resolve. The recorded, resolving base is **`df0fafe2fe6a06925c46be4ae57a8c98af131ab4`** (leading `d`); this is the base used for the actual diff, consistent with r297. No source substitution was made.

Actual five-file source delta: `service/native_package.py`, `native-dist-manifest.json`, `headless-dist-manifest.json`, `docs/agent-interface/NATIVE-PACKAGE.md`, `tests/release/test_native_package.py`; 835 insertions. Lot A production modules are unchanged. Source-to-docs successor supplies the r297 report. `git diff --check` for base-to-source PASS.

Read full r277 including original Lot B/C, r295, r296 and r297, the five-file delta, both conftests, and actual relevant NativeSupervisor CLI/registration/hello/serve, NativeRuntime construction/accounting/admission/hello/RPC, TLS/response validators and install-policy path walker. No applicable AGENTS.md was found in the checked repository/worktree discovery. A memory pass supplied evidence-boundary conventions only; current files and executions supply the technical conclusions. The P1B23 one-shot skill was inspected but is not applicable to this package-unit-review lane; explicit user authority permits the tests here.

Sole lasting write: this report in the root checkout. No candidate source edit, commit, delegation, actual socket/listener/model process, ML import/workload, hardware probe, Docker, network/download/install, crash experiment, operational cleanup or HELD action. Existing root untracked STT evidence was listed only, not read or changed. Temporary fixture teardown only. Free space approximately 3.0 GiB, below 6 GiB: only tiny inert ZIP/metadata/model fixtures ran; actual-work admission was not attempted or weakened.

Verified SHA256 pins:

- native_package.py: `ccb8611e2b1b227fb7ace1c9898d57314e5e18e5ceb90542528a1c0ef885f848`.
- native-dist-manifest.json: `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`.
- test_native_package.py: `133a938dad127e7cc82a66f953b8e8851becee95659b6089bbf86a029339f1f4`.
- r277: `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
- unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

## Material findings

**F1 — P2: selected wheel Requires-Dist is never checked against the declared closure.** `service/native_package.py:132–139` validates active requirements from installed distributions. `:158–165` reads the selected wheel METADATA but checks only Name, Version and Requires-Python, plus WHEEL tags. It neither validates that wheel's Requires-Dist closure nor compares those requirements to the installed metadata. Thus the exact artifact hash can authenticate an artifact whose own active dependency graph contradicts the manifest. This does not require hostile same-UID mutation, installation, ML or an ABI claim; an operator can select inconsistent same-name/version artifact bytes and get a metadata-verified result. The explicit caveat that wheel installation provenance is unproven does not establish artifact dependency closure.

Reproduced once with existing `NativePackageTests.setUp()` tiny fixtures. Rewrote only the fixture `packaging-1-py3-none-any.whl` with Name packaging, Version 1, supported `Tag: py3-none-any`, and these METADATA headers:

```
Requires-Dist: missing-review-dependency>=99
Requires-Dist: torch @ https://example.invalid/never-download.whl
```

Updated its declared hash. Patched `python_identity` to the existing nonqualifying Darwin fixture, and `importlib.metadata.distributions` to four inert installed records for the exact manifest names/versions with no requirements. The actual `dependency_closure`, artifact hashing, wheel parser and archive verifier all ran unchanged. `runtime_manifest` accepted and returned `d6f1eee1b46af42ed7d3caac5fb988bdcdab5c660666a170189d10b82cc6a737`. No URL was accessed. Existing final15 has no selected-wheel Requires-Dist negative case.

Required correction: validate active wheel Requires-Dist with the same finite marker/extras/URL/version policy against the exact distribution set, and establish the intended relationship to installed requirements. Add negative tests for an absent edge, incompatible version, active URL and extras; preserve inactive-marker handling. This remains metadata validation, not binary installation/ABI proof.

**F2 — P2: archive preflight can fully inflate an unchecked launcher member before refusal.** `service/native_package.py:250–262` checks exact names and checks each ordinary source member's declared size against its local counterpart before reading it. The synthetic launcher is different: `:262` calls `source.read` before checking its expected length. A wrong operator-selected archive can therefore consume memory proportional to an arbitrarily large decompressed launcher merely to refuse it. Exact hash selection and immutable-writer assumptions authenticate input identity, not a finite resource bound for processing malformed input.

Reproduced once without large allocation: copied the exact fixture archive member set into another tiny compressed ZIP, replacing only `native-worker/native-worker.py` with 131072 `X` bytes. An observational wrapper delegated `ZipFile.read` unchanged and recorded its target's declared size. Result: `native_package_launcher` refusal, **launcher bytes requested before refusal = [131072]**. No extraction or execution occurred. The same code has no upper bound before this read; no large bomb was built or run.

Required correction: check launcher size against `len(LAUNCHER)` before decompression and use bounded reads. Bound ZIP central-directory input/entry accounting before allowing it to allocate without a package-specific ceiling; both wheel (`:153–156`) and source (`:251–254`) constructors currently parse the central directory before their count checks. Ordinary exact source names eliminate arbitrary extraction/path-depth authority, and WHEEL/METADATA have 65536-byte declared output caps; compression ratio alone is not an additional finding where output is already capped. No untrusted archive extraction service is alleged. Add a tiny over-limit launcher/ZIP-accounting regression; retain the operator-managed premise.

## Behaviors supported by inspection and focused evidence

The interpreter check compares exact current executable, full sys.version/build/implementation and Darwin arm64; hashing requires a regular single-link current-owner file, owner executable bit, no group/other write and no-follow component walk. Fixture observations do not qualify a real arm64 build. Exact installed distribution set, normalized names, local-version refusal, active installed requirement markers and selected wheel name/version/supported tags are implemented. F1 limits the artifact closure claim. No ABI/readiness conclusion follows from metadata.

Private config/manifest/CA/cert/key reads require 0600 current-owner single-link regular files, bounded contents and stable identity with no-follow component walking. Immediate roots require 0700/current owner. SSL subsequently loads checked paths; this relies on the explicitly documented stopped-rotation/immutable writer and no hostile same-UID writer premise. It is not an FD-pinned SSL loader. Client/server role selection is explicit; TLS1.3 and CERT_REQUIRED remain, with peer DER pins at actual frame I/O. Secrets do not enter the native archive or supervisor inference parameters.

`main` passes the checked preflight config object to `serve_checked`; it does not reread the config. Duplicate registration bindings refuse before supervisor construction. It consumes the actual Qwen descriptor/inventory classes and NativeSupervisor/serve functions. The existing CLI remains separate and is not silently promoted to installed packaging. The real bundle verifier checks configured model bytes and descriptor relations.

The private `enrolled_runtime` constructs the actual NativeRuntime. Its native tuple is closed to deployment/configuration/lane/boot/runtime-manifest identities; the factory checks every configured field against the actual validated hello and refuses quarantine or closed import admission. NativeRuntime already checks four identity fields; the factory adds exact boot equality, without learning boot. The fake TLS connector invokes actual supervisor dispatch/hello and protocol framing/pin logic. The 15-method fixture tests stale boot and wrong peer; all-five comparison and quarantine/import-fault refusal are also source-inspected, not falsely counted as separately executed negative cases. This is a private consumer seam, not durable installed enrollment.

Archive membership is explicit, includes the native manifest itself and excludes keys, local config, dependencies, weights and Linux lock files. Two separately generated fixture archives had equal SHA256 in the additional inert observation. Source mutation refuses. Package archive hashing does not have a manifest self-reference cycle: the archived native source manifest is a static allowlist; the external runtime manifest holds the archive hash and is not an archive member. Its canonical hash then binds supervisor configuration/templates. This supplies a current-byte provenance relation, not an independently authenticated release-signature chain. F2 limits bounded malformed-archive refusal.

## Independent executions

All project interpreters: `/private/tmp/pab-0c-venv/bin/python -I -B`, `PYTHONDONTWRITEBYTECODE=1`, candidate cwd. Explicit sys.path prefix:

```
/Users/ben/.txcode/worktrees/py-audio-box/t3-0d06d538
/Users/ben/.txcode/worktrees/py-audio-box/t3-0d06d538/tests/service
```

Direct `unittest.defaultTestLoader.loadTestsFromNames`, `TextTestRunner(verbosity=2, failfast=True)`; exit nonzero on failure. Both conftests were inspected first: no pytest collection, maintenance autouse guard or HTTP listener fixture was activated.

1. Final package module `tests.release.test_native_package`: **15 PASS, 1.398s, exit 0**, exactly once.
2. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`, `tests.service.test_installed_host_policy.InstalledHost`, `tests.worker.test_native_protocol`: **26 PASS, 1.429s, exit 0**, exactly once.
3. Separate inert observations above: inconsistent selected-wheel requirements accepted; deterministic archive equality true; oversized launcher read before refusal. These are not extra unittest method counts. No test suite was rerun.

The r297 39/19 earlier selected results and its final15, and r296's final96 are historical parent/implementation evidence, not fresh executions here. Fresh total is 41 unittest methods. A new broad Lot A96 run was neither necessary for this native-package-only delta nor performed. No legacy digest regeneration/test/waiver.

## Parent-owned disposition

Correct F1/F2 and independently validate the affected partial source before adoption. No source repair or self-acceptance occurred here. Installed native enrollment/history, Docker control/mount/topology policy and supported empty-catalogue onboarding were explicitly excluded from r297's partial claim; their absence is not a newly discovered defect. They remain required before full Lot B acceptance. No reviewed real Python/wheelhouse was supplied; operator pin fixtures remain nonqualifying.

Actual dependency acquisition/install provenance/import/ABI, TLS bridge/firewall, native process/FD/resource/death/crash observations, Apple Silicon probe/load/generate/cancel/restart, audio/performance, clone lifetime and CPU/CUDA release rows remain separate mandatory gates. Actual native Metal stays UNAVAILABLE / UNQUALIFIED. Full B and release are not accepted; legacy digest RED remains unwaived.
