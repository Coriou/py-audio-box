# Native enrollment exact identity correction — r305 Sol

2026-10-25. F1/P3 implementation candidate for fresh independent review. No self-adoption, merge or push. No full installed activation/history/topology/onboarding claim; native Metal remains UNAVAILABLE / UNQUALIFIED.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e31d5aee`, branch `t3/native-enrollment-exact-identity`, clean at entry. Entry/base `bdd1259a8dc212e2aa73caeed9ddf252aae94884`.

Requested source `8a402214afca47528af8670198f52ab18c8114ff` cherry-picked as `80c6bef64d188e5c59d74862967274dfe8b79633` and requested r303 docs `e37385ea8c433eebd0fa1e4c7759ba942a453f8a` cherry-picked as `8e4b6caecb7b828e0e096dc70999920c0bd84c9f`. Fix SOURCE `5ee444f503cb282fe93d2bf48318089b7cb67839`; this report is a subsequent docs-only commit, whose exact HEAD is supplied in handoff.

Read r304, r303, r277 and NATIVE-PACKAGE.md; inspected native package enrollment comparisons and selected test fixtures/conftests. No applicable AGENTS.md found. Bounded memory registry search returned no relevant hits; technical conclusions use this checkout.

## Correction

`native_startup_inputs` retains canonical bytes of the validated saved document and compares them with canonical reconstructed snapshot bytes before TLS, and canonical final-record bytes after hello. Canonical JSON distinguishes true, 1 and 1.0, unlike Python dictionary equality. Existing document canonical-syntax, duplicate-key, nonfinite and private-file checks remain in force.

Inspected the other snapshot equality checks in this enrollment slice: both enrollment and reopen bracket comparisons express exact snapshot identity, so both now compare canonical bytes too. No broader package equality rewrite. Owner incarnation still uses exact literal path/device/inode/UID; write-once exclusive durable record creation, config/template/trust hashes, exact native tuple/hello, import uncertainty, Core fences and Linux policy are unchanged. The immutable operator-input/no hostile same-UID writer premise and nontransactional bracket limits remain.

Two focused tests each exercise five substitutions: version true, version 1.0, and numerically equal float owner device/inode/UID. Before-TLS substitutions refuse without requests or record repair; substitutions during healthy fake hello refuse at final record validation and retain changed bytes. Existing valid exact reopen/context/runtime and write-once test passes unchanged.

SOURCE SHA256:

- service/native_package.py: `1f6a1e779319e0b74b3adf80c4eb7e3e0be5b3c57f6bc9c97d6baf067e3009d8`.
- tests/release/test_native_enrollment.py: `e3d881bd805b5bcc8e9ad00b118ae1e42f3b875f3eff5fe571299425b184a8d1`.
- Unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known legacy digest RED remains unchanged/unwaived; not regenerated or rerun.

## Final finite execution receipt

Once after final source/test edits, direct unittest via `PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B`, explicit absolute source and tests/service sys.path entries, defaultTestLoader.loadTestsFromNames, TextTestRunner verbosity 2 and failfast. Observed Python `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`, not target 3.11.

Exact selection:

1. tests.release.test_native_enrollment — 10 tests.
2. tests.release.test_native_package — 27 tests.
3. tests.service.test_installed_host_policy.InstalledHost — 20 tests.
4. tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change — 1 test.

**58 PASS / 5.925s, exit 0**. Tool output is the receipt. `git diff --check` passed. No subsequent source/test edits. Conftests inspected to avoid maintenance autouse and server/listener fixtures; no pytest collection. Tiny inert fake TLS/certificate/model/ZIP fixtures and direct unittest are expressly authorized below 6 GiB; no free-space requirement was imposed on them and no disk cleanup was performed. Fixture-owned temporary teardown only.

No Docker, network, listener, download, install, actual child, ML or crash tests; archive help is in-process. No operational cleanup/HELD/STT operations, root checkout edits, delegation or adoption.

## Remaining bounded gates

Fresh independent review and explicit parent adoption of exact SOURCE remain pending. Installed owner startup under Core exclusion, native completed-history/publication/import-intent validation, entrypoint/lifespan/credential wiring, stopped boot/trust renewal, topology/mount policy and supported empty-catalogue onboarding remain separate Lot B work. Actual Darwin artifact provenance/install/import/ABI, real TLS/denied peers, process/FD/death/resource observations, Metal probe/load/generate/cancel/restart and clone/audio/performance qualification remain unrun. CPU and CUDA release rows remain separate mandatory gates. This finite fix supplies no installed or hardware readiness authority.
