# H4 Pocket managed provisioning r255 — implementation handoff

Source-review candidate only; actual availability/readiness is NOT established.

Base: `055d9359e236fe7889b84233e204e450faeb139e`.
Source commit: `2666a92` (Extend managed provisioning to explicit pinned Pocket CPU bundles).
Worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-ff01960f`.

Read r188 assignments 2/3, existing provisioning/tests/script/assets/pins, PocketBackend, PocketExecution and worker verifier constraints. No applicable AGENTS.md was found in the worktree or checked ancestors. Read both conftests; used unittest directly, avoiding pytest collection and listener fixtures.

## Implementation and operator inputs

The existing provision() stages, verifies, resumes, seals and atomically replaces active.json for Pocket as well as Kokoro. There is no second downloader, registry, acquisition framework or scheduler. Existing legacy CLI behavior remains; only managed CLI help changed. No installer, activation topology, production/core/recovery/contract or engine code changed.

Pocket's descriptor contains the existing exact manifest fields plus exactly `allowed_origins` and `config`. Backend is `pocket-tts`; language is an explicit `fr` or `en` identity. Checkpoint is bounded to the worker's 256-character limit. Revision is an operator-supplied immutable 40-character lowercase hex revision; every file must have that same revision. This deliberately excludes multi-revision bundles. License/access remain explicit nonempty operator metadata under the existing policy, not inferred consent or an assertion of license rights.

Versions must name exactly `pocket-tts`, `torch`, `tokenizers` and equal the trusted runtime_versions argument. No new upstream runtime version values were invented. The legacy acquisition path already records these three package identities. This does not independently qualify the installed runtime dependency closure or CPU torch build.

allowed_origins is a bounded unique list of HTTPS authorities, e.g. `https://fixture.invalid`. Each file origin must belong to it. Credentials, query parameters and fragments are refused, so authenticated/token-bearing download URLs are not supported. The production download seam checks redirect destinations against the same policy before following them. Trusted injected downloaders retain the existing responsibility for their own transfer bounds.

File entries retain exactly path, expected bytes, expected SHA256, origin, revision. Pocket allows 2..128 unique flat filenames with .safetensors/.yaml/.json suffixes, requiring config.yaml and safetensors. Maximum is 2 GiB per file and 4 GiB total, matching worker limits; descriptors/manifests remain within 65536 bytes. No pickle or arbitrary paths are accepted.

`config` is the explicit operator JSON object for the selected model architecture. It must include cloning weights_path. Nested *_path values can name authorized relative files or exact asset HTTPS origins; these compile to authorized flat relative paths. Other remote/path-bearing strings refuse. weights_path must select safetensors; the alternative non-cloning top-level weight config is refused. An optional config language must equal descriptor language. Architecture parameters are operator supplied rather than discovered from an installed package or hub. Descriptor/file/versions fields are closed; architecture-specific config keys are passed through rather than pretending to validate an upstream architecture schema.

config.yaml is generated locally as deterministic compact JSON plus newline (a safe YAML subset). The operator must provide its expected generated size/hash in the file entry. Its HTTPS origin records reviewed provenance; it is not fetched. Config generation must match those pins before storage/network work. The generated manifest omits config and allowed_origins, retains exact asset provenance and identity, and is accepted by existing verified_bundle and PocketBackend. Bundle directory identity hashes the complete descriptor including policy; active manifest_sha256 hashes the exact emitted raw manifest, rather than the descriptor. Readiness still requires an independently supplied actual qualification receipt.

## Evidence

Final command, executed with no pytest:

```
/private/tmp/pab-0c-venv/bin/python -I -B -c "import sys,unittest; sys.path.insert(0,'/Users/ben/.txcode/worktrees/py-audio-box/t3-ff01960f'); s=unittest.defaultTestLoader.loadTestsFromNames(['tests.service.test_provisioning','tests.worker.test_engine_pocket']); r=unittest.TextTestRunner(verbosity=1).run(s); sys.exit(not r.wasSuccessful())"
```

Result: 40 tests passed in 0.473 seconds. git diff --check passed.

New tests provision each explicit fr/en fixture identity, verify raw manifest digest/checkpoint/revision/language equality through worker verify_inputs, and construct the real PocketBackend with injected torch/Pocket/YAML modules. They assert the model loader receives absolute authorized weights/tokenizer paths after the adapter's local translation. These are injected constructor checks, not an upstream loader/model check. Additional cases cover idempotent reuse, sealing retry, interrupted partial resume without reacquiring verified complete weights, preserving an old active bundle on failed transfer, descriptor/asset negatives, redirect refusal and exact stream byte caps. Existing Kokoro and worker regressions remain passing.

Inert fixture pins only (not valid models, upstream pins or availability evidence):

- model.safetensors: 13 bytes, SHA256 `a7f18475aa48dffa7a4a769d1c17cd1a79ed98f3890686c368483c244145a2d2`.
- tokenizer.json: 2 bytes, SHA256 `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`.
- generated config.yaml: 85 bytes, SHA256 `28284a16109ff9d0f16aca0208cc6a3b6910d0eac2ea7731c71d2d4693f6b4ba`.
- file/model revision: forty `a` characters; runtime versions: `fixture`; origin: `https://fixture.invalid/<filename>`; checkpoint: `fixture-kokoro` inherited by fixture setup and used only as an opaque equality binding.

No external network, actual downloads/models, installs, Docker, listeners, process-crash campaign, HELD operation or delegation. Existing transfer deadline/cancellation tests exercised disposable inert transfer children. Test-owned temporary-directory teardown only; no retained operational resources were cleaned.

## Parent-owned remaining gates

Parent Sol independent review is still required. This implementation does not claim actual Pocket bundle availability, French/English model support, clone quality, offline upstream loader behavior, installed runtime compatibility, supported clean installation or release readiness. Those require independently reviewed real operator pins/config/license/access, actual runtime/model execution with network disabled and listening-based qualification under separate authority. A malformed or incomplete architecture config can still be refused by the real upstream loader; no injected test establishes that an arbitrary architecture config will load. No source fixture or manufactured pin satisfies those gates.
