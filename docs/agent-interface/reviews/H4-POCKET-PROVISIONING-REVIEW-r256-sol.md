# Independent Pocket provisioning review r256 — Sol

2026-10-24. **Bounded SOURCE GO. No material current issue found in the reviewed delta under the private, trusted-operator storage/configuration premise. NO-GO for actual-model availability, installation, language/audio qualification or release on this evidence.** Parent adoption remains parent-owned.

## Pins and scope

- Candidate: `/Users/ben/.txcode/worktrees/py-audio-box/t3-ff01960f`.
- Base: `055d9359e236fe7889b84233e204e450faeb139e`.
- Source: `2666a92ccbdd5991376459bceb60864bfa910552`.
- Report HEAD: `24be2487aa5f8c2e5973275a9c1ff78b8f002f71`.

HEAD was verified before and after review; candidate tracked/untracked status was clean. Source-to-HEAD changes only the r255 report. Base-to-HEAD changes `service/provisioning.py`, provisioning tests, one CLI help string and that report. Read the complete r255 handoff, complete r188 report including assignments 2/3, existing provisioning/tests, acquisition script/pins, asset verifier, PocketBackend, PocketExecution and actual worker `verify_inputs`. Inspected both conftests before execution: unittest bypasses pytest's maintenance and listener fixtures. No applicable worktree/checked-ancestor AGENTS.md was found. A bounded memory search supplied no task-relevant premise.

## Assessment

1. **Existing managed path retained.** `service/provisioning.py:274-384` extends the existing lock/stage/verify/seal/activate flow. There is no parallel acquisition framework or scheduler. `scripts/provision-tts.py:81-102` retains legacy dispatch; the implementation delta there is help text only. Existing Kokoro schema and asset expectations remain intact.
2. **Closed descriptor and deterministic local config.** `service/provisioning.py:36-146` requires exact descriptor/file/version fields, explicit fr/en identity, one lowercase 40-hex revision shared by every asset, operator size/hash/license/access inputs, unique flat allowed filenames and 2 GiB/file, 4 GiB/total ceilings. Architecture-specific config keys intentionally remain operator-owned. The recursive compiler handles dictionaries and lists, permits only authorized relative filenames or exact declared asset origins for `*_path`, and rejects path/URL-bearing values elsewhere. Top-level cloning weights are mandatory safetensors; alternate non-cloning config is refused. Generated compact JSON plus newline is a safe YAML subset and must match config.yaml's independent size/hash before root creation or transfer (`282-293`). Its origin is provenance, not a fetched config or a claim about upstream architecture.
3. **Transport and cancellation.** Allowed authorities and unique exact asset origins are checked at `80-107`; credentials, query and fragment are refused. The production Pocket branch forwards the authority list (`348-349`) to the opener whose redirect handler checks destinations before following (`154-168`). Reads request at most remaining bytes plus one overflow probe; overflow never reaches disk (`169-181`). Existing child deadline/cancel/reap machinery remains (`196-235`). Trusted injected downloaders explicitly own their bounds. This is not an independent bound on arbitrary operator callbacks or OS-uninterruptible I/O.
4. **Identity and activation.** Full descriptor hashes identify storage directories (`296`), while emitted manifest bytes exclude descriptor-only config/policy (`149-151`, `359`); active manifest_sha256 hashes those exact bytes (`32-33`, `378`). Consequently a policy-only change can create a distinct directory while retaining the same runtime manifest digest: this is consistent with runtime-byte identity, not a policy approval receipt. Existing bundles require exact asset set and exact serialized manifest equality (`319-328`). Partial transfers are discarded; verified complete assets are reused; sealing is retried before pointer replacement (`336-381`). Failed transfer/seal/activation preserves the previous pointer. Atomic replacement is not evidence of power-loss durability; no crash campaign was performed. Same-UID hostile mutation remains outside the expressly trusted storage premise.
5. **Consumer and distribution boundaries.** The inert generated fr/en bundles pass actual `verified_bundle` and worker `verify_inputs`, including raw manifest digest/checkpoint/revision/language bindings. The actual PocketBackend constructor receives injected runtime modules and translates approved relative paths to absolute bundle paths. This establishes adapter compatibility at that seam, not real upstream config/model compatibility. Archive allowlist contains provisioning, assets and Pocket adapter modules (`headless-dist-manifest.json:38,41,74`); new imports are stdlib, so this delta introduces no archive dependency. The release worker remains the existing Kokoro installation (`service/Dockerfile.release:15-30`), not a newly qualified Pocket install. Existing operator metadata in `requirements/tts-assets.json` remains Kokoro-oriented and supplies no real Pocket pins.

## Independent checks

All commands used candidate cwd and `/private/tmp/pab-0c-venv/bin/python -I -B`, explicitly inserting the candidate absolute path into sys.path. No pytest collection.

```python
import sys, unittest
sys.path.insert(0, '/Users/ben/.txcode/worktrees/py-audio-box/t3-ff01960f')
s = unittest.defaultTestLoader.loadTestsFromNames([
    'tests.service.test_provisioning', 'tests.worker.test_engine_pocket'])
r = unittest.TextTestRunner(verbosity=1).run(s)
sys.exit(not r.wasSuccessful())
```

**40 tests PASS, 0.449 s.** A separate run of `tests.worker.test_engine_kokoro` and `tests.service.test_provisioned_engine` passed **8 tests, 0.148 s**. Existing inspected transfer tests use bounded disposable children with mocked I/O; there was no external transfer. The second command also set PYTHONDONTWRITEBYTECODE=1 for subprocess inheritance.

Additional in-memory probes passed: nested-list exact-origin tokenizer rewrite; 13 refusals covering relative escape, absolute path, unapproved HTTPS origin, hub string, query-bearing filename, dictionary/list path values, remote/path strings under ordinary nested keys, duplicate allowed origins, duplicate asset origins and wrong generated config digest. These probes constructed an inert descriptor and called validation/compiler directly, without storage or network. `git diff --check 055d9359 HEAD` passed. No unexpected test failure occurred.

Only tiny inert temporary fixtures were used; observed free storage was about 3.6 GiB, below the 6 GiB gate. No large allocation, network, model acquisition/execution, dependency installation, Docker, listener, process-crash campaign, operational cleanup, HELD operation, delegation, source edit or commit. Test-owned temporary teardown only. Sole intentional lasting write is this report.

## Parent-owned gates

Adopt only the reviewed source scope. Independently supply and review actual Pocket model/config pins, revision, license/access acceptance and installed runtime closure; qualify offline upstream loading and exact CPU execution. Explicit fr/en bundle identity does not establish support, intelligibility, clone similarity or audio quality in either language. Actual model/French/English/listening evidence, supported installation, installed HTTP/MCP journeys and applicable CPU/CUDA/native Metal release qualification remain separate. No fixture hash, `fixture` package version, injected constructor or operator receipt label substitutes for those checks.
