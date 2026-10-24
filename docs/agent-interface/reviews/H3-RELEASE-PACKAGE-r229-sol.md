# H3 r229 Sol — finite host launcher and standalone source distribution

2026-10-24. **Implementation unit delivered for independent source/security review. No self-adoption, installed-runtime, audible-speech or release GO.** This completes the bounded package/finite-host-launch source unit on the already integrated activation and historical-incarnation implementation. Actual qualification remains separately parent-owned.

## Pins and authority

Checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-9941da0e`, branch `t3/headless-launch-package`. Entry was clean at `59a1dcb1be3487169e77df97f7625d451a2fa68b`; ancestry includes reviewed `d5bbabe7c2827e60958fed8e7267cf9bbcd24eed`. During implementation root advanced by two EXECUTION-LEDGER-only commits; own checkout was fast-forwarded to `acb2d37b53d8fd96d68557a3916308866ba9fb75` without merge/cherry-pick or source adoption. Source commit **`e0f9ecaa16b613ab3132e0a33b7c5956ba29eb5e`**, direct parent `acb2d37b53d8fd96d68557a3916308866ba9fb75`. Initial and post-source-commit status were clean; the final separate report commit contains only this report.

Read full normative r198 (six launch steps/Lot B/restart/failure), full r213 checkpoint, r228 review, partial-scope install instructions, and actual split init, closed install policy, host observer/authority/registration/history/handoff, activation factory and installed worker main. No applicable AGENTS.md found. Bounded memory registry query had no relevant hit and supplied no premise. No skill or delegation used. User explicitly authorized exact `pyproject.toml` and `headless-dist-manifest.json`, and bounded official PyPI metadata/index resolution after a cache-first attempt. Existing Poetry metadata/dependencies remain intact; the headless archive has a separate tool section.

## Delivered behavior

`scripts/headless-install.py` and new `service/package_install.py` supply finite `init`, `complete`, `up`, `stop`, `status` and stdlib `archive`. Operational commands require trusted Python 3.11 and the reviewed local Linux root authority; the archive/help paths use stdlib alone. Explicit real configuration, CPU profile/bundle/context and scope are required. No fixture qualification or inferred UID policy is supplied.

Init validates the explicitly supplied closed host policy through existing installed host inspection, then exclusively creates the protected per-scope authority leaf, `host-policy.json` and empty `launch.lock`. Their fixed identities/modes and protected ancestry are checked. Inputs are explicitly provisioned files/tree with fixed modes/identities, not silently copied or repaired; credential content is never read by the host observer or placed in policy/provenance/arguments/log output. The reviewed split initializer performs role provisioning and the permanently dropped Core child under held host exclusion. Complete requires the exact initialization-record digest and existing authority; ambiguous creation refuses and is retained.

First up retains exclusive creation intent, each created ID and the completed exact pair. The four fixed launcher document names are closed, canonical, root-private and bound to configuration/IDs; the protected history reader recognizes only those names in addition to existing policy/lock/provenance. Unknown files and incomplete creation still refuse. This is the necessary launcher-authority adapter; registration SQL, recovery writers, engine protocol, Core and Dispatcher were not edited.

Both exact image IDs must already exist locally. Create uses `--pull=never`, no build/download path, fixed role volumes, RO model/config binds, control-only credential bind, exact capabilities, namespaces, resource/tmpfs limits and loopback/network policy matching the actual existing observer. The same final control ID starts first and reaches a checked waiting phase before the PID1 worker starts. A bounded dropped-identity hello wait precedes the existing protected register-and-notify transaction. No duplicate registration writer, replacement placeholder or scheduler was added.

Control's installed command factory uses the reviewed activation/owner lifespan unchanged. It publishes a private, incarnation-bound startup result only after Uvicorn startup. The fixed exec reader checks PID1, exact notification and authenticated local API persistent readiness, exposing only bounded nonsecret phase/registration identity. Parent corroborates retained protected provenance, current host/process observation and fresh authenticated hello before reporting ready. Exec readers observe PID1 without pretending their own process is PID1; the activation path retains its original PID1 requirement.

Installed Uvicorn signal wiring requests ordinary drain on TERM/INT without forced-exit escalation or default post-shutdown TERM re-raise. Normal process exit 0 requires the retained owner lifecycle to be closed without error. Stop TERM/waits exact control, checks the existing clean terminal/reap/ack history under Core exclusion with stable journal/hello/topology brackets, then TERM/waits exact worker and checks stopped topology/history under ordinary lock order. Docker stop/KILL escalation, reset, prune, remove and implicit recovery do not exist here. Unknown/dirty/timeout outcomes retain all obligations. Same-container clean restart uses the already integrated historical-incarnation adapter and fresh registration; missing/recreated history refuses.

The separately extracted host source distribution contains its 93 exact manifest files plus the root `speech-box` command, including transitive source/data and pinned dependency locks, without requiring a checkout at destination. It is a source distribution, **not a bundled interpreter or wheelhouse**: a trusted Python 3.11 environment and matching explicitly supplied offline dependency wheels are provisioned separately using the included hash lock. Operational commands never install dependencies. `--help` and transitive package imports were executed outside the checkout with isolated Python and no repository PYTHONPATH.

The closed release Dockerfile has control and CPU worker targets sharing the existing pinned Python 3.11 CPU base; worker installs unchanged Kokoro lock and service lock together. No tini replaces PID1. Context excludes tests/studio/cache/unlisted sources; explicit runtime source/data/locks only. Release Compose is a fixed-policy reference without build/autopull, not a replacement for the finite registration transaction. Default legacy Compose and Dockerfiles are untouched.

## Dependency provenance

Cache-first probe failed because pinned FastAPI metadata was unavailable offline. Authorized resolver command (no package installation/build):

```
uv pip compile service/requirements.in --constraint service/requirements.lock \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-build --index-url https://pypi.org/simple \
  -o /private/tmp/pab-r229-py311.lock --quiet
```

The constraint input was the existing service lock from `acb2d37b53d8fd96d68557a3916308866ba9fb75`. Source is the official [PyPI simple index](https://pypi.org/simple/) and distribution metadata/index artifact digests consumed by uv. Target is CPython 3.11, x86_64 Linux/manylinux_2_28. The final lock preserves **all 32 versions and the complete existing artifact hash set**; no hashes were invented. An initial unconstrained metadata probe was discarded; its newer transitive candidates are not used. A further constrained resolution with the Kokoro lock succeeded, and the four overlapping pinned packages have no version conflict. Cached installed metadata for all 32 final versions declares Python 3.11 compatibility. These are resolver/metadata facts, not dependency installation, CPython 3.11 execution or image-build receipts.

- Final service lock SHA256 `67fd2bc0ebcb95a31eba99e2c7276435642dc0bc890905f69a866cfa1de4c5df`.
- Unchanged Kokoro lock SHA256 `6daa927a8493c6cbb41a5b64c4a806da9021d11a6ee1a10ab536a109f4230f66`.
- Manifest SHA256 `3db3de9a16ae339f9c12ea3a0bc175ed387f6d9bc5c0dadb0b7ce63f46f0e0c3`.

## Final checks

Both conftests inspected; no release conftest exists. Only `/private/tmp/pab-0c-venv/bin/python -I -B` and isolated unittest loading with explicit checkout/tests/tests-service paths were used. Interpreter is **Python 3.14.7**, not the shipped Python 3.11 runtime.

**Final combined result: 210 tests PASS, 53.990 seconds, exit 0**, including 28 new package/coordinator tests, with these exact modules:

- `tests.release.test_release_package`
- `tests.release.test_headless_install`
- `tests.service.test_installed_host_policy`
- `tests.service.test_installed_registration`
- `tests.service.test_installed_activation`
- `tests.service.test_split_init`
- `tests.service.test_install_child`
- `tests.service.test_host_install`
- `tests.worker.test_engine_lifecycle`
- `tests.service.test_production_api`
- `tests.service.test_production`
- `tests.service.test_production_admission`
- `tests.clients.test_production_clients`

New coverage includes deterministic archive build/extraction/help/transitive imports outside repo paths; manifest/legacy metadata; CPU base/context/locks; generated Docker arguments through the real closed topology validator; waiting-control/start-worker/register/ready order; retained partial creation and failed registration; dirty preflight/shutdown; timeout/nonzero exit without escalation; same-ID clean restart; registration/readiness binding; fixed live API probe and secret redaction; command factory startup publication and failed startup; repeated inert signal handlers; exact protected launcher record integration with real disposable SQLite registration; partial/changed/unknown authority negatives; and create-only authority with injected root/trusted-ancestor observations. Docker transport, host/UID/process, HTTP and signal delivery are injected. Files/SQLite/archives are disposable real fixtures. Development corrections included an existing-wrapper compatibility issue and a test-only nested trust-observation shim; the final result above includes the corrected fixtures.

Existing intentional `injected_close_unknown` shielded-future and worker-refusal diagnostics appear in passing output. They demonstrate refusal/retention, not successful unknown shutdown. `git diff --check` passes. All five changed/new Python files parse with `ast.parse(feature_version=(3,11))`; this is grammar evidence only. No broad pytest/guard/full suite was invoked.

Retained artifact: `/private/tmp/pab-r229-artifact-cn_lof1d/speech-box.zip`, SHA256 **`818f80bdbeaa851c08e2e47998b0052eb4e9652f781afa4de1dd29197b8de96e`**. Exact packaged changed files were byte-compared with source after commit; deterministic two-build comparison and extracted execution passed. No Docker image was built.

## Exact source files and SHA256

| File | SHA256 |
| --- | --- |
| `docs/agent-interface/HEADLESS-INSTALL.md` | `5567097b07a6120621248089454406cc9a6117903e8f9edef197b09c709d51d0` |
| `headless-dist-manifest.json` | `3db3de9a16ae339f9c12ea3a0bc175ed387f6d9bc5c0dadb0b7ce63f46f0e0c3` |
| `pyproject.toml` | `2748a65109f890e0c47de9867e78d7e09771ffb1fd0063c16129a4591590accd` |
| `scripts/headless-install.py` | `8a3b28e25c0a4b9ba89353dc3117bd52450cd1e14b2ea252d32aae5b9bc7dfd4` |
| `service/Dockerfile.release` | `bcf2b3aebaf0cb9d59938533375463ca90e3824e5de3e73a1c7692de33645bf4` |
| `service/Dockerfile.release.dockerignore` | `4fcef53ee8a13103cea211c4501ba7190777a17c32eccb54250a4d00412e2ffd` |
| `service/compose.release.yml` | `14c5955a62ea60d518193f73c64ede3e953bad7479aeac00ad6c2f544571deb0` |
| `service/entrypoint.py` | `14c00bbcba2a33826d2147f9561bfa2d0d81349ad6acfef03ed407f69f44c570` |
| `service/host_install.py` | `79769cfd98e25dd7f487ab3ed20288ffa7721939158c2647ac2d450d762e12df` |
| `service/package_install.py` | `73fe2c0b4b47ba742970e8e1d114a67056c71993e71c43ac69d6c7d44757baa9` |
| `service/requirements.in` | `69ba3b1add71749f6dbd76451a06ddfdefe3cba61567f4ed9fab15759ad80e57` |
| `service/requirements.lock` | `67fd2bc0ebcb95a31eba99e2c7276435642dc0bc890905f69a866cfa1de4c5df` |
| `tests/release/test_release_package.py` | `c622a577db68152981a092a6ff738666c49a58d5ce0cf43ab9be3ecbc8cc5276` |

## Boundary and handoff

No actual Docker build/run/pull/exec/stop, operational process qualification, network/listener, model download/provisioning/inference, ML/hardware, privileged mount/UID transition, HELD/maintenance operation, delegation, source adoption or unrelated cleanup was performed. Network use was limited to the explicitly authorized dependency metadata/index resolver. Only ordinary test-owned fixture cleanup occurred. No root source file was changed: **this report is the sole copy to `/Users/ben/Projects/py-audio-box`**.

Independent review of this cohesive implementation unit is next. Remaining actual qualification: trusted Python 3.11 dependency installation and image build, installed local Linux permissions/setgid/peer/mount/process authority, authenticated HTTP/CLI/MCP journeys, real CPU Kokoro audible speech, retained voice/artifact reuse, cancellation with confirmed descendant reap, and clean/failed restart behavior. No source/injected test or retained archive substitutes for those parent-owned receipts.
