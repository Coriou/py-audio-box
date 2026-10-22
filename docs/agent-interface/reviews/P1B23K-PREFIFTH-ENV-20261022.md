# K pre-fifth isolated environment restoration and validation — 2026-10-22

**PASS: isolated locked environment and all requested local campaigns. No integration or source acceptance by this test lane.** The prior admission measurement, MCP parity and compactness failures all pass unchanged in the restored environment. Full service/release: **1529 passed, 19 skipped**, exit 0. Every skipped Linux mechanism remains UNQUALIFIED.

## Exact provenance and authority

- Candidate checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-85d4bb33`.
- Exact candidate HEAD: `6130d1fecce3d978671ed1bfd509693d3aab3c97`, clean before restoration, before/after each campaign, and final verification.
- Frozen executable source/test base: `44e00b94d4f40ce7a058e3f84c26d571f7feb1f2`. Candidate difference is the 90-line recovery handoff only. `git diff --exit-code 44e00b94... -- service tests clients contracts pyproject.toml` passed before each campaign and finally.
- Parent: `/Users/ben/Projects/py-audio-box`. Entry HEAD was `7916f17ebe4608833269d958cd1c2b1430d78400`; it advanced externally during this long validation to `8e60e9fa80a2af345effcab49344de71ad3a7d17` (latest two commits are documentation handoffs). This lane performed no commit, checkout, cherry-pick or integration. Parent research/untracked files were left untouched.
- Sole repository write by this lane: this report in the parent root. No candidate write, source/test/dependency/lock mutation, delegation or global/Hermes modification.
- Read the candidate recovery handoff from section/line 1465 through EOF, both requirements locks and associated inputs, pyproject, 0C interpreter/runner receipts, release Dockerfile and runner instructions. The Docker runner was read, never executed.
- `flock` was unavailable. Acquired `/tmp/pab-k-prefifth-validation-lane.lock` via exclusive `mkdir`, marked owner `codex-env-20261022`; released only this owned lock after report generation. No concurrent broad campaign was launched by this lane.

## Restored environment and installation

Installed-interpreter inventory before selection: Xcode 3.9.6; Homebrew 3.10.19, 3.12.12, 3.13.11, 3.14.7; owned local 3.11.16. Host is Darwin 24.6.0 / x86_64. The original `/tmp/pab-0c-venv` was absent.

Selected actual `/usr/local/bin/python3.14` (Homebrew CPython 3.14.7), based on `phase-0/SKELETON-0C.md` and `phase-0/skeleton-0c/results.md`: these explicitly record original local 3.14.7, while authoritative packaged 3.14.8 is separate. This is restored local compatibility, not a packaged 3.14.8 receipt. Pyproject allows Python ^3.11; no assumption that 3.11 was the original interpreter was made.

Owned environment: **`/tmp/pab-0c-venv`**, executable **`/tmp/pab-0c-venv/bin/python`** (macOS real path `/private/tmp/pab-0c-venv`). `include-system-site-packages = false`. Owned cache/download/build/log root: `/tmp/pab-k-prefifth-env-20261022`.

Installation commands (exit 0):

```sh
env -u PYTHONPATH -u PYTHONHOME   UV_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/uv-cache   uv venv --python /usr/local/bin/python3.14 /tmp/pab-0c-venv
env -u PYTHONPATH -u PYTHONHOME   UV_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/uv-cache   uv pip install --python /tmp/pab-0c-venv/bin/python --require-hashes   -r service/requirements.lock -r tests/release/requirements.lock
```

uv 0.11.15 resolved 42 packages in 10.44s, prepared in 1m54s, installed in 49ms. Hash enforcement applied to repository-locked distributions; `cryptography==50.0.2` was built from the locked source distribution using installed native tooling and isolated build dependencies. No lock regeneration or direct-pin substitution. Build dependencies are build tooling, not additional runtime packages. `install.log` retains the complete installation receipt.

Lock SHA-256:

- `service/requirements.lock`: `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea`.
- `tests/release/requirements.lock`: `1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad`.

Runtime diagnostics parsed both complete locks and asserted every installed version against every pin: 32 runtime / 42 release-test packages. `uv pip check --python /tmp/pab-0c-venv/bin/python` exited 0, all 42 compatible. Actual required versions: MCP **1.30.0**, pytest **9.1.1**, tiktoken **0.14.0**. Imports of MCP, pytest, tiktoken, pydantic_core, cryptography, cffi, rpds and regex succeeded from the owned venv. No pytest bridge, `sitecustomize` injection, global package path or PYTHONPATH was used. The runner removes PYTHONPATH, PYTHONHOME, PYTEST_ADDOPTS and PYTEST_PLUGINS from child environments.

Actual native runtime: SQLite **3.53.4**, Python SSL and cryptography OpenSSL **3.6.4 (25 Aug 2026)**. `native.json` records native extension paths/SHA-256 and compiler/Rust versions; `environment.json` records interpreter, sys.path, package versions/origins and lock hashes.

The normal tiktoken 0.14.0 `get_encoding('o200k_base')` loader fetched only its tokenizer asset into owned `TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache`. Package source specifies `https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken` and expected SHA-256 `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`; actual cached bytes match exactly. Cache filename `fb374d419588a4632f3f557e76b4b70aebbca790`. No ML model asset was downloaded or run.

## Independent test receipts

All runs were sequential on the exact clean candidate, without intervening environment/source changes. Each actual child command uses:

```sh
PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider ...
```

The owned stdlib subprocess runner `/tmp/pab-k-prefifth-env-20261022/run.py` does not import pytest or change test behavior; it guards candidate HEAD/cleanliness/frozen executable equality, sanitizes environment, runs the exact child command, captures log/exit/duration and checks HEAD/status/limits afterward. No ignore, extra skip, xfail, test repair or source edit.

1. `focused`: `tests/service/test_admission.py::test_lotk_measurement_counts_descendants_not_root_and_flat_growth tests/service/test_adapters.py::test_http_cli_mcp_parity_and_errors tests/release/test_compactness.py -q -rs` — **3 passed in 6.68s**, exit 0 (wrapper 7.87s). This passed before any broad run. The previously failing actual test functions all pass unchanged; therefore the conditional same-test baseline-failure comparison was not triggered. No claim that all possible failures are external is made.
2. `targeted`: `tests/service/test_recovery.py tests/service/test_admission.py -k 'kfifth or kpre or lotk or lotp or b22' -q -rs` — **915 passed, 1 skipped, 408 deselected in 1300.01s**, exit 0 (wrapper 1301.04s).
3. `four-b2`: `tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs` — **1377 passed, 1 skipped in 1458.99s**, exit 0 (wrapper 1460.15s).
4. `full`: `tests/service tests/release -q -rs` — **1529 passed, 19 skipped in 1496.18s**, exit 0 (wrapper 1497.50s). Compactness is included; no module exclusion.

All four JSON receipts show exact HEAD unchanged and empty repository status before/after. Inherited runner/pytest RLIMIT_FSIZE remained `(9223372036854775807, 9223372036854775807)` before/after each run. This lane never called setrlimit or changed parent/pytest/shell limits. These independently executed counts supersede no source review and do not rely on the earlier 164-case green claims.

Exact full skip inventory:

- 1 at `tests/service/test_recovery.py:3404`: actual Linux RLIMIT_FSIZE evidence PENDING; macOS E1 model is not proof.
- 10 at `tests/service/test_worker_transport.py:65`: Linux peer credentials required.
- 1 at `tests/service/test_worker_transport.py:109`: Linux peer credentials required.
- 1 at `tests/service/test_worker_transport.py:127`: Linux peer credentials required.
- 6 at `tests/service/test_worker_transport.py:146`: Linux peer credentials required.

All **19 remain UNQUALIFIED**, including actual Linux 4 MiB enforcement and peer credentials.

## Evidence index and remaining gates

Owned retained evidence root: `/tmp/pab-k-prefifth-env-20261022/`. Includes `install.log`, `environment.json`, `native.json`, `run.py`, tokenizer cache, uv cache, and `focused`, `targeted`, `four-b2`, `full` each with `.log` and `.json`. These temporary paths are retained, not durable repository artifacts. No cleanup of venv/cache/disposable roots or other worktrees was performed; only the lane's own coordination lock was released.

The environment-restoration and requested local-validation gates are met. Source acceptance/integration remains owned by the parent and independent reviewer; this test lane does not self-approve the candidate. Actual Linux/ext4/standard Unix VFS/SHM/4 MiB/peer-credential qualification, pinned host authentication and launcher/freeze proof, installed/clean-host, C replay/four original REDs/full C matrix, C1/C2, B3, root integration/equality/restart proof and final Astra remain separate gates. Hardware/browser/listening/acoustic, physical ENOSPC/power-loss/daemon/VM/controller durability and public/unattended acceptance remain unqualified. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.

No Docker, audio/model execution, paid/E2E runner, push/PR/deploy, integration, global/Hermes mutation, delegation or user/research cleanup occurred. Network use was limited to isolated test-dependency installation/build tooling and the verified tokenizer asset. All other worktrees and STT research were preserved.

Evidence SHA-256:

- `install.log`: `61e01e9852632ec8bcc79733f1f10d919e6a2c22bc9b9ab06a7cbbba1468f95a`.
- `environment.json`: `308db2bebdb12103b3924077126e49290152c1105e48f0d7514912da03c07495`.
- `native.json`: `7ae118997fb75d11b2cdb46b9c2251555942539bdc5d72a245f70ef0babbd207`.
- `focused.log`: `e0f5f58643d9ede151b0bbee0bc87aed94a52412c5fb4c160ef54481a383c0b6`.
- `focused.json`: `1cba98a75968f47d208640eece9ab1ead561353e56637d85e296c742c1e8545c`.
- `targeted.log`: `64cc2bcf9fd742b7c6bbf19aebd9b6e0b8bf64d1d3ec29c54d55ed031d4bbd36`.
- `targeted.json`: `688e63b504a3711bef93a10595a9528fefafb4de09c8da947a1c95c22e776523`.
- `four-b2.log`: `993967f56992363869d9397479067e1c46a0f5eca7fcd5075243f537851832f5`.
- `four-b2.json`: `26de3b5d3a3a78773083a6cc4a564e9f3c6650449ea7f57dda502bf856c5d552`.
- `full.log`: `825c76860cdc692d58af45e6fc95320c9b5dcc037d70bbd8a54be7f048ee8a15`.
- `full.json`: `888d6d49bf8cb652935eb8aefc4441d4efbaf3f0e29d4cb30daadd79be976ace`.
