# P1B23 isolated runtime comparison — 2026-10-23

**PROVISIONING BLOCKED; comparison NOT eligible. Original archive chain remains incomplete. Native cause unresolved; T2 NO-GO retained.**

## Scope and preservation

Ordinary dependency/runtime provenance lane only. Entire ENV-HEALTH report read, including archive gap and limits; root release Dockerfile and service/release locks inspected; T2 runtime/version context read. No native report inspection or crash experiment. No delegation, source/test/lock/commit/other-worktree edits, global installation, PATH/opt-link change, admin request, bypass, cleanup, Docker, model/audio/paid E2E, or product campaign. Sole repository write is this report. No applicable ancestor/report-path AGENTS.md found. A broad parent find was interrupted before returning content; subsequent instruction checks used exact paths.

Owned evidence: `/tmp/pab-runtime-comparison-20261023` (new directory; retained). System `/usr/bin/python3` only orchestrated read-only hashing, cache inventory and official HTTPS availability checks; it did not load the 3.14 native packages. Existing 3.14.7 interpreter was not executed.

Root HEAD before/after: `c6833aec0a52a577987e21f157041fc79d86dc58`. Tracked files unchanged; original status unchanged before report. Full current-venv content manifest: 1818 files, before/after exactly equal, manifest SHA-256 `f16352769f5cd7bf0c6a85de6e46283469c6da77f6d4033032841b3cde66afca`. Includes pyvenv.cfg and resolved interpreter file; no existing venv overwrite. Original research status preserved. Other worktrees were not opened or changed.

## Exact frozen inputs

- service/requirements.lock SHA-256 `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea` (32 pins).
- tests/release/requirements.lock SHA-256 `1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad` (42 pins).
- Docker Python target `python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83`. No image pulled or run.

## Original installation audit

Retained install.log records uv installation of all 42 exact pins and local build of cryptography 50.0.2. All 42 have cache metadata receipts and expanded wheel directory mappings. Each mapping has common installed files and zero byte-hash mismatches (excluding installer-generated RECORD/INSTALLER/REQUESTED, uv_cache.json and bytecode). This strengthens local installed-to-expanded-cache correspondence only; it does not reconstruct original compressed archives or independently authenticate download metadata. Full per-package paths/counts are in installed-cache-map.json. Metadata receipts may include unused architecture variants (cryptography arm64 receipt is NOT attributed as the installed x86_64 artifact).

No original downloaded wheel or sdist archive survives in the named uv cache: 0 of 42 original archives verified against lock accepted hashes. Retained archive scan found only two identical locally built cryptography wheels:

`sdists-v9/pypi/cryptography/50.0.2/H2iW-HzfmBGgavKU/cryptography-50.0.2-cp314-abi3-macosx_10_12_x86_64.whl` and its `src/target/wheels/` copy. Each SHA-256 `f705dcdb17ccae1fcd727f6e77cdc5fe5c24a9a1fa0cfe63ecc0be89aab4683b`; neither is an accepted lock archive hash. This is expected for a local build and is not evidence of a lock violation. Original cryptography sdist is missing, so source-archive hash verification cannot be completed. The wheel native member SHA-256 `d84badbcd1742e80c1ed582049efb47f17ac1f9b8c8832b5d06584ff5a36a4a5` matches installed `_rust.abi3.so`. Build-isolation dependencies are present in cache but are not promoted into the 42-pin acceptance set.

Normal Homebrew cache inventory was limited to top-level/cache-download filenames containing python@3.14/python-3.14, sqlite, or openssl@3. No matching archives found. Installed Python 3.14.7, SQLite 3.53.4 and OpenSSL 3.6.4 formula/INSTALL_RECEIPT snapshots retained with hashes in bottle-receipts.json; these are local receipt evidence, not independent bottle authenticity. Exact original bottle checksum cannot be compared without the missing archive. No unrelated user cache contents read.

Missing original archives for every pin below (expanded cache correspondence available for all):

- `annotated-doc==0.0.5`
- `annotated-types==0.8.0`
- `anyio==4.15.1`
- `attrs==26.1.0`
- `certifi==2026.7.22`
- `cffi==2.1.1`
- `charset-normalizer==3.5.2`
- `click==8.5.0`
- `cryptography==50.0.2`
- `fastapi==0.143.2`
- `h11==0.16.0`
- `httpcore==1.0.9`
- `httpx==0.28.1`
- `httpx-sse==0.4.3`
- `idna==3.20`
- `iniconfig==2.3.1`
- `jsonschema==4.26.0`
- `jsonschema-specifications==2025.9.1`
- `mcp==1.30.0`
- `opentelemetry-api==1.45.1`
- `packaging==26.3`
- `pluggy==1.7.0`
- `pycparser==3.11`
- `pydantic==2.14.1`
- `pydantic-core==2.50.1`
- `pydantic-settings==2.16.0`
- `pygments==2.21.0`
- `pyjwt==2.15.1`
- `pytest==9.1.1`
- `python-dotenv==1.2.4`
- `python-multipart==0.0.32`
- `referencing==0.37.0`
- `regex==2026.9.29`
- `requests==2.34.2`
- `rpds-py==2026.9.1`
- `sse-starlette==3.5.0`
- `starlette==1.8.0`
- `tiktoken==0.14.0`
- `typing-extensions==4.16.0`
- `typing-inspection==0.4.4`
- `urllib3==2.8.0`
- `uvicorn==0.54.0`

## Provisioning availability and blocker

Installed uv is `/usr/local/bin/uv`, version `0.11.15 (Homebrew 2026-05-18 x86_64-apple-darwin)`. With `UV_PYTHON_INSTALL_DIR=/tmp/pab-runtime-comparison-20261023/python` and `UV_CACHE_DIR=/tmp/pab-runtime-comparison-20261023/uv-cache`, exact command `uv python list 3.14.8 --all-versions --show-urls --output-format json` exits 0 with `[]` (no exact installed/discoverable or managed download entry). This is a catalog limitation, not a claim that 3.14.8 does not exist. No uv install/download attempted, no other patch substituted.

[Official Python 3.14.8 directory](https://www.python.org/ftp/python/3.14.8/) returns HTTP 200 and lists `python-3.14.8-macos11.pkg` (75,240,265 bytes), source tarballs and signature/Sigstore files. [Official macOS installation documentation](https://docs.python.org/3.14/using/mac.html) describes Administrator privilege and system framework/application/bin installation locations. That installer workflow violates the owned-directory/no-admin constraints. No package was installed, extracted with ad hoc relocation, or signature bypassed. No published checksum was obtained for an acceptable owned standalone binary. Source compilation was not substituted for the requested managed/official installation workflow.

A bounded attempt to locate a newer official Astral download catalog at `https://raw.githubusercontent.com/astral-sh/uv/main/crates/uv-python/download-metadata.json` returned HTTP 404; one directory-discovery request to the corresponding GitHub API path also returned 404. No further search/download loop. A current trusted uv catalog with an exact Darwin x86_64 3.14.8 entry and checksum remains the concrete prerequisite; alternatively a supported official owned-directory installation with published checksum. No global uv upgrade performed.

**New Python path: NONE. New venv: NONE. New runtime/build/native versions, architecture/checksums, dependency install receipts and pip check: NOT AVAILABLE because provisioning did not occur.** Existing evidence directory is not an installed comparison environment. No tokenizer cache copy/download needed.

## Conditional ordinary health matrix

Predeclared in health-plan.json before any health execution: at most five fresh commands, 40 seconds each, five minutes total, stop first nonzero/signal/timeout, no automatic retry. Normal plugin autoload; bytecode/cacheprovider disabled only for readonly repository behavior. No allocator/GC/fork stress, product test bodies, candidate collection, or native reducer. Rows:

1. stdlib imports and owned tiny SQLite CRUD/close.
2. jsonschema/rpds/pytest imports and full new runtime/build/native inventory.
3. One owned trivial pytest with normal plugin trace.
4. Read-only uv pip check of new venv (equivalent dependency consistency check if pip absent).
5. Root-only `--collect-only` for tests/service/test_core.py, no test bodies.

**Executed health commands: 0. Exits/durations: N/A.** Blocker reached before provisioning, so no health or plugin-disabled fallback was run. Official availability request duration 0.219944 seconds; uv list exited 0, shell provisioning-inventory wrapper 1.254797 seconds (includes before manifests, not isolated uv timing).

## Comparison limits and next handoff

Current installed Homebrew 3.14.7 x86_64 runtime remains preserved; prior report measured framework/Clang 17/PGO/LTO, SQLite 3.53.4 and OpenSSL 3.6.4. Those runtime version claims are prior-report evidence, not remeasured by executing it here. A Darwin standalone 3.14.8 would change patch level, distributor/build/compiler/configuration, framework/startup behavior, stdlib/native module binaries, linked SQLite/OpenSSL and possibly extension build/link outcomes. Exact differences cannot be enumerated until installation exists; no single-variable causal comparison or Linux Docker equivalence is claimed. Docker additionally changes OS/libc/filesystem/process and library envelope.

Carry forward: exact locks and root pin, preserved old venv and local manifests, 42-package expanded-cache correspondence, built cryptography archive/native hash, missing original archive list, and exact managed-catalog blocker. Parent selects any next comparison after a trusted exact owned runtime is available. Even all-green future ordinary health would mean eligible comparison only, never native clearance or candidate acceptance.

Native V1/V2/V3, T2 NO-GO, lifetime incomplete, guard HOLD, all 19 actual Linux mechanisms, C1 mandatory `b2_pinned_launcher_context_unavailable`, B3/C1/C2, final global Astra, hardware/browser/acoustic/activation, Apple Silicon, and unexplained A1 inventory drift remain unchanged. No test ownership/candidate GO/native-fixed claim.

Evidence JSON files retained: current-venv-before/after, root-before, preservation, original-artifact-audit, installed-cache-map, retained-cache-inventory, homebrew-cache-inventory, bottle-receipts, availability stdout/stderr, official-availability, official-mac-install.html, health-plan. No cleanup.

Final verification: `git diff --check` exit 0; post-report tracked hashes unchanged and root status differs from baseline solely by this new report. Current venv full before/after manifests match.
