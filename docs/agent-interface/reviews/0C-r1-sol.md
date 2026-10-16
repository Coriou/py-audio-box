# 0C round 1 — independent Sol-low adversarial review

**GO for the bounded 0C skeleton candidate `7ab54bb4bf3b30f95a7debc9a9198267d1ac3bdb`.** No blocking finding in this lot. This is not integrated Phase 0 approval, Phase 1 acceptance, installed-product/browser acceptance, runtime qualification or final Astra approval.

Reviewed 2026-10-16 as the separate review agent, gpt-6.1-sol / low. Candidate worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e6cca322`, branch `t3/build-isolated-0c-skeleton`, base `a9163ad66daf500d405e984330ef03dbd0815da6`. HEAD matched exactly and worktree was clean before and after review. Only this root review document was written; no candidate/source/contracts/ledger/legacy/studio edits, commits, subagents, push, PR or deployment.

## Scope and inspected evidence

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, shared architecture and Phase 0/1 boundaries in IMPLEMENTATION-PLAN, CONTRACTS 1.3.0, EDITOR-0B including correction restrictions, SKELETON-0C, ACCEPTANCE-0C, all 34 changed files and their evidence. Inspected scripts before execution. No legacy remote e2e, model download/inference, private audio, STT, Voicebox or cloud work.

The plan's 0C output is directories/toolchains, cheap discovery/MCP fixture, release Compose skeleton and evidence matrix. Durable jobs/storage/owner fencing belong to 1A; shared authenticated client behavior to 1B; installed browser save/reload/task integration to 1C. Those missing implementations do not invalidate this explicitly synthetic skeleton.

## Actual independent checks

- Protected-path diff against the exact base was empty for apps, lib, contracts, studio, poetry.lock, PRODUCT, DESIGN, shared ledger and reviews. `git diff --check` passed. This is preservation evidence, not a rerun of legacy tests.
- Snapshotted existing Docker containers, volumes and networks under `/tmp/pab-review0c-r1/` before testing. Sequential bounded test/build/Compose activity; fresh random Compose projects and volumes; no host data mounts. Both projects were removed with only their own volumes/network. Original volumes survived; unrelated containers were not stopped or removed. No prune.
- Existing test image `sha256:324b1080defc0980967cf5237028c5785b27408f25cf8f7be91f0f2658e63a0d`: independently enumerated and SHA256-compared all **50 packaged source/contract/test files**, zero mismatches against checkout. Control image `sha256:c670144f2bf5ead7abec7293237c98744ca7c1adde2d0b7f4dd6ab3eaa83e054`; worker `sha256:e0918eaf7be61358e1cd526cfd1a03f9e2b4be8e26b43c95f0e016adc507623c`, matching packet identities.
- Ran test image with network none, read-only root, 2 CPU/1 GiB/PID128 and 64 MiB temporary tmpfs: **4 passed in 9.81s**. Actual loopback HTTP, CLI subprocess, maintained SDK stdio client/server, invalid fields, unavailable tools, 3072/3073 direct-byte boundary and exclusive download are exercised. No source overlay.
- Ran `python -m tests.release.measure` in networkless/read-only test image: reproduced **574 initial tokens**, tools/list 1820 UTF-8 bytes/423 tokens plus discovery CallToolResult 536 bytes/151 tokens, tiktoken 0.14.0/o200k_base. Status envelope 1264 bytes/284 tokens; offline receipt raw 157 bytes, job raw 434 bytes, list20 raw 280 bytes. Receipt/job/list are offline fixtures, not implemented endpoints.
- Independently initialized a real SDK stdio session (negotiated `2025-11-25`), serialized the SDK client's actual tools/list and discovery results, and reproduced 423+151=574 tokens. Discovery and missing-asset error text JSON equalled structuredContent. These counts include both body copies, exclude JSON-RPC framing and host prompts, and prove no real agent-host task economics.
- Additional isolated probes: Draft 2020-12 schema checks passed for all **41 operation request/response pairs** and six eager MCP input schemas. Injected bad hash, wrong range and interrupted transfer into the actual CLI: each failed compactly with no destination or sibling temporary file left. Existing-destination preservation/exclusive hard-link publication is covered by the four-test suite and source inspection. No durable publication claim follows.
- Rebuilt control target from exact source, memory limit 2 GiB: succeeded with cached dependency/compiler layers and reproduced the identical control image ID. This is a cache-backed rebuild, not an independent uncached npm/compiler run. Original evidence explicitly records Node24 npm engine warnings and successful TypeScript/Vite build; compiled HTML was independently served below.
- Registry manifest inspection independently matched Dockerfile's official Node index digest `a5f4366e834695d5f363b920c104953a535d3a9864d1bdf74566a0ebe01daee5` and Python index digest `48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83`. Official PyPI version metadata confirmed all eight direct runtime/test pins published and not yanked. Every locked distribution hash matched its official PyPI release metadata: **32 runtime / 42 test packages**. Installation uses `--require-hashes`.

## Packaging challenge resolved within 0C scope

**Non-blocking limitation, `service/compose.release.yml:10-11,24,48-50`: default host access fails on this daemon.** Re-ran the reviewed `tests/release/smoke.py`: internal HTTP, compiled evaluation HTML, private UDS ping, nonroot/read-only containment, worker network none and read-only assets all passed; its actual result was `host_loopback_reachable:false`. Thus published-port syntax alone does not establish host access. The source-level distinction is control's `fixture.internal:true`; this is not attributed to guessed hardware characteristics.

Tested the previously unstarted override, with a random project/loopback port and fresh named volumes:

```sh
docker compose -p <owned-random-project> -f service/compose.release.yml \
  -f service/compose.local-ui.yml up -d --no-build --pull never
```

Combined config sets `internal:false` (`service/compose.local-ui.yml:4-6`). Host requests then returned `/` **200/410 bytes**, `/evaluation/` **200/365 bytes**, `/v1/capabilities` **200/214 bytes**, and `/skeleton` explicitly reported fixture=true/ready_for_speech=false. Inspected running worker remained `NetworkMode:none`. Cleanup passed and preserved pre-existing volumes. Changing this explicit control-network choice makes the fixture externally usable from loopback without relaxing worker isolation. It permits control bridge egress; no offline-control claim is made for that override. Actual browser interaction, full JS execution/accessibility and service UI integration were not tested.

This satisfies an executable packaging **decision** for 0C: default contained internal fixture plus disclosed opt-in host evaluation. It does not satisfy a default fresh-install/first-task gate. Before 1C/release acceptance, the owner must choose and qualify the supported host-access topology and test actual browser journeys. The candidate already discloses default host failure and untested override, so this review's successful override adds E2 evidence without rewriting its historical packet or hiding the remaining default limitation.

## Other challenged boundaries

Progressive catalogue exposes only four named canonical fixture reads plus status and one-operation schema inspection. Future operations are marked unavailable and cannot execute through the detail endpoint. HTTP/CLI/MCP share canonical validators and errors; ordinary responses contain metadata, with audio solely behind explicit content retrieval. Direct MCP caps decoded bytes before HTTP; CLI uses 64 KiB ranges, validates metadata/hash and publishes exclusively. No opaque executable command bus or ML discovery imports/scans/downloads were found.

Canonical `source_types` is an unchanged required const of intended sources, not current acquisition availability. Empty engines/profiles, only implemented read operations, explicit synthetic MCP descriptions, X-Speech-Fixture header and `/skeleton` availability explanation make it safe within this fixture. No acquisition operation is advertised. A future generic consumer must not treat source_types alone as installed importers; that is a shared-owner availability design concern, not grounds to fork 1.3.0 here.

Node24 release packaging versus Node25.2.1/npm11.6.2 package engines is disclosed and produces warnings rather than demonstrated build failure. Accepted 0B explicitly calls its pins UI evaluation pins, not a runtime mandate. [Official Node release status](https://nodejs.org/en/about/previous-releases) confirms 24 LTS/25 EOL; preserving 0B while selecting maintained release Node24 is justified. [Official Python3.14.8 release](https://www.python.org/downloads/release/python-3148/) exists; [official MCP SDK documentation](https://py.sdk.modelcontextprotocol.io/v1/) explicitly identifies v1 as its maintenance line with v2 current stable. No claim that v1 is latest major is needed for this bounded SDK wire fixture.

UDS ping is only a private connectivity probe, with no execute payload, owner-fenced protocol, reconnect/cancellation/reap or artifact adoption. Writable staging is not a durable attempt namespace. Worker has no service/contracts code, DB mount, Docker socket or supplied credential environment. These observations support E2 container boundary only. Keep E1 schema/measurements, E2 synthetic processes/containers, E4 SDK fixture-versus-real-host distinction, and pending E3 hardware/E5 listening separate. Accepted 0B localStorage, 60-second/tiny PCM, split/replacement and other limitations remain unchanged; no browser acceptance is inherited from an HTML fetch.

## Handoff

No blocking corrections requested for this exact 0C candidate. Integration owner may proceed with its own exact-SHA integration verification and ledger decision. Preserve the default host-access limitation and the now-tested bridge alternative explicitly; neither this GO nor four passing tests authorizes a Phase1/release/runtime claim. Final Astra global review remains separate.

Final inventory note: all original volumes remained; both owned Compose namespaces were absent. One additional anonymous volume appeared during the review window with no project ownership label. Its origin was not established, so it was left untouched; no claim of global Docker volume-set equality is made.
