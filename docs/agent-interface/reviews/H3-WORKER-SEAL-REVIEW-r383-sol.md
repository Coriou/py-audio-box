# r383 — bounded r382 worker-seal adversarial review

2026-10-25. **BOUNDED SOURCE GO. No blocking defect found in the r382 worker slice. Full r378 family SOURCE NO-GO; production installed export UNAVAILABLE; release NO-GO; legacy CLOSED.** This is independent source review, not adoption, merge, activation or qualification authority. Parent keeps the candidate unmerged for the next integration slice; final Astra review follows the complete family.

## Exact authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-e320873b` was clean before and after execution. Verified direct-parent chain:

- `359e8aeb2bcfab20f9fb8c5d3fd77997ce28f6db`: report only, sole changed file r382 report.
- `8749e75519f41d9052978ed43a7127e637006c2f`: worker source candidate.
- `ff9a4d01eb53daad9ec8987b4b6d375f39ab1ae1`: reviewed r380 report pin.
- `c9b391f9e231e7f0b92b455fac6f226970b92d5a`: initialization source.
- `c73e109272a3e27a038824ca72fb698a4e6578da`: r379 report.
- `3ac457d8532d6fe62764ae760b621a698e30e393`: pure primitives.
- `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`: base; ancestry verified.

No applicable AGENTS.md found in candidate descendants/ancestors or root report destination ancestors. Read full root r378 and r381, candidate r382/r380, complete r382 source/test diff and relevant converter, managed-reference, worker-child, lifecycle and configuration context. Root r378 SHA-256 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`; r381 SHA-256 `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`. Diff whitespace check passed. Memory registry search had no relevant hits; conclusions use current source and independent execution.

Sole persistent write: this root report, mode 0600. No source writes, scratch runner/log writes, commit, merge, delegation or self-adoption. Existing unrelated root untracked evidence was left untouched. Authorized fixture lifecycles created/disposed only inert temporary files.

## Findings and bounded acceptance

No blocking findings within the expressly selected slice.

1. **Closed startup authority.** `worker/runtime_supervisor.py:576-589,592-635` requires explicit new policy/configuration and a separate mandatory expected worker image; old/local selection cannot consume that image argument. `export_startup` validates detached configuration, image equality, actual executable hash and declared immutable closure before serving. Constructor repeats the check and derives image/principal/reference authority from the checked configuration (`:22-42`), refusing meaningful parallel converter/image/reference maps and installed Pocket selection. Empty maps supply no alternate authority. Expected image is trusted launcher input, not Docker self-attestation. Complete real closure acquisition and running-image corroboration remain later brackets.

2. **Mandatory exact seal, bounded transport.** `worker/runtime_supervisor.py:99-108` adds the six-field export seal only for the explicitly configured family. `service/installed_export.py:75-115` binds canonical full config/converter/profile/principals and strict integer publication format 2. `service/host_install.py:1163-1180` preserves the closed old four-field parser and validates the selected new shape through it. `service/supervisor_runtime.py:15-46,91-160` detaches configuration and selects the new validator only explicitly, including dropped-reader startup/hello. These are authenticated transport seams, not owner activation. Existing maintenance hello extensions remain subject to the existing refusal discipline; no optional export key is admitted to the old parser.

3. **Independent work gate.** `worker/runtime_supervisor.py:348-373` compares configured principal membership, canonical converter registration and the full frozen profile snapshot, not only profile ID or Python numeric equality. Qualified/installed/valid/ready/measured plus a nonempty qualification receipt are independently required before source acquisition. Local converter hash/closure is checked again. Prepare, start and retained source acquisition all invoke this gate (`:123-124,186-187,208-211`). Unqualified configuration may assert its inputs in hello but cannot do new export work. An injected receipt is not operational qualification.

4. **Private descriptor and uncertainty boundary.** `worker/runtime_supervisor.py:374-402` uses existing `managed_reference.acquire`, then independently checks regular type, UID/GID10001, mode0600, single link and O_RDONLY via `worker/managed_reference.py:27-34`. It enforces positive size <=24 MiB, bounded complete PCM/hash metadata, stable device/inode/size/mtime/ctime across reading and rewind. Failure after acquisition closes the FD. Acquire failure quarantines the supervisor; existing acquire (`managed_reference.py:58-100`) requires helper exit zero before returning, kills/waits on uncertainty, closes transferred FD and retains an unknown process handle. No root direct-open shortcut or permissions relaxation was introduced. Principal membership is checked before helper use; public source ownership and full Authority remain existing owner/child responsibilities, not claims established by a source ID alone.

5. **Compatibility and publication fences.** The legacy local converter mapping/direct-open branch remains in place. Pocket still uses its existing helper branch; new-family configuration projects only validated Kokoro fields for existing bundle checks and does not enter Pocket. Native activation, owner/public installed export gates, scheduler, child authority authentication, descendant reaper, proposal/publication/ACK and capacity ordering are unchanged. Retained source FD is passed through the existing isolated-child handoff and closed by the supervisor (`runtime_supervisor.py:208-227`); `worker/runtime_child.py:99-123` authenticates Authority and treats converter output as a proposal. Compatibility here is source inspection plus selected inert tests, not broad native/Pocket/v1 operational regression qualification.

## Independent finite execution

Inspected `/private/tmp/r382-tests.py`, exact candidate path/cwd and all selected test code before use. Read `tests/conftest.py` and `tests/service/conftest.py`; there is no worker conftest. No pytest collection or server fixture was invoked. Ran exactly once:

`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r382-tests.py`

CPython **3.14.7**. **42 methods PASS in 0.660 s; forbidden-event attempts `[]`.** Counts: 12 ExportSealTests, 5 AudioExportTests, 3 ExportWorkerBoundaryTests, 5 ManagedReferenceTests, 7 ConditioningSupervisorTests, 10 InstalledExportContractTests. This independent result is separate from r382's implementation-run timing. Audit hook installed before application/test imports blocks subprocess.Popen, os.system, socket.bind and socket.connect. Fake Popen/process/UID functions are injected; no real child or identity drop occurred.

One further inline invocation using the same interpreter flags and pre-import audit hook exercised **six adversarial refusals, all PASS; forbidden attempts `[]`**. No helper script was written. Reproduction uses existing ExportSealTests.setUp/qualified/source fixtures and real tiny FDs, mocking only converter FD acquisition/helper transfer and host ownership observations:

- Change source mode0600 to0644; reject and close returned FD.
- Return O_RDWR source FD with otherwise private metadata; reject and close.
- Add a second hardlink; reject and close.
- Wrap read_bounded to fchmod the source after reading valid unchanged PCM/hash bytes; reject changed ctime and close.
- Rewrite actual inert executable bytes, restore0555, retain configured old hash; verify_local_converter rejects.
- Change configured full profile thread limit while retaining old hello seal; validate_seal rejects.

Existing tests additionally exercise foreign owner/modes, traversal/symlink/alias, bad source hash, profile numeric substitution, principal/converter/image mismatch before helper, unknown helper wait with retained fake process and closed FD, closure writable mutation, strict versions/closed keys and old hello refusal. No finite-boundary exhaustion at 24 MiB was allocated: the positive/maximum check and bounded reader were source-inspected. Mocked ownership is not Linux DAC or kernel cross-UID evidence. No new diagnostic framework or repeated fault campaign was introduced.

## Remaining parent-owned gates

Actual converter pin/complete closure inspection, explicit image build target, daemon selector/commands/mount subpath and assets inode cross-view, credential corroboration, image/config/seal/boot registration brackets, owner activation/AudioExports, retained history/readers/160 MiB accounting and public >1 MiB HTTP/CLI/MCP upload/export/download/reopen/replay/generated-asset journey remain UNWIRED. They must keep production unavailable and are not blockers to this bounded worker slice. Integrated cancellation/timeout/output/descendant uncertainty/publication/ACK/capacity evidence belongs to the later complete integration review.

No Docker, network, listener, application child, FFmpeg, ML/hardware, HELD/STT inspection, actual migration/install/download, actual pin acquisition, operational cleanup or heavy-gate waiver. All actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification **UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. Seven real-child reference-publication tests remain **BLOCKED/UNRUN**. **Bounded worker SOURCE GO only; full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
