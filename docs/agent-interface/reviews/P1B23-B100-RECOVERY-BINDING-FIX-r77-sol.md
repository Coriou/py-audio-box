# P1B23 B100 recovery binding fix r77 Sol

2026-10-24. **PREPARATION SUCCESSOR; acceptance=false. NO-GO for actual dispatch until fresh independent review and explicit parent approval.** Implemented the concrete R1/R2/R3 corrections and focused supervisor UNKNOWN-publication handling. This report is implementation evidence, not an independent approval. No Docker/daemon/volume access, product/guard/pytest import, campaign/test retry, image rebuild, provisioning, commit or delegation occurred.

Read r76 in full, including its findings, positive source/PAX/HOST findings and preserved decision boundary, and the full r75 preparation and r74 recovery scope. The current implementation brief authorizes only this private successor and bounded stdlib HOST checks. The memory registry search returned no relevant premise. No tracked source was edited; this report is the sole repository write. Existing unrelated untracked artifacts were preserved.

## Selected successor and exact identity

Preparation root: `/private/tmp/pab-r77-recovery-prep-20261024`. **Only its `selected-READY/` subdirectory is selected for future execution.** Root-level files, `selected-final/`, earlier manifests, fixture approvals, failed checks and draft seals are retained preparation history, not alternate execution entries.

| Selected artifact, relative to `selected-READY/` | SHA256 |
| --- | --- |
| `frozen-READY.json` | `a84c0deaf693b431c73a29a94e2594e4e25446d5793f8a5ac74a99974eef4f49` |
| `bootstrap.py` | `457c959d9405d1e78e212c1f53febf2ae79b17ce41a6284b7806135886c07614` |
| `launch-boundary.txt` | `9e594accc9b8945d6eb5515790a55b0ead628e89570e7b25ea077baccc14da4c` |
| `entry.py` | `6c06c831612a15544c1ab2a542cf799f7e2ba1ad41b180a1e0c2d62008d1346a` |
| `worker.py` | `8f1c54e57523013fd834c1ef8aab11e93a9111223098dcdd0fe8739234137ddc` |

The manifest freezes executable and data inputs, including the bootstrap and launch text. It does not include its future independent review or approval. The separate whole-preparation `review-bundle-FINAL.json` freezes selected bytes, concrete argv, provenance, test sources/receipts and retained history: SHA256 `abbf26b249ec86b10f6579faef1bd3d5353039112b85528f984b7cdd29d2c4c7`.

`reader.py`, `codec_v2.py`, `capture.py`, `identity.py`, `privateio.py`, both concrete argv arrays, bindings and old-proof baseline are byte-identical to r75. Their equality is retained in `unchanged-positive-bindings-FINAL.json`. PAX is the sole transport. The image, candidate pins, volume, two old full container IDs, readers, helper names, resource/security constraints, order and original bounds remain unchanged. No LP1 or additional volume microgate was introduced. r76's positive PAX/source/retained-HOST findings remain accepted within their stated limits; its existing 26-check PAX suite was not rerun.

## R1: parent and independent-review binding

The stdlib boundary requires exact parent-pinned manifest and authorization digests. Approval binds the independent report's absolute path and SHA256, the manifest reviewed by that report, an explicit GO verdict and explicit parent acceptance. The report bytes themselves must contain these exact separate LF-terminated lines:

```text
R77-INDEPENDENT-REVIEW: GO
R77-REVIEWED-MANIFEST-SHA256: a84c0deaf693b431c73a29a94e2594e4e25446d5793f8a5ac74a99974eef4f49
```

A NO-GO marker is rejected, as are a missing GO marker, wrong report hash, wrong reviewed manifest or unaccepted review. These markers are a review/parent binding convention, not a cryptographic claim that an arbitrary report is independent. Parent must actually obtain and assess a fresh independent review of the selected manifest. r76 remains NO-GO and cannot satisfy this convention.

Required future private parent JSON fields:

```json
{
  "authorize_one_recovery": true,
  "parent_accepts_independent_review": true,
  "acceptance": false,
  "frozen_sha256": "a84c0deaf693b431c73a29a94e2594e4e25446d5793f8a5ac74a99974eef4f49",
  "reviewed_manifest_sha256": "a84c0deaf693b431c73a29a94e2594e4e25446d5793f8a5ac74a99974eef4f49",
  "independent_report_path": "PARENT_ABSOLUTE_FRESH_INDEPENDENT_REPORT_PATH",
  "independent_report_sha256": "PARENT_EXACT_REPORT_SHA256",
  "independent_review_verdict": "GO",
  "synthetic_fixture": false
}
```

No operational approval was created. All fixture approvals/reports are explicitly SYNTHETIC/NONOPERATIONAL and have `synthetic_fixture:true`; the execution boundary refuses them before private module loading. Tests deliberately call validation/module-loading functions separately to exercise source behavior without granting execution authority.

Future entry is the exact concrete argv array in `future-entry-FINAL.json`: `/usr/local/bin/python3 -I -S -B -c LITERAL_LAUNCH_BOUNDARY ROOT MANIFEST_SHA PARENT_APPROVAL_PATH PARENT_APPROVAL_SHA BOOTSTRAP_SHA`. ROOT is `/private/tmp/pab-r77-recovery-prep-20261024/selected-READY`; manifest/bootstrap pins are above. Parent must use the exact reviewed literal from `launch-boundary.txt` in its invocation, not execute a mutable bootstrap pathname or dynamically trust whatever launch file is present. The 1439-byte stdlib launch boundary checks bounded stable authorization/bootstrap bytes against parent invocation hashes **before exec** of the verified bootstrap bytes. Parent also pins its literal launch text by the published hash above.

Freeze manifest first, obtain fresh independent review second, then parent creates/fixes approval bytes and their invocation digest. This avoids recursive self-hashing. The launch read and bootstrap's subsequent authorization read both check the same pinned digest; thereafter the accepted authorization bytes are carried in the snapshot. Supervisor and worker record and verify the same authorization digest and snapshot identity. Worker never reopens the approval pathname.

## R2: validation before private execution and one immutable input snapshot

The bootstrap imports stdlib only. It reads the manifest and every selected file with bounds, no-follow/component checks and stable regular-file signatures, then validates every length/hash before executing any private module. It also verifies selected provenance sources and all197 AUTH candidate mappings before private execution. No already-executed worker supplies this gate.

Private modules are compiled from admitted immutable `bytes` through a closed private-import cache and per-module import hook. No private path import or top-level private code runs before the gate. The snapshot holds manifest, approval/report bytes, bindings, provenance, reader and argv bytes; the selected-file map is read-only. The bootstrap requires exact reader-prefix/subset equality plus a fixed literal main suffix, and requires both39-element argv arrays to embed those exact reader bytes in `python -I -S -B -c SCRIPT MODE`.

Worker selected-input reads/hashes resolve to snapshot bytes. Thus create argv and later inspect predicates read the same admitted bytes even if corresponding files are replaced afterward. External operational observations, new command receipts and old-proof reads remain actual fresh reads; they are not falsely served from a snapshot.

Supervisor publishes one exclusive0600 snapshot envelope and passes its SHA256 and authorization digest with the **verified literal bootstrap bytes** to an isolated worker interpreter. Worker bounds/reads the envelope once, verifies that identity, authorization and all embedded manifest/file/report/reader bindings before private execution, and retains the same snapshot identity in its own evidence. Envelope replacement or a different worker snapshot fails before mutation. Unchecked approval/provenance/argv pathname reloads no longer supply operational decisions.

The parent literal launch boundary starts the360s alarm before admission. The worker bootstrap starts its330s alarm before worker admission; entry/worker preserve those start times rather than restarting their watchdog budgets. The outer worker capture remains345s; reader/phase/alarm/resource limits are unchanged. This is bounded software behavior, not a guarantee against uninterruptible filesystem stalls.

## R3 and supervisor publication

Node017 now requires enclosing `STOP` and integer `exit==1`, plus the exact result object: selected node, integer pytest_exit1, collected=true, setup passed/call failed/teardown passed, accepted=false, with no extra fields. Existing selection, index, seconds, two-stream set, stream lengths/hashes and failure-output checks still apply. Missing/different/Boolean exit, wrong node or extra result fields refuse content acceptance. `transport_verified` is established separately before content reconciliation and remains true when content is inconsistent; `content_verified` stays false. Historical FAIL is not promoted.

Supervisor completion parsing and worker-final decoding are inside a refusal handler. Missing/torn/malformed final publication yields UNKNOWN even after a synthetic successful/reaped worker CLI. Early publication, inventory/readback and late publication errors reach an independent exclusive `unknown.slot` path when possible; there is no command retry. Normal completion publication follows inventory validation, with final publication files explicitly excluded from that inventory. A FINALIZATION-PENDING record is never success.

Supervisor's original16MiB allowance is partitioned into14MiB ordinary capture/snapshot,1.5MiB normal finalization and a separate512KiB emergency slot reserve. Worker still excludes the original supervisor16MiB and worker-final16MiB from its ordinary4GiB budget. The combined allowance is not enlarged. An emergency UNKNOWN record/slot or nonzero/unknown supervisor OS outcome overrides any earlier completion-looking publication. Future actual-receipt acceptance must reconcile supervisor OS exit0, complete worker evidence, valid final slot/inventory and absence of UNKNOWN/incomplete publication; a JSON state alone is insufficient.

A full filesystem, hard stall, alarm exit124 or process death can prevent any handback. UNKNOWN publication is best effort within the existing deadline; neither local CLI termination nor emergency publication establishes remote cancellation. All objects, prefixes and lane remain held; no retry or cleanup follows uncertainty.

## Final source-bound HOST evidence

Only the final selected receipts below qualify the selected successor. Both isolated stdlib subprocess suites exited0, COMPLETE, with leader reaped/descriptors closed, exact pending/completion argv and raw stdout/stderr hashes reconciled. Sources are bound through each source-binding JSON and the selected manifest. Exact nested subprocess argv, spawn, raw streams and OS exits are retained for worker-snapshot negatives.

| Receipt in `selected-READY/` | SHA256 |
| --- | --- |
| `host-suite-v3.complete.json` | `0ec37f8e535d347a3acdba38e391c71e729c58ab8884bc38741323b47696261a` |
| `host-suite-v3.stdout` (441 bytes) | `82b9f3eed334403e20fc3ab598fe546fe0ab3d8f8424c1d255b94e95780ae629` |
| `supervisor-suite-v3.complete.json` | `6cc386db61c6636b5fea22eda6bbfd16b6b5df95aae4bc05aab50fbd00e04dc4` |
| `supervisor-suite-v3.stdout` (686 bytes) | `0a09f4e73f198aa34a6188fd343e778468ebabc2d2049721e1b39221855557d8` |

Both stderr streams are empty, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The25 HOST checks cover positive synthetic admission with operational refusal; stale/wrong report bindings, wrong reviewed manifest, unaccepted and NO-GO approvals/reports; supervisor→worker approval replacement; altered transitive module before import with a never-created mutation sentinel; post-admission argv/bindings replacement with unchanged snapshot reads; different worker snapshots;017 missing/wrong/Boolean exit, extra result field and wrong selected node; missing/torn finals and publication UNKNOWN. Actual daemon mutation dispatches are zero. These checks validate the shared selected-input reader used by create/inspect; they do not claim an actual helper create/inspect lifecycle.

Four additional integration fixtures run the actual selected supervisor control flow with a synthetic worker-call adapter: missing final, torn final, final-inventory publication exception and initial source-publication exception. Every case returns1 with checksummed UNKNOWN and acceptance=false. No synthetic adapter invokes Docker. This covers the focused publication issue beyond calling only its decoding helper.

Preparation history is retained honestly: initial25+4 checks passed on earlier bytes; v2 HOST setup failed on duplicate exclusive bootstrap copy and its dependent supervisor fixture lacked a snapshot. Distinct corrected v3 fixtures then passed. `selected-final/` exposed an UnboundLocalError after an inline-comment edit swallowed initialization; it returned UNKNOWN, and that source/failed receipt remains retained. `selected-READY/` removes that defect and has its own25+4 passing, exact-source receipts. None of these HOST preparation fixtures is a campaign/pytest retry or operational receipt; no failed history was overwritten or represented as passing.

## Preservation, inventory and remaining gate

Before/after complete no-follow censuses match for the **entire** r75 preparation root, both older proof roots and the held lane directory: path/type, regular hashes/lengths, link targets, mode/UID/GID/nlink/device/inode/mtime/ctime, and directory identities. `preservation-check-FINAL.json` retains the reconciliation. No old root, object or lane was changed.

The new root has459 regular leaves,55 directories and24298125 regular bytes; all directories0700, all leaves0600/current UID/single-link, no symlinks. Preparation used umask077 and exclusive new file/root creation; draft updates are retained through history/seals. `preparation-inventory-FINAL.json` SHA256 `768eeb87a5e34a1d6b5da963d437535bf6314673909a9a8a57ec3fa7d3a9881d` inventories457 leaves plus its two explicit self/publication exclusions. The checksummed512KiB `preparation-final-FINAL.slot` SHA256 is `28bf864e6436977b1c28d3d811222c222ca797e9862f15c02a448be273064d76`; inventory and slot were reread, and directories fsynced. The freeze utility's convenience console count adds3 rather than2 to the inventoried leaves; the authoritative path set is457+2=459. This bookkeeping display discrepancy has no selected runtime role. The authorized repository report is separately excluded from private seals.

Remaining gate: fresh independent source/HOST-evidence review of **selected-READY's exact manifest**, then explicit parent acceptance/report binding and private approval digest before any actual execution. There is no operational authorization in this handback. Cross-version Linux PAX serialization, daemon behavior, live volume content, actual recovery, hard-stall publication, product/nativeT2/platform/hardware/durability/release acceptance remain unqualified. Historical017 FAIL, retained=false and018–099's82 UNRUN are untouched. A later actual recovery still requires its separate independent receipt audit.
