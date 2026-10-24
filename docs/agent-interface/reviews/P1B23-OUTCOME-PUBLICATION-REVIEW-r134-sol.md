# P1B23 outcome publication — independent r134 Sol review

2026-10-24. **Bounded SOURCE / retained inert-HOST GO for the r133 publication correction. No blocking finding in this scope. Operations remain NO-GO.** The one-line global declaration repairs the proven publication scope error without resetting the IO latch or qualifying the earlier child failure. This review grants no execution, build, retry, release or cleanup authority and contains no operational approval markers.

## Identity and review boundary

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-2ddc93b1`, branch `t3/fix-outcome-publication-scope`, clean at documentation `d8ffc4e9386c927741d27b1c27ac7dbe9d5529e1c`, source `5c8c507eb61be6ff5ce121ddb290e80cc9484e1c`. Immutable baseline is documentation `ff9c4efad932b5b61fe79eb14fab393811a6eb38`, source `b3b34cb9ac5f9cbb6db791179cd91a93b3c97da2`; historical `/Users/ben/.txcode/worktrees/py-audio-box/t3-19f692dd` independently remains clean at that documentation ref.

Read full r131 actual-failure report, r132 independent failure/proven-P1 review and r133 implementation report. Inspected actual controls publication/qualification/schema/handback source, real guard serializer/reader, standalone checker, retained capture driver, raw outputs, fixtures and preparation history. Used only source reads, Git reads, hashes, JSON/AST and source compilation; read the two actual PAX members into memory, with no extraction. Did not run the checker, pytest, collection, guard/product/helper modules, Docker, lane, build or operational code. No delegation, private edits, commits or cleanup. Sole write is this report. No applicable AGENTS.md was found in checked workspace/report/candidate ancestors or tracked workspace inventory. Memory supplied only the general evidence-tier boundary, not a technical premise for this verdict.

## Exact diff and pins

Against the immutable documentation baseline, the source commit changes exactly four files. Comparing directly to the earlier source ref also includes the already-existing r120 preparation document; that inherited documentation is not an r133 source change.

- Controls: exactly one added `global diagnostic_io_failed` at `publish_outcome` entry.
- Runner: only the controls digest literal changes.
- Manifest: only controls and runner digest values change.
- New `diagnostics/publication-tests/check_outcome.py`: standalone stdlib focused checker, outside the tests tree and ordinary `test_*.py` / `*_test.py` collection naming. No collection was executed.

Independently rehashed all 193 file-map entries and four derivative-map entries: all 197 match. Membership is unchanged; only the controls and runner values differ. The checker is outside that map. The exact diff leaves guard, real Observer, original target, product recovery, timing/deadline/channel logic and all 39 mode predicates frozen. Source compilation of the changed Python files and frozen guard succeeded without executing them.

SHA256 independently verified:

| Artifact | SHA256 |
|---|---|
| controls | `fbea91ee219305f934b319581f58b0b3ad3b6839abbbcf877298751a1c9512db` |
| runner | `ab618bb7a23a264dbd3376a1b709a979d24fe8ba632ab19529d5c4efe7f1a245` |
| manifest | `7cf25d609bffe786f6f2cf911c09d9852d38af6d9f3d9865419e9f2fb185c3c0` |
| checker | `5033d39028fcb126c122b5c1cd7323273b1a7a2d44aa91bc7209bb1be0ff7a45` |
| guard | `d3a011ffcc50e785f39abba54f07b51fe733b132425fff8dfc9a9848c92d254d` |

## Publication correctness

The baseline assignment in the validation-exception handler made `diagnostic_io_failed` local throughout `publish_outcome`, so its earlier record/completeness reads were unbound. The global declaration correctly selects the existing module latch, matching `publish_diagnostic`. There is no new initialization/reset, broad fallback, exception-class change or native-success waiver.

Source preserves original exit normalization and phase capture, duplicate/unknown phase rejection, saved qualification copy, teardown/final snapshots, finite identity and external node/private binding. The real validator still checks child decoding, parent operands, qualification equality, schema and diagnostic completeness; exit0 without fully qualified PASS raises. Publication validation failure latches IO failure and records RED. The real bounded atomic JSON serializer remains the final handoff. Missing/incomplete records cannot become positive baseline evidence through this fix.

The exact nested finally still publishes diagnostics/outcome, restores the previous survival binding and closes the mapping. Successful publication preserves a propagating primary exception; an unexpected publication exception can supersede it with the primary retained as context. The fix does not claim to eliminate that pre-existing exception behavior. Post-handback IO assertion remains mandatory on normal flow; nonzero pytest/result handling is not replaced by publication status.

## Retained evidence reconciliation

Independently verified exact inventory membership, every listed size/SHA256, current UID, regular single-link leaves0600, directories0700 and absence of symlink leaves across all six r133 roots. Inventories exclude themselves. Verified capture checker hashes and raw stdout/stderr hashes. These are retained capture records, not a fresh execution or independently witnessed process lifecycle.

Final before root `/private/tmp/pab-r133-inert-e07f3a4e-22dc-425c-a7c9-ef917e554875`: OS0, five labeled assertions, expected UnboundLocalError for both finite identities after deferred terminal failure. Its retained `controls-source.txt` is the successor copy made by the generic driver; the argv explicitly selects the original controls copy under the initial `93b659e8-...` root. Independently compared that selected copy to immutable baseline bytes: exact match. Thus the before regression is genuinely against the unfixed source. Final checker revision is identical to the committed checker.

Final after root `/private/tmp/pab-r133-inert-a0238163-d3cf-4513-881f-063544a88ae2`: OS0, 131 labels, empty stderr. Raw stdout SHA256 `886a2a2315ed63614bf09180f4a9c01d4e3b3f759c37d2f3a360ad681fc55ad9`; final inventory95 entries SHA256 `240e9790bbd95b9d4b0405dc019a58de6db856b878e1ba626e51e2861dac1e87`. Before inventory11 entries SHA256 `3caaeac4f0a3ab393ca18e8ffff1d2ba56409233ee2a3e63706bf4e055a236b1`.

History is preserved: initial before OS0; first after OS1 with raw FileNotFoundError caused by the checker's external-binding loop reading its deliberately missing child leaf; corrected intermediate OS0; expanded OS0; final before/after OS0. Read the initial traceback and checker revision. Correction moved binding checks onto coherent records; owning source remains the same one-global correction. This was an inert checker preparation failure, not a hidden operational retry.

## Meaningful coverage and limits

The checker AST-extracts actual publication, decoder/schema/validator functions and exact handback finally; separately extracts real `atomic` and `metadata`. It does not import or execute the whole guard/helper/program. Its inert OS adapter delegates stdlib calls and adds O_NOFOLLOW to temporary opens. No canned validator PASS exists. Synthetic guard objects supply state, while real filesystem reads, validation and JSON/atomic publication run in the retained checks.

Both identities cover coherent COMPLETE/PASS; coherent completed-child failed qualification with exit1 and teardown FAILED as COMPLETE/FAIL; prelatched IO preservation; missing, malformed, mixed-identity and invalid-qualification records; injected read/write OSError; saved qualification/phase/exit/snapshot preservation; external binding rejection; and later guard mutation not rewriting persisted captures. False exit0 invokes the real validator, latches RED, and the extracted exact IO assertion raises. Handback cases exercise restoration/close and primary exception identity/context using inert mappings.

Pinned PAX SHA256 `5f75f562e9827425b1072705c09659c5a2d6afa14fa0851b698a42f0f101ef6b` independently matches. Its actual018 child and parent JSON equal the retained COLLECTED_CONTROL raw9-incomplete fixture records exactly. Raw9/E/TK, INCOMPLETE child, false/false/false/true/true qualification and first failure terminal remain RED/FAIL. The017 version is explicitly a synthetic identity projection. Neither actual r131 final nor teardown snapshots exists: supplied snapshots are inert fixtures, not recovered actual evidence.

Coverage gaps are nonblocking for this one-line scope correction: 131 counts assertions, not independent cases or runtime qualifications. The full pytest/plugin/Observer/guard process, actual mmap/fork/scheduler/deadline behavior, native cause, faulting OS subprocess exit and complete operational serializer environment are not exercised. Injected reader/writer errors do not enumerate all filesystem failure points; mutation checks cover supplied state/persisted records rather than every live alias. The escaped-primary handback case supplies result1 and does not exercise the real outer `result='ESCAPED'` path end to end. Disabled diagnostics, all duplicate/unknown phase variants, every exit enum, oversized/duplicate-key JSON and every binding/schema permutation lack dedicated retained cases. Source inspection and the mechanically frozen implementation support these unchanged paths; no new blanket coverage claim follows. This review does not repeat r132's physical-PAX or full ACTUAL retention audit.

## Disposition and finite next stage

Accept the minimal global declaration plus focused standalone checks and mechanical pins as bounded SOURCE / retained inert-HOST evidence. No further source fix is required for this publication defect. A supported finite next stage is a source-only diagnostic design for observing the interval after child final gate2 and before hook stage1, preserving deadline, channel, guard predicates and qualification rules; it must not presume a child cause or silently alter the positive contract. No needless checker rerun or generic platform gate is recommended here. Any operational successor bundle or approval is separate and has not been prepared or authorized.

Actual r131 remains FAIL/RED; native child cause and NativeT2 remain UNKNOWN. Publication repair alone cannot qualify018. Campaign remains **17 PASS /018 FAIL /81 UNRUN**; separate017 PASS remains separate. Resources and lane remain HELD under owner `9a2e44f4-dd52-479d-9f6f-514eaee9ecbb` per the retained authority/evidence; no live lane operation or resource-state probe occurred in this review. No acceptance, execution readiness or release is conferred.
