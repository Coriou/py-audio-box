# P1B23 SOURCE integration checkpoint — r149 Sol

2026-10-24. **Integration PARTIAL / dormant / validation pending; awaiting fresh independent Sol review. No Phase GO or operational authority.** Parent accepted r148 bounded plan/accounting and explicitly commissioned this one reversible source merge in an isolated branch. No source semantics were edited in integration.

## Frozen identity and ancestry

- Implementer: gpt-6.1-sol, low; no delegation.
- Own fresh T3 worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-410df38e`; branch `t3/source-integration-r149`. Initial tracked/untracked status clean.
- Parent root: `/Users/ben/Projects/py-audio-box`, branch `feat/agent-voice-workspace`; initial and pre-report-copy HEAD `a48f7d6a289269f9e6bbc6433058a6e27e45e717`. Initial tracked status clean; unrelated untracked STT research left untouched.
- T3 start pin equals current parent root pin `a48f7d6a289269f9e6bbc6433058a6e27e45e717`; no fast-forward necessary. Root→own starting ancestry is exact equality.
- Common ancestor `5ef14a18ba27e57dac6b74a30c086468060d4a07`; candidate documentation `1685451c9145d7d0b5cdd557ce11d5c25e96442d`; accepted source `bf7686711981465f064214d736b90f4417d841c1`.
- Historical candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-2f8963ab`: same HEAD and tracked-clean before/after, untouched.
- Merge `0d54ed08c023964a22134590f0e9fee42045e99c`; result tree `3c84cf7239d11c580b69ff78571d52e11ab1d0cf`.
- Ordered merge parents: `a48f7d6a289269f9e6bbc6433058a6e27e45e717` then `1685451c9145d7d0b5cdd557ce11d5c25e96442d`. Performed `git merge --no-ff --no-commit 1685451c9145d7d0b5cdd557ce11d5c25e96442d`, then a merge commit with hooks/signing disabled. No reset/rebase/squash/cherry-pick/checkout replacement or root branch mutation.

Full r146/r147/r148 and canonical PRODUCT/plan/protocol read; ledger current acceptance read. No applicable AGENTS.md found in checkout or checked ancestor paths. Memory registry search had no relevant r146/r147/r148 hit; no memory premise used.

## Preservation and documentation

Before merge, refreshed ancestor→root non-docs delta: **empty**, treating diagnostics Markdown as non-doc owned content (only docs/ is excluded). Refreshed ancestor→candidate ordered inventory: exactly33 paths and exact r148 modes/blobs. Source→candidate has no owned-content change. Canonical PRODUCT/plan/protocol equal on both sides and result.

Git reported **zero conflicts**, including zero semantic non-doc conflicts; no manual conflict resolution. Deliberate indexed-tree review confirmed exact union of both tracked path sets, with175 root-only paths preserved byte/mode exact and30 candidate-only paths retained. All274 root review files and all102 candidate review files retain their original bytes/modes; no deleted historical reports. No entire docs side was selected.

Root-relative documentation delta is exactly:

- `docs/agent-interface/phase-1/RECOVERY-P1B2.md`: candidate's historical append; original root bytes remain its exact prefix, candidate blob retained exactly.
- Added candidate historical reports `docs/agent-interface/reviews/B-COMBINED-CONTROLS.md`, `B-CONTROLS-IMPLEMENTATION.md`, `B-GUARD-IMPLEMENTATION.md`, each exact candidate blob.

Every other existing root document is byte/mode unchanged. In particular latest `EXECUTION-LEDGER.md` equals root exactly, retaining r148 acceptance and current commission. The ledger is the sole existing candidate path differing from result; result uses the newer root ledger. No conflict edits were needed. Outside docs/, root→merge changes are exactly the33 owned paths below: no hidden semantic delta or private proof added. All root-only tracked paths are preserved.

Canonical SHA256 (unchanged):

- `PRODUCT.md`: `a03974c734b1e80d4d15bc9adca9940fca78efdfd01ad894144b958515050879`
- `docs/agent-interface/IMPLEMENTATION-PLAN.md`: `fc74d47d23461c09eef7206b69269fcd12e01042552fe377bcf321c4d705a790`
- `docs/agent-interface/ORCHESTRATION-PROTOCOL.md`: `9ad225c96de2b5857f5d449f372a980c5dd769c3b25e1039583c7eef4787160d`

## Exact owned identity

All33 modes/blobs independently matched candidate and accepted source again **after merge commit**. Ordered inventory SHA256 (one `mode blob path` line plus LF per entry): `20c27ed22bf8da39b5e17eacaed32f4d655479a9b4d335d0de76db7f9cc49277`.

```text
100644 714a7cb5a68bda5e1d69a4efa2e905a261b0bcfd diagnostics/b-controls/HOST-overhead.py
100644 53e355a6f8f726ae606c9b06a09c12dba5c1b6a1 diagnostics/b-controls/HOST.py
100644 8b79e136168298d59c05e52c25c93f5b7f7a7168 diagnostics/b-controls/PLAN.md
100644 0b299239f9e25605f6234a54928a4f37bdabcd2b diagnostics/b-controls/runner.py
100644 06107f6cd20f3e0a22f43cbb952c40e8deaa923f diagnostics/p-one-node/FREEZE.md
100644 c3a1a1a4e3e0450483cbceabc8b02c00edf43936 diagnostics/p-one-node/HANDOFF.md
100644 5fd81204cadd5715187f3c860167809b4ab9f932 diagnostics/p-one-node/QUALIFICATION.md
100644 4e406acd567248ffc1ebae4ab8411554cc1920da diagnostics/p-one-node/dummy-writer.py
100644 77dc5e9f449054fff2186f3ad1367e0d7d2f1d2d diagnostics/p-one-node/entry.py
100644 342b31ac3c3213d0305c680c6050cf0f6ba75c30 diagnostics/p-one-node/finite-driver.py
100644 c517baaac698ed0cefdf237547ce1fdc7ca4254a diagnostics/p-one-node/host-check.py
100644 0bf502b0d271d607e37d93dabaf5817145e6d739 diagnostics/p-one-node/identity_semantics.py
100644 f5d386ac31654759caf5c8224dfbec80a697b377 diagnostics/p-one-node/inputs/P1B23-P-R5-ACTUAL-RECEIPT-AUDIT-r1-sol.md
100644 ca2ab5c9aef04514c3ec15bd7f1c2bd53d4cebcb diagnostics/p-one-node/inputs/P1B23-P-RESTORATION-RUNTIME-r5-20261023.md
100644 0aa1a15a9ebe5531d7f9576963211781efad9bb7 diagnostics/p-one-node/inputs/P1B23-P-RESTORATION-SOURCE-r5-sol.md
100644 57f8447b71c5133adbca68a48bd73b79592f5231 diagnostics/p-one-node/inputs/P1B23-P-WITNESS-SOURCE-r3-sol.md
100644 3dcc4ad8f6cb57ae5fc74197e63414508980d8cc diagnostics/p-one-node/launcher.py
100644 7534965d0cca2642f6d350c019d342885aac94e3 diagnostics/p-one-node/one_node_observer.py
100644 bc3014fab0347f269f110d14d1fcbef9dd292ad3 diagnostics/p-one-node/qualification-host-check.py
100644 85959cfd8116854eda8942576e2606852fea8921 diagnostics/p-one-node/qualify.py
100644 bb87a805e261b80eb8459f228c1f9a9a4a9a1164 diagnostics/p-one-node/runtime-identity.py
100644 4b13dd7d07fb4e9ca989132c5a3cbf403794088b diagnostics/p-one-node/source-manifest.json
100644 46b0695dd7a38449dc967cf46ae80d67a945ad5b diagnostics/publication-tests/check_final_gap.py
100644 162518a2505f7b45bb62427b31a78db28c575926 diagnostics/publication-tests/check_outcome.py
100644 1d5ab19787b96bf2b963f105dc1d6cc91fd51518 service/core.py
100644 9dae99df8d741a0cb4d3ac508d30776dc72e930a service/dispatcher.py
100644 55f9ef4fef62e6b94cb2e01e4334f518aa453318 service/recovery.py
100644 db8944d31a4ef99653f01581b2d912445a5d1b93 tests/_maintenance_wait_guard.py
100644 ce22fd030bb7ab1db038e0ce0565f9cf680d5bb5 tests/conftest.py
100644 bc536af3dabb35980a415c8704436efa2431695a tests/service/test_child_wait_guard.py
100644 3a89b1b14cd78b5f20023dd02e1ffdb2095d896b tests/service/test_core.py
100644 e1db49ea53816c68376db1f94f93d8ebc1b5ce88 tests/service/test_dispatcher.py
100644 9080bb2a813fefdf945312bb29b545596f050760 tests/service/test_recovery.py
```

## Verification and limits

Read-only Git status/diff/ls-tree/merge-base/ancestry and SHA256 comparisons establish the above. `git diff --cached --check` passed before merge commit. All24 owned Python paths passed stdlib AST parse and `compile(source, filename, 'exec')` without executing compiled code, importing candidate modules or writing bytecode. This is syntax/source identity evidence only. Own checkout was clean immediately after merge. Historical candidate HEAD/tracked status and root HEAD/tracked status rechecked unchanged before this report write.

No Docker, tests, checkers, pytest collection, runtime, build, lane/resource probe, activation, cleanup, push or delegation. Hooks were disabled for Git merge/commits. Only this new report is written after merge, committed separately, then copied alone to root; no ledger update/root commit. Other historical worktrees/private proofs were not modified or opened. Root untracked STT content was not opened or altered.

Source-content identity carries bounded source review history; it does **not** relabel old receipts as naming `0d54ed08c023964a22134590f0e9fee42045e99c` or the later report commit. Mapped manifest/sourcecommit literal pins remain historical until a separate reviewed binding. No frozen manifest, image, private preparation or approval marker changed. Identical executable content alone neither creates new actual evidence nor closes a consumer's literal-SHA requirement.

## Unresolved gates and parent handoff

- Closed81 indices019–099 remain UNRUN. No selector/private prep in this unit. Future runner/HOST/manifest/PLAN allowlist and full operative consumer closure are a separate commission/review; no more diagnostics. New selector bytes and their source/image binding require their own review. Capacity and admission remain parent-owned.
- Separate C/T2 branch `1d741cb328228e67f99a096e9d6a9c1acd1f01b6` was not imported. Current recovery tests still have zero `_lotc` definitions. Combined C/current ownership/guard integration and validation remain open.
- Native V1 positive-predecessor failure and snapshot SIGSEGV/masked signal, V2 SQLite UTF8 and V3 missing SHM, plus later helper/collection crashes remain distinct unresolved rows. Native cause UNKNOWN; Linux evidence does not qualify Darwin.
- Original-entry/affected preservation, ordered four-B2 and full service/release then complete offline suite remain pending on a declared combined source. No historical prefix or syntax check replaces them.
- B3 bounded transient cleanup, C1 installed persistent topology and C2 same-volume restart/fault matrix remain downstream; A1 global-inventory limitation remains open. D1–D5, shared HTTP/CLI/MCP/browser walking slice and later real-engine/profile/listening/full-product gates remain as canonical plan.
- Historical campaign **17 PASS /018 FAIL /81 UNRUN**, separate017 PASS and accepted current018 one-sample PASS remain unchanged and separately attributed. r135 campaign closed, ONE sample STOP honored; no all100 or reliability claim.
- Owner `c9b6506a-0f20-4440-8ddd-be41f35bb82b`, all resources/proofs/lane remain HELD as carried from accepted reports; no live state probe or mutation.

Parent must obtain **fresh independent Sol review of this exact merge/result/report** before adopting the integration branch or merging to root. Parent alone owns ledger/adoption and later separately bounded validation authority. This implementer supplies a reviewable checkpoint, not self-GO.
