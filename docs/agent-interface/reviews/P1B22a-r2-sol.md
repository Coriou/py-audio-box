# P1B2.2a independent adversarial review — round 2

2026-10-16 · gpt-6.1-sol / low. **GO for bounded B2.2a source/E1 acceptance only** at exact HEAD `2f7d0705686c3124321b73dffca3599af2a153b1`, frozen source/tests `9d273aad1d50be38ecb66ec6cd4dbad06e1b1c84`, production `ba120cb81c967a4fd33f969ca0e5c65960cc83bd`. Round-1 F1 and F2 are closed within the evidence below. No new concrete blocking finding. This approves only the private accepted-intent/evidence prefix stopping under pending authority. Parent integration verification, remaining B2.2 transitions, B2.3 trusted host transport, B3, C1/C2, final Astra and public/unattended acceptance remain pending.

## Identity, scope and basis

Inspected `/Users/ben/.txcode/worktrees/py-audio-box/t3-c72faa32` read-only. Initial and final HEAD match the requested SHA; status is empty before and after all execution. Production exactly matches `ba120cb`; source/tests exactly match `9d273aad`; the latter-to-HEAD delta is handoff-only. Fresh-root baseline `177cdd7` to HEAD changes exactly `service/recovery.py`, `tests/service/test_recovery.py`, and `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. No Core/initializer/supervisor/reconciliation/host/contracts/dependencies/packaging/ledger/prior-review change. `git diff --check 177cdd7 HEAD` passes. No applicable ancestor/scoped AGENTS.md found, including the root report destination ancestry.

Read the complete root P1B22a-r1-sol findings, reproducers and minimal fixes; complete P1B2-BOUNDARY-ASTRA architecture/records/plan-resume/B2.2 requirements; complete candidate RECOVERY-P1B2 original handoffs, B2.1 corrections, original B2.2a and latest responses; ledger split/correction entries, orchestration protocol, P1B21-r3, original P1 implementation decision/exit matrix, R1 authority/final-review and B1 original/correction/final-review context. Inspected the complete production delta, added tests, actual protected readers, archive grounding, persistence/publication and worker startup seams. Memory registry search returned no relevant B2 entry; no memory-derived acceptance evidence is used.

## F1 — high — closed: original snapshot survives committed acceptance and actual process death

Locations: `service/recovery.py:1248-1287, 1303-1328, 1339-1363`; process-death regressions `tests/service/test_recovery.py:1316-1343`.

The fixed receipt names derive from the validated accepted recovery ID. The canonical staging descriptor contains exactly revision/recovery_id/intent_digest/source_digest/snapshot_digest. Its full accepted-intent digest transitively binds plan, proof, original source/history, archive identity and selected lane/epoch. The bundle is the exact bounded original SQLite backup, including committed WAL content. Its physical SHA-256 and actual schema/logical rows/current proposal are checked by the grounded archive oracle; a hash-shaped assertion is insufficient.

Both normal locks span validation and writes. Required marker, full accepted receipt document and pending fence precede staging; staging creation descriptor, exact bundle, file fsync and receipt-parent fsync precede the Core accepted-intent transaction and pending-step transaction. No authority transition or automatic restore is introduced. Restart reads the owned bundle, verifies descriptor/private mode/owner/regular-file/single-link/stable identity/size/bytes, checks its accepted digest and actual original snapshot contents, repeats barriers, then uses those bytes for the final archive. Caller bytes cannot repair missing or corrupt staging after Core acceptance.

A recorded creation descriptor with missing/empty/partial/wrong output quarantines. Exact complete durable output is verified and re-barriered, preserving bytes/inode rather than rewritten. Descriptor and bundle names consume the existing finite receipt namespace, 2,048-entry/16 MiB ceilings and finite deadline. The staging addition does not grant arbitrary path or SQL authority. Existing final snapshot/manifest/original-journal no-remint rules remain.

Independent stdin probes used fresh `TemporaryDirectory` installations and existing fixture builders. Before fork, deleted the fixture snapshot bytes. Child ran the real helper without a snapshot argument and called `os._exit(73)` at each of `core-intent-commit`, `pending-receipt-commit`, and `snapshot-parent`; parent verified exit 73 and deleted its intent/observation variables. It then launched a **fresh Python interpreter** using `subprocess.run([sys.executable, '-c', resume], input=json.dumps(...))`. Input carried only root paths, scope and the owned accepted-document path. The fresh interpreter reloaded that document from disk, resumed twice at `now=1000` without snapshot bytes, checked direct historical Core and Journal refusal before both invocations, and compared final archive bytes directly with the staging bundle. **All three cases passed**. Thus qualifying restart execution cannot use inherited Python snapshot RAM. The snapshot-parent case also verifies durable archive output without its evidence-step receipt.

A separate actual-death probe at `fence-parent` passed owned-document receipt-only resume twice after TTL, worker refusal before resume and Core refusal after Core acceptance. One preliminary probe incorrectly required historical Core refusal before a receipt-only intent reached Core; it stopped at that assertion and was discarded. Astra explicitly permits that DB-only historical constructor before Core pending exists. The corrected receipt-only probe and the three qualifying committed-window probes above passed; no implementation change was made.

The focused suite additionally passed staging descriptor create/write/fsync/parent/recorded and bundle create/write/fsync/parent interruptions, actual file-fsync and parent-sync failures, and twice-refused postacceptance missing/changed bundle, malformed descriptor, public mode, symlink, hardlink and oversize cases even when original caller bytes are supplied. No further minimal fix is requested for F1.

## F2 — medium — closed: all completed predecessor receipt evidence is checked on every invocation

Locations: `service/recovery.py:1190-1221`; regressions `tests/service/test_recovery.py:1346-1363`.

Before new prefix authority writes, the helper iterates every validated predecessor accepted record, derives its four exact Core step documents and completion certificate, then compares actual protected receipt certificate bytes against canonical `{certificate,steps}` and original journal bytes against the immutable before-image. Fixed names derive from the predecessor recovery ID. The ordinary DB validator has already checked actual SQL step bindings/contiguity, original history retention and grounded control archives. Therefore the comparison is anchored in actual validated accepted history rather than an unrelated caller certificate label. It is repeated even after the current pending fence supersedes the previous completed fence.

The full/focused suite passes two completed predecessors followed by a third pending intent: changed and missing **oldest** journal/certificate refuse twice; both unchanged variants resume. Independently extended this to **three** coherent completed predecessors and a fourth pending prefix. Used recovery/archive ID pairs `1/2`, `3/4`, `5/6`, then `7/8` (each digit repeated 32 times). After the fourth prefix was durable, separately changed the first byte of the **middle** predecessor's `.journal` or `.certificate.json`, preserving length/path/mode, or deleted that file. Both changed and both missing variants refused on each of two `now=1000` invocations. Both unchanged controls returned the identical pending result twice. **All six independent cases passed**, with no evidence rewrite/drop. This tests an interior predecessor rather than only the latest fence or oldest special case. No further minimal fix is requested for F2.

## Preserved boundaries

Initial exact source/history/media/affected-set/schema/configuration/proof/consumer coverage and expiry checks remain before receipt acceptance. Accepted exact same-intent replay may pass TTL only with unchanged ownership/death-coverage inputs. Current-row/reference/archive exclusions remain identity-specific; no blanket history-table exclusion. Earlier plans, accepted records, resolutions, certificates and control evidence stay bound. Missing/corrupt journal keeps its exact status/bytes and requires coherent authority and every retained registration/possible-owner coverage; it is never treated as a clean receipt.

B2.1 structural allocation provenance, actual archive/completion grounding, canonical job/context and prescribed unsuccessful-output contracts remain unchanged. B1 no-remint/self-only history projection/private WAL-aware inspection and R1 inherited-uncertainty fences remain intact. Both publication wrappers retain pending/permanent retirement checks before method bodies. Both startup paths refuse Core-pending state twice; receipt-only ordering retains its explicit cross-volume distinction. No journal rotation, jobs/attempts/events/dispatch/ACK/reservation/monitor/fault/high-water change, epoch completion, resolution or receipt completion occurs in this prefix. Snapshot remains evidence only.

Programmatic `apply` remains unconditionally `P1B2_apply_unavailable`; packaged and host apply route to that refusal. Enrolled activation stays closed. No new activation flags, free confirmed-dead Boolean, arbitrary path/SQL writer or late host-authentication prerequisite was introduced. E1-mock closed proof inputs exercise internal validation only and do not authenticate actual host death. Later transition-kernel/host work is outside this acceptance.

## Fresh verification

Existing interpreter only, no installs; bytecode and pytest cache disabled. Focused and full commands ran sequentially on unchanged source/tests in the exact candidate:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**278 passed in 85.58s, exit 0.** Includes staging faults/protected-input negatives, actual child death/no snapshot retention, prior receipt evidence, original B2.2a and B1/B2.1/packaged lifecycle regressions.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**548 passed, 18 skipped in 190.44s, exit 0.** Sequential after focused completion, on the same frozen source/tests. Skips remain unqualified. No redundant affected-full or heavy campaign ran.

Independent probes described above ran through `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'` from the candidate, importing `installation.__wrapped__`, `prefix_fixture`, `b2_intent` and `b2_receipt`, with disposable roots only. The combined corrected committed-death/fresh-interpreter/middle-history probe exited 0; corrected receipt-only actual-death probe exited 0. Fresh interpreter children inherit bytecode suppression. These are separate reproductions, in addition to the candidate tests; no probe source was written into the candidate.

Final exact HEAD/status and source/production comparisons remain unchanged/clean; diff check passes. Only this designated root review report was written. No candidate/source edit, commit, subagent, push/PR/deploy, installation, Docker/network/models/audio/paid/E2E runner, prune or user cleanup. Test-owned disposable roots only; unrelated research/worktrees preserved.

## Evidence limits

E1 local source/SQLite/filesystem/process and mocked host projections only. Process death and barriers do not establish physical power-loss/ENOSPC/daemon/VM/controller durability, actual-host proof, installed lifecycle, engine/audio/hardware or clean-host qualification. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** Remaining B2.2 tranches/B2.3, B3, C1/C2, parent integration, final Astra and public/unattended acceptance remain pending.
