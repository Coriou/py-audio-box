# K pre-fifth correction — independent Sol-low review

2026-10-22 · gpt-6.1-sol / low. **Bounded GO for source inspection / E1 design only; required validation gate NO-GO / PENDING.** No concrete new source blocker found. This does not authorize integration as a fully validated correction, C acceptance, production maintenance or activation. Parent must combine the separate locked-validation lane receipts and obtain the remaining integrated/C/final-Astra gates.

## Exact identity and authority

Reviewed ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-85d4bb33`, exact HEAD `6130d1fecce3d978671ed1bfd509693d3aab3c97`, baseline `7916f17ebe4608833269d958cd1c2b1430d78400`, frozen source/tests `44e00b94d4f40ce7a058e3f84c26d571f7feb1f2`. Candidate clean before inspection and at final verification. Frozen-to-candidate delta is only 90 handoff lines. Baseline-to-candidate delta is exactly recovery.py, append-only recovery tests and append-only RECOVERY-P1B2 handoff. Fresh AST source comparison: only existing `_pending_prefix_locked` changed; only `_unfinished_source` added; no definitions removed. Baseline tests/handoff remain exact byte prefixes. `git diff --check 7916f17 HEAD` passed. Both executable files equal frozen bytes.

Fresh SHA-256: recovery `8c37d444276f95a0ddb5bdfbaa4983b2e2b5b8bc75b69441dfffa5df75a16597`; tests `d4db87959423ca5b36b9da6335efe9ea866a883755fc59d0c232932bd2e0f5ef`.

Read canonical orchestration, current plan/implementation plan and relevant execution-ledger history; full original P1B23C-PREFIFTH review, correction brief and exit matrix; KERNEL-ASTRA, C-ACCEPTANCE-ASTRA, K-PREBUNDLE review; candidate RECOVERY-P1B2 from section 1465 onward. Inspected actual callers, snapshot/after-image projection, retained-stage/history validators, certificate/finalization writers, fresh-exec/death harness and appended matrix. No applicable AGENTS.md found in the inspected worktree context. Memory supplied only exact-checkout/read-only/evidence-tier guidance, not acceptance evidence.

Historical C remains clean at `58e3eb9f40a8db5ca162fe8f64922efb13f224c4` in t3-adbf212d; older C remains clean at `6cfe3973f41db92679531e38660d88cf72ce27eb` in t3-4a8727d4. Both were read-only identity checks. Parent has preexisting untracked STT research; it was not changed or cleaned. Sole authored repository file is this report. No fixes, commits, delegation, installs, Docker, model/audio/paid E2E, network, limit changes or broad pytest execution.

## Source findings and exception audit

No blocking source finding established in this bounded audit.

- `service/recovery.py:1763–1780,1824–1828,1901–1902`: certified unfinished receipt continuation now validates through the shared helper before `_core_certificate`, `_finish_receipts` or fifth insertion. Fourth without either certificate file falls through the ordinary full-validation path before creation. Existing finalization temporary/registration checks and actual-stage retained receipt validation precede snapshot projection. This closes the original monitor-only path structurally and also covers unaffected source drift.
- `service/recovery.py:2126–2194`: reached Core/receipt creation descriptors require exact actual outputs; completion-bound journal/fence is valid when its attempt exists. Without that attempt the prescribed earlier target is required. Missing, partial, wrong, unrecorded or out-of-order outputs refuse; no future unattempted finalization output is demanded. `_unfinished_source` calls this before snapshot substitutes accepted original journal bytes. Output validation is therefore not inferred from the reconstructed hash.
- `service/recovery.py:480 onward,615,2077–2085`: snapshot retains exact schema/version, bounded complete SQL population, installation/volume/key binding, inventory/media and source projection. `_verify_core_rows` compares every declared output table, exact healthy integrity and the three resolution families before reconciliation projects prescribed before rows. Full evidence-projection equality, coherence, original consumers and affected reservations remain required. All these preexisting definitions are byte-identical to baseline; no weakened oracle, broad history exclusion or repair was introduced.
- **Fourth-only exception, `service/recovery.py:1830–1838`:** explicit return requires both protected existing Core certificate descriptor and output. `_prefix_exact` at 1519 opens read-only with NOFOLLOW, compares bytes/mode and re-barriers existing files; it cannot create them. The branch calls neither certificate creation nor finalization and returns only `core-complete-receipt-pending`. SQL validation and retained immutable evidence reads remain. This is observational in file-content/namespace/SQL terms, not zero-fsync semantics. A missing half refuses; with both absent the full reconstruction path is required before any creation. No path from this return can mint fifth or release a freeze.
- Genuine fifth-completed replay returns without current reconstruction. Existing `_receipt_only_successor`, pending-stage prerequisites and all-history descriptor/certificate validation remain unchanged. Completed A does not resume unfinished B; explicit B takes full validation before writes. F1 owner, F2 actual stage, F3 predecessor/own receipt distinction and F4 both history directions are retained. Process/session/connection/snapshot caps, both locks, Unix VFS, TTL and no-attempt frozen-hash semantics are source-identical.

## Exit-matrix and oracle audit

Appended tests are meaningful contract tests rather than only helper-call assertions. `_kpre_attempt` (4590) uses actual nonempty production P preparation and actual child `os._exit(73)` at the requested hook; retained accepted intent/bundle are read back. `_KPRE_FRESH` (4317 onward) reloads protected owned intent in a fresh interpreter; only OS/host transport is modeled. It supplies no caller snapshot/expected-row repair. Refusal catches only ValueError/OSError/SQLite errors; writer traps raise AssertionError and cannot masquerade as expected refusal. `_kfifth_fresh` repeats each invocation twice.

Original matrix mapping:

1. Monitor RED at 4803 checks fifth absent and the independent exact after-image oracle before resuming; protocol writers are trapped and SQL/receipt bytes checked afterward. Two distinct P/death setups, each with two fresh refusals, cover direct K. The four C/direct-K rows on fresh integrated C remain a later mandatory gate, not claimed by these K tests.
2. Twelve reached fourth/finalization prefixes at 4828 distinguish valid output restart from intent-only quarantine. Positives use accepted-after-TTL and replay without changing accepted intent; corrupt attempts at 4973 cover four output kinds × five damage modes.
3. Sixteen output/source mutations × certified/final-fence prefixes at 4851 cover output families, integrity, resolution sets, unaffected SQL sequence/history/typed references/media/key/installation/volume. Existing prebundle tests cover no-attempt/hash and earlier K/P invariants. Owner tests at 5031 retain the original capsule/proof and exercise same-container hello/new registration/different-container coverage refusal before writable connection access.
4. Twenty-three actual successor prefixes at 4918 exercise A observation, explicit B completion/quarantine and both directions; nine descriptor kinds × three corruptions × two directions at 4939 cover all completed histories. Writer/scan/capture traps guard A replay. Completed-current changes and valid hello/registration replay are covered at 4955/5091.
5. Focused/four-B2/full validation and fresh C completion remain pending. Source inspection does not satisfy this row.

Independent short probe: system `python3` AST comparison and byte-prefix/frozen equality checks passed without importing production modules. A separate read-only SQLite `mode=ro` probe of the two retained `/tmp/kfifth-baseline-red-20261022/test_kfifth_actual_death_monit{0,1}` roots independently confirmed monitor `fault`, all five ordered SQL steps and actual `completion-fence-dirsync` trace. This corroborates the baseline failure's durable outcome; it is not a fresh baseline execution or proof of corrected runtime behavior. The supplied baseline log ends with two expected failed refusal tests, not a green receipt.

## Validation gate and remaining limits

**No pytest tests executed by this reviewer.** `ls -l /tmp/pab-0c-venv/bin/python` returned absent. No environment was installed and no competing broad lane started. Separate task `pab-k-prefifth-locked-validation-20261022` owns restoration/qualification and sequential campaigns; its later receipts are not available or accepted in this report.

Implementer-attributed alternative-environment results remain: 164 correction cases passed; target 913 passed/1 skipped/1 failed; four-B2 1376 passed/1 skipped/1 failed; full collection error; available partial 1526 passed/19 skipped/2 failed. None is a green full gate. Python 3.11 Path/lstat assertion, MCP 2 versus locked 1.30.0 and unavailable tiktoken require restored locked validation, not waivers or source repairs here.

Required parent commands after interpreter/lock qualification, sequentially on frozen-equal source, with unchanged inherited limits:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py -k 'kfifth or kpre or lotk or lotp or b22' -q -rs
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

No actual Linux 4 MiB cap, ext4/Unix-VFS/SHM/peer-credential or installed host proof follows. All 19 skipped platform mechanisms stay unqualified. C1 pinned-host-reader production refusal remains mandatory; A1 Docker inventory drift remains unexplained without inferred cause/waiver. Freeze persists through host-observed fifth and matching outputs. C/B3/C1/C2, integrated restart/source equality, final global Astra, hardware/browser/listening and public/unattended activation remain pending. Apple Silicon remains documentation-only qualification scope.

**Return: bounded source GO; full validation/integration gate NO-GO pending locked receipts and subsequent gates.**
