# P1B23 fresh-interpreter child-wait guard — independent Sol-low review

2026-10-23. **Bounded guard NO-GO on exact `5b698566aed6b52451a822e3e930c0a4824bf674`, frozen code `145f2441de030ae1ecc605b4f1f3a8767daff467`.** Two test-only issues remain: the ordinary journal-symlink refusal has no accepted phase contract, and helper aggregation accepts a complete receipt pair attributed to a foreign owner. No integration, combined lifetime/source acceptance, T2 or native-cause clearance.

## Identity and boundary

Inspected ABS `/Users/ben/.txcode/worktrees/py-audio-box/childwaitguard-20261023`; porcelain empty before and after all executions. No applicable AGENTS.md found. Launch baseline `0c71e1bc85ed7fe3293be6bffe671867eadc5e27`; chronological replay `dd9b3b3`, `2f06c68`, `beb3a1a` matches the handoff. The complete tree at beb3a1a equals original `4259415daaf5acecfe53ed9d44f9895207a00d1b`: both tree IDs `8cd3ff1d04a9ba17b9def364b9373e4432f0bca2`. Frozen-to-HEAD changes only the handoff. Original-to-HEAD production/dependency diff is empty. Reversing the sole import and all 17 fresh-launch substitutions restores original recovery-test bytes exactly. Full baseline changes are the earlier three production lifetime files and their tests, then this test guard/handoff; they are not represented as a new production fix here. Diff-check passes.

Read candidate RECOVERY-P1B2 lines 2024–EOF, including the entire 2182–EOF guard appendix; entire RUNTIME-LIFETIME-ASTRA, root LIFETIME-r1/r2, T2-r1, canonical orchestration protocol/implementation plan and relevant ledger chronology. Inspected the entire guard delta and fresh-launch inventory/callers. LIFETIME-r2's separate F2 remains owned by its correction lane; no imports or edits from that worktree.

Sole repository write is this root report. No fixes, commits, delegation, other-worktree writes, installation, parent-limit lowering, native crash generation, native causal/allocator/GC experiment, Docker/network/model/audio/paid E2E, campaign retry, push/deploy/reset/clean/prune. Owned temporary scripts and retained pytest basetemps are under /tmp. Initial root research artifacts were present and untouched.

## G1 — journal-symlink has an ordinary OS-refusal witness gap

Locations: `tests/_maintenance_wait_guard.py` observed_session/semantic_phase; `service/recovery.py:411`, `:3024`, `:3109`; `service/persistent_init.py:23`.

The preserved four-B2 log really stops on `test_lotp_early_refusals_precede_original_writer[journal-symlink]`, PID 79524/raw 18944/empty phase, with one call failure and its repeated teardown error. Its log hash matches the frozen receipt. That historical log does not retain the inner exception, so this review does not retrospectively prove its cause.

One newly authorized isolated execution of the same row, in a fresh pytest interpreter, adds only a test-owned wrapper around the original os._exit. At exit 74 it writes the current exception class and at most 16 function/line phase markers, without message, payload, SQL, protocol or allocator changes. Result: **1 failed / 1 teardown error, 2.07s, exit 1**, owned PID **86100**, raw **18944**, empty guard phase. Captured class **OSError**, bounded path:

`_run_bounded_maintenance:542 → observed_session:199 → _maintenance_session:411 → prewrite:3109 → _preparation_prestate:3051 → _observe_b2_journal:3024 → read_file:23 → test mapped-open:2900`.

Thus the real session wrapper IS installed and sees the prewrite path; this is neither a pre-install refusal nor a wrong-module wrapper. It catches only ValueError, whereas the real O_NOFOLLOW open raises OSError for this setup. The production parent then sanitizes the failed child as ValueError. The observed refusal precedes the original writable open. No signal occurred, and this isolated observation establishes no race or production runtime defect. The guard correctly refuses to certify the row with its current contract.

**Minimal TEST-ONLY fix brief:** establish a named journal protected-read phase witness for this exact node/parameter and actual no-follow symlink refusal, before original writable open. Bind acceptance to that phase plus actual exception class and a precise symlink/no-follow condition (or exact OS error if separately captured), rather than any OSError anywhere in the session or exit 74 alone. Preserve writer sentinel, source/receipt-byte assertions and unchanged production exception/protocol. Add negatives proving AssertionError, a different OSError phase and missing witness still latch/report outside raises, plus a fresh-helper version to exercise parent aggregation. Do not merely add this node to an allowlist or broaden normal 74 acceptance. No repair is authored here.

## G2 — receipt owner is not bound to the acknowledged helper

Location: `tests/_maintenance_wait_guard.py:aggregate`.

Ack/done equality is checked, but event `owner` is never compared with ack.pid. A harmless independent synthetic receipt probe publishes ack/done `{pid:100,node:owned-helper}`, an enrollment `{owner:999,pid:777}`, and a matching accepted terminal from owner 999. **aggregate returns `[(777,0)]` instead of rejecting it.** This is an actual fail-closed validation gap against the stated unowned-receipt contract; it is not evidence of a foreign OS process having been waited or signalled. The existing “unowned” test supplies only a terminal without enrollment and misses this complete foreign-owner pair.

Each directory represents exactly one original helper interpreter; nested helpers have separate directories. Require each event's owner to equal the directory's validated ack PID, with exact integer/PID/schema checks, and preserve nested per-directory ownership. Bind the ack node/identity to the launch context rather than only comparing two self-reported files. Add the complete foreign-pair regression. Ack/done are also read without the 4-KiB pre-read size check used for event records; apply the same bound before parsing them. Keep atomic publication/private namespaces and missing completion rejection. These are test-only receipt improvements, not production protocol work.

## Positive guard coverage and limits

All 17 known maintenance-capable recovery python-c launch sites use fresh_run, including _KPRE_FRESH callers and nested helpers. The sole remaining recovery subprocess.run is the WAL-only SQLite writer, which does not invoke maintenance. Other inspected Core/Dispatcher/admission/release fresh bodies do not enter this seam. Bootstrap is explicit per test script, inherits caller environment, and adds only an interpreter-local tests path; no sitecustomize, ambient PYTHONPATH, production opt-in or dependency change exists.

The first fork per active maintenance operation enrolls the exact returned child; subsequent operation forks remain with their own owner. Original wait results are returned; no extra wait/reap/signal is introduced. Real semantic refusal/unrelated-child and nested-helper controls pass. Missing ack/completion, orphan, duplicate, oversized event and terminal-only unowned tests pass. Failure survives helper exit and parent aggregation; report override outside pytest.raises and session stop work. Fresh rejected AssertionError exits 86 and does not reach the external writer sentinel. Expected 73 remains tied to declared hook/node/phase, and SIGKILL to named transport/cleanup cases. Synthetic status 11 tests generate no signal. These positives do not close G1/G2 or establish exhaustive fault tolerance.

Atomic files are bounded per event, helper event count and nested directory count; ack/done pre-read bounds and identity binding have the gap above. Frozen helper metadata was not retained by the implementer, so no post-run independent ack population audit is claimed. Reviewer runs retain their own basetemps.

## Actual commands and results

All pytest work used candidate cwd and this prefix:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS \
 PYTHONDONTWRITEBYTECODE=1 \
 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache \
 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

- `tests/service/test_child_wait_guard.py -q -rs -x --basetemp=/tmp/pab-childguard-r1-sol-tests`: **14 passed, 6.37s, exit 0**. Includes fresh normal/semantic/nested/concurrent helpers, missing completion, malformed receipts and AssertionError masking rejection.
- `tests/service/test_recovery.py -k 'lifetime_wait_guard_unit or lifetime_guard_real_semantic or lifetime_guard_reporting or lotk_actual_death_fresh_owned_resume' -q -rs -x --basetemp=/tmp/pab-childguard-r1-sol-controls`: **10 passed, 1660 deselected, 5.05s, exit 0**. Includes exact wait tuple preservation, unrelated child, synthetic signal/report masking and real declared fresh resumption.
- Isolated phase command used the same environment prefix, replacing `-m pytest` with `-c 'import sys; sys.path.insert(0,"/tmp"); import pytest; raise SystemExit(pytest.main(["-p","no:cacheprovider","-p","pab_childguard_r1_phase","tests/service/test_recovery.py::test_lotp_early_refusals_precede_original_writer[journal-symlink]","-q","-rs","-x","--basetemp=/tmp/pab-childguard-r1-sol-phase"]))'`. Plugin `/tmp/pab_childguard_r1_phase.py`; bounded result `/tmp/pab-childguard-r1-phase.jsonl`; failure described in G1. No repeat followed.
- Locked Python stdin static/receipt probe: **exit 0**, `/tmp/pab-childguard-r1-sol-audit.json`, including G2 counterexample, replay equality, reverse-bootstrap equality, stage log hashes and package/limit checks.

Verified CPython **3.14.7**, pytest **9.1.1**, MCP **1.30.0**, tiktoken **0.14.0**; FSIZE `(9223372036854775807,9223372036854775807)`, CORE `(0,9223372036854775807)`. No parent limit modification.

Frozen implementer receipts remain attributed: 14 guard, 85 Core/Dispatcher, 177 recovery passes; recovery 124 actual waits including eight fresh-helper children. Four-B2 769 passed/one skipped/one failed/one teardown error, 49 actual waits including four fresh-helper children, pytest exit 1; full NOT STARTED. Independently matched all four stage-log SHA-256 values to `/tmp/pab-childguard-frozen-receipts.json`; their temporary ack/completion populations were unavailable. No four-B2/full or unknown blocked sequence was rerun.

## Handoff

Return G1/G2 to the test-guard owner for a minimal correction and fresh independent review. The entire later combined source-plus-guard candidate still needs its own review and qualification. The original lifetime baseline has no acceptance here; LIFETIME findings and separate close-F2 work remain separate.

V1 original failures/three native crashes, V2 invalid UTF-8 and V3 original SHM loss remain UNRESOLVED. New guard/lifetime greens are not causal clearance. All 19 Linux cap/peer-credential mechanisms, actual Linux 4 MiB/ext4/Unix-VFS/SHM, mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2, install/durability/T2/final Astra/hardware/browser/acoustic/activation remain pending. A1 inventory drift remains unexplained; Apple Silicon remains documentation-only. Existing limits, strict source/history joins, H/charges and finite cleanup/kernel-latency qualifications remain intact.
