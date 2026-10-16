# P1A2 round-1 correction candidate

2026-10-16. Implementation handoff, awaiting independent Sol-low re-review. No self-GO, integration/installed acceptance or activation.

## Identity and integration

Assigned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ae3b1e13`, branch `t3/typed-integrity-fence`, initially clean at `68262bb`. No other checkout or subagent. Inspected the complete `b6d16ef..54730c631f8708dd577be5a794215df205c3df7d` history: exactly two A2 commits, oldest first. Cherry-picked source `801cfbba590f28f9a999d810f3c4b50099c3b039` as `55d8adb`, then report `54730c631f8708dd577be5a794215df205c3df7d` as `840af60`. Before correction, service/tests diff between original candidate and imported `840af60` is empty. No omitted precursor was found.

Correction source/tests: **`76e693f47d232e0ce3be590dfa97cdbb3da83d7c`**. Final documentation-only handoff SHA is supplied with the response. Read PRODUCT, start/protocol, CONTRACTS 1.3.0, complete P1 Astra decision including A2/invariants/exit matrix, R1 fault-authority decision, PERSISTENCE-P1A2 and the complete mandatory P1A2-r1-sol review. No applicable AGENTS.md found. Memory registry supplied no task-specific decision.

Correction changes exactly `service/persistence.py`, `service/reconciliation.py`, `tests/service/test_persistence.py`, `tests/service/test_dispatcher.py`, `docs/agent-interface/phase-1/PERSISTENCE-P1A2.md` and this report. Imported A2 also contains its original Core/Dispatcher changes. No worker, contracts/fixtures, dependencies, packaging, legacy, ledger, reviews or evidence files changed.

## F1: typed reference adapters

Removed scalar-prefix inference and arbitrary JSON-column traversal. Explicit table/column adapters traverse the canonical contract definitions for jobs, frozen worker plans and bundle catalogues, every project revision, asset/artifact metadata, original manifest/artifact/message, publication-work destination and artifact binding, and dispatch authority/job/context/allowlist/profiles/outcome. Closed receipts and reservation-demand shapes explicitly carry no media references. Other current scalar tables/columns are covered by the existing exact column inventory and carry no media fields. Unknown tables, columns, nested schema fields, union discriminators and reference value types refuse conservatively. No prefix supplies identity or ownership.

Only declared asset reference fields and the `asset` schema's ID create media references. Non-prefixed IDs remain opaque valid IDs. Request text, principal, key, provenance prose, descriptions and instructions retain their literal content without becoming references. Core's supported queued speech request `asset-management is useful.` stays healthy through repeated reopen/dedup. Historical/render and non-prefix positives remain covered. VoiceDesign description/instruction is tested through a structurally/relationally valid frozen design job in the real jobs table: synthetic submit supports speech only. That design snapshot still retains its genuine prospective output reference and can therefore report unavailable; the test proves that authored prose adds no false reference, not executable VoiceDesign support.

One old reference test stored `{"asset_id":...}` as a fake project revision. It now uses the declared project's `assets` location; the existing separate real render/project-history tests exercise complete snapshots and both historical revisions.

## F2: separate integrity authority and availability cost

Added `p1_integrity(id, revision, state, token)` in the same Core SQLite authority; singleton ID 1, integrity revision **1**. Public contracts/worker protocol **1.3.0**, Core user_version **1**, private P1 format **1**, and R1 resource protocol **2** remain unchanged. No R1 measurement, monitor transition or reset is reused.

Every inventory scan and page fingerprint revalidation commits `healthy|unresolved -> checking(random token)` BEFORE defect-bearing database/filesystem observation. An unresolved completed diagnostic may be inspected again, but its prior fault is carried into completion. Begin denial performs zero collection and blocks the session. Inherited checking refuses observation altogether. Only the same owner object's uninterrupted token can commit diagnostics/history and the healthy/unresolved completion transition in one transaction. Failed diagnostic/completion write leaves the precommitted checking row; no live retry clears it. Page fingerprint mismatch participates in that completion transaction, so invalidation is durable even if the changed name later disappears.

The initializer checks exact authority columns/revision. Missing authority next to populated old diagnostics explicitly creates **unresolved**, never healthy. Empty/new A1 diagnostics can establish an initial healthy authority before their first observation. This does not enroll populated pre-P1 roots or supply B2 recovery.

**Selected availability cost:** death before, during or after an otherwise healthy scan, before completion commits, leaves checking and requires later explicit operator recovery. Ordinary coherent clean restart remains available. A valid exact reap receipt cannot establish that an earlier integrity observation was healthy. No B2 acknowledgement/reset, recovery, GC, fresh publication authority or public activation is added.

Startup already calls inventory before queued cancellation/release and runtime reconciliation. Its inherited-checking refusal now reads the authority before any inventory observation. Existing metadata/job/idempotent reads open no epoch. The Dispatcher integrity gate and pending lifecycle/stop scheduling are unchanged; tests retain an executing reservation and withhold publication/ACK/fresh admission even when the runtime supplies the original exact synthetic receipt over two restarts.

## Bounds and write costs

Historical object records, scan summaries and append-only history are counted/summed with SQLite `length(CAST(... AS BLOB))` before any diagnostic row is fetched or JSON-decoded. Aggregate historical bytes must fit the remaining finite scan byte budget, and object/history counts must fit the finite entry ceiling. No diagnostic tail is silently omitted. A regression puts an oversized invalid JSON record in the store and proves it is refused before decoding.

Observation still has the existing one-second deadline and entry/file/metadata ceilings. Begin/completion transactions, SQLite busy timeout, historical aggregate SQL queries and diagnostic serialization/write/commit costs occur **outside that observation deadline**. A page now performs a full bounded attested revalidation and writes diagnostic history; this adds SQLite write cost and consumes finite retained-history capacity. This is neither a hard realtime bound nor indefinitely unattended throughput. No history compaction or accounting credit is implemented.

## Regression matrix

| Boundary | Evidence/result |
| --- | --- |
| F1 real queued text/principal/key and dedup over reopen | Healthy, no false media diagnostic. |
| Description/instruction/reference prose; historical project text | No prose reference; genuine typed references remain retained. |
| Non-prefix asset, publication/work, frozen render and all revisions | Existing positive tests pass. |
| query_only before begin | Collection count exactly zero; no observation is asserted; clean writable restart works. |
| Real transient orphan, query_only only AFTER observation, immediate os._exit | Independent SQLite reader proves checking committed before collection; no close/retry; transient removed by owning parent; two restarts retain checking and perform zero collection. |
| Healthy completion denial and page completion denial | Immediate exit leaves checking; two restart refusals. |
| Healthy death before/mid/after collection | Explicit availability-negative process barriers; inherited checking never sampled healthy. |
| Successful page invalidation, changed name removed | Durable refusal over two restarts. |
| Missing authority with populated old diagnostics | Explicit unresolved transition. |
| Exact synthetic runtime receipt after open epoch, three failure/death cases | No publication, release, ACK or fresh key over two restarts; receipt/binding and executing charge retained. |
| Pending lifecycle stop, original receipt conflicts, normal clean receipt restart | Existing Dispatcher tests retained; final suite result below. |
| Historical aggregate bound | Oversized record never JSON-decoded. |

## Commands, failures and evidence limits

All runs use existing `/tmp/pab-0c-venv`, `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`; no dependency install. Disposable synthetic roots and bounded subprocesses only. Child immediate-death cases use `subprocess.run(timeout=10)` and are waited/reaped; parent removes only its own transient name. No separate heavy campaign ran concurrently.

Affected command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_dispatcher.py tests/service/test_core.py tests/service/test_admission.py
```

219 passed in 119.31 s, `/tmp/p1a2-r1-affected.log`. This preceded final context/binding/profile adapter tightening, combined byte-budget refinement and two additional healthy-death cases; final-source full verification below supersedes it.

Final-source focused command: same interpreter/options, `tests/service/test_persistence.py tests/service/test_dispatcher.py -k 'inventory or integrity_open_epoch'`: **29 passed, 79 deselected in 8.45 s**, `/tmp/a2-final-focused2.log`.

Final-source full local suite, run once after freezing correction source:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service tests/release
```

**269 passed, 18 skipped in 108.34 s, exit 0**, `/tmp/p1a2-r1-final-full.log`. This includes the final Core/Dispatcher/Persistence/Admission suites and all new regressions on source `76e693f47d232e0ce3be590dfa97cdbb3da83d7c`. Skips are not installed acceptance. `git diff --check` passed. No source/test edits followed this run.

Development failures were investigated rather than waived: initial three failures concerned inspection of already completed fault diagnostics and a fake revision shape; corrected while retaining inherited-open refusal. Three new Dispatcher assertions initially expected a failed job, but the established integrity gate deliberately preserves the outstanding receipt-phase job; corrected to require receipt retention, no success/release/ACK. VoiceDesign synthetic submit was unsupported, then an edited fixture had a reference-text consistency error; replaced with an oracle-validated stored design snapshot. Final combined-budget development exposed one extra history-budget reason for an already exhausted empty-history scan; zero remaining budget now correctly permits zero historical bytes. Intermediate logs `/tmp/a2-dev*.log` and `/tmp/a2-final-focused.log` are not acceptance evidence.

No Docker campaign: A1 global inventory drift remains unresolved; its runner was not weakened or repeated. No models/private audio/cloud/STT/Voicebox, remote E2E, `scripts/run-e2e-test.sh`, push/PR/deploy, resource cleanup outside owned temporary synthetic data, or public activation. Evidence is E1 local source/SQLite/process and in-process synthetic runtime receipts. It does not prove installed PID-1 authority, physical ENOSPC, power-loss/controller/VM durability, acoustic/hardware qualification or clean-host installation. B1/B2/B3/C1/C2 and global Astra acceptance remain pending.

Next is independent Sol-low re-review of exact correction source, inherited fence/page ordering, typed adapters and the preservation matrix; parent integration verification follows. No self-approval is supplied.
