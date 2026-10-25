# r389 — independent bounded retained-history review

2026-10-25. **Bounded r388 SOURCE NO-GO: one blocking integrity finding, R389-F1.** The coherent current 37-method allowlist passes, but an additional finite probe shows that the new-family retained reader accepts an explicitly uncertain SQL dispatch phase. Full r378 family SOURCE NO-GO; owner activation and installed export remain UNAVAILABLE. Release NO-GO; legacy CLOSED. No adoption or activation authority is granted.

## Verified scope and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-e7dc96ee` was clean before execution and after all checks, at `44b7b33594306f7ef7b9c8928f91e219b6394ab7`. Verified direct parent chain: report-only HEAD -> source `2c49778718df7d85b7806302f53ba3441358acf0` -> reviewed daemon/snapshot report `a40d1ca2798273e0b812dea9ab304947ab73c499` -> snapshot correction `184a3d5ff1cd4d6f87b94e482eae8a8e7d67a45b`. Required base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` is an ancestor. Base-to-HEAD `git diff --check` passed. HEAD changes only the r388 report; the source delta from a40d1ca has the four service files and two test files reported by r388.

Read FULL absolute ROOT reports H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md, H3-SNAPSHOT-REVIEW-r387-sol.md, H2-PUBLICATION-FORMAT-CONTRACT-r372-astra.md and H2-EXPORT-V2-GATE-r377-astra.md under `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`; full candidate H3-EXPORT-RETAINED-HISTORY-r388-sol.md; complete source diff, focused test module, relevant private-copy, SQL/schema, publication/Authority, reconciliation, registration and dispatch consumers. Inspected `/private/tmp/r388-tests.py` and all five follow-up runners before execution. Read both conftests; direct unittest does not invoke their HTTP listener/uvicorn fixture. No applicable AGENTS.md found in candidate descendants, candidate/root ancestor chains or report destination directories. Memory supplied the historical canonical-numeric-identity caution only; current source and fresh execution control this decision.

Sole persistent write is this ROOT report, mode 0600. No source, runner or log writes, commit, merge, delegation or self-adoption.

## R389-F1 — reject uncertain/nonterminal SQL dispatch phases before declaring clean history (blocking)

Locations at the exact candidate: `service/recovery.py:3131-3137,3154-3169`, reached by `service/installed_export.py:299-309`; related pre-Core predicate at `service/installed_export.py:345-349`.

`installed_clean_history` requires terminal/reaped jobs, retained reservations, SQL ACK and receipt, returned commands and terminal/ACKed journal records, but does not require the corresponding SQL dispatch phase to be `acknowledged`. Its detached validation and `_allocation_coherent` also do not impose that phase requirement. Consequently, changing only `dispatches.phase` to `unknown` in otherwise clean retained export history still lets `ex.retained_snapshot(..., family=ex.FAMILY)` return successfully. The phase is included in the projection/digests, so this is acceptance of freshly inspected contradictory state, not a stale digest bypass.

This matters because `unknown` is an existing uncertainty fence: Dispatcher writes it on uncertain operations and explicitly checks it in retry handling (`service/dispatcher.py:251,513,539,561,595,645`). The new retained reader feeds Helper.clean, retained registration pre/post snapshots and the stopped receipt selector. It must refuse uncertain history rather than certifying it clean merely because ACK/reap copies remain. The runtime is currently unavailable, so this finding is a source acceptance defect, not a claim of an exercised production activation exploit.

The pre-Core reader explicitly rejects `unknown`, but accepts any other phase with ACK=1 and a receipt. The same probe's `started` phase was accepted by both readers, despite not being the persisted terminal ACK phase. Dispatcher actually persists `acknowledged` at `service/dispatcher.py:547`; `acked` is not the phase spelling.

One finite independent probe used `ExportRetainedHistoryTests.retained_export()` once, then, for each phase, executed only:

```python
with database(path) as db:
    db.execute('UPDATE dispatches SET phase=?', (phase,))
ex.retained_snapshot(s.roots, public, family=ex.FAMILY)
ex.startup_read(path.parent, s.config, s.gate, hello, incarnation, family=ex.FAMILY)
```

All original receipt/journal/job/command facts were retained. Exact observed receipt:

```text
baseline phase [('acknowledged', 1)]
unknown retained ACCEPTED
unknown preCore REFUSED ValueError installed_history_unclean
source preserved True
started retained ACCEPTED
started preCore ACCEPTED
source preserved True
CPython 3.14.7; forbidden attempts []
```

Probe exit 0 means observation completed; it does not mean the mutated states passed a correct safety oracle. Source preservation compares the fixture's bytes immediately after each deliberate SQL mutation with bytes after both readers. The probe ran from stdin, wrote no helper/log, and used the same before-import audit hook and exact candidate paths as the allowlist.

Finite correction requested: for the explicit new family, require the SQL dispatch terminal phase `acknowledged` consistently in clean retained/detached selection and the pre-Core predicate, alongside existing ACK/receipt/reap/returned-command joins. Do not broaden or change old v1/native/B2 semantics. Add focused phase-mutation refusal/preservation coverage (including `unknown` and a pre-ACK phase), preserving the positive acknowledged history/re-registration case. No new framework or broad campaign is requested.

## Other reviewed boundaries

No second blocking finding established in this bounded review. Explicit FAMILY/SPLIT and exact enrollment/config/launch/evidence select new v2 readers; live SQL checks coherent user_version/marker/DDL, while detached schema/marker selection does not invent an historical user_version. The initial fresh registration reader retains its empty-history checks. Registration retains its dropped INSERT writer and canonical pre/post brackets, with non-registration history preservation covered by the focused fixture.

Retained SQL opens only the bounded private WAL-aware copy (`installed_export.py:294-300`); source identity/namespace checks remain in `recovery.private_database:74-114`. Canonical retained job/context/dispatch identity, source metadata, manifest output, public artifact/intent, converter, full frozen export profile and permitted principal checks are present at `installed_export.py:367-406`, together with existing publication provenance validation. They do not authenticate coordinated replacement of all trusted copies.

The reconciliation extension selects 160 MiB media-name / 16 MiB scalar allowances only after explicit family/policy and exact enrolled database validation (`reconciliation.py:138-146`). Existing 2048-entry and finite time bounds remain, every regular name charges fully, and detached inventory sums regular-name sizes. No global MAX_BYTES increase or v1/native/B2 budget expansion was found. Current registration selection binds lane, images, volumes, container coverage and start ordering; stopped receipt selection checks exact enrollment, registered pair and canonical retained receipt. These checks remain subject to R389-F1's clean-history predicate defect.

Pre-Core facts inspect private control SQL/media, with checked gate/seal/configuration, selected registration, hello lane/boot and control incarnation. They do not read protected supervisor receipts or activate an owner. Their later independent owner/runtime checks remain required; fixing R389-F1 must not add protected receipt access to this reader.

## Independent coherent execution receipt

Ran the current 37-method allowlist **once**, using `/private/tmp/pab-0c-venv/bin/python -I -B -` and an in-memory transformation of the inspected `/private/tmp/r388-tests.py` (SHA-256 `538bc3277e52c112e02aca011741d4bed57e067a8bb2035307d00ecf5cbae603`). Before application/test imports its audit hook blocks subprocess.Popen, os.system, socket.bind and socket.connect. Exact cwd/sys.path select the candidate and its tests. Asserted CPython version tuple `(3,14,7)` before executing the runner.

The only adapter inserted before suite construction was:

```python
from unittest.mock import patch
from service.core import Core
class V1NativeHistory(NativeHistoryTests):
    def setUp(self):
        with patch('tests.service.test_native_lifecycle.Core',
                   side_effect=lambda root, **kw: Core(root, **dict(kw, _publication_creation=1))):
            super().setUp()
NativeHistoryTests = V1NativeHistory
```

This changes native fixture seed creation only; no production gate/source is adapted. Independent result: **37 PASS, 0 errors/failures, 74.654 s, exit 0; CPython 3.14.7; forbidden attempts `[]`.** Composition: 12 retained-history, 9 export-daemon, 4 snapshot, 2 selected v1 daemon, 2 initialization, 1 v1 registration, 5 terminal/detached-history and 2 native-history methods. Both native methods exercised v1 fixtures, including positive twice-reopen and negative changed publication. Receipt is retained in this review turn's tool output, not a new disk log.

This is a fresh coherent current selection, not a relabelling of r388's 36 PASS / 1 ERROR run or its later focused runs. Those prior failures remain attributed in r388, including the erroneous `acked` phase expectation corrected there. Passing the allowlist does not cover the additional phase mutation demonstrated above.

Fixtures use tiny actual SQLite/files/flocks and approximately 1.1 MiB inert PCM, with fake Docker/UID/journal facts. They copy local inert completed history into a freshly initialized disposable new-family enrollment; they do not demonstrate an actual installed public journey. Fixture disposal only, through existing lifecycle; no operational cleanup.

## Parent-owned exits and limits

Correct R389-F1 and obtain focused independent source closure before accepting this retained-history slice. Owner activation/prepare/precheck/Core/postcheck/reconciliation, notification consumption, AudioExports ownership, truthful capabilities and the installed public HTTP/CLI/MCP upload/export/download/range/reopen/replay/generated-speech journey remain later parent-owned work. Final complete-family Astra review remains required. Missing operational pins/production availability are not additional blockers to this bounded reader slice.

No actual Docker, network, listener, application child, FFmpeg, ML/hardware, HELD/STT, installation/migration/build/acquisition/download, heavy waiver or actual-root cleanup. Seven real-child publication tests remain **BLOCKED/UNRUN**. Actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification remain **UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. No disk/RAM refresh or qualification receipt is inferred. **Bounded r388 SOURCE NO-GO pending R389-F1; full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
