# r390 — R389-F1 terminal dispatch phase candidate

2026-10-25. Implementation candidate awaiting separate fresh Sol review; no self-adoption. Full r378 family SOURCE NO-GO. Owner/public activation and installed export remain UNAVAILABLE. Final complete-family Astra review remains later. Release NO-GO; legacy CLOSED.

## Pins and authority

Own isolated checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-f3e21aa0` started clean at `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`. Verified ancestry, then fast-forwarded ONLY this checkout to `44b7b33594306f7ef7b9c8928f91e219b6394ab7`. Read-only source `/Users/ben/.txcode/worktrees/py-audio-box/t3-e7dc96ee` verified clean at that exact pin. Source candidate **`ef9746065548dfba2e6408a5102c32ce3590b3d7`**, direct child of 44b7b33. This report is a separate documentation commit.

Read FULL absolute ROOT `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-RETAINED-HISTORY-REVIEW-r389-sol.md` and normative `H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`, candidate r388 report and relevant reader/clean/detached/helper/dispatcher/schema/fixture code. No applicable AGENTS.md found in own checkout or ancestor chain. Lightweight memory query found no relevant entry. No ROOT/original writes, push, delegation, adoption or activation.

## Narrow correction

`service/installed_export.py` adds one pure `_clean_dispatches(rows)` predicate over existing SQL projection rows: every dispatch must have phase exactly **`acknowledged`**, ACK=1 and non-null receipt. Pre-Core uses it in place of its `unknown`-only exclusion. `service/recovery.py::_installed_detached_source` applies the same predicate only when the explicit family is `installed_export.FAMILY`. The existing clean-history path calls that detached validator; retained snapshots, Helper.clean, registration pre/post snapshots and stopped receipt selection therefore share the requirement. Existing terminal/reaped jobs, retained reservations, returned commands and journal terminal/ACK/receipt joins remain required. Pre-Core still reads control facts only, without protected receipts.

No SQL mutation, repair, new framework, publication/ACK protocol or runtime gate change. Default v1/native/B2 branches do not invoke the new predicate. Actual Dispatcher spelling is `acknowledged` (`service/dispatcher.py` ACK completion); `acked` is refused by the new family.

## Focused evidence

Inspected `tests/conftest.py` and `tests/service/conftest.py` before imports. Direct unittest does not invoke the listener/uvicorn fixtures. Both executions used exactly `/private/tmp/pab-0c-venv/bin/python -I -B -` with stdin runners, asserting CPython `(3,14,7)`. Before application/test imports, each installed:

```python
import sys
forbidden=[]
def audit(event,args):
    if event in ('subprocess.Popen','os.system','socket.bind','socket.connect'):
        forbidden.append(event);raise RuntimeError('forbidden:'+event)
sys.addaudithook(audit)
```

Then exact root `/Users/ben/.txcode/worktrees/py-audio-box/t3-f3e21aa0` was inserted into sys.path alongside its `tests/service` and `tests`, and selected as cwd. No disk runner/log was created; receipts remain in tool output. No native adapter was selected.

1. Entire focused `tests.service.test_export_retained_history.ExportRetainedHistoryTests`: **13 methods, 12 PASS / 1 ERROR, 56.674 s, exit 1**, forbidden attempts `[]`. New phase matrix passed, as did acknowledged export reopen/canonical mutations, pre-Core positives, clean retained re-registration, private WAL/source identity and other retained cases. The one error was the old speech test reusing `TerminalRestart.terminal_fixture`, which explicitly seeds SQL phase `acked`; the corrected production reader properly refused it.
2. Replaced that test with `test_legacy_acked_speech_fixture_refused_without_repair`, asserting the unchanged fixture is `('acked',1)` and refuses without source changes. Moved positive stopped-receipt coverage to the completed-export fixture, whose actual local Dispatcher persisted `acknowledged`; no phase overwrite is used to make a legacy fixture pass. Reran ONLY the two changed methods (`test_dispatch_phase_refusal_across_clean_history_readers` and `test_legacy_acked_speech_fixture_refused_without_repair`): **2 PASS, 15.848 s, exit 0**, forbidden attempts `[]`.

This is coverage across those executions, not a claimed all-green single 13-method run. The unchanged broad 37-method campaign was not rerun. Final `git diff --check` passed.

The matrix mutates only SQL phase, keeping ACK=1, receipt, terminal/reaped job, returned commands and journal evidence intact: `unknown`, `started`, `bound`, `observing`, `receipt`, `ack_pending`, `acked`, `idle`. Each refuses with `installed_history_unclean` through six seams: retained snapshot, pre-Core, direct detached validator, clean-history validator, Helper.clean and stopped receipt selector. Detached mutations update both projection and summary phase and recompute source digest, avoiding a stale-digest-only negative. Each reader preserves SQL/WAL/SHM bytes and source file identities/modes/ownership/link counts/sizes/mtime/ctime and namespace; the SQL phase remains the deliberately mutated value. Explicit fixture restoration to acknowledged follows the negative matrix and verifies retained/clean/pre-Core positives. Existing positive re-registration passed unchanged.

Fixtures use actual tiny SQLite/WAL/files/flocks and approximately 1.1 MiB inert PCM, with fake UID/Docker/journal facts and local injected export execution. They do not prove installed activation, actual converter work or real descendant reap. Fixture disposal uses existing fixture lifecycle only.

## Limits and next gate

Fresh independent Sol review of the exact source candidate is next, parent-owned; no review commissioned here. Full-family Astra review remains later. Owner prepare/precheck/Core/postcheck/reconciliation, notification consumption, AudioExports ownership, truthful capabilities and installed public HTTP/CLI/MCP product qualification remain outside this fix and unavailable.

No actual Docker, network, listener, application child, FFmpeg, ML/hardware, HELD/STT, actual installation/migration/build/acquisition/download, operational cleanup or heavy waiver. Seven real-child publication tests remain BLOCKED/UNRUN. Actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification remain UNRUN. CUDA and native Metal remain mandatory, unqualified/UNRUN. No operational pins/readiness or resource refresh inferred. Full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.
