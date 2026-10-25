# r391 — independent R389-F1 terminal dispatch phase review

2026-10-25. **Bounded R389-F1 SOURCE GO: the phase acceptance finding is closed at the exact candidate below. No additional blocking finding established in this narrow review.** Full r378 family SOURCE NO-GO; owner activation and installed export remain UNAVAILABLE. Release NO-GO; legacy CLOSED. This is not adoption, merge or operational authority.

## Verified scope and pins

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f3e21aa0` was clean before execution. Verified exact direct chain: report-only HEAD `580f0570c133113e26a84fe8ca7ff6c3ca39965a` -> source `ef9746065548dfba2e6408a5102c32ce3590b3d7` -> `44b7b33594306f7ef7b9c8928f91e219b6394ab7`. Required retained-history base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` is an ancestor. Base-to-HEAD `git diff --check` passed. HEAD changes only r390; source changes only installed_export.py, recovery.py and the retained-history test module.

Read FULL absolute ROOT H3-RETAINED-HISTORY-REVIEW-r389-sol.md and normative H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md; full candidate r390 and r388 reports; complete correction diff/current focused test module, both conftests, inspected runner `/private/tmp/r388-tests.py`, relevant fixture/inert runtime and reader/helper/registration/SQL/Dispatcher paths. No applicable AGENTS.md found in checkout descendants, checked ROOT/candidate ancestor chains or report destination directories. Memory supplied the historical canonical-representation/source-only caution; current pinned source and independent execution control this disposition.

Sole persistent write: this ROOT report, mode 0600. No source/runner/log writes, commits, merges, delegation or self-adoption.

## R389-F1 closure and preservation

`service/installed_export.py:313-315` defines one pure SQL projection predicate requiring phase exactly `acknowledged`, ACK=1 and non-null receipt. Dispatcher actually persists this spelling at `service/dispatcher.py:547`. Pre-Core uses the shared predicate at `installed_export.py:349-354`, after explicit FAMILY/configuration/enrollment/gate selection and private database opening. It still reads control facts only, without protected receipt access.

`service/recovery.py:3028-3043` invokes the same predicate only for identity-selected `installed_export.FAMILY`, after detached source/row validation. `installed_clean_history` reaches that validator at recovery.py:3157; existing terminal/reaped jobs, retained reservations, summary/projection/digest equality, returned commands and terminal journal/ACK/receipt/UID joins remain at recovery.py:3131-3176. The new condition does not replace these checks. Default v1/native/B2 paths do not invoke it; no legacy phase semantics or production writer changed.

The retained snapshot calls clean-history at installed_export.py:309. Helper.clean and retained registration snapshots use that reader (daemon_helper.py:358-367,391-401); registration also checks clean history before its existing writer (433-438). Stopped receipt selection calls clean-history at installed_export.py:421. Thus the phase requirement reaches retained, detached, pre-Core, clean, helper and stop selection, including registration pre/post snapshots. There is no repair or phase-write operation in this production delta.

Current `tests/service/test_export_retained_history.py:147-195` is a meaningful regression for the original repro: it changes only phase while keeping ACK=1, receipt, returned commands, terminal/reaped job and journal evidence. It changes both detached summary/projection phase and recomputes source digest. Eight values (`unknown`, `started`, `bound`, `observing`, `receipt`, `ack_pending`, `acked`, `idle`) each refuse through six seams with the precise `installed_history_unclean` oracle. Exact equality in production rejects every other non-acknowledged spelling as well. The test keeps a WAL writer open, checks DB/WAL/SHM bytes and device/inode/mode/UID/GID/link/size/mtime/ctime plus namespace after every reader, and verifies SQL remains at the deliberate mutation. Explicit fixture restoration occurs after refusal checks; readers do not perform it.

Positive acknowledged export retained/reopen, stopped receipt, pre-Core and clean re-registration cases passed. The legacy speech fixture still seeds `acked`; the current test explicitly asserts refusal without repair, rather than weakening production or relabelling that fixture as a positive. No v1 fixture adapter was needed for this selection. No additional standalone phase probe was necessary because the inspected matrix directly reproduces and strengthens r389's two-phase observation across all six seams.

## Independent coherent receipt

Executed the entire current `tests.service.test_export_retained_history.ExportRetainedHistoryTests` exactly once, from stdin using `/private/tmp/pab-0c-venv/bin/python -I -B -`. Before application/test imports, installed an audit hook that records and raises on `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`; asserted CPython version `(3,14,7)`. Set cwd to the exact candidate and sys.path to that checkout, its tests/service and tests. Direct unittest loads no pytest listener/uvicorn fixture. Asserted selection count 13; no source, fixture or runtime adapter applied.

**13 PASS, 0 failures/errors, 56.513 s, exit 0; CPython 3.14.7; forbidden attempts `[]`.** Raw receipt is retained in this review turn's tool output. This is one coherent fresh current run, not a reinterpretation of r390's initial 12 PASS / 1 legacy-fixture ERROR or its subsequent 2-method PASS. Those historical receipts remain accurately separated in r390. No broad 37-method campaign or seven real-child publication tests were run.

Fixtures use tiny actual SQLite/WAL/files/flocks, approximately 1.1 MiB inert PCM and fake UID/Docker/journal facts. Local in-process ASGI/owner execution uses an injected converter/reap proposal, then copies enumerated local history into disposable new-family enrollment. This proves reader/source behavior only, not installed activation, real conversion or actual descendant reap. Disposal used only existing fixture lifecycle.

## Parent-owned next work and limits

Parent next owns completion of new-family owner prepare/precheck/Core/postcheck/reconciliation, notification consumption, AudioExports ownership, truthful capabilities and public HTTP/CLI/MCP upload/export/download/range/reopen/replay/generated-speech journey, then final complete-family Astra review of the complete unmerged new family. Existing owner gates remain explicit at entrypoint.py:603 and daemon_helper.py:479,487. This narrow GO does not accept the complete family or authorize adoption.

No actual Docker, network, listener, application child, FFmpeg, ML/hardware, HELD/STT, installation/migration/build/acquisition/download, operational cleanup or heavy waiver. Seven real-child publication tests remain BLOCKED/UNRUN. All actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification remains UNRUN. CUDA and native Metal remain mandatory, unqualified/UNRUN. No fresh disk/RAM, operational pin or readiness receipt is inferred. **R389-F1 bounded SOURCE GO; full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
