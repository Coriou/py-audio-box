# H3 r214 Sol — independent first-round partial Lot B boundary review

2026-10-24. **NO-GO for source adoption of this boundary slice: R214-1 blocks the new notification helper's uncertain-child exclusion guarantee.** The ordinary focused suite passes. This decision concerns only the private host projection, fixed exec notification, WaitingControl pre-owner gate and read-only installed startup reader. It is not a Lot B exit assessment. Actual installation, activation, operations and release remain unapproved.

## Pins and review authority

Inspected candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-6073a1b5`, branch `t3/packaged-headless-activation`, clean HEAD `40a3aab17d83e871a7f16263fde1545c7a9ae8f2`. Source `c6d71b725052b6e772634d5981e8ef1574db3e0c` has direct parent/base `ed36a42d7601edfd0b5a87ce6ea7348300145662`. Verified exactly five source-checkpoint files, 677 insertions: `service/entrypoint.py` (215), `service/host_install.py` (73), `service/recovery.py` (91), `tests/release/test_headless_install.py` (233), `docs/agent-interface/HEADLESS-INSTALL.md` (65). Source-to-documentation HEAD changes only r213; source/tests/install instructions are identical at both pins.

Read complete root normative r198, including startup/notification, restart and failure obligations; candidate r212 and r213; the full delta and relevant authority, lock, snapshot, provenance and validation dependencies. No applicable ancestor AGENTS.md was found. A bounded memory-registry search returned no relevant hit and supplied no premise. No delegation or skill was used.

The authorized trust seam is the trusted host successful-registration channel plus independent DB/configuration/local-process/authenticated-worker-hello checks. Control is not required or permitted to read protected provenance/receipts; its provenance digest is a label, not proof. The historical-incarnation restart dependency is an authorized separate successor, not a defect charged against this partial checkpoint.

## R214-1 — preserve unreaped sentinel through notification pipe teardown

**Blocking, high priority.** Candidate `service/host_install.py:1212–1221`: after communication failure, failed finite reap appends the process and raises `_RegistrationChildUnreaped` at 1218. The unconditional `process.stdin.close()` in `finally` at 1221 can itself raise and replace that sentinel. `installed_launch_authority` retains duplicate launch-lock/authority descriptors only in its `except _RegistrationChildUnreaped` branch at 919–921; an escaping OSError misses that branch and the outer finalizer closes the original descriptors at 930–932. Retaining the Popen object alone does not retain launch exclusion.

Concrete trigger: communicate times out, kill is attempted, the two-second wait also fails, and pipe close raises. Result: caller receives the close error while the child remains uncertain, and the designed exclusion-retention path is bypassed. A subsequent coordinator may acquire launch exclusion while the earlier notification child is still unaccounted for. This is a source error/lock-lifetime defect even though no success is returned. It does not depend on permitting actual Docker execution in this review.

Independent inert reproduction used the real `_installed_notify` parser and a valid fixture projection, replacing only Popen with a child whose `communicate` raises TimeoutExpired, `kill` returns, `wait` raises TimeoutExpired and stdin `close` raises OSError. Observed exactly:

`observed=OSError; sentinel=False; context=_RegistrationChildUnreaped`

The repro ran with the permitted isolated interpreter, real disposable registration fixture and injected host/privilege observations; no external child was launched. No persistent repro file or source edit was made.

Required correction: preserve the primary failure, especially `_RegistrationChildUnreaped`, through stdin/output teardown; attach cleanup failures without changing the exception used for exclusion retention. Add a focused regression for failed communicate + failed reap + failed pipe close, asserting both the escaping sentinel and retained authority exclusion through the wrapper. Also cover a reaped primary failure with close failure so diagnostics retain the actual primary error. Review output-file context teardown for the same masking pattern. Do not broaden child operations, retry the notification or erase retained state.

## Other boundary findings

No additional blocking defect established by this review:

- `entrypoint.py:24–65` closes and bounds the canonical public projection, binds exactly two role/process records to the retained registration and validates host cgroup associations. `host_install.py:909–916,1159–1195` routes through successful protected registration, validates retained provenance/DB membership and current host observation, and brackets provenance identity and host state around notification. It returns delivery only. No protected key, seed, credential content or receipt access is granted to control.
- `host_install.py:1198–1209` fixes Docker context parsing, exact control ID, exec user, isolated Python/import path and bounded stdin; callers cannot choose helper action/path/command. Five-second communication and two-second reap limits exist, subject to R214-1. Killing/reaping the local Docker client is not evidence that a remote exec process has been reaped; actual behavior remains unqualified.
- `entrypoint.py:75–163` uses a fresh private directory, retained directory descriptor, exclusive no-follow publication, fixed private single-link file checks, fsync and path-to-descriptor corroboration. Existing directories/files refuse. Partial/malformed content refuses rather than being repaired or overwritten. After the first complete checked read, bytes and full identity remain fixed; byte-identical inode substitution refuses. The monotonic wait budget belongs to this WaitingControl instance; it does not constitute a global launcher/startup deadline or disconnected-time readiness receipt.
- `entrypoint.py:166–200` requires local Linux PID1, brackets start ticks/PID namespace and private cgroup view, holds the checked control lock through authenticated hello/private read/second hello/local observation/notification reread, and constructs no Core or listener. The host cgroup association comes from the trusted host; local control verifies `0::/` plus its own incarnation. Post-Core binding remains explicitly caller-owned and unimplemented in this partial slice.
- `recovery.py:3458–3546` explicitly requires SPLIT, uses the carried stable WAL-aware private copy, opens only copied SQLite read-only/query-only with trusted schema disabled, bounds schema/rows/bytes/query time, verifies installation/configuration/root/registration/image/lane/boot and process bindings, and refuses B2 namespaces/pending recovery plans. Close and private-copy teardown failures refuse; primary reader errors are restored when teardown also fails. No original SQLite connection or receipt/provenance reader is added.
- The recovery change is append-only. Existing activation guards, legacy startup and B2 writers are unchanged. Full reconciliation/admission, ordinary command, packaging and restart are outside this partial adoption decision; their absence is documented rather than treated as delivered functionality.

These are bracketed source checks under the declared trusted-channel model, not atomic immunity to arbitrary privileged mutation or actual Linux identity/permission evidence.

## Independent checks and limits

Read `tests/conftest.py` and `tests/service/conftest.py` before execution; no release conftest exists. Ran only `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit absolute candidate and candidate/tests import paths and unittest loading of exactly:

- `tests.release.test_headless_install`
- `tests.service.test_installed_registration`
- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

**141 tests PASS, exit 0, 29.485 seconds.** The green suite does not cover the combined cleanup failure reproduced above. `git diff --check ed36a42 40a3aab` passed. Candidate HEAD and clean status were rechecked after execution.

Evidence uses real temporary files, SQLite/private copies and lock APIs with injected privilege/host/runtime observations. No actual Docker exec/build/run/pull, proc qualification, mount/UID transition, network/listener, model/ML/hardware, pytest guard/full suite or HELD operation occurred. No source edits, commits, delegation, adoption or unrelated cleanup. Only ordinary test-owned fixture cleanup occurred. Sole persistent write is this root r214 report.

Parent-owned next step: correct and independently re-review R214-1 within this boundary slice. Historical-incarnation work remains separately authorized; no Lot B exit, actual installation or release approval follows from this report.
