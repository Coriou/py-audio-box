# r375 — R374-F1 retained migration owner-lock mode correction

2026-10-25. Implementation candidate only. R374-F1 is addressed in source and
focused inert fixtures; fresh independent review, then Astra gate and adoption
remain parent-owned. No self-adoption. Release NO-GO; legacy CLOSED.

## Authority and exact pins

Sole writable checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-9c5f400f`,
branch `t3/publication-lock-mode-fix`. Initial clean HEAD verified exactly
`96a1b249c1b6a505e1d32c500adca10355115458`. Verified ancestry and fast-forwarded
only this checkout to `5fa085a6401e21170b786114523b2c801c5a6831`.
Read-only source `/Users/ben/.txcode/worktrees/py-audio-box/t3-b413d79c` HEAD
verified at that same candidate pin. Source/test correction commit:
`87f33eb1367cce6ed84ca3b73d10ec4146b9b207`. This report follows separately.

Read full root `docs/agent-interface/reviews/H2-PUBLICATION-V2-REVIEW-r374-sol.md`
and normative `H2-PUBLICATION-FORMAT-CONTRACT-r372-astra.md`, plus full candidate
`docs/agent-interface/reviews/H2-PUBLICATION-V2-r373-sol.md`. No applicable AGENTS
found. Memory registry consulted for prior context only; current pinned source
and the supplied contract govern this correction. No root/original-source writes,
merge, push, delegation or operational authority exercised.

## Narrow correction

`service/publication_format.py` now uses one descriptor predicate for the existing
same-UID, regular, single-link, zero-length checks plus exact
`stat.S_IMODE(info.st_mode) == 0o600`. Unsupported modes refuse; no chmod, repair,
replacement or deletion is introduced. The held descriptor is checked before and
after nonblocking flock acquisition. Its retained name is recorded alongside the
existing held stream.

The existing `_same` identity boundaries now also revalidate every held descriptor
and corroborate its device/inode against its retained no-follow pathname: before
the original mode=rw connection, under BEGIN IMMEDIATE before DDL, before COMMIT,
and after committed-pair verification. This extends the existing boundary checks,
not a new migration framework. Locks remain held through existing close handling.

Only this module and its focused test module changed. Exact v1/v2 formats, typed
row/rowid preservation, provenance, installed/native v1 gates, no-dispatch migration,
finite exits, and all other r372 boundaries remain unchanged. Original F2 >1 MiB
closure and the previous 97 tests / nine contracts are attributed r373/r374 evidence,
not rerun or relabelled as fresh evidence here.

## Guarded focused evidence

All application/test execution used CPython **3.14.7** at
`/private/tmp/pab-0c-venv/bin/python -I -B -`, with an inline runner. Before any
application/test import, the runner asserted the version and installed an audit
hook refusing `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`
with `FORBIDDEN_REAL_OPERATION:<event>`. Explicit import paths were this exact
checkout and its `tests/service` directory; cwd was this exact checkout. Loaded
only `/Users/ben/.txcode/worktrees/py-audio-box/t3-9c5f400f/tests/service/test_publication_format.py`
through importlib and its existing unittest load_tests hook. Read both conftests,
the selected tests and imported ASGI/inert fixture helpers; neither conftest was
activated. No broad pytest. Seven real-child reference-publication methods remain
BLOCKED / UNRUN.

Final focused run: **26 tests PASS / 6.423 s**, exit 0; retained raw output:
`/tmp/pab-r375-focused-final.log`. `git diff --check` passed.

New regression coverage:

- Modes 0666, 0644, 0660, 0640, 0400, 0700, 01600, 02600 and 04600 refuse with
  exit 2 before the original mode=rw connection. The test guards SQLite connection
  arguments, compares the complete logical SQL dump, and verifies retained lock
  device/inode/mode/UID/GID/link/size/mtime and empty contents unchanged.
- Descriptor mode changes via fchmod on disposable held test descriptors after
  acquisition and after original transaction validation refuse at the existing
  pre-writer/pre-DDL boundaries, preserving logical v1 and retaining the changed
  0666 evidence. Product code performs no permission changes.
- Existing active flock refusal remains passing; the same released valid 0600
  retained lock then migrates successfully. Existing direct speech/two-file prepare
  row preservation, finite rollback/unknown-close tests, v1 admission/replay and
  exact format tests all pass in this focused selection.

First focused run: 26 tests / 6.486 s, one failed subcase assertion, raw output
`/tmp/pab-r375-focused.log`. Mode 0400 correctly refused earlier at O_RDWR open
with PermissionError; the test had demanded the later descriptor-specific reason.
Adjusted only that assertion to accept exit 2/refused for 0400 while retaining
all unchanged-state checks. Other modes still require foreign-owner-file. No
product weakening or failed evidence cleanup; final run above follows that fix.

## Remaining gates

Fresh independent review, then Astra gate, parent adoption and any actual migration
remain parent-owned. No actual conversion, root migration, registration/enrollment,
Docker, network/listener, application child, FFmpeg, ML/hardware, HELD/STT,
operational cleanup, install/download/acquisition or heavy waiver occurred. Only
disposable inert fixture creation and ordinary fixture teardown ran.

Actual conversion/enrollment and Linux/amd64, Intel Mac Docker Desktop, OrbStack:
UNRUN. CUDA and native Metal mandatory, unqualified/UNRUN. Installed/native v2 and
legacy remain CLOSED; old registration-writer limitation remains as reported in
r372/r373/r374. Source fixtures do not establish installed completion or release
qualification. **Release NO-GO; legacy CLOSED.**
