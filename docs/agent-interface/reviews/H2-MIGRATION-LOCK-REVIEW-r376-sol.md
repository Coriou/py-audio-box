# r376 — independent narrow R374-F1 migration-lock review

2026-10-25. **Bounded SOURCE GO for the narrow R374-F1 correction.** No blocking finding in the reviewed change. Exact private retained-lock mode is enforced on descriptors, with held-descriptor/path identity revalidation. This closes the remaining r374 P2 at source/inert-fixture tier. Fresh Astra gate, adoption and actual migration remain parent-owned. **Release NO-GO; legacy CLOSED.**

## Pins and scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-9c5f400f` verified clean before and after execution at HEAD `531b3eff72ff6b9c8eee6da44f41055bf58fe179`. Its sole HEAD change is the r375 report. Parent correction `87f33eb1367cce6ed84ca3b73d10ec4146b9b207` changes only `service/publication_format.py` and `tests/service/test_publication_format.py`, atop report `5fa085a6401e21170b786114523b2c801c5a6831`, implementation `052abeac6a352e5c6a844098f58342195f09bd3f`, and ancestor base `96a1b249c1b6a505e1d32c500adca10355115458`. Root HEAD remains that base. Commit chain and base ancestry verified; base-to-HEAD and narrow diff whitespace checks pass.

Read full root r374 and normative r372, candidate r375/r373, narrow source/test diff, migration implementation, selected test module, both conftests and the imported inert export/ASGI fixture helpers. No applicable ancestor/relevant nested AGENTS found. Memory registry supplied general bounded-review discipline only; findings use current source and fresh receipts. Root has pre-existing untracked evidence/reports; only status names were inventoried, no STT contents read.

Sole persistent write: this root report, mode 0600. No source/test/runner/log writes, commits, merge, delegation or self-adoption. Disposable SQLite/filesystem fixtures and their ordinary teardown only.

## Source findings and closure

- `service/publication_format.py:236-241`: `_owner_info` requires regular type, exact `stat.S_IMODE == 0o600` (including rejection of special bits), same UID, one link and zero length. Existing ownership/type/link/size checks are retained, rather than replaced by the mode check.
- `service/publication_format.py:255-270`: existing known-owner namespace, bounded listing, no-follow descriptor open and nonblocking exclusive flock remain. Descriptor predicate runs before and after flock. Path device/inode must match the held descriptor. Unsupported locks are refused without chmod/chown, repair, replacement or deletion.
- `service/publication_format.py:244-252`: `_same` retains original namespace/ancestor identity checks and now checks each held descriptor plus its no-follow retained pathname device/inode. The parallel lists are initialized together and appended together in the sole acquisition path; no input controls them.
- `service/publication_format.py:307-315`: held checks run before original mode=rw open and again under BEGIN IMMEDIATE before first DDL. Additional checks at 333 and 339 surround commit/committed verification. A detected pre-DDL mode/identity refusal retains logical v1; rollback/unknown classification remains intact. These are finite boundary checks, not continuous protection against a malicious same-UID actor mutating files between checks.
- `service/publication_format.py:280-295,352-364`: process maintenance lease, existing control flock and identity check, lock lifetime through DB/private-copy closure, and uncertain-close ownership fence are preserved. No recursive Core construction or lock-order change. Product code introduces no permission-changing or evidence-deleting operation.

No publication/provenance, admission, named-format DDL, installed/native gate, row-preservation or other ownership-lock module changes occur in this correction. Original F2 large-export and F1 provenance closure remain attributed r373/r374 evidence; the broad 97 tests/nine contracts were not rerun or presented as fresh.

## Independent guarded execution

One inline run with exact `/private/tmp/pab-0c-venv/bin/python -I -B -`, asserted CPython 3.14.7, cwd the candidate. Explicit sys.path entries: candidate root and its `tests/service`; importlib loaded only the exact candidate `tests/service/test_publication_format.py`, using its `load_tests` hook. Audit hook installed before application/test imports rejects `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`. Neither conftest activated; no broad pytest.

**26 focused tests PASS / 6.311 s**, including:

- Nine unsupported retained modes (0666, 0644, 0660, 0640, 0400, 0700, 01600, 02600, 04600): refusal before original writer; complete logical SQL dump unchanged; retained lock identity/mode/UID/GID/link/size/mtime and empty contents unchanged (`tests/service/test_publication_format.py:319-344`). 0400 may refuse earlier at O_RDWR; this correctly remains exit 2/refused.
- Held descriptor mode changes before writer and before DDL: refusal with unchanged logical v1 and changed-mode evidence retained (`tests/service/test_publication_format.py:346-367`).
- Active 0600 flock refusal without reaping, followed by successful migration after release of the same valid lock (`tests/service/test_publication_format.py:304-314`).
- Existing direct speech/two-file prepare preservation, exact-format/admission/replay, rollback/commit ambiguity and uncertain-close fences.

Additional independent finite predicate probe used one disposable empty regular descriptor, fchmod through all **4096** permission/special-bit combinations, verified actual mode each time and called `_owner_info`: **only 0600 accepted**, all 4095 others raised `foreign-owner-file`. This is descriptor-policy evidence, not 4096 migrations.

The same inline command then attempted one independent pathname replacement probe: create disposable v1 direct history/0600 retained lock; wrap acquisition to rename the retained path and create a new empty 0600 pathname; guard SQLite against original mode=rw. Observed result was **`(2, 'refused:local-identity-changed')`**, earlier than my assertion expecting `owner-changed`. Therefore the overall inline command exited **1** on that reviewer probe assertion, after the 26-test PASS and all-mode PASS. No retry, tuning or pass relabelling. The post-result SQL dump and both-path assertions were not reached; do not treat them as executed preservation evidence. The observed refusal and writer guard support refusal before original writer; descriptor-specific pathname rejection is additionally established by source at 249-251, not by this probe. The completed focused cases above independently establish logical preservation. The final audit-counter assertion was likewise not reached; no forbidden-event receipt appeared. Raw receipts remain in tool output, with no new log file.

## Disposition and remaining gates

**R374-F1: closed at bounded source/inert tier; narrow candidate SOURCE GO.** No operational migration acceptance or parent adoption. Fresh Astra gate remains parent-owned. Seven real-child publication methods remain **BLOCKED / UNRUN**; broad 97/nine-contract historical receipts unchanged and not requalified.

Actual conversion/loudness, converter/image/runtime corroboration, registration/enrollment, actual Linux/amd64, Intel Mac Docker Desktop and OrbStack: **UNRUN**. CUDA and native Metal: **mandatory, unqualified/UNRUN**. Installed/native v2 remains closed; old registration writer still lacks a universal publication-version refusal, as retained in r372-r374. No actual-root migration, Docker, network/listener, application child, FFmpeg, ML/hardware, HELD/STT access, operational cleanup, installation/download/acquisition or heavy waiver. Inert source fixtures do not establish installed completion. **Release NO-GO; legacy CLOSED.**
