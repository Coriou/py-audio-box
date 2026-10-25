# r384 — explicit daemon initialization and sealed launch boundary candidate

2026-10-25. **Candidate awaiting independent Sol review; no self-acceptance.
Full r378 SOURCE NO-GO; installed export UNAVAILABLE; release NO-GO; legacy
CLOSED.** Owner activation, retained history and the public journey remain the
next slice. This is executable initialization/registration wiring, not installed
product acceptance.

## Authority and pins

Own isolated checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-1f5ee68c`
started clean at `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`. Verified ancestry
and fast-forwarded ONLY this checkout to
`359e8aeb2bcfab20f9fb8c5d3fd77997ce28f6db`. Read-only source checkout
`/Users/ben/.txcode/worktrees/py-audio-box/t3-e320873b` still had that exact HEAD
at final inspection. No applicable AGENTS.md found in this checkout or ancestors.

Source candidate **`c6e6fcc2883cb5f9ad6a5206976a3f1088011fc9`**, direct child of
359e8ae. This report is a subsequent documentation commit. No original/root
writes, parent adoption, push, delegation or independent review commissioning.
Read full root r378, r381 and r383 and candidate r379/r380/r382 reports and
relevant implementation/test seams. Root report SHA-256:

- r378 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`
- r381 `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`
- r383 `4617c07c962b8a9a5ff06aa7f36392d9262f45d837d3f25ece23a74c22554651`

## Implemented boundary

- Distinct private `service.daemon_client.export_main` selects the singleton
  family from code, never a marker/request flag. It validates closed r378 config
  and credential production-write principal coverage before Docker calls. The
  imported credential document is checked again in the helper. Existing ordinary
  client defaults remain v1; new selection requires installed service validation
  dependencies. No new public command or enable-export option is introduced.
- Existing eight volumes, labels, create-exclusive handles, authority lock,
  initialization intent and phase/nonce protocol remain the coordinator. Exact
  family-aware commands select the new fixed helper and child entrypoints.
  Initializer supplies FAMILY to the reviewed r380 Core callback; dropped child
  entrypoints carry it to `_split_core` and the reviewed registration writer.
  Descriptor closure, permanent drop verification, synchronous collect/reap and
  enclosing exclusions remain in the same existing paths.
- Worker command selects the new policy and mandatory separate expected worker
  image. The sole additional worker mount is enrolled control-volume subpath
  `core/assets` -> `/references/assets`, RO/noCopy. Inspection checks exact
  requested Subpath/NoCopy (including numeric/bool substitutions), actual volume
  name/source suffix/RO destination and the complete mount inventory. Unsupported
  daemon create errors propagate; there is no whole-control or bind fallback.
- Explicit new-family local facts include assets. Worker startup selects these
  facts only with its checked export configuration; dropped hello reader selects
  the reviewed extended parser. Helper cross-checks imported input views and
  assets against its own private control/assets inode plus both runtime views.
  Inspected image/command/config, full profile/principal seal, lane/boot and
  container start times bracket registration and notification.
- New gate uses strict integer version 2 and the exact new launch/evidence
  literals with closed fields. It binds configuration/enrollment digests and
  reruns selected observation/hello validation. Old gate/parser selection remains
  strict v1. Registration still inserts only through the existing dropped writer;
  pre/post snapshots and all other SQL rows must remain identical.
- Helper registration prestate has an explicit new-family **initial-history-only**
  RO reader: exact enrolled v2 schema/documents, existing 2048-row/16 MiB scalar
  limits/deadlines, exact fresh fixed rows apart from registrations, empty private
  media namespaces and stable private DB identity. No v2 data reaches a v1
  snapshot/history reader. This does not implement retained export history.
- `worker-export` is a separate recipe requiring explicit FFmpeg, libsndfile and
  espeak package versions and executable SHA-256, verifies the executable and
  removes write bits. It inherits no unversioned worker apt layer and has a
  refusing direct CMD. Existing targets and the original final/default target
  remain in their original order. Package transitive closure, resolved alias-free
  immutable dependency list, finished image identity and measured qualification
  still require later authorized acquisition/review. No real pins or fake hashes
  were added. Compose adds comments documenting only the new subpath reference;
  it remains non-authoritative and does not enable Compose registration bypass.

## Deliberate remaining launch gap

The new control command names `entrypoint.daemon_export_main`, which explicitly
refuses until checked owner activation exists. New helper `up/status/stop` refuse
before runtime create/start, and `ready` independently refuses for this family.
This prevents an initialization marker, SQL enrollment, seal or fake notification
from enabling the public journey. The new registration/observation functions are
executable and tested under injected held-helper calls; the ordinary new-family
runtime does not yet invoke them because owner activation is the next slice.

Next slice must implement selected owner precheck/Core/postcheck/reconciliation,
AudioExports/capability gates and notification consumption, and all retained
startup/up/status/stop/reopen readers with canonical export joins and separate
160 MiB media-name / 16 MiB scalar budgets. Current initial-history reader must
not be broadened by dropping its empty-history checks or raising global limits.
Public upload/export >1 MiB, generated assets, full/range download/reopen/replay,
and integrated failure/cancel/timeout/reap/publication/ACK/capacity remain open.
No new lifecycle, CoreAuthority, publication or ACK logic was changed.

## Focused inert evidence

Read both conftests before imports; no pytest collection/server fixture. Runner:
`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r384-tests.py`, exact cwd
and sys.path this checkout, CPython **3.14.7**. Before application/test imports,
only sys was imported and an audit hook installed blocking subprocess.Popen,
os.system, socket.bind and socket.connect.

Final retained log `/private/tmp/r384-final-tests.log`: **28 methods PASS in
18.834 s; forbidden-event attempts `[]`**:

- Nine new ExportDaemonTests: fake client install and preflight; real tiny
  helper import/init -> new Core -> registration writer; fixed new child/hello
  handoffs; exact mount/image command positives and refusal matrix; unsupported
  subpath one-call refusal; credential coverage; strict gate versions/keys;
  seal/profile/principal/image/boot/start/assets registration bracket mutations;
  unavailable runtime has no create/start/notify and cannot return ready.
- Two selected v1 DaemonTests: full install/up/status/stop/same-ID restart and
  security/mount/command matrix.
- Two selected ExportInitialization tests: fresh/reopen and transactional writer
  prestate; one v1 Registration exact retry.
- All 12 ExportSealTests from r382.
- Two existing AdapterBoundsTests: output/nonzero exit and timeout retains every
  root until fake child reap.

Development evidence retained in `/private/tmp/r384-tests-*.log`: initial import
found an accidental syntax error in the edited child descriptor tuple; corrected.
A five-test baseline passed. First new 11-test run had two fixture errors reading
an authority FD after its held scope closed; corrected fixture reads. Subsequent
11, 12 and 13-test runs passed as cases were added; final expanded 28 passed.
Every completed test run reported forbidden attempts `[]`; initial syntax failure
occurred before tests ran. `git diff --check` passed before source commit.

Fixtures use real tiny SQLite/files/flocks with fake Docker/Popen/pipes and
injected UID/mount observations. Registration journey patches the helper runtime
entry to exercise the bounded seam while production runtime remains unavailable;
it is not an owner activation or actual launch. No kernel UID/DAC, real mount,
Docker subpath support, actual converter closure or installed readiness is proven.
Scratch runner/logs are local evidence, not distributed qualification receipts.

No Docker/network/listener/application child, FFmpeg/ML/hardware, HELD/STT,
actual install/migration/build/acquisition/download, operational cleanup or heavy
waiver. Existing managed-reference kill/wait behavior was not changed or expanded
into a new campaign. Seven real-child publication tests remain **BLOCKED/UNRUN**.
All actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification remains
**UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. Separate
Sol review is next; full Astra review belongs after the completed family.
**Full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
