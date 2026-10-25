# r380 — bounded fresh export initialization and registration writer

2026-10-25. Implementation candidate for the initialization/writer subslice of
r378 conditions 1+2. **Full-family SOURCE NO-GO. Production export UNAVAILABLE;
release NO-GO; legacy CLOSED.** No adoption or independent acceptance claimed.

## Authority and pins

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-1b97856c` was clean at
`1bec52ea8e1fedd8411099c0afc5001ff1c3031a`. Fast-forwarded only this checkout to
`c73e109272a3e27a038824ca72fb698a4e6578da`, whose parent source primitives are
`3ac457d8532d6fe62764ae760b621a698e30e393`.

Source candidate: **`c9b391f9e231e7f0b92b455fac6f226970b92d5a`**. This report is a
subsequent documentation commit. No root/original writes, merge adoption, push,
delegation or independent review commissioning. No applicable AGENTS.md found.

Read full root r378, candidate r379 gap matrix and r372. Input SHA-256:

- r378 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`
- r372 `a41dc92e7380e791478897300e42df98a8663b46876df04e4858f0ef6185f5c1`

The r378 source was read from
`/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`;
it is not present in the candidate checkout. The source worktree named in the
brief was read only.

## Concrete implementation

- Explicit private `installed_export.FAMILY` selection, separate from unchanged
  singleton SPLIT filesystem ownership. Legacy configuration/installation
  parsers remain closed and reject the new records. New config version is strict
  integer 2; scope/policy/launch/evidence use the four r378 literals. Enrollment
  document/SQL revision remains 1. New launch grammar reuses detached existing
  field validation through a private projection; it does not change daemon
  commands or activate a launcher.
- `initialize_split(..., family=FAMILY)` validates new config, launch, installation
  ID, host scope and digest before exclusive initialization. It retains existing
  root/directory locks, role transition/intents, synchronous child callback and
  receipts protocol. New-family use requires a separately supplied synchronous
  child callback; the old production launcher refuses this selection. No public
  entrypoint or default is changed.
- `_split_core(..., family=FAMILY)` carries a checked internal enrollment
  expectation into Core and creates publication-v2 transactionally through the
  existing Core initializer. Admission, dispatch, inventory and revision-1
  enrollment follow, then an exact fresh SQL postcheck. The small necessary
  `persistence.initialize_enrollment` branch validates the new record only when
  Core carries that expectation; local/default/native enrollment is unchanged.
- Core checks selected root identity and, for an existing store, validates a
  source-preserving private copy before owner mutation; rechecks the opened DB
  before WAL/initializers. New admission checks the selected family again after
  duplicate lookup and before preallocation. `_publication_creation=2` alone
  supplies no new-family expectation. Old installed/native v1 gates remain.
- Exact finite SQL validation uses existing canonical v2 DDL and pure detached
  row/document validators, all sqlite_master entries including indexes/triggers,
  coherent user_version/P1, exact canonical enrollment, registration bindings,
  integrity/FKs, 2048 rows, 16 MiB scalar budget and finite SQL deadlines. No
  retained SQL is executed. Native/B2/unknown/partial table sets refuse. Fresh
  fixed rows and empty private media namespaces are separately checked. This is
  an initialization inventory, not the later retained media/history scanner.
- Completion requires the same canonical private intent/digest, selected family,
  host/config and role inodes. A complete exact fresh DB after a lost child reply
  may finish the missing done marker under control exclusion. Partial Core DDL
  or enrollment is refused without reconstruction. Old v1 lost-reply quarantine
  stays unchanged. Already enrolled roots still refuse initialization completion.
- New registration-writer branch requires explicit family selection, exact new
  enrollment/evidence/configured images, inherited held control lock, private
  no-follow files and identities. Exact v2/schema/enrollment validation occurs
  before BEGIN IMMEDIATE, under it, before INSERT and before COMMIT. Canonical
  prestate mismatch refuses; only registration INSERT is issued on the target.
  Rollback, existing-row retry and non-registration preservation are tested.
  Writer constructs no Core, performs no target DDL and opens no receipts.
  Old branch now explicitly requires coherent v1; shipped old writers are not
  retroactively changed. Their missing universal version check remains a fact.

## Focused inert evidence

Read `tests/conftest.py` and `tests/service/conftest.py`; no pytest collection or
listener fixture was invoked. Used exactly CPython 3.14.7 at
`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r380-tests.py`.
Before importing any application or test module, the runner imported only sys,
installed an audit hook raising on subprocess.Popen, os.system, socket.bind and
socket.connect, then inserted this exact checkout in sys.path and imported tests.
Final retained local log: `/private/tmp/r380-final-tests.log`; runner is a local
scratch artifact, not a distributed executable or qualification receipt.

Final run: **30 methods PASS, 6.498 s; forbidden-event attempts `[]`**:

- 13 new `ExportInitialization` methods: fresh exact v2 and same-family Core
  reopen; same-intent completion; changed config refusal; mixed markers/unknown
  index; copied enrollment/old-family refusal; registration retry/prestate and
  other-row preservation; lost-child-reply completion; partial schema refusal;
  unknown trigger/B2-like namespace/copied root refusal; inherited-lock refusal
  and transactional rollback; v1 publication DDL/native-like table refusal;
  extra media namespace refusal; opened-owner format recheck before admission.
- Four selected `SplitInit` methods: fresh/private-copy/reopen, pre-Core
  completion, validated completion without duplicate child, old unmarked-Core
  quarantine.
- Three selected `Registration` methods: exact retry, stale prestate rollback,
  precommit rollback.
- All ten existing `InstalledExportContractTests` pure methods.

The new fixture composes existing injected ownership observations with real tiny
SQLite/files. Its child callback calls the dropped-core function synchronously
with mocked UID/GID/group observations; no child process, mount or identity drop
actually runs. New writer fixtures hold real flock exclusion. Refusal fixtures
compare retained file bytes where indicated. Native/B2 cases here are alien-table
namespace fixtures, not actual native/B2 operational roots. Current unchanged
legacy parsers reject new configuration/enrollment in executable fixtures; no
archived binary is executed.

Development runs exposed and corrected a converter/host-registration function
name collision, an unsupported Core context-manager use in the test, a mutation
fixture missing autocommit, a SQL quoting error, an attempted reuse of the
v1-only reconciliation enrollment parser, and an overly specific refusal regex.
The last was corrected to accept role-identity refusal. The fresh inventory now
uses its own finite empty namespace check, keeping production reconciliation
v1-only. None of those failed runs attempted a blocked audit event. Final source
`git diff --check` passed before commit.

## Remaining parent-owned r378 gaps

This finishes the bounded executable initialization/writer functions and inert
fixtures, not the full integrated condition 1 or 2 launch journey. Production
entrypoints still do not supply the new family callback/expectation. Private
container entrypoint/descriptor wiring, credential-principal corroboration,
sealed hello, actual pre/post observation brackets and their image/seal/boot
change fixtures remain in the later integration slice. This candidate must not
be advertised as a ready installation based on a DB or marker alone.

Conditions 3–5 remain open: explicit-pin image target and complete immutable
closure acquisition; daemon-only launch/mount-subpath inspection and assets
cross-view; authenticated seal and supervisor mappings/full frozen-profile
checks; dropped managed-reference helper; owner activation/AudioExports wiring;
truthful measured capability gates; installed upload/export >1 MiB public
HTTP/CLI/MCP/download/reopen/replay journey; retained export canonical joins and
160 MiB media-name/16 MiB scalar accounting in every installed reader;
failure/cancel/timeout/reap/ACK/capacity fixtures. Existing production readers,
hello parsers and activation remain v1-only, so the new family is unavailable.
No readiness is inferred from unqualified fixture hashes/profiles.

Fresh separate Sol adversarial review of this exact slice remains parent-owned;
final Astra complete-family review comes later. Neither was delegated here.
No Docker/network/listener/application child, FFmpeg/ML/hardware, HELD/STT,
actual migration/install/download, resource cleanup, heavy-gate waiver or actual
pin acquisition occurred. All actual CPU/Linux/Intel Mac Docker Desktop/OrbStack
qualification is UNRUN. CUDA and native Metal remain mandatory,
unqualified/UNRUN. Seven real-child reference-publication tests remain
BLOCKED/UNRUN. **Full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
