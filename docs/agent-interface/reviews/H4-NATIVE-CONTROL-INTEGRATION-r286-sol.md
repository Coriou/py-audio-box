# Native control integration — r286 Sol

2026-10-25. Implemented coherent bounded SOURCE integration, ready for parent
fresh independent Sol review. This is an implementation report, not independent
acceptance. Installed native activation remains CLOSED; actual Metal remains
UNAVAILABLE / UNQUALIFIED. Lot B packaging/enrollment and Lot C hardware work
remain future authority gates.

## Authority and exact pins

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ef4727f6`, branch
`t3/native-runtime-integration`. Entry HEAD verified before editing:
`bd85e1c4c41e1b1927d6f4cc2385041ac357158d`, clean.

Source commit: **`434e26747736e303e116bdde0b413282755b1f30`**.
This report is a separate subsequent commit.

Read full normative r277, historical partial r278/r279, and r280–r285, including
F1–F5 corrections and the empty/committing/terminal stop variants. r277 SHA256:
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
No AGENTS.md found in repository/worktree discovery. Memory-registry keyword
search returned no relevant technical entry. Both conftests were inspected before
execution; no pytest collection, maintenance hook or listener fixture ran.
No delegation occurred.

Available storage observed 3,359,976 KiB, below 6 GiB. All executions used tiny
owned temporary fixtures with injected native admission capacity. The real native
6 GiB plus declared demand gate remains unchanged. No actual socket/listener,
native subprocess/signal, ML/model, hardware probe, Docker, network, download,
installation, crash experiment, operational cleanup or HELD action occurred.
Fixture teardown touched only its own disposable test resources.

## Implementation

`service/native_runtime.py` implements the real client operation path, using the
actual native canonical framing, TLS 1.3 certificate pin checks and closed typed
response validator. The production connector uses bounded TCP connect/TLS wrap;
tests inject only the channel, not the runtime or supervisor handlers. Every
reply is checked against deployment/config/lane/original boot/runtime/binding and
request nonce. Separate three-lifecycle/one-transfer semaphores preserve bounded
slots. Communication is at most two seconds, with active operations additionally
bounded by the original deadline. No endpoint/path/model configuration comes from
a public request.

The owner invokes `detach` before lifecycle executor work. Clone bytes are read
from the owned Core asset using the existing exact size/hash/PCM readers; the
executor receives immutable bytes and Authority, never Core/SQLite. Prepare
writes bounded private evidence, obtains one allocation and binds it before
start. Start transfers exact authorized input in finite chunks, repeats freshness
checks and never grants a new launch on replay. Interrupt events let cancellation
stop upload/import between bounded commands; stop remains an independent finite
RPC and never supplies death proof. Empty accepted-stop and frozen committing/
terminal decisions are understood without converting them into STOPPED receipts.

The private control spool is outside scanned staging, mode0700 with mode0600
no-follow/non-hardlinked files, exact snapshot/binding names and a 64 MiB declared
worst-case quota / 64-record ceiling. ResourceStore also measures this spool as
part of the existing control disk budget. Native demand includes the detached
reference, output/import copies and bounded snapshot overhead. Partial output,
missing/changed input, conflicting complete evidence or retained import links
blocks new admission on reopen. There is no automatic deletion, restart repair,
allocation eviction or relaunch. Existing-byte replay compares exact bytes.

Collect validates the lane-selected native receipt before importing anything.
It then checks proposal/Authority tuples, sole output digest/size, complete PCM,
provenance and existing native-child proposal oracle. Partial output remains in
the spool. A complete private inode is linked without replacement into the fixed
control-owned mode0770 stage as `output.wav`; the private import alias is removed
and both directories fsynced before returning. Core still independently verifies
owner, nlink, hash/PCM, manifest/provenance and publication relations. The configured
local import UID must equal the actual control UID. No native UID becomes a local
owner and no receipt gains an engine_uid field.

First import after the original cutoff cannot succeed. Exact already-materialized
local output/receipt replay does not expire historical evidence. Dispatcher
checks its deadline again before first publication, while existing durable
publication-intent replay retains its original recovery fences.

`worker/runtime_protocol.py` now has one common Qwen relation helper parameterized
by the closed internal Linux/native lane. Frozen qwen-mps selects native
MPS/float32/sdpa, public metal and packaged_native. Existing Linux CPU/CUDA
relations and explicit Linux MPS refusal remain. Full descriptor provenance
hashing is unchanged. Missing qwen-mps execution is refused in live/history paths.
`worker/native_protocol.py` delegates native relation decoding to that helper and
contains the shared pure `terminal_receipt` validator. Selection comes from the
frozen profile, never a received receipt's version/method. Native expected
configuration/lane/boot metadata is itself frozen in the private execution context.

`NativeQwenContexts` holds bounded trusted principal/profile/version entries,
enrolled native configuration and exact inventory metadata. It compares the
actual owned immutable version on resolution and checks freshness 0..300 seconds.
It does not inspect Docker-local paths as native model evidence. Admission uses
one slot, descriptor RAM, one inference process and zero dedicated VRAM for the
native lane. Existing native supervisor independent disk/RAM/full sealed-bundle/
reference/freshness gates remain authoritative at prepare/start. Existing
`current_native_hardware` was reused unchanged; history does not expire probes.

The only supervisor addition is `register_snapshot`, an internal provisioning
helper absent from wire OPS. It extends an existing trusted principal/version/
profile/context template to an exact allocation, preserving bundle mappings,
execution and inventory; only the authored request and actual job/attempt tuple
can vary. It cannot enroll a new principal, bundle or profile. Fixtures explicitly
invoke it on the trusted configuration side before the first prepare request.
The previously reviewed terminal/watchdog/journal handlers were not rewritten.
This is not public self-registration or an installed enrollment route.

Dispatcher performs immutable handoff, binds restored Authority before observation,
interrupts transfers after durable cancellation and stops a cancellation that
arrived during prepare before launching inference. It shares receipt validation
with Reconciliation, recovery and the Linux SupervisorRuntime reader. Native
receipts are stored unchanged. Resource release still follows validated reap,
durable Core terminal outcome and the existing reservation transaction; ACK is
last. Invalid native/Linux substitution retains the executing reservation.
Recovery's existing installed Linux policy explicitly rejects native history;
no native deployment is admitted through its Linux UID/PID1/enrollment rules.
The remaining installed-clean-history checks also invoke the shared validator.

`headless-dist-manifest.json` includes the new production module. Public schema,
DDL, installed policies, Core publication implementation and legacy `lib/tts.py`
are unchanged. One existing synthetic production test receipt was corrected to
include its explicit `reaped_children=0`, matching the already-closed Linux
journal receipt rather than weakening the shared validator.

## Executed checks

All commands used `/private/tmp/pab-0c-venv/bin/python -I -B` and explicit
`sys.path[:0]=['.','tests/service']`, with direct unittest selections only.

Final focused source invocation:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=[".","tests/service"];unittest.main(module=None,argv=["unittest","tests.service.test_native_runtime","tests.service.test_native_lifecycle","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=0)'
```

**73 PASS, 11.192 seconds.** A subsequent additional test, on the same production
source, passed separately: native publication storage uncertainty after durable
adoption, retained reservation/no ACK, same-root reopen, publication-intent replay
and one artifact/one launch (**1 PASS, 1.319 seconds**). Thus 74 focused methods
passed; no single 74-method invocation is claimed.

The new service tests enter actual Authority → NativeRuntime → fake pinned TLS
channel routing actual NativeSupervisor dispatch → actual native_child.run_socket
→ infer with fake ML → encoder/parent receiver → exact fake wait/receiver close →
durable receipt → output transfer → actual Dispatcher/Core publication. They
reopen the SAME control/native fixture, replay original completed output, update
trusted boot metadata and complete next work. Clone input uses logical IDs and
actual bounded transfers; repeated start makes one launch. Public ASGI HTTP,
existing CLI and MCP generation/job/artifact transports use the same native
runtime. Their owned immutable version is explicitly seeded SOURCE fixture data,
not clean installed onboarding.

Coverage includes all response tuple fields/pins, Linux receipt substitution in
live and detached consumers, UID/symlink refusal, bad output/retained partial spool,
quota/configuration refusal, cancellation during upload and output import,
first-late-import failure, historical completed replay, nonterminal restart UNKNOWN,
stale/future resolution, native pids/VRAM accounting and reservation/ACK order.
The retained worker tests cover frame/shape/PCM/hash/input offset/replay/gap,
bundle and independent admission, launch/wait/receiver/journal uncertainty,
TERM/KILL/identity, F1–F5 barriers and original-boot terminal reads. This is a
focused combination matrix, not exhaustive kernel/concurrency/crash qualification.

Final selected regression invocation: **32 PASS, 16.707 seconds**:

- `tests.service.test_qwen_lifecycle.QwenLifecycleTests`
- `tests.service.test_pocket_lifecycle.PocketLifecycleTests`
- `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`
- `tests.service.test_production_reconciliation`
- six `tests.worker.test_engine_kokoro.EngineTests` methods:
  `test_exact_arguments_and_detached_identity`,
  `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`,
  `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`,
  `test_bundle_binding_before_backend`.

Separate fresh interpreter: `tests.worker.test_qwen_import_seam` plus
`tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`:
**6 PASS, 0.404 seconds**. Root/legacy/extracted archive complete loaders use fake
upstream. No real dependency/ABI or executable archive qualification is claimed.
Final total is **112 distinct selected methods passing**, across those invocations.
The Linux regression fixtures emit ResourceWarning messages for unclosed test
SQLite connections; these were not suppressed or represented as failures.

Intermediate failures were corrected and are not hidden: initial native service
setup exposed the outer execution shape's missing private native metadata allowance;
a fake receiver recursively called its patched exchange; fixtures initially used
a nonexistent wait-counter name and mutated a shared fake response object. Lifecycle
fixtures initially mismatched saved-version normalized hash metadata, trusted
principal, plan hash and finite disk demand. The first selected production regression
failed because its synthetic receipt omitted reaped_children; the fixture now
supplies the exact Linux field. An initial storage-fault test expected success from
before-adoption failure; actual existing Core correctly failed incomplete adoption.
The final replay test injects after durable adoption and passes. No production
recovery fence was altered to satisfy that expectation.

`git diff --check`: PASS before source commit. `lib/tts.py` SHA256 remains
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
The historical broader legacy digest RED is not waived or regenerated.

## Remaining gates and limits

Parent fresh independent Sol review of the exact source commit is required before
adoption; no independent review/self-acceptance is asserted. Real TLS listener,
certificate enrollment/rotation/private key loading, bridge/firewall reachability,
Darwin process/FD/start identity, actual arm64 interpreter/dependency closure,
free-RAM behavior, durability/crash behavior, Metal load/speech/performance,
acoustics and release qualification remain unexecuted.

The internal exact-snapshot provisioning helper is a SOURCE seam, not Lot B
native enrollment or a public owned-version installer. Existing installed discovery
and Linux deployment policy remain closed to native activation. Lot B must supply
that distinct native package/configuration/enrollment variant and dependency
closure. Lot C must perform actual authorized Apple Silicon qualification with
unchanged headroom gates. Nonempty clone conditioning still retains UNKNOWN;
arbitrary cancellation-point clone readiness is unresolved. CPU Docker and CUDA
remain mandatory separate release rows.
