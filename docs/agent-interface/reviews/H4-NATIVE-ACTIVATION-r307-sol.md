# Native first-start owner implementation — r307 Sol

2026-10-25. Finite SOURCE candidate for independent review, no self-adoption.
Implemented the explicitly permitted empty-history alternative. This is not full
native installed service activation, completed-history acceptance or Metal readiness.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-59e53153`, branch
`t3/native-startup-activation`. Clean initial parent
`5a734b15302084fceed9efcfdfad87dde3f7bfd2` (integrated r303+r305 after r306).
Source commit `68581f8470d20061383780aac5e75316a1f7fda7`.
No applicable AGENTS.md discovered in repository or checked ancestors.
Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277, r296/r303/r305/r306;
inspected current host_install/entrypoint/recovery/reconciliation/native_runtime,
Core, Dispatcher, enrollment and selected test fixtures/conftests.
Memory registry search returned no matching enrollment/startup entry; no memory
fact supplies implementation authority.

SHA256 at source commit:

- native_activation.py: `b6cf067d9af2c621de5f4039959771bc86a6c781db63ba6846b9184f21622a32`.
- native_package.py: `2acee59ab9a92a8aa708932b4ef4cef55685cfa44c0319a3b5e07a760885fc2e`.
- test_native_activation.py: `a31b6c1a2644fb7135077a0e693bb549c7a0c5a13832b51fac094f60ff9ab6ca`.
- unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
  Known legacy digest RED remains unchanged/unwaived.

## Implemented product slice

`entrypoint.create_native_owner` takes explicit private enrollment/control/templates
paths, owner root and staging. No public runtime/configuration object, environment
lookup, automatic enrollment or boot learning. It constructs the existing Dispatcher;
NativeActivation calls actual native_startup_inputs on its control-writer thread.
The NativeRuntime is therefore created, attached, and used for Core import ownership
on the correct owner thread. TLS receives no Core/SQLite callbacks. Existing detached
I/O executor, scheduler, publication/import-intent, frozen-plan, cancellation,
resource and close logic remains authoritative.

Precheck requires the enrolled private root to be completely empty and separate
from spool/staging. Runtime spool and staging must also be empty, even of completed
imports. Core records whether it initialized a fresh namespace. After Core lifetime
exclusion, activation requires that fresh flag, repeats canonical enrollment and
owner/trust/template snapshots around healthy exact configured-boot hello, checks
zero native launches, and repeats empty transport checks. A competitor that
initializes Core in the precheck/acquisition gap causes refusal; normal Core
exclusion handles an outstanding competitor. No SQLite snapshot or Linux history
reader is used to approve native retained work.

Native checked owners enter the existing enrolled activation predicates rather
than the permissive unenrolled foundation path. No profile field is promoted;
existing reconciliation/resource/integrity/readiness gates still apply. The
private owner responsibility is retained strongly in the existing entrypoint
lifecycle registry through startup, constructor timeout and failed close. Successful
close/refusal cleanup removes it; failed close remains unknown and is never retried.
A rejected reuse cannot erase the original owner's responsibility.

Both source package allowlists include native_activation.py. Linux installed
activation retains its exact class path; Linux history refusal and native receipt
exclusion are unchanged. NativeRuntime, recovery, reconciliation and host_install
policy implementations are unchanged.

## Finite execution receipts

Inspected root and service conftests: pytest maintenance guard and service
socket/uvicorn fixtures were not activated. No release conftest exists. Direct
unittest only, `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute
worktree and tests/service sys.path additions, TextTestRunner verbosity=2,
failfast=True, exit nonzero on failure. Interpreter observed Python **3.14.7**,
not target 3.11. Tiny fake TLS/filesystem/model metadata fixtures, no real TLS,
child, ML, listener or installed execution.

1. Development selection: seven new startup tests, **7 PASS / 1.088s**, exit 0
   (tool receipt 8b35be).
2. After adding retained close responsibility: eight native tests plus six
   focused regressions, **14 PASS / 4.110s**, exit 0 (04734d):
   `NativeActivationTests`; Linux `ActivationTests` methods
   `test_post_core_private_reader_under_lifetime_lock`,
   `test_changed_post_acquire_hello_refuses_and_releases_core`,
   `test_late_reconciliation_hello_change_stays_closed`,
   `test_close_requests_stop_but_waits_for_receipt`;
   `NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt`;
   `ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.
3. Inspection found reuse cleanup could drop the original lifecycle record.
   Corrected finish to require identical Dispatcher owner; strengthened live reuse
   test. Exactly the two affected reuse/failed-close methods rerun:
   **2 PASS / 0.559s**, exit 0 (7a4d54). No broad unchanged rerun.

No failed test run. Final diff check passed. Subsequent native_package change is
docstring-only. New tests cover owner-thread construction, reconciliation/close,
retained root/staging refusal, acquisition-gap competing initialization and trust
change, wrong boot/no learning, reuse, and failed close retention/no retry.
Fixture-owned temporary teardown only. No Docker/network/listener/download/install,
actual child/ML/crash, HELD/STT, operational cleanup, root-checkout writes, merge,
push or subdelegation. Tiny fixtures are below the authorized heavy ACTUAL gate.

## Explicit remaining gate and limits

**Every initialized Core refuses, including a clean zero-job prior start.** Failed
post-acquisition startup can leave a newly initialized root; it is retained and
refuses reuse, without repair. No completed native history is accepted. Native
completion support must bind exact schema, frozen Authority, native terminal and
publication records, durable import-intent/completion and reconciliation proofs
before any retained root may activate. Boot/trust renewal remains stopped,
reconciled, separately designed work. Immutable operator input/no hostile same-UID
writer premise persists; snapshots are not hostile-writer/ABA isolation.

The factory is a private owner startup seam, not a native ASGI lifespan or CLI
launcher. Authenticated installed HTTP/CLI/MCP credential/lifespan wiring, native
control topology/mount policy, fresh owned preset/clone onboarding and completed
history remain parent-owned Lot B work. Existing public Linux journeys are intact.
Do not advertise a complete installed native journey from this candidate.

No reviewed arm64 wheelhouse is available. ActualMetal remains **UNAVAILABLE /
UNQUALIFIED**: real Darwin dependencies/ABI, authenticated TLS/topology, hardware,
model/MPS, process/resource/cancellation/restart, clone lifetime and audio/performance
qualification are unrun. CPU/CUDA release qualification remains separate.
Independent fresh Sol review follows; this report neither approves nor adopts itself.
