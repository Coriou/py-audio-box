# Native control F6–F8 correction — r288 Sol

2026-10-25. Implementation complete for fresh parent Sol review before adoption.
This report does not grant independent acceptance. Native Lot A is SOURCE only;
installed native activation stays CLOSED and actual Metal is UNAVAILABLE /
UNQUALIFIED. Lots B/C remain future work.

## Authority and pins

Owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2`, branch
`t3/native-control-integration`, initially clean at
`fdc70aa18f95f745cf09a37ccae8149791e6e1e9` (base
`bd85e1c4c41e1b1927d6f4cc2385041ac357158d` plus root r287 review).
Requested unmerged historical source and report were cherry-picked in order:

- `434e26747736e303e116bdde0b413282755b1f30` →
  `ceecf828b5aba035b627565e32f92c88992a67dc`.
- `e187bf2061ec61a458aafee82d68472ef9097086` →
  `d319205afa81ecbe8efe604de62b45c9b230dfa5`.
- Correction source: **`9ed05f744ac3342dacb114e301e33fe4cfbc254b`**.
- This report is a separate subsequent commit.

Historical `t3-ef4727f6` was not edited. Read full r287 review, r286 report,
r277 normative contract and r285 F1–F5 review. No AGENTS.md found in worktree
discovery. A bounded memory-registry keyword search returned no relevant hit;
no memory-derived technical authority was used. Both `tests/conftest.py` and
`tests/service/conftest.py` were inspected before tests; root-level conftest does
not exist. No pytest invocation/collection, maintenance guard activation or HTTP
listener fixture occurred. No delegation or adoption occurred.

## Corrections

**F6:** `NativeSupervisor.prepare` now invokes the existing internal
`register_snapshot` for a previously absent exact binding. It constructs and
validates Authority, admission and execute-message relations, compares principal,
complete context excluding only authored request, and complete profile inventory
against an existing trusted registration template. The template therefore fixes
owned version, resolved settings/voice controls, execution descriptor, model and
bundle mappings, references, hardware inventory, deployment metadata and bundle
capabilities. The new exact registration copies trusted execution, actual bundle
paths and inventory; none is enrolled by the request. Registration independently
checks the candidate, including inventory freshness and typed execution equality,
before insertion. Existing prepare checks still verify sealed bundle bytes,
headroom, quotas and original deadline before journal creation.

No wire operation/public enrollment/trust field or registry was added.
`register_snapshot` remains internal. Fresh job/attempt/owner/staging tuples may
vary; authored request remains subject to existing closed plan/admission/worker
relations and cannot change the frozen settings or voice. Existing replay and
conflicting allocation fences remain. F1–F5 watchdog/terminal code is unchanged.

The lifecycle fixture no longer reaches from control into `register_snapshot`.
First generation, same-root reopen and next Core allocation traverse actual
Authority → NativeRuntime → canonical fake pinned TLS → actual supervisor prepare
without a prepare fault callback. Only the separate publication storage-fault
case installs its own fault callback. Seeded trusted principal/version enrollment
and explicit trusted boot refresh remain fixture configuration, not public
onboarding. New negatives cover principal, version, profile, model/revision,
descriptor/reference metadata, bundle maps, inventory/architecture/deployment,
native configuration and capabilities, with no journal or exact registration
created. Plan hashes are recomputed where applicable.

**F7:** NativeRuntime now requires a trusted pre-resolved numeric IP endpoint;
gateway hostnames and scoped IPv6 addresses are explicitly unsupported. Operator
enrollment must perform hostname resolution separately, outside RPC. Production
connect uses one numeric socket address, with no getaddrinfo/create_connection or
multi-address loop. `_rpc` passes its absolute cutoff to the connector; remaining
time is recomputed before TCP, TLS wrapping/handshake and after handshake, then
the same cutoff continues through framing. TCP taking 1.5 seconds of a 2-second
budget leaves 0.5 for TLS. Expiration refuses before the next blocking phase and
closes owned raw/TLS descriptors, with no wire request. TLS handshake is explicit
(`do_handshake_on_connect=False`); TLS 1.3, CERT_REQUIRED, certificate pin and no
fallback remain. Connector injection now receives an absolute cutoff too.

**F8:** Strict production receipt validation is unchanged. Five synthetic Linux
receipt constructions now supply `reaped_children=0`: Pocket public prepare
success/cancel (two), dispatcher runtime (one), installed activation runtime (one),
and the additional audited worker engine lifecycle runtime (one). These runtimes
inject confirmed death and create no OS children; zero is their truthful injected
count, not a measured actual supervisor reap. The installed Pocket journey
inherits the public Pocket fixture. Existing synthetic fixtures with explicit
counts 1/3 and actual Linux supervisor receipts were not relabelled.

One additional fixture correction in `test_installed_activation.SelectedRestart`
sets `Mock(spool=None)`: a generic Mock otherwise fabricates a spool attribute
that the r286 private-spool accounting attempts to interpret as a filesystem path.
No production disk-accounting fallback was added.

Eight changed source/test files: `service/native_runtime.py`,
`worker/native_supervisor.py`, `tests/service/test_native_runtime.py`,
`tests/service/test_native_lifecycle.py`,
`tests/service/test_pocket_preparation.py`, `tests/service/test_dispatcher.py`,
`tests/service/test_installed_activation.py`, and
`tests/worker/test_engine_lifecycle.py`. No Core publishing, DDL, install activation,
public schema, protocol schema or legacy loader change.

## Executed evidence

All project/test Python runs used `/private/tmp/pab-0c-venv/bin/python -I -B`
from the owned cwd, with explicit absolute root and `tests/service` paths in
`sys.path[:0]`. Direct dispatcher import additionally required the absolute
`tests` path. File-editing scripts used host Python only for text edits, with no
project imports. No actual socket/listener, process launch/signal, real ML/model,
hardware probe, Docker, network, download/install, crash experiment, operational
cleanup or HELD action occurred. Existing fixture teardown touched only its own
disposable data. Storage was 3,323,484 KiB available, below 6 GiB: only tiny inert
fixtures/archive bytes ran with injected capacity. Production headroom gates
remain unchanged. Printed CPU/CUDA/MPS load labels are synthetic.

For unittest selections below the command shape was:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2","/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2/tests/service"];unittest.main(module=None,argv=["unittest", <quoted selections>],verbosity=0)'
```

1. **80 PASS, 11.782 seconds, final corrected source:**
   `tests.service.test_native_runtime`, `tests.service.test_native_lifecycle`,
   `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`,
   `tests.worker.test_native_qwen.NativeQwenTests`,
   `tests.worker.test_engine_qwen`. Original 74 plus six new methods: remaining
   connector budget, expiry before handshake, expiry before TCP, hostname refusal,
   fresh allocation authorization and altered trusted-template refusal.
2. **32 PASS, 18.322 seconds:**
   `tests.service.test_qwen_lifecycle.QwenLifecycleTests`,
   `tests.service.test_pocket_lifecycle.PocketLifecycleTests`,
   `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`,
   `tests.service.test_production_reconciliation`, plus six
   `tests.worker.test_engine_kokoro.EngineTests` methods:
   `test_exact_arguments_and_detached_identity`,
   `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`,
   `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`,
   `test_bundle_binding_before_backend` (each fully qualified).
3. **6 PASS, 0.485 seconds, separate fresh interpreter:**
   `tests.worker.test_qwen_import_seam`,
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.
4. **46 run, 45 PASS / 1 ERROR, 42.223 seconds:**
   `tests.service.test_pocket_preparation.PocketPreparationTests`,
   `tests.service.test_pocket_preparation.PocketPreparationInventoryTests`,
   `tests.service.test_pocket_preparation.PocketPublicJourneyTests`,
   `tests.service.test_installed_pocket.InstalledPocketJourney`,
   `tests.service.test_installed_activation`,
   `tests.service.test_installed_restart.TerminalRestart`.
   Pocket prepare/save/generate/reopen and inherited installed Pocket journey pass.
   The sole error was SelectedRestart's fabricated Mock spool described above.
5. **2 PASS, 3.883 seconds after the fixture correction:**
   `tests.service.test_installed_activation.SelectedRestart`,
   `tests.worker.test_engine_lifecycle.DispatcherEngineTests`.
   Thus all 46 distinct methods in selection 4 have passing evidence, with one
   targeted rerun rather than a claimed all-green 46-method invocation.
6. **5 direct dispatcher cases PASS:** direct calls to
   `test_serialized_responsive_cancel_unknown_and_retry`,
   `test_timely_completion_wins_cancel_and_lost_ack`,
   `test_late_or_stale_proposal_cannot_publish` for `(False,False)` and `(True,True)`,
   and `test_replacement_only_confirmed_terminal_increments_generation`.
   Each used its own TemporaryDirectory and `pair.__wrapped__` generator, closed
   in finally; no pytest collection, autouse guard or listener. Initial import
   failed because `tests` was absent from explicit sys.path; adding that path
   permitted import of the existing maintenance helper without activating it or
   invoking fresh_run.

Final distinct coverage: **165 unittest methods with passing evidence plus five
direct dispatcher cases**, across the listed invocations. Repeated native runs
are not extra distinct passes. Earlier native runs exposed the old hostname UID
fixture (78/79 pass), then passed 79 and 80 before test refinements. Two later
80-method runs each had one negative-test construction error: unsupported version
name, then changed version ID without matching request selector. Both were
corrected; the final 80-method invocation above passes. No production relaxation
was used. The installed selection emitted an injected close-unknown diagnostic;
SQLite ResourceWarnings were not suppressed. No remaining failed selected case
is being claimed green without a corrected passing execution.

`git diff --check`: PASS. r277 SHA256 remains
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
`lib/tts.py` remains
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
The known broader legacy digest RED is unwaived and was not regenerated or run.

## Parent gates

Fresh independent Sol review of exact correction source is required before parent
adoption. No source self-acceptance, actual TLS deployment, certificate/private-key
enrollment, bridge/firewall reachability, real DNS/operator binding qualification,
Darwin process/FD/durability, ABI/dependency closure, actual Metal load/generate/
cancel/restart, acoustic/performance or release claim follows. Lot B must supply
packaging/enrollment and trusted numeric endpoint configuration. Lot C needs
separate actual Apple Silicon qualification with unchanged headroom. Nonempty
clone conditioning remains UNKNOWN; arbitrary cancellation-point readiness is
unresolved. CPU Docker and CUDA remain separate mandatory release rows.
