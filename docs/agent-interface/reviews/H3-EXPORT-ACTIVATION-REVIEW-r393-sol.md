# r393 — independent owner activation and r378 composition review

2026-10-25. **Bounded owner-activation SOURCE GO; full r378 SOURCE NO-GO / finite evidence incomplete.** No demonstrated production-code defect was found in the inspected composition. R393-E1 below remains a required integrated uncertainty/retained-close fixture, not a proven code defect or a demand for an unlimited fault campaign. Release **NO-GO**; legacy **CLOSED**. No adoption, merge, activation or operational qualification authority is granted.

## Authority, pins and inputs

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-a145b721` was clean before and after execution. Verified direct chain:

- `37dd1d44fe1d18e4d2eb4d5b9f7570b4ff7de5ed`: report-only r392 HEAD.
- `48da311dfce93f3a5a6ad09dd1448ee251e3153c`: owner activation source, direct parent 580f057.
- `580f0570c133113e26a84fe8ca7ff6c3ca39965a`: reviewed history-phase report, direct parent `ef9746065548dfba2e6408a5102c32ce3590b3d7`.
- Full linear ancestry through retained-history, snapshot correction, daemon wiring, worker seal and initialization to base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` verified with parent hashes. Base-to-HEAD diff whitespace check passed.

No applicable AGENTS.md found in candidate descendants, candidate/ROOT ancestor chains or review destination directories. Read FULL absolute ROOT inputs under `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`:

- H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md — SHA-256 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`.
- H3-EXPORT-INITIALIZATION-REVIEW-r381-sol.md — `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`.
- H3-WORKER-SEAL-REVIEW-r383-sol.md — `4617c07c962b8a9a5ff06aa7f36392d9262f45d837d3f25ece23a74c22554651`.
- H3-SNAPSHOT-REVIEW-r387-sol.md — `1da16379dc57b41cbbd1153fa69137f26f7160505c917cdb4a159e99c9fc99a4`.
- H3-HISTORY-PHASE-REVIEW-r391-sol.md — `351d44bff1c393bd1c02a5a8c966c6012fe49af43a6ac4ae5dd26efd17862eaf`.

Read candidate r379/r380/r382/r384/r386/r388/r390/r392 reports, owner diff and current activation/Core/Dispatcher/helper/worker/private-reader/reconciliation/source-FD and selected test seams. Earlier reviews remain bounded evidence; their GO statements were not promoted to composition acceptance. Memory registry quick search returned no relevant hits; no historical memory fact controls this decision.

Sole persistent write: this ROOT report, mode **0600**. No source/runner/log writes, commit, merge, delegation or self-adoption. Existing unrelated ROOT untracked evidence was left untouched; no HELD/STT content was opened. Inert temporary fixture disposal used existing test lifecycles only.

## R393-E1 — required finite uncertainty composition evidence remains missing

**Acceptance-blocking evidence gap, not a demonstrated production-code defect.** r378 finite exit 5 requires focused unconfirmed-helper cases with retained uncertainty, no publication and no premature ACK/capacity discharge. The current fixtures establish three distinct facts but never compose the public helper-unknown path with retained owner close:

- `tests/worker/test_export_seal.py:150-164` runs actual managed_reference.acquire with fake Popen/socket endpoints: unknown waits retain the fake process and close the transferred FD.
- `tests/worker/test_export_seal.py:113-118` injects acquire refusal into verify_export and establishes supervisor failed/quarantined hello.
- `tests/service/test_export_owner_activation.py:281-295` starts an already bound/running public export, injects UNKNOWN observation, checks no artifact/ACK and executing reservation, then explicitly clears UNKNOWN at line 291 before cancel and injected reap. It does not close an owner while the helper remains uncertain. The selected existing close-thread test independently covers lifecycle cancellation, not this helper failure.

The distinction is concrete: worker prepare calls verify_export before journal allocation (`worker/runtime_supervisor.py:127-128,139-153`). A helper unknown-wait refusal can therefore reach the owner before a runtime ID is returned. Dispatcher catches the prepare failure and retains phase unknown (`service/dispatcher.py:517-521`); a row with runtime None stays quarantined (`:566-570`). Installed close waits for the outstanding row (`:390-397,559-574`), and close timeout must leave the owner/notification responsibility retained through `_owned_app` (`service/entrypoint.py:438-444,476-495`). A bound UNKNOWN-observation fixture does not execute those same transitions.

**Finite requested correction:** add one inert fresh checked-family public upload/export fixture that routes actual acquire with fake Popen/FD transfer and unknown waits through real worker verify_export into owner prepare failure, then requests ordinary owned-app shutdown. Assert retained fake helper handle, closed transferred FD, supervisor quarantine, unknown dispatch/no invented runtime ID, no artifact/publication/ACK, reservation not discharged, and owner/notification/lifetime exclusion still retained when close cannot finish. Assert a retained lifecycle record on close timeout/cancellation; do not turn close failure into successful shutdown or forge a reap receipt. The fixture must explicitly account for any retained test threads/handles; no product repair, automatic unquarantine or cleanup shortcut is requested. If a narrow fixture seam is needed, keep it test-only. Production changes are justified only if that fixture exposes a defect.

No additional acquisition, actual process, infinite-fault matrix or new recovery protocol is needed. This report does not silently waive the gap merely because the separate seams pass. Resolve this one finding before final complete-family Astra review; parent owns implementation and commissioning.

## Accepted composition observations

1. `entrypoint.py:298-368,375-407,681-684` selects FAMILY/LAUNCH explicitly, detaches closed configuration, compares canonical combined admission profiles/budget, refuses caller AudioExports and carries the checked enrollment expectation into Core. Actual credential production-write principals corroborate the configured export principals before model/owner construction. The factory preserves the existing retained lifespan owner, rather than creating another scheduler or lifecycle.
2. `dispatcher.py:66-103` performs precheck, checked Core acquisition and postcheck before installed_checked. `core.py:229-260,300-303` preserves private-copy expectation validation and opened-store validation before WAL/initializers; the creation-version argument alone grants no installed authority. Post-owner reads run under the Core lifetime exclusion. `reconciliation.py:373-378` now threads the same authenticated family/public expectation into the reviewed inventory; admission still waits for reconciliation and authenticated live hello. Checked parser shape alone does not create readiness.
3. `entrypoint.py:82-125,295-368` and `installed_export.py:319-365` keep selected new gate/seal shape, strict integer publication format, closed keys, enrollment/configuration/registration/control-incarnation/live-boot brackets. `daemon_helper.py:420-468` preserves INSERT prestate, poststate and all-other-row identity, repeated image/topology/assets/seal observations around registration/notify. New helper ready/status/stop/up uses the existing coordinator; same-family restart invokes the exact stopped receipt selector at helper lines 532-540, not a loose equality substitute.
4. `worker/runtime_supervisor.py:352-402` independently checks configured principal, canonical converter and full frozen profile, measured eligible profile, immutable local converter, then the dropped managed-reference descriptor. O_RDONLY/type/UID/GID0600/single-link/positive bounded size/PCM/hash/stable identity and rewind remain checked. No caller converter/reference map or manufactured qualification receipt was added. Source FD transfer in owner tests is injected; real helper behavior is independently exercised with fake processes.
5. `dispatcher.py:249-267,326-359` keeps idempotent lookup before new-work readiness/converter gates; exact checked family plus v2 is required for installed submit/retry/availability. Unqualified configuration does not advertise export. Existing v1/native remain excluded; the selected v1 activation/helper checks pass. This is finite source regression evidence, not broad native qualification.
6. `core.py:707-713` explicitly sets the new-family private output copy to0600 before writing. Actual owner-generated >1 MiB retained history now reopens through the checked reader, with no copied local rows in that owner journey. Existing publication intent/files/public commit, terminal receipt, ACK and reservation fences remain in their existing order; no protocol rewrite or early uncertainty discharge was found. r391's acknowledged-only SQL predicate is preserved and independently retested across six reader seams.

## One coherent independent execution

Read both candidate conftests before application imports. Their pytest listener/uvicorn fixture was neither collected nor invoked. Executed exactly one current affected selection from stdin:

`/private/tmp/pab-0c-venv/bin/python -I -B -`

Before application/test imports, imported only sys, installed an audit hook recording and raising on `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`, and asserted version `(3,14,7)`. Set cwd to the exact candidate and prepended only candidate root, tests/service and tests. Direct unittest runner; selection count asserted 37. No fixture/source adapter or broad pytest run. Raw allowlist/results remain in this review turn's tool output; no disk runner/log created.

**37 PASS; zero failures/errors; 68.773 s; exit 0; CPython 3.14.7; forbidden-event attempts `[]`.** No retries or additional executable probes. Existing matrices already exercise the concrete boundaries; a standalone probe would not close R393-E1 without the requested composed fixture.

Exact allowlist (all methods in each of the three named classes at this immutable pin, plus five explicitly named methods):

- tests.service.test_export_owner_activation.ExportOwnerTests — 11:
  `test_bad_output_no_publication_new_retry_refused`, `test_explicit_family_and_caller_exports_refuse`, `test_factory_uses_same_lifespan_and_checked_configuration`, `test_fresh_owner_public_large_export_reopen_replay`, `test_generated_kokoro_asset_uses_same_export_resolver`, `test_helper_status_stop_reopen`, `test_owner_seal_and_notification_changes_refuse`, `test_stale_output_no_publication_new_retry_refused`, `test_timeout_receipt_no_publication_new_retry_refused`, `test_unknown_descendant_retains_capacity_until_cancel_reaped`, `test_unqualified_configuration_does_not_admit_or_advertise_export`.
- tests.service.test_export_daemon.ExportDaemonTests — 9:
  `test_child_adapter_selects_fixed_new_entry_and_hello_configuration`, `test_credential_principal_refuses_before_core`, `test_exact_subpath_commands_and_inspection_refusals`, `test_fixed_child_entrypoint_and_closed_family_request`, `test_install_init_new_core_writer_sealed_registration`, `test_private_client_install_exact_family_and_preflight_refusal`, `test_seal_profile_principal_image_boot_and_assets_brackets_refuse`, `test_subpath_daemon_failure_has_no_fallback`, `test_up_uses_existing_coordinator`.
- tests.worker.test_export_seal.ExportSealTests — 12:
  `test_actual_acquire_unknown_wait_retains_process_and_closes_fd`, `test_closed_new_hello_old_exact_and_readiness_refusal`, `test_closed_startup_inputs_and_transport_family`, `test_constructor_refuses_parallel_caller_maps`, `test_full_frozen_profile_principal_and_converter_before_helper`, `test_helper_fd_verified_retained_readonly_and_closed_on_bad_hash`, `test_helper_uncertainty_quarantines_without_second_scheduler`, `test_local_closure_rechecked_before_source_acquisition`, `test_main_explicit_family_selection_only`, `test_startup_requires_trusted_image_and_actual_immutable_closure`, `test_transport_requires_matching_seal`, `test_unqualified_profile_cannot_acquire_or_dispatch`.
- tests.service.test_daemon_volumes.DaemonTests.test_install_up_status_stop_same_id_up.
- tests.service.test_export_initialization.ExportInitialization.test_fresh_exact_and_same_family_reopen.
- tests.service.test_export_retained_history.ExportRetainedHistoryTests.test_dispatch_phase_refusal_across_clean_history_readers.
- tests.service.test_installed_activation.ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- tests.service.test_installed_activation.ActivationTests.test_direct_cleanup_cancel_close_thread.

The public owner journey starts with real tiny fresh initialization and fake inspected Docker/sealed worker facts, acquires the actual installed owner, uploads managed PCM, publishes 1,152,044-byte PCM24 through real owner SQL/publication checks, verifies full/range/CLI download and hashes, closes and reopens its own produced history, then replays the same key/job without new prepare/allocation/reservation despite lost live readiness. HTTP/CLI/MCP projections agree. Generated Kokoro managed-asset reuse follows the same resolver with injected small speech PCM; no inference occurs. All converter output, terminal/reap/ACK journal and Docker/UID observations remain explicit inert injections.

This new coherent PASS does not erase r392's historical 37-method 36 PASS / 1 stale-fixture FAIL (68.752 s), its corrected 3 PASS follow-up (10.001 s), or final 11 owner PASS (32.105 s). The current registration-only fixture correctly asserts `initialized` at test_export_daemon.py:64; it has not become a ready-owner fixture. Historical r390/r388 failures and bounded later corrections remain as recorded in their reports.

## r378 finite disposition and parent gates

1. Fresh exact v2, same-intent completion and same-family reopen: source mechanisms present; current fresh/Core/owner reopen evidence plus bounded r381 evidence accepted within SOURCE scope. No old-volume migration or repair opened.
2. INSERT-only registration and immutable brackets: composed helper uses reviewed writer and retained readers; selected current bracket/refusal matrix passes. Historical shipped-writer absent universal version check remains explicit; this does not qualify an archived binary.
3. Sealed checked owner/configuration/credentials/profiles/source FD and truthful capability: bounded SOURCE GO. Missing operational pins are closed required inputs, not distributed fixture defaults. Actual closure completeness is unqualified.
4. Fresh installed injected large public journey/generated asset/full-range/hash/reopen/replay and HTTP/CLI/MCP: current owner-produced fixture PASS; condition satisfied at the authorized inert SOURCE evidence tier only.
5. Timeout/bad/stale output, cancel/bound-descendant uncertainty, canonical retained copies, acknowledged phase and separate helper unknown waits pass. **R393-E1 leaves full finite exit incomplete** until the single public helper-unknown -> retained owner-close fixture is supplied and independently reviewed. No production defect is inferred solely from absent evidence.

Final Astra complete-family review is parent-owned only after independent findings are resolved. This review does not commission it, merge, self-adopt or declare r378 complete.

No actual Docker/network/listener/application child/FFmpeg/ML/hardware, HELD/STT content, installation/migration/build/download/acquisition, operational cleanup or heavy waiver. No fresh disk/RAM observation. Seven real-child reference-publication tests remain **BLOCKED/UNRUN**. Actual converter/image pins, alias-free complete runtime closure, measured profiles, actual installed CPU/runtime and resource qualification remain **UNRUN**. Linux CPU, Intel Mac Docker Desktop and OrbStack are all **UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. **Bounded owner SOURCE GO; full r378 SOURCE NO-GO pending R393-E1; release NO-GO; legacy CLOSED.**
