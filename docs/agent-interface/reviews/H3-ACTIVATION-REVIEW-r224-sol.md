# H3 r224 Sol — independent first-round installed activation review

2026-10-24. **NO-GO for bounded source adoption of this candidate.** Two demonstrated lifecycle/order defects remain below. This decision concerns the installed activation source unit only. Missing packaging is outside this assignment and is not a new gate. Actual Linux installation, packaging, audible speech and release remain unqualified; parent owns correction, subsequent review and adoption.

## Pins and review boundary

Inspected clean candidate `/private/tmp/pab-r223-sol`, branch `impl/r223-installed-activation`, HEAD documentation `8b5e5aafee91d599e260027b25545abddba7b061`. Verified direct parents: source/tests `d692a3d715dc8d5cca6c2fc21f17150bd3f8d44d` directly follows base `663af006833bbbe8304928b5c49aa168d1268283`; documentation directly follows source. The source delta is exactly eight files: `service/api.py`, `service/auth.py`, `service/dispatcher.py`, `service/entrypoint.py`, `service/execution_context.py`, `worker/runtime_supervisor.py`, `tests/service/test_installed_activation.py`, `tests/worker/test_engine_lifecycle.py`. HEAD adds only the r223 report.

Read complete r198 (including six launch steps, Lot B, restart/failure and verification rules), r213, r216, r222 and r223, and traced changed code with its Core, startup-reader, policy, reconciliation, credential and runtime dependencies. No applicable AGENTS.md found in checked ancestors/scoped paths. Bounded memory-registry search returned no relevant hit and supplied no premise. No skill or delegation. Sole persistent write is this root report; no source edit, commit, adoption or unrelated cleanup.

## R224-1 — blocking: repeated startup cancellation loses the acquired owner

`service/entrypoint.py:302–309` shields acquisition initially, but its CancelledError handler awaits `acquisition` without shielding it. A second cancellation cancels that task while its `asyncio.to_thread` function continues. Assignment to `owner` never completes. The finally at `320–324` consequently skips owner.close and closes the notification. Neither app.state nor another lifecycle retention path receives the eventual owner.

Independent inert probe used the real ActivationTests disposable registration/Core fixture and real Dispatcher. A wrapper constructed Dispatcher, retained it only in the probe, and paused its return using threading events. Factory captured that wrapper before lifespan entry. After acquisition reached the pause, the probe cancelled startup twice, released the wrapper, then inspected the real owner. Result:

`notification_closed=True; owner_alive=True; closing=False; app_retains_owner=False`.

This is an owner/admission leak, not proof of safe failed shutdown: the control writer remains active and has received no close intent, while lifespan has finished and invalidated its notification dependency. The probe explicitly closed its captured owner afterward. No worker job or external operation was needed. The initial probe captured the factory before installing its wrapper and therefore did not exercise the intended interception; that disposable process was interrupted, and the corrected bounded probe above supplied the result.

Correction must retain acquisition/owner responsibility across repeated cancellation and ensure acquired owners receive cleanup, including retaining the owner on failed close. Add a regression with cancellation during acquisition followed by a second cancellation before acquisition completes. Assert no untracked live owner, close intent where ownership was acquired, and preserved obligations on failed close. Shielding only the first await is insufficient. The analogous unshielded preparation wait at `299–300` should be covered by the same cancellation audit.

## R224-2 — blocking: queued cancellation releases before final activation reconciliation

`service/dispatcher.py:129–132` performs startup queued-cancellation release before `initialized` completes and before the later reconciliation hello (`417–427`). `_activation_open` (`172–174`) checks enrollment/post-owner verification only; it does not require reconciled, nonquarantined readiness. Therefore a launch that ultimately refuses installed activation can already have changed reservation accounting.

Independent probe used the real disposable Core/Admission/Production fixture, created a queued reservation, marked its job cancelled/never_started using the existing test fixture pattern, and configured the runtime to change boot on hello 5 (the existing late-reconciliation negative). After owner reconciliation, the result was:

`persistent_activation=False; quarantined=True; reservation.state=retained`.

Thus the cancellation release occurred despite failure of the final installed hello. No terminal receipt was fabricated; the defect is the ordering of an unstarted accounting release relative to the required activation gate. This contradicts r198's requirement that the new scope activate only after exact configuration/registration/hello/reconciliation and r223's checked healthy-startup claim for queued release.

Defer installed queued-cancellation release until the complete activation/reconciliation decision is healthy, retaining the resource, persistence, integrity and orphan checks. Preserve legacy enrolled inspect-only behavior. Add a combined queued-cancellation plus late-hello-refusal regression asserting the reservation stays queued/accounted; retain the existing healthy-startup release positive. Do not repair by weakening the late hello check.

## Other inspected properties

The expected pre-owner waiting gate and independent Core lifetime acquisition exist. Exact-type InstalledActivation checking, one-use `started` flag and runtime/root binding precede Core; `postcheck` repeats the private-copy DB/configuration/registration/notification/incarnation/hello bracket under Core exclusion. No new protected receipt/provenance read or digest-as-proof claim was found. SQLite owner operations stay on the control writer thread; the private reader uses its separate copied database. The configured production factory builds frozen Kokoro contexts; the low-level dependency check itself only tests contexts for non-None, so its trust remains internal rather than a public authorization boundary.

Normal orphan/resource/integrity reconciliation remains. Installed close sets closing/stop intent and continues the existing observe/collect/publication/ack path; the passing shutdown test preserves executing capacity after stop send and exits only after an injected receipt. Timeout raises while leaving the writer alive. That retained-obligation behavior does not cure R224-1, where no close is sent and the owner reference is lost.

Explicit configuration/credential readers require root:10001 and exact 0440/0640 modes, regular single-link inputs, no-follow traversal, bounded reads and descriptor/path identity checks. Credential principals bind to configured contexts; scopes and bearer strings are closed and no default secret or secret logging was added. Bundle/mapping checks reuse the closed configuration and sealed-tree validator; the worker requires explicit fixed installed policy/configuration and stores its mapping through MappingProxyType. Legacy local-owner verification remains the default. These are source/trusted-provisioning checks, not hostile-filesystem or actual mount/permission qualification.

Installed capabilities use configured profiles/engines and current check time, lowering readiness on closed activation. Installed status uses Dispatcher readiness/configuration digest rather than the synthetic fixture status. Qualification/measured labels remain provisioned profile claims, not evidence generated by this review. No public gate publisher, discovery/download fallback or readiness fabrication was introduced in the inspected delta. Existing synthetic test coverage below passes; it does not resolve the two additional negative combinations.

## Independent checks and limits

Read both conftests before execution. Used only `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit candidate root, candidate/tests and candidate/tests/service import paths. Ran the seven authorized unittest modules: `tests.release.test_headless_install`, `tests.service.test_installed_activation`, `tests.worker.test_engine_lifecycle`, `tests.service.test_production_api`, `tests.service.test_production`, `tests.service.test_production_admission`, `tests.clients.test_production_clients`.

**71 tests PASS, exit 0, 28.036 seconds; Python 3.14.7.** Both additional corrected probes above completed successfully and demonstrated the stated defects. `git diff --check 663af006 HEAD` passes. All eight changed Python files independently parse with `feature_version=(3,11)`; this proves grammar only, not Python 3.11 runtime/dependency compatibility. Candidate remained clean after verification.

No pytest/maintenance guard/full suite, Docker build/run/pull/exec, actual Linux proc/mount/UID qualification, network/listener, model inference/download, ML/hardware, HELD operation or production evidence mutation occurred. Files/SQLite/locks in tests are real disposable fixture objects; identities, host/hello/reap/backend observations are injected. No actual descendant-death, speech or installation acceptance is inferred. Packaging remains the already separate successor, not a blocker invented for this bounded review.
