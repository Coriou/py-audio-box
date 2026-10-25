# r352 Sol — native live availability within unchanged trust

2026-10-25. Implementation candidate for r351 commission 1; **fresh independent Sol review and parent adoption remain pending**. Legacy reconciliation CLOSED. Metal UNQUALIFIED. Actual release NO-GO.

## Authority, scope and pins

Sole implementation worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e5f5061e`, clean starting HEAD `103ad35cf4e6628e8604e45f08f998c2c954eccc`. No root checkout writes, merge, push, delegation or self-adoption. Read FULL r351 (SHA256 `cb18433122c203d7c115bcd735befba570cdd40d223ed4ce987992c344fbf54e`), PRODUCT, current NATIVE-PACKAGE and r343 contract; inspected current Qwen verification/hardware, private protocol/runtime/supervisor/child, context derivation, selection, admission, activation and renewal seams. The memory registry search yielded no relevant technical premise. No applicable AGENTS.md was found by repository/parent discovery.

Only SOURCE/inert work was authorized. No actual host availability query, hardware/model probe, ML model/import/acquisition, download/install, network/listener, Docker, OS child execution, host crash, HELD/STT work or operational cleanup was performed. Existing fixture process handles, TLS frames, ASGI, SQLite/files and backend functions remain inert. No live free-memory measurement or gate waiver: the supplied below-6-GiB constraint and unchanged 6-GiB heavy gate remain separate from source tests.

Exact final tested-source tree and receipt pins are recorded below. This report is a handoff, not an independent review or release acceptance.

## Implementation

`worker/native_availability.py` owns one bounded in-memory observation for the exact enrolled MPS device/inventory and supervisor incarnation. It retains at most one provider flight; concurrent callers share it, waits are bounded to one second, a hung provider cannot spawn replacement threads, and late results never revive a timed-out observation. No persisted lease, probe history, timer/scheduler, signature or extra epoch framework was introduced. The existing 300-second freshness bound and eight trust epochs are unchanged.

The installed default provider is implemented: it compares actual native host metadata and queries Metal's `MTLCreateSystemDefaultDevice` through ctypes, releasing the returned device through Objective-C. It does not import torch/ML, spawn a process, download, load a model or perform inference. No real host query was executed here; the host API wiring test executes the provider function with mocked library calls and all journeys explicitly inject inert providers. Availability remains distinct from actual runtime ABI, performance and hardware qualification.

The closed private `availability` operation uses the existing authenticated TLS 1.3 channel and per-frame peer pins. Request/response envelopes bind fresh request IDs, deployment/configuration, boot/lane, immutable inventory digest and runtime manifest. The control validates exact static host/device/index/support shape and current positive primitive-typed availability aged 0..300 seconds. Old hello snapshots do not supply a current observation. Two bounded availability RPC slots are separate from lifecycle/transfer slots; refresh traffic cannot consume those stop/collect/ACK slots. No public API/trust/timestamp field was added.

Static native validation now accepts expired immutable enrollment timestamps, retaining model/profile/topology/trust and exact native identity checks. The original inventory/context bytes remain the identity and measurement input. Resolution, preset derivation, selection and Admission use current availability separately; native prepare/start and supervisor registration/start enforce it independently. Input verification is followed by another check before launch intent. A prepared Authority can acquire current availability without changing its snapshot, deadline, plan, voice, origin, settings or measurement binding. The native child validates the immutable static input; its owned supervisor supplies the current-availability launch gate.

Activation/control reopen attempts to reacquire availability over the same pinned channel. A failed association does not suppress authenticated clean-history activation/collect/reads: future work still has to pass the same live checks. Discovery reads the last validated metadata only, expires it conservatively, and never probes, resolves owned media, loads ML or acquires models. Explicit renewal commit also requires current availability; boot/trust changes and all r343 compatibility/unknown/capacity/history fences remain in force.

The cancellation fixture exposed a shared registration check in terminalization. Terminalization now explicitly uses static validation only, so a failed/blocked probe cannot poison reap, receipt construction, collect or ACK. No historical receipt/Authority is reused as new allocation authority. The elapsed public tests compare prior complete SQL rows (including reservations and dispatch Authority) and retained media/spool/stage bytes, templates, enrollment and versions. Freshness alone spends no capacity; new jobs retain the existing charges.

The source archive allowlist includes the new availability module and the already-required `service/job_control.py`, `service/production_selection.py` and `worker/engines/qwen_prepare.py`, which were missing. This changes archive/runtime identity and is an **upgrade**, not old-installed-binary renewal compatibility. No upgrade migration is implemented. Fresh installation must begin with the corrected package.

## Source evidence and limits

The new preset journey registers its first-ever public preset after 2701 injected freshness seconds, then generates again after 1501 seconds and after another 1501 seconds with a control-only reopen and the same live supervisor. It has three provider calls, three jobs, one boot, no predecessor/renewal namespace and unchanged enrollment/templates/old result bytes. HTTP, CLI and MCP share existing generation/registration operations.

The saved-clone journey uses existing public upload/prepare/results/save/generate, then generates after two 1501-second windows/control reopens. Three provider calls serve four total jobs within retained limits. Old versions/reference/artifact/Authority/reservation/media bytes remain exact. A further reopen with a failed provider preserves public old-key replay, job/artifact reads and retained bytes while new generation refuses. Measurement tests retain the full immutable execution hash across refresh and reject a changed measurement hash; no readiness or measured qualification is manufactured.

Negative fixtures cover failed/unavailable/stale/future/timeout observations; malformed primitive fields; request replay/foreign inventory/manifest/boot/shape/peer; changed incarnation/static identity; resolution, selection, admission, control and supervisor prepare/start; prepare-to-start expiry with positive refresh or refusal; concurrent refresh coalescing and slot saturation; cancellation of a prepared request while refresh waits; and continued owned stop/reap/collect/ACK after availability failure. Existing affected unknown/history/import/pair/renewal/origin/capacity fences are included in the finite selection below. This is finite fixture coverage, not exhaustive adversarial or installed qualification.

Inspected `tests/conftest.py` and `tests/service/conftest.py` before execution. Used direct unittest under `/private/tmp/pab-0c-venv/bin/python -I -B`, verified Python **3.14.7**. Pytest/conftests, real HTTP fixture listeners, Docker and actual children were not invoked. Names saying “actual child/inference” in retained tests refer to the fixture's in-process functions and fake handles, not OS child/hardware execution.

Development receipts were retained in the tool transcript: 27 PASS initially; a 10-test new-seam run had two failures and one setup error (frozen fixture monotonic clock/late-result assertion, terminalization incorrectly refreshing, and tuple-key canonicalization in the assertion); a three-test focused correction run still had the terminalization failure; the corrected terminalization test passed. Further focused runs were 11 PASS, 5 PASS and 2 PASS. These were implementation diagnostics, not hidden clean receipts or adoption authority. The final finite selection was run once, with no broad unchanged rerun; its result and any strictly focused correction are recorded below.

## Final receipt

The once-run finite selection below completed **161 tests in 159.662 seconds: 160 PASS, 1 FAIL**. The failure was `ProductionSelectionTests.test_metadata_runtime_topology_and_unknown_unmeasured_are_conservative`: an intentionally incompatible fake runtime acquired a native marker, metadata inspection continued into a nonexistent native cached-availability method, and conservative exception handling lost the explicit topology reason. Added an immediate metadata refusal for incompatible topology. **Only that failed test was rerun: 1 PASS in the focused correction receipt.** No broad unchanged selection was repeated. The 160 other passes precede this one-line correction; there is no claim that a second all-green 161-test run occurred on the final tree.

Pins (Git tree excludes this report so the source snapshot is independently reproducible):

- Once-run selection source tree: `abb08998c3aedee573e5779c7ad775bdf48e8d56`.
- Final corrected source tree: `36279792bab58fc7fcb4f400a61142076aac9919`.
- Final staged binary diff SHA256 before report: `c4754dd6a89f888e51c76f02691ebea031e97a434052b4d8b6dde6c9280a5ddd`.
- Selection script `/private/tmp/r352-final-selection.py`: SHA256 `9d50b4a0f3e704e1f7dd53f88c9ed0f664efad1d690d2de82f1a5288d17dd822`.
- Raw once-run log `/private/tmp/r352-final-unittest.log`: SHA256 `8ca3b3edcb5572e19bcf74ce28c9c632a75e26fb7fc22698493969992a7cbc2f`.
- Focused correction log `/private/tmp/r352-focused-correction.log`: SHA256 `c64b8766cbccb60422fdfed97174c8251e9f71ed796a17c2ba789a2f9b10f11d`.
- New availability module SHA256: `e8d3e7fbeecd3d58e795a17bd5877a8e08c3ba8b79ff8a271419be1f36962a95`.
- Native runtime SHA256: `2992b6769333c0ce44334978f7dd1f9d4e3cfebb2f8e4b7412ae3b2fbf74f9e3`.
- Native supervisor SHA256: `106a40aa2230a2387c8d11e0ee0e78ee539feafbbf3f8a66040d5586b0add1e9`.
- Native protocol SHA256: `1b078f5070c361e731dbf459244de4753551e147e4d229afedce13276e95b8b0`.
- Corrected selection SHA256: `c3b8c9a735dcfc71c819b1eef96fe0206120fd5e41e8253139ce6bbe8f7b0404`.

Exact execution command:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r352-final-selection.py
```

The script prepends only this worktree's absolute path to `sys.path`, uses `unittest.defaultTestLoader.loadTestsFromNames(NAMES)`, runs with `TextTestRunner(verbosity=2)`, and exits nonzero on failure. Its exact finite `NAMES` list is:

```python
NAMES = [
    'tests.worker.test_native_availability.AvailabilityTests',
    'tests.release.test_native_live_availability.LivePresetTests',
    'tests.release.test_native_live_availability.LiveCloneTests',
    'tests.release.test_native_live_availability.LiveActivationTests',
    'tests.worker.test_native_protocol.NativeProtocolTests',
    'tests.worker.test_native_supervisor.NativeSupervisorTests',
    'tests.service.test_native_runtime.NativeRuntimeTests',
    'tests.service.test_native_lifecycle.NativeLifecycleTests',
    'tests.service.test_production_selection',
    'tests.release.test_native_activation.NativeActivationTests',
    'tests.release.test_native_enrollment.NativeEnrollmentTests',
    'tests.release.test_native_onboarding.NativeOnboardingTests',
    'tests.release.test_native_managed_clone.ManagedCloneTests',
    'tests.release.test_native_managed_clone.ManagedCloneDerivationTests',
    'tests.release.test_native_origin_voices.OriginCloneTests.test_cancelled_clone_two_renewals',
    'tests.release.test_native_origin_voices.OriginCloneTests.test_failed_clone_two_renewals',
    'tests.release.test_native_origin_voices.OriginPresetTests.test_public_preset_stable_across_two_renewals',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_two_epochs_offline_old_history_new_allocation_and_stale_path',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_second_boot_during_commit_refuses_without_commit',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_host_cumulative_limits_and_no_restart_identity_reuse',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_sealed_row_change_and_unknown_namespace_refuse_without_repair',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_eight_total_epochs_and_ninth_refuses_before_intent',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_control_spool_capacity_refuses_before_queued_job_and_replay_is_free',
    'tests.release.test_native_renewal.RenewalCompletionTests.test_current_peer_inventory_mismatch_refuses_activation',
    'tests.release.test_native_history.NativeHistoryTests.test_unknown_staging_evidence_refuses',
    'tests.release.test_native_history.NativeHistoryTests.test_missing_import_completion_refuses',
    'tests.release.test_native_package.NativePackageTests.test_archive_deterministic_bytes',
    'tests.worker.test_native_qwen.NativeQwenTests.test_native_freshness_and_linux_separation',
]
```

Focused correction used the same interpreter/flags and absolute worktree path with `loadTestsFromName('tests.service.test_production_selection.ProductionSelectionTests.test_metadata_runtime_topology_and_unknown_unmeasured_are_conservative')`. `git diff --cached --check` passed after correction. No further runtime test was needed for this report-only addition.

## Handoff

Fresh independent Sol review must assess the actual diff, including the native default provider, private association/session relation, clean-history activation behavior, terminal static split, selection/measurement identity and unchanged renewal/retention fences. Parent owns adoption. No installed topology/peer denial, ABI/wheelhouse, actual MPS/audio/performance or hardware stop/restart/upgrade acceptance follows. CPU Docker is the next separate lane; CUDA and Metal release obligations remain mandatory. No UI, STT, universal router or storage reclamation was added.
