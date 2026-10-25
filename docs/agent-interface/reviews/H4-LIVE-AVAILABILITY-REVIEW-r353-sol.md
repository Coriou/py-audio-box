# r353 Sol — independent live availability SOURCE review

2026-10-25. **Bounded GO for r351 commission 1 / P1 SOURCE exit on the exact corrected r352 candidate. No blocking source finding. Parent adoption remains separate. Actual release NO-GO; Metal UNQUALIFIED; legacy reconciliation CLOSED.** CPU Docker usability is the next separate commission, not a blocker for this exit.

## Authority and verified pins

Read-only candidate: `/Users/ben/.txcode/worktrees/py-audio-box/t3-e5f5061e`. Verified HEAD `c7b64eefdda3bd4032d7c7a560af4b692e056b97`, base `103ad35cf4e6628e8604e45f08f998c2c954eccc`, and empty `git status --porcelain` before review and after tests. No applicable repository or explicit ancestor AGENTS.md was found. Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or operational actions.

Read FULL r351, r352 and PRODUCT; inspected the complete base-to-HEAD diff (23 files, 875 additions / 54 deletions), including all test changes, and surrounding provider, protocol, runtime, supervisor, child, hardware/static validation, context, selection, Admission, activation, renewal and archive consumers. The memory evidence-gate skill supplied scope/receipt discipline only, not a technical premise about this candidate.

Verified final source tree `36279792bab58fc7fcb4f400a61142076aac9919`: its only difference from HEAD is the added r352 report. Reproduced binary diff SHA256 against base: `c4754dd6a89f888e51c76f02691ebea031e97a434052b4d8b6dde6c9280a5ddd`.

Verified SHA256 pins:

- r351: `cb18433122c203d7c115bcd735befba570cdd40d223ed4ce987992c344fbf54e`.
- PRODUCT: `b1352e25cff039890dc130863e9ab0effdd0441e57ebc97e0e1d49a6e2c00d14`.
- `native-dist-manifest.json`: `6c159c968f0602bcec40ede55cc2797f54dc8a1a2623dd628c233dcfaa0fa747`.
- Availability: `e8d3e7fbeecd3d58e795a17bd5877a8e08c3ba8b79ff8a271419be1f36962a95`.
- Runtime: `2992b6769333c0ce44334978f7dd1f9d4e3cfebb2f8e4b7412ae3b2fbf74f9e3`.
- Supervisor: `106a40aa2230a2387c8d11e0ee0e78ee539feafbbf3f8a66040d5586b0add1e9`.
- Protocol: `1b078f5070c361e731dbf459244de4753551e147e4d229afedce13276e95b8b0`.
- Corrected selection: `c3b8c9a735dcfc71c819b1eef96fe0206120fd5e41e8253139ce6bbe8f7b0404`.

## Findings and adversarial assessment

**No actionable blocking finding or failing reproduction in the inspected scope.** This is a finite source/inert review, not exhaustive defect clearance.

The provider is implemented in the installed default path, not merely injected by tests. `worker/native_availability.py:47` compares actual native host metadata, calls Metal through explicitly typed ctypes functions, converts the returned pointer to availability, and sends Objective-C `release` for a non-null device. The pointer stays local to that provider thread; only an immutable inventory value leaves it. No ML import, inference, process or acquisition is required by this path. The selected wiring test mocks both native libraries: it verifies the call/release wiring without executing the actual host API. Actual Objective-C/Metal ABI, allocation/lifetime behavior and host responsiveness remain installed qualification limits.

`Availability.get` maintains one pending provider flight under a lock. Callers share its deadline/event; expiry does not clear a still-running flight, so a hung provider cannot accumulate replacements. Provider exceptions fail closed. Completion after the monotonic deadline cannot populate the cache or result; a completed timed-out flight must be consumed as failure before another flight is possible. The one-second bound is the provider wait; transport has its existing separate bounded RPC budget. A timed-out native call is not forcibly killed, and a permanently stuck flight deliberately leaves new work unavailable. This does not establish native crash containment.

The observation is bound to canonical immutable enrollment inventory and manifest. Exact host/device/index/support shape is compared separately from available/probed-at fields; wire parsing rejects boolean indices, integer booleans and malformed timestamps. Positive availability must be aged 0..300 seconds. Protocol request/response validation preserves configuration, boot, lane, binding, operation, runtime identity and fresh request nonce; the runtime checks its exact native context and inventory, and both frame directions retain TLS 1.3 peer checks. A captured reply cannot authorize another request. Supervisor dispatch independently checks current boot/lane/inventory digest/manifest before serving availability. No public timestamp or refresh API was added.

Static enrollment accepts expired identity observations while retaining closed execution, topology, profile/model, native and immutable hardware relations. Current checks occur at preset/clone resolution, selection, Admission, control prepare/start and supervisor registration/start. Start checks again after input verification, and checks frozen phase/cancellation before launch intent. Refresh does not alter Authority, deadline, frozen plan/context, measurement digest, voice/origin or old receipts. Historical prepare replay or terminal start idempotence does not allocate or launch new work. The static child and terminalization paths are bounded by supervisor launch authority and owned reap/receipt validation respectively; neither supplies a new-work bypass.

Availability has separate two-slot RPC semaphores on both sides. Refresh does not spend lifecycle/transfer slots; the provider runs outside the availability lock, and the watchdog's signalling path remains separate. The tests cover refresh saturation, stop/collect/ACK while refresh waits, cancel during prepared refresh, refusal at expiry between prepare/start, successful refresh at start, and terminal proof after post-launch probe failure. This is fixture concurrency evidence, not a guarantee of zero scheduling delay or listener qualification.

Control reopen starts with no persisted availability lease and attempts a fresh authenticated association. The caught availability refusal in startup permits only the existing authenticated clean-history activation/read path; future resolution/admission/prepare/start still check independently. Exact hello/enrollment/trust/history checks remain. Discovery uses cached metadata only, conservatively expires it, and never invokes the provider. Explicit stopped renewal still checks current availability at commit, exact boot/trust/configuration and retained history/capacity; the eight-epoch bound is unchanged.

The public preset journey first registers after 2701 injected seconds, then generates after two further 1501-second advances/control reopen under the same boot. The saved-clone journey crosses 3002 seconds with existing upload/prepare/save/generate and control reopens. Provider counts demonstrate refresh rather than timestamp restamping. Retained SQL rows, Authority/reservations, templates/enrollment, versions and media bytes remain exact; new jobs alone consume their existing charges. The failed-provider reopen preserves old replay/job/artifact reads while refusing new work. Measurement tests keep full execution identity exact and refuse a changed measurement hash; no qualification is created.

Archive `members`, `archive`, `verify_archive`, runtime-manifest validation and package serving consume the updated allowlist. It includes `service/job_control.py`, `service/production_selection.py`, `worker/engines/qwen_prepare.py` and `worker/native_availability.py`; verification requires exact member names and local bytes, not just file presence. The deterministic archive test passed. The changed archive/runtime identity is an upgrade; unchanged-manifest renewal checks do not make an old installed binary compatible. Documentation correctly requires a fresh corrected package and claims no upgrade migration.

## Independent receipt and historical receipt reconciliation

Inspected `/private/tmp/r352-final-selection.py` before execution: its absolute sys.path entry selects this candidate; its finite NAMES list matches the r352 report, includes the corrected topology test through the whole selection module, and counts 161 tests. Inspected both conftests: direct unittest does not activate their pytest maintenance fixtures or listener fixture. Inspected fake TLS/process/backend seams; no actual child or provider was required.

Executed the exact command **once**, from the candidate worktree:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r352-final-selection.py
```

Receipt: Python **3.14.7**, selection **161**, `Ran 161 tests in 160.572s`, `OK`, process exit **0**. Raw output is retained in this review's tool transcript (execution session 68696); no additional log file was written. No focused extra probes or reruns were needed. `git diff --check base HEAD` passed and candidate remained clean at the pinned HEAD.

Verified script SHA256 `9d50b4a0f3e704e1f7dd53f88c9ed0f664efad1d690d2de82f1a5288d17dd822`. Independently read and hashed the historical implementation logs: final log `8ca3b3edcb5572e19bcf74ce28c9c632a75e26fb7fc22698493969992a7cbc2f`, focused correction log `c64b8766cbccb60422fdfed97174c8251e9f71ed796a17c2ba789a2f9b10f11d`.

Those historical receipts remain **160 PASS / 1 FAIL in 159.662s**, followed by **one focused PASS in 0.598s**. The original failure lost `runtime_topology_incompatible` in discovery. Comparing original tree `abb08998c3aedee573e5779c7ad775bdf48e8d56` to corrected tree shows exactly the immediate `return reasons` correction. They are not a prior combined green run. This r353 run supplies the separate coherent final-tree 161-pass receipt.

## Commission exit and limits

The unchanged-trust freshness P1 is closed at the bounded SOURCE level: automatic implemented host-provider association, more than eight freshness windows under one boot, preserved immutable history/measurement identity, current future-work gates, inert negative/concurrency coverage and unaffected retained renewal/history fences. Parent may adopt this exact source candidate; this report performs no adoption.

No actual host provider/hardware probe, network/listener, Docker, installation/download, ML, actual OS child, host crash, HELD/STT or operational cleanup was performed. Fixture temporary files/SQLite/ASGI/fake TLS and handles are source evidence only. The supplied below-6-GiB constraint and 6-GiB heavy gate remain unchanged, without a live resource measurement or waiver. Installed topology/denied-peer, arm64 ABI/wheelhouse, real Metal/audio/performance, stop/restart/renewal/upgrade and CPU/CUDA/Metal release qualification remain parent-owned gates. Existing unsupported history and finite retention limits are not repaired or waived by availability refresh. Release remains NO-GO; proceed next to the separately commissioned CPU Docker usability lane.
