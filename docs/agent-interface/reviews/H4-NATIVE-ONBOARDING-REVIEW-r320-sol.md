# Independent native owned-preset onboarding SOURCE review — r320 Sol

2026-10-25. **Bounded NO-GO for adoption of r319 SOURCE pending F1.** Exact independent r319 selection, once: **92 PASS / 56.069s / exit 0**. A separate tiny adversarial probe demonstrates that a previously derived allocation can authorize a canonically different execution descriptor without reconstruction from the original enrolled template. No self-adoption, installed, hardware, acoustic or release qualification. Metal remains **UNAVAILABLE / UNQUALIFIED**.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-09593ec2`, branch `t3/native-qwen-onboarding`. Exact clean entry and final HEAD `716a106148d170cd8debdc3d594b48dbf7bc13fb`; base and merge-base `4c8f8c4d1ec5a663ca69ec4d6809b505a63c3146`; SOURCE `dee425feb460a44987560a59e9fadc2c36c16c2d`. SOURCE-to-HEAD changes only the r319 report. Base-to-SOURCE changes six files: execution_context, API, native_protocol, native_supervisor, onboarding tests and NATIVE-PACKAGE documentation. `git diff --check` passed. No applicable AGENTS.md found in candidate discovery/checked ancestors.

Read PRODUCT, NATIVE-PACKAGE, normative r277, r318, r319 and relevant r271; inspected actual diff and Dispatcher/Production/CoreVoiceStore, execution decoder, supervisor registration/prepare, enrollment/runtime/launch fixtures and archive membership consumers. Read both conftests before direct unittest; pytest/autouse maintenance and listener fixtures were not executed. Applied the memory one-shot evidence-gate skill for authority/evidence boundaries; technical conclusions come from current candidate source and independent execution.

Sole lasting write: this root-checkout report. No candidate edit, commit, merge, push, delegation or operational cleanup. Only authorized tiny ASGI/fake TLS/backend/process-handle/filesystem/SQLite/ZIP observations and fixture-owned teardown. No network/listener, Docker, install/download, actual child, ML, crash, HELD or STT. Observed available disk 2,918,680 KiB, below 6 GiB; no heavy ACTUAL operation or gate waiver.

SHA256:

- service/execution_context.py: `16e8ad3d97e74938b4a834eee3a3d2d16c59f85a71f93ca64ef87d1c8adaf970`
- worker/native_protocol.py: `5413ad0f4fc1ad4ba92aa67eb613ce7655d707161d973cd5a75c6eae748234d8`
- worker/native_supervisor.py: `25a393ec37c0f460e559ebfa9e39cb9f31649d63ccec0299ed916e596e7e544d`
- tests/release/test_native_onboarding.py: `ccf519b2f131684a49b0044bb41376d3d00439e57296c6dfb57ca1c5f9a5f17e`

## F1 — P2: derived registration fast path bypasses original-template canonical authorization

`worker/native_supervisor.py:223` searches **all working registrations**, including previously derived owned allocations, using Python structural equality. Only when that finds no match does the new code at lines 228–235 reconstruct from `_preset_templates` and compare canonical bytes. Python considers `1 == 1.0`; canonical JSON distinguishes them. The execution decoder accepts both numeric representations for sampling parameters and compares their settings by value. Consequently a derived registration effectively supplies template authority for a canonically different descriptor on subsequent registration.

Independent fixture observation (session 21340, chunks 71f742/80456e, exit 0):

```
descriptor top_p 1.0 -> 1 context Python equal True canonical differs True
BEFORE REFUSED native_registration
ORIGINAL ACCEPTED
AFTER ACCEPTED
original template count 1
```

Only `execution.descriptor.top_p` changes; the immutable version, its settings/digest, resolved settings/digest, request, principal, profile and native identity remain unchanged. Thus the changed descriptor is refused against the original enrollment, then accepted after the legitimate derived context is registered. The retained original-template dictionary still has one row: storing a separate snapshot alone does not prevent the fast path from using derived rows as authority. `prepare` invokes `register_snapshot`, so this is an actual private supervisor authorization consumer, not a test-only hook.

Reproduction body, using `/private/tmp/pab-0c-venv/bin/python -I -B -` and explicit candidate plus candidate/tests/service sys.path, asserted Python 3.14.7:

```python
import copy
from tests.release.test_native_onboarding import NativeOnboardingTests
from worker.native_protocol import owned_preset_context
from worker.transport import Refused, canonical
from contracts import validate as oracle
c = NativeOnboardingTests(); c.setUp()
try:
    original = copy.deepcopy(c.f.snapshot)
    original[1] = owned_preset_context(original[1], original[0]['principal_id'],
                                      c.f.f.e.preset, 'probe')
    original[0]['input_sha256'] = oracle.plan_hash(original[1]['plan'])
    changed = copy.deepcopy(original)
    changed[1]['execution']['descriptor']['top_p'] = 1  # original is float 1.0
    assert changed[1] == original[1]
    assert canonical(changed[1]) != canonical(original[1])
    try:
        c.f.s.register_snapshot(changed)
        raise AssertionError('unexpected initial acceptance')
    except Refused:
        pass
    c.f.s.register_snapshot(original)
    c.f.s.register_snapshot(changed)  # incorrectly accepts
finally:
    c.doCleanups()
```

Impact is bounded: no effective sampling-value change, foreign-principal access, public request descriptor injection, model substitution or hardware execution was demonstrated. Canonical public resolution normally constructs the original descriptor itself. Nevertheless this violates the explicitly required independent exact derivation/original-template boundary and makes authorization depend on prior allocations. Correct the fast path so only original enrolled templates confer authorization, with canonical primitive identity on both original exact registrations and owned derivations. Preserve exact original registration support and finite allocation limits. Add refusal coverage before and after legitimate derived registration, including numeric substitutions and repeated registration. Parent owns implementation and fresh independent review.

## Other source assessment

The public route remains the existing closed `voice_preset_register` HTTP/CLI/MCP operation. Authentication supplies principal; no public trusted config/descriptor/path/model or self-enrollment input is added. Dispatcher requires checked installed readiness; Production delegates to existing CoreVoiceStore transaction/CAS/ownership/idempotency/immutable-version logic. No DB seeding, DDL, new API, scheduler, UI, STT, profile promotion or voice fallback appears in the delta.

The pure packaged helper binds version ID to principal, target and the full trusted template, and copies rather than mutates it. This includes explicit model/bundle, profile, settings and native deployment/configuration/lane/boot/runtime identity. Production reloads the owned version; NativeQwenContexts compares the expected version before admission/freeze/submission. Ambiguous selectors and clone templates refuse. Existing exact registered versions retain their route. Shared helper remains in `worker/native_protocol.py`, explicitly present in native-dist-manifest.json; deterministic/archive-extra-member checks passed. Contracts/fixtures, lib/tts.py and Linux Kokoro/Pocket implementations are unchanged against base.

Fixture corrections are legitimate test enrollment synchronization: NativeFixture replaces native identity/working registrations after supervisor construction, so onboarding setUp/prepare explicitly copy those test registrations into the original-template snapshot. The public journey starts with an empty catalogue, registers through CLI and replays through HTTP/MCP, generates through canonical HTTP, verifies published asset SHA256 and closes/reopens the same owner/transport root. Same registration and generation replay retain one fake backend launch. It uses actual native lifespan/Core/runtime/protocol/child-function consumers with fake TLS/backend/owned process observations, not a real child or hardware. No production synchronization hook was introduced. This passing journey does not cover F1's post-registration primitive substitution.

## Independent execution receipt

Exactly one r319 92-case selection: session 77616, initial chunk c8ca4a, terminal e20086. `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute candidate and candidate/tests/service sys.path. Printed/asserted Python **3.14.7**, full runtime `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`. `defaultTestLoader.loadTestsFromNames`, asserted count 92, `TextTestRunner(verbosity=2, failfast=True)`, failure maps to nonzero exit.

Selection: NativeOnboardingTests (3), InstalledPresetTests (3), NativeHistoryTests (44), NativeActivationTests (8), NativeLaunchTests (8), NativeEnrollmentTests (10); full tests.worker.test_native_protocol, tests.clients.test_production_clients and tests.service.test_production_api; NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt; NativePackageTests.test_archive_deterministic_bytes and test_extra_archive_member_refuses. **92 PASS, 56.069s, exit 0.** Expected injected close_unknown shielded-future traceback and argparse refusal usage appeared with passing assertions. No selection rerun.

Supplemental probe history is separate: initial session 15705/chunk b82f62 exited 1 during probe construction (`settings_mismatch`) because it changed version settings without updating their digest. It never reached the intended authorization check and is not passing evidence. The corrected descriptor-only probe above reached both authorization paths and completed exit 0. No candidate changes or operational repairs were made.

## Remaining parent-owned gates

Fix F1 and obtain fresh independent SOURCE review before adoption. Preserve zero-job or fewer-than-64 succeeded single-attempt non-clone retained-history rules, exact existing schema/Authority/import/receipt/publication bounds and same immutable boot/trust. No renewal, history waiver or automatic cleanup.

Managed clone onboarding, Docker/native topology/mount/firewall/reachability and stopped/reconciled renewal remain deferred slices, not additional current blockers. No reviewed arm64 wheelhouse or compatible installed ABI is supplied. Actual installation/import, TLS denied peers, hardware/model/MPS generation/cancel/restart, process/death/resource behavior, clone lifetime, audio/performance and release acceptance remain unrun. CPU and CUDA remain mandatory qualification rows. Unchanged lib/tts.py SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; known legacy digest RED remains unchanged/unwaived, not regenerated or rerun.
