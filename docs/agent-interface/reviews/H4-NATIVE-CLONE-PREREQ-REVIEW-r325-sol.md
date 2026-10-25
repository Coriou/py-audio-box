# Native clone prerequisite independent SOURCE review — r325 Sol

2026-10-25. **Bounded GO for r324's clean completed exact-enrolled clone history prerequisite and Pocket discovery correction. No blocking finding in this scope. Lane 1 remains INCOMPLETE; release and actual hardware qualification remain NO-GO.** This is an independent review, not adoption, and does not require the whole managed onboarding journey as a prerequisite blocker.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-cfd9ac5f`, branch `t3/managed-native-clone`. Base/merge-base `e4f710bb55e34e1623d13637671c8302713f135d`; SOURCE/tests `7b120c28b42003b55c698d3e00f5144448f00783`; report HEAD `573306ad7efda2ad6e04e89c9c433ee20a45aec8`. HEAD resolves the supplied `573306a`. Candidate status was empty before and after observations. HEAD differs from SOURCE only by the r324 report. `git diff --check` passed. Root HEAD was the base; its pre-existing untracked STT evidence was observed only through status and left untouched.

Sole lasting write: this root report. No candidate edits, commit, merge, push, delegation, operational cleanup, network/listener, Docker, install/download, actual child/ML, crash, HELD or STT work. Fixture-owned temporary teardown only. Disk observation: 2,834,232 KiB available, below 6 GiB; only expressly authorized tiny inert observations ran, no heavy ACTUAL gate was attempted or waived.

Read full r324, r323 advice, NATIVE-PACKAGE and PRODUCT; inspected the actual seven-file source/test diff and its ownership, Core startup, native protocol/execution, supervisor authorization/input/lifetime, runtime transport, discovery, production/context and public/client consumers. No applicable AGENTS.md was found in candidate/root file discovery or checked candidate ancestors. Memory supplied only bounded evidence conventions; current source and receipts support the technical conclusions. The P1B23 one-shot skill was inspected for scope; this is authorized direct unittest SOURCE work, not a P1B23 operation.

Verified SHA256:

- `service/native_activation.py`: `73a59c0e63b86c8e47c4d7713cc55a493d65ec20240e1162c4c95a794ec8e806`
- `service/api.py`: `4186ad45a072adce88c361c249b3ba3726bda06a2a3ec79218e3f4bcd00f99b1`
- r324 report: `821883a7a0150b5922a55a19109e63b30fe56498bb9b761bbc433fe3b1bb8ec6`

## Review findings

No blocking source defect found in this bounded change. The production delta is two localized changes; fixture extensions explicitly identify seeded prerequisite evidence.

1. `completed_history` no longer rejects every clone. It still requires the existing exact schema, frozen Authority/context/native identity, succeeded single reaped attempt, command order, retained admission demand, receipt, publication and durable import proofs. `native_execution` is the existing Qwen relation validator: it joins clone version, reference ID/hash, exact reference transcript/configuration, model/revision mapping, execution descriptor, frozen asset shape and profile. Neither a historical allocation nor derived preset becomes new enrollment authority.
2. The added `VoiceStore(core)` constructor only holds Core; it performs no catalogue DDL. `_asset` uses `Core.asset` to verify managed membership/schema, no-follow regular single-link owned media and digest, then checks principal ownership (including the existing bounded published-output fallback). Core invokes retained history under lifetime exclusion before WAL/schema initialization and before attaching `core.voices`. Thus the corrected pre-schema access is valid and read-only.
3. Current asset metadata must equal the frozen asset canonically, including JSON primitive representation. The subsequent bounded read checks exact length/hash and mono PCM16 rate/frame/length through existing `pcm`. Detached control `input` must equal the verified owned bytes. The fixed `input` member is added only for clone histories; preset histories still refuse it. This does not introduce caller-selected paths or accept additional spool members.
4. Missing, altered, foreign or non-single-link references refuse. Existing and supplemental refusal tests compare retained file bytes, SQL dump and ACK count across failed activation and assert import closure remains closed with no completed imports when a runtime was acquired. The new path performs no SQL/file repair or ACK, and the final import-closure discharge remains after all proofs. Core's existing lifetime-lock/directory handling is not a claim of zero filesystem syscalls or zero pre-existing chmod behavior.
5. Native independent authorization is unchanged: `register_snapshot` matches original `_preset_templates`; dynamic derivation remains restricted by `owned_preset_context` to `custom_voice`. Exact enrolled clones can allocate, but arbitrary managed clone derivation cannot borrow preset authority. Native input transfer retains exact size/hash/PCM and replay/offset checks; child/reference mappings bind voice revision, checkpoint/revision and reference text. No reclamation policy was introduced. Cancellation evidence keeps owned, control and native reference bytes before confirmed reap and afterward; cancelled/failed/unknown histories still do not gain startup acceptance.
6. Operation detail now honors the configured prepare/results/save capability instead of blanket installed refusal. Installed Pocket regression checks all three against discovery, including `implemented_read` for results, absent preset registration and unqualified speech state. Supplementary detail checks cover supported Pocket, unsupported Kokoro and native-Qwen-style capabilities, and preserve `voices_list` read availability. Those supplementary capability stubs test the predicate, not engine installation or readiness. Actual context source confirms Pocket advertises preparation while native Qwen does not.

Original preset derivation/history, Core single-owner ordering, publication/import fences and Linux policy are unchanged by the diff and remain represented in the selected regressions. Historical NATIVE-PACKAGE text still says non-clone history; r324 explicitly describes the narrow superseding candidate. Consolidating that already-known historical operator-document debt remains with the installed lifecycle deliverable, not a new blocker here.

## Independent coherent final receipt

Read `tests/conftest.py` and `tests/service/conftest.py` before execution. Direct unittest avoids pytest's autouse maintenance guard and listener fixture. All test observations used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute candidate and `tests/service` paths inserted into `sys.path`, candidate cwd, version assertion `(3,14,7)`, and `TextTestRunner(verbosity=2, failfast=True)`. Actual runtime:

`3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`

**One coherent final selection: 84 PASS / 70.952s / exit 0 / 0 failures / 0 errors / 0 skips.** Tool session `26944`, initial chunk `aadda4`, terminal chunk `c0f212`. Counts were asserted per block and at total before running, then actual `testsRun == 84` was required for exit 0. This independently closes the fragmented r324 selection receipt; it does not rewrite its earlier failures.

Selection, in order:

- `tests.release.test_native_history.NativeHistoryTests`: all 44.
- `NativeCloneHistoryTests`: exactly five named methods below; none of its inherited 44 history methods selected.
- `tests.service.test_native_lifecycle.NativeCloneLifetimeTests`: one cancellation/reap test.
- `tests.release.test_native_onboarding.NativeOnboardingTests`: all 6.
- `tests.service.test_installed_pocket.DiscoveryTests`: 1.
- `tests.service.test_pocket_preparation`: 13 module-loaded tests.
- `tests.service.test_installed_preset.InstalledPresetTests`: 3.
- `tests.clients.test_production_clients.ProductionClientTests`: 3.
- `tests.service.test_production_api.ProductionAPITests`: 5.
- `tests.worker.test_native_supervisor.NativeSupervisorTests`: exactly three named methods below.

Reproduce from the pinned candidate cwd (no pytest):

```sh
/private/tmp/pab-0c-venv/bin/python -I -B - <<'PY'
import sys, unittest, importlib
assert sys.version_info[:3] == (3,14,7), sys.version
root = '/Users/ben/.txcode/worktrees/py-audio-box/t3-cfd9ac5f'
sys.path[:0] = [root, root+'/tests/service']
print(sys.version, flush=True)
loader = unittest.TestLoader()
suite = unittest.TestSuite()
def add(module, cls=None, methods=None, expected=None):
    m = importlib.import_module(module)
    tests = (loader.loadTestsFromModule(m) if cls is None else
             unittest.TestSuite(getattr(m,cls)(n) for n in methods) if methods else
             loader.loadTestsFromTestCase(getattr(m,cls)))
    assert tests.countTestCases() == expected
    suite.addTests(tests)
add('tests.release.test_native_history','NativeHistoryTests',expected=44)
add('tests.release.test_native_history','NativeCloneHistoryTests',[
    'test_clone_keeps_exact_reference_and_transcript_across_replay',
    'test_changed_owned_reference_refuses','test_changed_detached_reference_refuses',
    'test_missing_owned_reference_refuses','test_foreign_owned_reference_refuses'],5)
add('tests.service.test_native_lifecycle','NativeCloneLifetimeTests',expected=1)
add('tests.release.test_native_onboarding','NativeOnboardingTests',expected=6)
add('tests.service.test_installed_pocket','DiscoveryTests',expected=1)
add('tests.service.test_pocket_preparation',expected=13)
add('tests.service.test_installed_preset','InstalledPresetTests',expected=3)
add('tests.clients.test_production_clients','ProductionClientTests',expected=3)
add('tests.service.test_production_api','ProductionAPITests',expected=5)
add('tests.worker.test_native_supervisor','NativeSupervisorTests',[
    'test_input_complete_replay_conflict_gap_and_hash',
    'test_clone_conditioning_nonempty_blocks_terminal',
    'test_input_wrong_complete_hash_quarantines'],3)
assert suite.countTestCases() == 84
result = unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
print('RECEIPT',result.testsRun,len(result.failures),len(result.errors),len(result.skipped))
sys.exit(0 if result.wasSuccessful() and result.testsRun == 84 else 1)
PY
```

## Supplementary adversarial observations

Separate from the coherent 84, session `4317`, initial `81a4c5`, terminal `cb2275`: four evidence-preserving refusal probes passed. Each explicitly selected a new subclass method using the existing `NativeHistoryTests.refusal` helper; inherited methods were not loaded. Mutations after clean completion were:

- Clone current asset metadata `frames` changed to `float(frames)` via the assets metadata row, preserving the rest. This tests canonical frozen/current comparison beyond a simple file hash mismatch.
- Clone detached `spool/*/input` removed.
- Clone owned reference given a second hard link at fixture-root `held-reference-link`, leaving content intact.
- Non-clone history given an otherwise extra fixed `input` file with `b'not-a-clone'`.

The helper snapshots files after mutation, compares SQL dumps and ACK counts before/after refused reopen, and checks import closure where available. All four passed without production changes. These are fixture-owned mutations, not candidate edits.

The fifth probe in that run errored before any endpoint request: my stub omitted `installed_checked`, causing `AttributeError: 'D' object has no attribute 'installed_checked'` at `service/api.py:109`. Run receipt is therefore **5 run / 4 passed / 1 probe error / 5.427s / exit 1**, not a clean five-test pass and not a candidate finding.

Corrected only the in-memory stub by adding `installed_checked=True` and `installed_activation=SimpleNamespace(configuration=config)`. Reran only that detail probe: chunk `4982e9`, **1 PASS / 0.731s / exit 0**. In-process authenticated ASGI GETs used a read credential and checked `voice_prepare`, `voice_results`, `voice_save`, `voices_list` for three capability sets: Pocket supports the first three; Kokoro and Qwen support only speech generation. Expected details were respectively implemented/implemented_read/implemented, unavailable for all three unsupported operations, and implemented_read for voice listing throughout. Installed Pocket and Kokoro predicate checks used a validated inert Pocket configuration with capability engine overridden; Qwen used `installed=None`, matching native's separate configuration path. This matrix is bounded route-predicate evidence, not a new public engine journey.

No broad rerun followed successful checks. Candidate pins/status and diff whitespace were rechecked afterward.

## Remaining parent-owned gates

Proceed to the next managed clone implementation, without marking Lane 1 complete: one supported provisioned native clone capability, writer-owned uploaded reference and exact transcript or explicit supported x-vector mode, typed preparation and separate explicit audition/reference provenance, independently reconstructed reference authorization from original templates, immutable save/reuse without per-voice enrollment editing, canonical HTTP plus thin CLI/MCP, and the unseeded empty-catalogue upload → preparation/registration → audition → save/reuse → verified artifact → clean replay journey. Its ownership/transcript/model mismatch, CAS/idempotency and provenance tests remain necessary. Seeded clone fixtures here do not demonstrate that exit.

No source/inert result establishes installed runtime, kernel/process containment, acoustic similarity/intelligibility, measured performance, restart/fault/renewal or release qualification. Metal remains **UNQUALIFIED**, with no reviewed arm64 wheelhouse. CPU and CUDA actual release rows remain mandatory. Topology, pinned real TLS/mounts/denied peers, compatible ABI/model/hardware and supported terminal-history/boot renewal remain separate.

Legacy `lib/tts.py` SHA256 independently remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; the control still records `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`. Digest **RED unchanged/unwaived**; no reconciliation or metadata regeneration. No self-adoption or whole-Astra-Lane-1 GO is issued.
