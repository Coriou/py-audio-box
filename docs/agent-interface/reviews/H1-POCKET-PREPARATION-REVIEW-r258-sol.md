# H1 Pocket preparation independent review — r258 Sol

2026-10-24. **Bounded SOURCE GO for parent adoption of r257. No blocking safety or semantic finding.** Actual Pocket CPU model execution in French/English, listening/clone quality, installed activation, cross-UID resources and hardware-family qualification remain separate parent-owned gates. Nonempty interrupted conditioning snapshots and mid-flight restart remain UNKNOWN; this review does not authorize their repair or cleanup.

## Pins and scope

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-fb2cfaa6`, branch `t3/pocket-preparation-audition`, base `fa2a524fe609cc7d6f42e31b3ec01f609743227e`, source `89366614abfb566155e8aecb304fcd08918d4ff0`, documentation HEAD `b91835c47f92f13acf1ff6cb8d348945757d6696`. HEAD, branch, base ancestry and clean candidate were independently checked; source-to-HEAD adds only r257's report. Actual base-to-source delta is the stated 14 files, including 13 Python files and the explicit distribution manifest.

Read full r257, r188 including assignment 3, r238/r239 and r248; inspected the actual delta, contracts/oracle, saved-reference production, shared detached reader, supervisor verification/finalization and archive closure. No applicable repository or ancestor AGENTS.md found. The bounded memory search supplied no implementation premise. Both conftests were inspected: unittest/direct calls avoid pytest collection, the maintenance guard and listener fixtures.

Sole lasting write is this root report. No candidate/source edit, commit, delegation, actual ML, Docker, download/install, network/listener, process-crash exercise, operational cleanup or HELD operation. Tests used small owned inert fixtures, injected backend/transport/death and identity policy, in-process ASGI, ordinary temporary teardown and inert archive/import/help subprocesses.

## Source assessment

- `worker/engines/pocket_prepare.py:15-65` closes preparation as a distinct descriptor with literal phase/transform and exact frame-aligned ticks. Strict classic PCM16 mono validation precedes extraction; selected original PCM bytes are copied unchanged into a canonical WAV at the original rate. No cleaning, denoise, resampling or transcript interpretation is represented. Original digest/size/frame/rate bindings survive independently of audition rate. `worker/runtime_protocol.py:120-160` binds the complete plan, owned mappings, explicit audition text, empty reference transcript, profile limits and the existing 1-MiB publication ceiling.
- `service/production.py:16-44` resolves the owned managed original and checks its actual selected bytes before submission/reservation. `service/execution_context.py:89-116` supports trusted preparation/generation templates without public descriptor/path authority. Nonempty Pocket reference text refuses before reservation and maps through `service/domain.py:243-247` to `unsupported_control` with a reference_text explanation. The public test confirms no reservation is left.
- `worker/engines/pocket.py:109-155` shares the existing bounded synthesis body; preparation forwards the exact selected WAV and exact frozen audition text to the reviewed backend seam. AudioResult rate/channels/frames/finite samples/size remain measured. `worker/runtime_child.py:42-75` obtains both results before staging either fixed file, then proposes the ordered pair. Audition failure stages neither output; a later staging failure may retain evidence but cannot publish half metadata.
- `service/core.py:491-549` independently measures both files and reconstructs reference bytes from the managed original. Coherently rehashed fabricated reference bytes still fail `prepared_policy`. Both hashes pass through the existing role/publication oracle and atomic pair intent/barrier. The oracle retains exact original/segment/transcript/provenance and distinct reference/audition semantics. Save selects a committed candidate's prepared reference; generation reloads that immutable owned version. Existing fixed synthetic and single-output Kokoro branches retain their verification policies.
- **Authorized shared reference dependency:** `service/reconciliation.py:439-509` tracks schema paths rather than subtracting IDs globally. Only schema- and frozen-plan-oracle-validated prepared voice_prepare roles suppress the two future role declarations and resolved reuse reference declaration. Original inputs, saved references, committed outputs, manifests/artifacts and publication intents remain independently traversed. Missing original and missing committed audition regressions block. `service/recovery.py:2953-2983,3021-3029` performs its closed document checks and calls this same walker; no incompatible detached decoder or reader waiver was introduced. Legacy absent-role paths remain unchanged.
- **Authorized shared resource dependency:** `service/resource_store.py:139-210` uses FD-relative no-follow staging traversal with stable directory/name identity checks. ENOENT tolerance requires a previously observed directory, literal conditioning immediately under the current executing Pocket stage, exactly one executing reservation, validated stored context, matching current stage, stable control owner/mode/device and continued absence. Unknown disappearance, files/symlinks, substitution/reappearance, root loss and permission failures are not blanket-suppressed. Existing sizes already counted remain conservative; `check_disk` retains max highwater plus full charges. This adds no reap/release/ACK authority. The trusted exclusive supervisor/helper premise remains essential; this is not hostile same-UID writer proof or actual cross-UID permission evidence.
- `worker/runtime_supervisor.py:93-103` independently verifies the new exact type through shared verify_inputs before journal mutation. It remains dispatch mode pocket; unchanged ECHILD/receiver/helper-before-terminal ordering at lines 265-324 still applies. Nonempty helper failure retains quarantine. No new lifecycle mode, receipt interpretation, persistence schema or cleanup authority appears.
- The explicit manifest includes the new module. Archive tests build/extract the actual package, compare exact membership/bytes and verify isolated lazy local import closure with heavy imports forbidden. Installed policy remains `owned-kokoro-v1` / `docker-kokoro-split-v1`; trusted PocketContexts and direct mappings are integration seams, not automatic installed Pocket readiness. The inert public journey explicitly installs its trusted templates.

No reproducible blocking finding was found, so there is no severity/file-line defect entry or supplemental failure reproduction to report.

## Independent execution

All commands ran from the pinned candidate using `PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B` (verified **Python 3.14.7**). No retained r257 PASS result was counted as independent execution. Commands below are the exact runner bodies; C denotes the absolute candidate path above, substituted literally in execution.

```python
# Focused: python -I -B -c BODY
import sys, unittest
sys.path[:0] = [C, C + '/tests/service']
modules = '''tests.service.test_pocket_preparation tests.worker.test_runtime_conditioning
 tests.service.test_pocket_lifecycle tests.service.test_engine_admission
 tests.worker.test_engine_lifecycle tests.worker.test_engine_pocket
 tests.worker.test_engine_kokoro tests.service.test_production
 tests.service.test_production_admission tests.service.test_production_reconciliation
 tests.service.test_production_api tests.service.test_reference_publication
 tests.worker.test_production_synthetic tests.contracts.test_reference_roles
 tests.clients.test_production_clients tests.release.test_release_package'''.split()
result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
sys.exit(not result.wasSuccessful())
```

**142 tests PASS, 33.555 s, exit 0.** Expected injected `lifecycle supervisor unavailable; lane quarantined` diagnostic occurred with overall PASS. Includes exact PCM/ticks, frozen text, rates/provenance, pre-reservation refusals, fabricated bytes, pair failure atomicity, saved-reference reuse, real in-process HTTP/CLI/MCP adapters, same-root Dispatcher reopen after completion and empty-at-reap cancellation, opaque interrupted namespace UNKNOWN, resource interleavings and archive closure.

History runner used the identical body/path setup with modules `tests.service.test_installed_restart tests.service.test_installed_registration tests.service.test_installed_host_policy`: **87 tests PASS, 63.513 s, exit 0**. This run was justified by the shared reader/resource changes; its Docker/host/death observations are injected, not installed qualification.

Direct command: `PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r257-resource-regressions.py`: **17 direct cases PASS, exit 0**. Read the complete wrapper and selected function bodies before invocation. Wrapper SHA-256 `6e89046790ce619368d438c4aa8f7f60b5b0d88269eba19cf7164e577762c7f9`; it inserts candidate, candidate/tests and candidate/tests/service explicitly. Allowlist: queue/quota/free/growth; quota/free fault latching; private nonrollbackable snapshot; atomic reserve rollback; unresolved monitor acquire/reserve/release and queued cancel; acquisition rollback; checking/healthy/fault fence-write transitions; revision/missing/old/invalid/empty-old inherited protocol damage; and observation outside SQLite transactions. No crash/death-test function selected. Importing pytest assertion/MonkeyPatch support does not collect or activate its fixtures.

Total **246 passing methods/direct cases (142 + 87 + 17)**, with additional mutation/interleaving loops inside methods. `git diff --check fa2a524 b91835c` passed. Independently parsed all 13 changed Python files with `ast.parse(..., feature_version=(3,11))`: PASS; this is grammar compatibility only, not Python 3.11 runtime evidence. Candidate remained clean at the exact documentation HEAD after checks. No independent log files or wrapper copies were written.

## Handoff

Parent may adopt this bounded source candidate. Actual backend/model pins and dependencies, French/English speech and similarity listening, installed mapping/readiness and authenticated runtime behavior, real reap/helper permissions/resource measurement and the required CPU/CUDA/native Metal family evidence remain separately authorized work. Successful inert audio forwarding is not acoustic qualification. Completed empty-namespace restart is covered; arbitrary nonempty interruption and mid-flight restart remain UNKNOWN and blocked under existing rules.
