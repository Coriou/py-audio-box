# H4 Pocket helper — independent complete r252 Sol review

2026-10-24. **Bounded SOURCE NO-GO for the complete candidate: one P2 shipping omission.** The new helper is absent from the explicit headless source archive even though the archived supervisor invokes it. The reviewed checkout implementation satisfies the selected empty-at-reap repair semantics in the independently executed inert suites; no additional blocking source defect was found. This is not actual cross-UID/kernel, PID1/authentication, installation, audio, activation or release acceptance. Parent owns correction and adoption.

## Pins and authority

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-3133b78f`, branch `t3/pocket-helper-r251`, clean HEAD **`bc91810dd466e86c26138c9f5082bcb9e54d2233`**. Base **`54b5fb5a5a76ae9f82f041fb687805c39a782956`**; guard cherry-pick **`82339f9164605cf593cefff85d7df8fa39492755`**; guard report **`18735de3ed870de0d2aa7e81dccc74804e7b354a`**; repair source **`1979f60bef60b5463a0ca9c5b5affb0d111406f7`**. Repair source is exactly six files, 569 insertions/53 deletions. Assessed the final repair against guard semantics, not duplicate cherry-pick history. Shared root was observed at `6f901762ea4e798d17c1055bf301645d2d11ed12` with guard integration already present.

Read r248 FULL as normative, r249/r250 FULL and r251 FULL, and r244–r247 findings/responses. R248's selected empty-directory removal supersedes r247's earlier proposed retained-namespace attestation. Reviewed all six changed source/test files, actual supervisor operation/serve/watchdog paths, child stage setup, journal persistence/validation, Authority/closed decoder, admission and reconciliation configuration semantics, Dispatcher observation/collect/terminal/release/ACK, installed filesystem policy, Docker context and explicit archive builder/manifest. No applicable AGENTS.md found in candidate tree or checked ancestors, including report-root ancestors. Bounded memory registry search returned no relevant hit and supplied no premise.

Only lasting write: this shared-root report. No source edit, candidate change, commit, delegation or adoption.

## R252-1 — P2: the shipped archive omits the executable repair helper

**Invocation:** `worker/runtime_supervisor.py:301–305` launches `python -m worker.runtime_conditioning` for every started Pocket finalization. **Missing declaration:** `headless-dist-manifest.json:93–96` lists the other runtime modules but omits `worker/runtime_conditioning.py`. **Concrete producer:** `service/package_install.py:207–227`, `archive()`, copies only that explicit manifest allowlist; Docker's `COPY worker` cannot restore a file absent from an extracted source distribution.

Independent inert reproduction called the real `archive(candidate, temporary_zip)` and inspected its actual ZIP entries and archived supervisor bytes:

```text
archive_bytes 391321
supervisor_present True
helper_present False
supervisor_invokes_helper True
prior_pocket_modules {'worker/engines/pocket.py': False,
                      'worker/engines/pocket_execution.py': False}
```

Equivalent reproducible probe, with `root` the exact candidate Path and ordinary disposable `TemporaryDirectory`:

```python
from service.package_install import archive
import zipfile
archive(root, temporary_zip)
with zipfile.ZipFile(temporary_zip) as z:
    assert 'speech-box/worker/runtime_supervisor.py' in z.namelist()
    assert b'worker.runtime_conditioning' in z.read(
        'speech-box/worker/runtime_supervisor.py')
    assert 'speech-box/worker/runtime_conditioning.py' not in z.namelist()
```

Thus the archive cannot supply this runtime module. If Pocket execution is supplied/enabled from that distribution, housekeeping cannot import its entry point: nonzero helper exit quarantines the lane before terminal persistence, so clean completion/restart is unavailable. This is a missing artifact, not evidence that uncertain cleanup falsely releases capacity.

The boundary matters: current installed policy and `main()` remain Kokoro-only, and the archive already omits the earlier Pocket engine modules. This does **not** demonstrate regression of today's supported installed Kokoro journey or claim Pocket is currently activatable through that installer. A Docker build directly from the full checkout includes the new helper via `COPY worker` and the `!worker/*.py` ignore exception. Nevertheless the complete repair adds a mandatory executable dependency without carrying it through the existing source distribution; “installation later” does not make the missing file present. Existing archive tests assert listed files/import selected modules and do not check this new dependency.

Required bounded correction: include the helper in the explicit distribution manifest and add an inert extracted/archive dependency assertion for it. Account explicitly for the pre-existing Pocket package omissions at the later Pocket installation gate; no installer activation/redesign is requested here. Re-review the concrete correction before removing the complete-candidate NO-GO. No candidate changes were made by this reviewer.

## Lifecycle and filesystem assessment

`tick` retains the existing engine `waitpid(-1, WNOHANG)` ECHILD condition, and requires receiver completion for a started allocation. Leader exit alone or a pending receiver does not launch housekeeping. The final proposal is drained before freezing the receipt's binding, UID, original deadline, timely flag and engine reap count. `finalizing` is installed under `tree_lock` before helper creation; both stop and watchdog signalling skip it. The lane stays occupied and no blocking helper wait holds the lock. A watchdog signal between ECHILD and acquiring that lock precedes helper creation, so it cannot target the housekeeping process. Normal command mutation remains serialized by `serve`; execution-client threads only read allocation/proposal state.

The helper uses its exact Popen handle, one-second wait, explicit kill and another bounded wait on timeout/wait error. Confirmed helper refusal/timeout leaves the journal nonterminal and marks the lane failed; unconfirmed helper death keeps the finalizing active lane occupied. Subsequent tick cannot count housekeeping as engine work or terminalize it. Popen failure also enters the failed/nonterminal path. No successful receipt is returned by collect and ACK requires terminal state. The original Dispatcher requires STOPPED observation and a bound receipt before `_terminal`; resource release and ACK follow durable terminal handling. Failed cleanup does not manufacture this authority. A stopped prepared record has its explicit no-start branch; a started record with missing stage fails helper validation.

`runtime_conditioning.py` opens fixed `/staging` with Linux O_PATH/directory/no-follow/close-on-exec flags and validates root UID/GID0 mode1733, matching installed policy. It opens only the exact allocated single-component stage, readably for fsync, requiring owner10001, child group, mode0770, same device and stable name/FD identity. Literal conditioning must be a real child-UID/GID mode0700 directory on that device. Removal is one descriptor-relative `rmdir`; no private-directory open, enumeration, recursive deletion, file unlink, chmod or chown exists. Nonempty rmdir refuses atomically. Successful removal fsyncs the stage and rechecks its binding; verified absence requires a valid bound stage. Exact helper UID10001, child primary GID and empty supplementary groups are specified and checked. No DAC_OVERRIDE or extra capability is introduced.

This rests on the normative trusted control owner and dead exclusive child writer-exclusion premise. The identity checks are not a hostile privileged/concurrent writer guarantee. Tests inject installed ownership over real same-UID filesystem operations; they do not prove real cross-UID permissions. Journal-save failure after actual empty rmdir keeps the tested running supervisor nonterminal/failed and does not retry terminalization. Existing journal crash/durability semantics and startup quarantine remain, without a new receipt, phase, schema, recovery waiver, boot cleanup or historical latch reset.

## Acceptance and narrow recovery correction

Success and cancellation reopen each scenario's same real Dispatcher database and staging root, preserve original job/artifact/saved version and principal ownership, assert healthy reconciliation and absence of conditioning, then admit a different job to executing state. Cancellation creates the directory before intent, delays confirmed reap and verifies both directory retention and executing reservation beforehand. Runtime/backend/death observations are injected, while actual helper filesystem logic and actual Dispatcher/Core/readers run.

The completed fixture database and explicit asset/output files are copied into small inert enrollment support. Real Reconciliation rows are checked against persisted inventory and latest history; actual complete `snapshot`, `_installed_detached_source`, `installed_clean_history` and diagnostic eligibility readers run. This is test enrollment, not Pocket installation evidence or an AST fragment. Existing retained empty/nonempty UNKNOWN history remains blocked. Wrapper/backend WAV and translated YAML sentinel cases refuse removal and retain opaque bytes/UNKNOWN. Mid-flight restart and nonempty interruption remain UNKNOWN, not clean-restart acceptance.

The parent-authorized `recovery.py:1865–1873` change matches Admission and live reconciliation: the existing closed execution decoder selects full typed execution, otherwise legacy effective identity. There is one expected hash, no either-hash fallback. Pocket success/cancel and Kokoro typed clean history pass; typed effective-identity old hash, arbitrary hash and semantically valid descriptor/Authority mutation with stale configuration reject. Legacy effective-identity history remains accepted. Authority already invokes the decoder; this introduces no alternate parser or broader recovery redesign.

## Independent verification

Inspected root/service conftests first. Used only explicit unittest imports with **Python 3.14.7**, `PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd and these absolute import paths:

```python
sys.path[:0] = [
    '/Users/ben/.txcode/worktrees/py-audio-box/t3-3133b78f',
    '/Users/ben/.txcode/worktrees/py-audio-box/t3-3133b78f/tests/service',
]
r = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(n) for n in names))
sys.exit(not r.wasSuccessful())
```

Named suite, **82 PASS, 17.463 s, exit0**:

```python
names = ('tests.worker.test_runtime_conditioning',
 'tests.service.test_pocket_lifecycle', 'tests.service.test_engine_admission',
 'tests.worker.test_engine_lifecycle', 'tests.worker.test_engine_pocket',
 'tests.worker.test_engine_kokoro', 'tests.service.test_production',
 'tests.service.test_production_admission',
 'tests.service.test_production_reconciliation',
 'tests.service.test_production_api', 'tests.clients.test_production_clients')
```

Installed regressions, **87 PASS, 61.852 s, exit0**:

```python
names = ('tests.service.test_installed_restart',
 'tests.service.test_installed_registration',
 'tests.service.test_installed_host_policy')
```

One bounded focused repeat investigating the disclosed prior five-second Dispatcher fixture wait: `ExecutionCleanHistoryTests` plus `PocketLifecycleTests.test_dispatcher_generate_artifact_and_cancel_reap`, **3 PASS, 4.971 s, exit0**. The full 82 suite also ran those cases successfully. The implementation's earlier isolated timeout did not reproduce; its cause remains unexplained. No deadline extension, production fallback or endless rerun was used, and these passes do not establish absence of timing flakiness. The wait helper polls asynchronous fixture state for five seconds; the production helper timeout is separately tested through process seams.

The expected injected `lifecycle supervisor unavailable; lane quarantined` diagnostic appeared with overall OK. Archive probe above PASS as a reproduction of the omission; temporary archive removed by ordinary fixture teardown. `git diff --check 82339f9 HEAD` PASS; all six Python files parse with Python3.11 grammar (not Python3.11 runtime execution). Candidate remained clean at the pinned HEAD.

Observed `/private/tmp` available **3,873,812 KiB**, below the 6-GiB actual gate. Only small inert fixtures and normal owned fixture teardown ran. No pytest collection/autouse maintenance guard, listener, actual ML, Docker, network/download, process crash, operational cleanup, HELD operation or historical-root modification. Lazy-import tests may use inert interpreter probes. No PR or merge.

Parent must resolve R252-1 and adopt the corrected source separately. Actual installed mounts/caps/UID switching, kernel permission enforcement, authenticated transport, PID1 engine/helper reap, resource budgets and audio/listening qualification remain unproven. Nonempty snapshot interruption and mid-flight restart remain conservative UNKNOWN by design.
