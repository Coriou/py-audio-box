# H4 saved-reference Pocket lifecycle — independent r245 Sol review

2026-10-24. **Bounded SOURCE NO-GO for the complete candidate: one P2 durable inventory producer/reader incompatibility.** The generation/admission/publication slice passes its named inert checks, but the new retained namespace representation is not supported by an existing closed history reader. Parent adoption is separate. No actual clone, install, runtime containment or release qualification follows.

## Pins and review scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-b646dab9`, branch `t3/pocket-cpu-generation-lifecycle`, clean at documentation HEAD `191c92a7920884022e737b23c0b0d6ca1cecd5e4`. Base `d803efb80451f0035d1e4c380f3fccac488e9709`; source `39a26b32b8b8d4100ad9245eb312a3b1e2184c3e`. Source delta is exactly 11 files, 477 insertions/73 deletions. Source-to-docs changes only r244. Shared root entered at the base with pre-existing untracked STT evidence, left untouched.

Read r244 FULL, r188 including assignment 3, r240–r243 and r195/r196; inspected the actual eleven-file delta and relevant unchanged consumers: catalogue ownership/immutable registration, Core input/publication, Authority, Dispatcher, admission/resources, reconciliation, recovery/history, installed policy, supervisor journal, child, descriptors, legacy PocketBackend and adapter tests. No applicable AGENTS.md was found in checked candidate ancestors/tree. Root/service conftests were inspected before execution. A bounded memory registry search returned no relevant hit and supplied no premise. No delegation, source edit, commit or candidate change. Sole lasting write is this report.

## R245-1 — P2: new retained directory is outside the closed durable history contract

**Producer:** `service/reconciliation.py:206-219`, persisted at `390-396`. An exact Pocket attempt's empty control-owned mode0700 `conditioning` directory becomes `classification=committed_referenced`, `reason=retained_conditioning_namespace` and is written into the existing inventory tables/history.

**Existing consumer:** `service/recovery.py:3027-3032` accepts `committed_referenced` only with `retained_metadata`; the new row raises `installed_history_inventory`. Moreover, `3046-3051` requires every object of that classification to be a regular managed asset in namespace `assets`, with asset hash/UID/link checks. Merely adding the new reason to an enum would still leave the directory incompatible. This is a semantic classification mismatch as well as a reason mismatch.

Independent reproduction used `PocketLifecycleTests.setUp()` and `proposal()` with the inert backend, then `Reconciliation(...).reconcile_startup()`. The exact directory row was emitted and persisted unchanged. Executing the unchanged AST statements comprising the reader's `permitted` assignment and immediately following inventory loop against that actual row raised `ValueError('installed_history_inventory')`. This isolates the exact closed consumer without inventing enrollment, touching an installed root or invoking recovery. It is **not** a full installed-history/recovery run. The fixture's overall scan was complete but blocked while its live/unreaped allocation remained; the probe does not claim an otherwise healthy installed state.

This finding is narrower than “all recovery is broken”: `_validate_source` at `recovery.py:1354-1372` accepts bounded textual reasons, and no existing Kokoro/synthetic path was found that emits this new Pocket row. Current installed policy remains Kokoro-only and does not activate Pocket. Therefore this does not demonstrate regression of an already supported installed Kokoro journey. It does demonstrate that the candidate introduces durable state which the existing installed clean-history contract cannot represent. Calling the lane unactivated accurately limits deployment claims but does not close that shared producer/reader incompatibility for a complete lifecycle candidate.

**Required resolution:** agree a narrowly typed retained namespace representation and coherent existing readers, or keep this as explicitly incomplete work without adopting the new reconciliation exception as completed lifecycle integration. Preserve no-follow inspection, UNKNOWN for uncertain/nonempty/foreign data, and the absence of deletion authority. Do not relabel a directory as an asset, broaden acceptance indiscriminately, delete the directory to make history pass, or bypass recovery checks. A correction requiring recovery ownership must return to the parent; this review authorizes no redesign or edit. Add a source regression that carries an actual retained namespace row through the applicable closed reader.

## Other assessed boundaries

- **Principal/catalogue/reference binding:** `execution_context.py:81-112` accepts trusted detached principal templates and owned local mappings, validates exact descriptor and bytes without ML, closes the public request, and fixes saved version selection. `production.py:60-76` reloads the actual principal-owned immutable catalogue version/assets and verifies Core-managed reference bytes/PCM16. `core.py:436-443` independently checks ownership, exact metadata and required IDs on submission/allocation. `voice_store.py` preserves append-only version/digest semantics. Caller path/context/reference overrides refuse. Stable immutable trusted mappings are the premise; no hostile-host or concurrent privileged writer framework is assumed.
- **Private semantic consistency:** `runtime_protocol.py:27-67,81-116` closes context/descriptor/mapping shapes, exact fixed runtime paths, CPU/model/manifest/language/profile identity, one clone version/reference, complete saved reference hash/format/frame/rate fields, null seed, empty reference text, no fallback/reuse/variants and finite budgets. Revision remains descriptor/raw-manifest bound. Existing oracle supplies plan/settings hashes, bundle catalogue compatibility and current profile matching. A descriptor budget mutation that remains semantically valid changed the complete Authority binding in the independent probe.
- **Admission and supervisor:** existing request/voice/assets/profile plan hash remains distinct from full Authority binding. Reservation configuration hashes the complete private execution snapshot. The single transaction/queue remains; admission selects the exact resolved profile and requires its measured nonsynthetic qualification assertion. Assertions in tests are injected, not qualification. Supervisor `runtime_supervisor.py:90-104` checks closed execution/fixture, exact separately registered bundle/reference mappings, then independently verifies all selected bytes with fixed managed UID10001 **before** journal allocation. Missing mapping defaults closed. Child repeats verification. No second scheduler or public mapping authority was added.
- **Owner separation and pre-ML limits:** `pocket.py:24-56,151-200` extraction preserves component no-follow, final owner/type/link/mode, full manifest/assets/reference hashes and corrected strict RIFF validation. Only trusted integration supplies managed owner UID; child fixes it to 10001. Namespace/snapshot ownership still uses executing child UID. `runtime_child.py:68-80,137-141` applies AS/file/process ceilings and numerical-library thread variables before lazy generation/backend imports. File ceiling includes the larger reference/output. Tests mock limits and redirect runtime mounts/ownership; they prove ordering/arguments, not installed enforcement or readability.
- **Output/publication:** adapter measures exact backend rate/frames/mono finite samples and PCM16 bytes/digest. Child stages one exclusive no-follow fsynced output, and Core independently measures it against exact descriptor rate/output limits and derived `pocket-tts-execution-sha256:` provenance (`core.py:532-535`). The 22,050-Hz fixture reaches real Core publication. Stale input/provenance/rate negatives refuse. Existing owner/attempt/input/deadline/publication barriers remain. Reference provenance is retained; generated origin does not prove actual speech.
- **Cancellation/reap:** unchanged Dispatcher still requires exact confirmed reap before capacity release/ACK; publication or stop intent alone is insufficient. New tests exercise injected success/cancel/reap and retain conditioning data before/after reap. Unknown/nonempty namespaces remain blocked and untouched. No namespace deletion is introduced. Existing result-versus-cancel-intent race semantics remain.
- **Compatibility:** public protocol 1.3.0 and lifecycle-v1 frame shapes remain unchanged; Kokoro source ID spelling/rate and synthetic behavior remain unchanged. The expanded private context needs updated paired readers; unchanged version strings do not prove universal mixed-version compatibility. R245-1 prevents a claim that all durable readers are coherent. Installer/configuration remains closed to Pocket and supplies no new reference mount activation route.

## Independent execution

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B --version`: **Python 3.14.7**. Candidate cwd, explicit absolute import paths, inherited `PYTHONDONTWRITEBYTECODE=1`. No pytest collection, listener or maintenance fixture. Exact named command:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-b646dab9","/Users/ben/.txcode/worktrees/py-audio-box/t3-b646dab9/tests/service"];names=("tests.service.test_pocket_lifecycle","tests.service.test_engine_admission","tests.worker.test_engine_lifecycle","tests.worker.test_engine_pocket","tests.worker.test_engine_kokoro","tests.service.test_production","tests.service.test_production_admission","tests.service.test_production_reconciliation","tests.service.test_production_api","tests.clients.test_production_clients");r=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names));sys.exit(not r.wasSuccessful())'
```

**61 methods PASS (9 new + 52 existing), 12.991 seconds, exit 0.** One existing injected test printed `lifecycle supervisor unavailable; lane quarantined`; suite completed OK. Backend/transport/reap are injected; no peer authentication/PID1/death-proof claim follows.

Additional small inert probe, same interpreter flags/import paths, exit 0:

```python
p = PocketLifecycleTests(); p.setUp()
try:
    a, proposal, stage = p.proposal()
    scan = Reconciliation(p.core, p.stage)
    summary = scan.reconcile_startup()
    row, = [r for r in scan.rows.values() if r['name'] == 'conditioning']
    assert row['classification'] == 'committed_referenced'
    assert json.loads(p.core.db.execute(
        'SELECT record FROM p1_inventory_objects WHERE id=?', (row['id'],)
    ).fetchone()[0]) == row
    tree = ast.parse((root/'service/recovery.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_installed_detached_source')
    idx = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'permitted' for t in n.targets))
    fragment = ast.Module(body=fn.body[idx:idx+2], type_ignores=[])
    try:
        exec(compile(fragment, 'service/recovery.py:3027-3032', 'exec'),
             dict(source={'inventory':[row]}, owners=[row['owner']], digest=digest))
    except ValueError as exc:
        assert str(exc) == 'installed_history_inventory'
    else:
        raise AssertionError('new reason accepted')
    (stage/'conditioning'/'opaque').write_bytes(b'x')
    assert Reconciliation(p.core,p.stage).reconcile_startup()['blocked']
    assert (stage/'conditioning'/'opaque').exists()
    changed = copy.deepcopy(a.context)
    changed['execution']['descriptor']['max_text_bytes'] += 1
    assert Authority(a.job,changed,a.allowed_assets,a.profiles).binding != a.binding
finally:
    p.doCleanups()
```

Probe imports: stdlib `ast,copy,json`, candidate `PocketLifecycleTests`, `Reconciliation`, `service.recovery.digest`, `service.worker_transport.Authority`; `root` is the absolute candidate Path above. Also independently parsed all 11 changed Python files with `ast.parse(..., feature_version=(3,11))`: PASS, **grammar only**, not Python 3.11 execution. `git diff --check d803efb HEAD`: PASS. Candidate remained clean.

## Remaining limits and parent gate

Actual child UID20000+ differs from control UID10001. Even an empty actual child-owned mode0700 directory fails the new control-owner condition and remains UNKNOWN; retained nonempty conditioning also remains UNKNOWN. This is conservative retention, not permission to inspect/delete or evidence of installed restart success. R244 candidly discloses it. Treat cross-UID restart/inventory integration as an unresolved activation blocker in addition to R245-1, not a successful restart result inferred from same-UID fixtures.

Known snapshot operations and per-file ceilings do not provide aggregate filesystem quota or private cross-UID monitoring. Real memory/thread limits, model headroom, mounted immutable bundle/reference readability, authentication/containment, forced termination/reap and installed restart remain unqualified. Preparation/audition and installer/provisioning are outside this generation-only lot.

Disk observation: 3,868,528 KiB available on `/private/tmp`; only authorized small owned inert fixtures and their normal teardown ran below the 6-GiB heavy gate. No ML/model loading, actual inference, Docker, network/download, listener, process crash, operational cleanup or HELD-resource access. Parent owns resolving the narrow source finding, independent correction review/adoption, and separately authorized actual clone listening/resource/install/release gates.
