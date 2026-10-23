# P1B23 product failure — independent actual partial-receipt audit r34 Sol

2026-10-24. **Bounded GO for the honesty/integrity of the r33 partial receipt; product diagnosis remains HOLD.** The stopped exit-1 writer is proved in retained inspection, not an accepted expected pytest RED. Smallest next proposal: recover the existing volume with two existing finite RO readers, without another writer, cleanup, lane release, build or runtime campaign. This is a source-reviewed recovery proposal for parent disposition, not operational self-GO.

## Scope and independent verification

Read full r33 actual report, r32 gate and r28 global review. Targeted deep read covered finite-driver writer/main/capture/checked_program/source verification/readers/terminal publication; launcher capture, bindings, run_container/status gate, retention, run/cleanup/finish/root/supervisor; runtime identity and entry selection. Read retained authority/intent, raw captures/inspections, slots and generated source context. Read-only Git, text, stat, JSON and stdlib hashes only; no candidate/product imports, Docker queries/mutations, DB opens, pytest, restart/retry/cleanup or delegation. Scoped root ancestry checks found no AGENTS file. A broad ancestor file search was interrupted and is not evidence of global absence. Memory registry search supplied no relevant evidence. Sole repository write is this report, mode0600; no commit/source change. Unrelated STT artifacts preserved.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-dc329cf6` is clean at documentation `c06e2f49b48264b51d65b0a9d1506b973ee4f839`; executable pin is `84477f38b4b6de52994e21e6cf6f3bafec801db6`. Independently rehashed all12 authority pins and all193 repository payloads plus two derivatives: no mismatch. Launcher SHA256 `dbc5d32afb11bfc4254deaa6945f94e8da7b5e6f65b20d0b4b36eaac2267eb7d`; finite-driver `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`.

Actual root `/private/tmp/pab-p-one-r16-f78d3a20-d739-414c-b77e-ab1388470850`: independently recomputed all50 inventory lengths/hashes and all20 raw streams from10 completion records, zero mismatch. Exactly55 regular leaves: inventory plus five present exclusions intent.slot/final.slot/evidence-hashes.json/handback.json/supervisor-complete.json; declared release-intent.json is absent; no extras. Actual and supplemental `/private/tmp/pab-blocked-pipe-r33-sol-20261024` trees are0700 directories/0600 regular leaves/no symlinks. Both512KiB slots have valid internal body SHA256 and NUL padding. Supplemental execute COMPLETE/exit1,77 stdout bytes/zero stderr, reaped/closed/no group kill agrees with STOP/UNKNOWN; supervisor records COMPLETE/worker_exit1 without inferring remote completion.

Authority SHA256 `c32089e60e7946ee6a3a1943bc9f9035e290eb4fcb7f82d67cacf1a5831cafe2` and r32 review SHA256 `120dceda2305b9626b7f92d9a5e7016a1a85272304a7a1aeeff3f5122226fa34` match the slot owner. Actual authority says approved ONE-PRODUCT-DIAGNOSIS, current parent attribution, exact candidate/image/node/runtime/bounds, acceptance=false/product_release=false. Launcher only checks nonempty authority files: semantic scope comes from this retained authority, not that syntax check. Legacy owner SOURCE/DOCUMENTATION constants remain historical and do not prove successor identity; current hashes and explicit authority bind the successor.

Final slot: STOP-OR-UNKNOWN-HOLD, retained=false, lane_released=false, acceptance=false, error `RuntimeError: STOP lifecycle status: writer`, host_bytes8430547. Inventory charge8424560,4GiB software cap/16MiB supervisor reserve. Intent remains PENDING; handback remains FINALIZATION-PENDING. These are earlier snapshots, not competing success receipts. Operations: volume-create CONFIRMED, writer-create CONFIRMED, writer-start TERMINAL. No readers/removals/release intent. Current lane owner equals both slot owners; device/inode exactly `[16777223,1007146242]`. It remains held. This is host evidence plus retained daemon observations, not a fresh daemon refresh or durability/physical-quota claim.

Excluded evidence SHA256 independently matches r33:

- intent.slot `57a44bb6571803dc2aa15c3d477cd770b3784f57d17d9bf5633d058ad7cd2930`
- final.slot `fa41362b52ca786806e8db9b2cee4137eda45fd5124df43bd9374693f0bff7df`
- handback.json `ae4bb2d77a755421c6aa40ea977ac4b57d00ca733b9118b15909fe1844cc89ae`
- evidence-hashes.json `1ff0cb7544e9f07c0b8cff79cfcd5541be906cff267ce09813f05219f874abec`
- supervisor-complete.json `38ac805d806377978399c526e9c70b49992ff28cdc4b0bc64e10cab643c46f89`

## Exact stop and causal limits

Invocation `f78d3a20-d739-414c-b77e-ab1388470850`; token `ef4f585be7dd62b800a26275b0e817046208947b8341a736`.

Writer full ID `e8dd28a5058383c8e4ca012cf1be0570943b06691bd61c318cca35b6e045bc18`; image `sha256:60a969256c4fb44fd8e0b830eb13223a62e75f1bb9518eadaa91b466489273f5`; volume `pab-p-one-f78d3a20-d739-414c-b77e-ab1388470850`. Retained full prestart/stopped inspections bind that ID/image/name/labels, numeric10001, command, /app, CPU2/RAM2GiB/PID256, networknone/read-only root/capdropALL/no-new-privileges, no restart/autoremove/logging and sole /tmp named volume RW. Volume binding is local/Options null/Scope local with exact labels. Stopped State: exited/Pid0, Running/Restarting/Paused/Dead false, ExitCode1, OOMKilled false, Error empty; start22:12:15.633745235Z, finish22:12:18.324732628Z on Oct23 UTC. Attach COMPLETE/exit1/empty streams; wait CLI COMPLETE/exit0 and exact bytes `1\n`. Stop is established before lifecycle_status rejects. No OOM, timeout or daemon Error cause is established.

finite-driver.py:342-431 deliberately encapsulates exceptions: initial admission/source/identity/ENTRY failures become only `state['failure']=type(exc).__name__`. Finally it reaps namespace descendants, stops monitoring, checks source and identity-after/equality, classifies terminal, and attempts checksum-slot publication. Postcheck exceptions become terminal_failure type only; publication exceptions are swallowed into UNKNOWN. Writer normally emits no attached output; identity output and ENTRY streams go to volume files. Thus empty attach streams fit many paths and identify none. Even a recovered RuntimeError type may be insufficient to distinguish source/freshness/memory/census checks because exception messages are not retained. Do not invent the missing message or rerun to obtain it.

Expected pytest exit1 is explicitly required at driver line414; with successful admission/postchecks/no failure it produces TERMINAL and **writer return0**. Launcher requiring writer0 therefore does not unconditionally misclassify expected diagnostic RED. Observed writer1 means non-TERMINAL driver result or an earlier uncaught failure; exact cause requires files. The launcher correctly refuses to label it successful, but its run() ordering aborts before retention. This is a failure-evidence retrieval gap, distinct from the unknown underlying runtime/product bug.

Generated context from retained r31 root `/private/tmp/pab-p-one-r16-836ee3d8-f0ca-4582-bbdc-c7069c4faa9c/context/source` equals repository manifest except exactly one extra derivative `diagnostics/p-one-node/dummy-writer.py`. Independently hashed all196 mapped context files; all match. Retained built-image receipt names the actual image above. verify_source merely iterates the merged files/derivative_files hashes under /app; it neither compares generated manifest bytes to repository source-manifest.json nor rejects that extra dummy mapping nor invokes dummy code. There is **no unconditional augmentation mismatch in this source**. This does not freshly prove image filesystem bytes. Identity-before invokes unchanged CPython3.14.8/42-version/native SQLite helper before ENTRY and captures combined stdout/stderr into identity-before; nonzero rejects before ENTRY. Any actual identity failure is still unknown. The two runtime identity files must match byte-for-byte for terminal success; do not infer equality from host/image metadata.

## Files to retrieve, without assuming their existence

Archive the entire /tmp census, not a selected success-only list. Candidate files from the exact path are:

- `pab-seed`; `pab-intent.slot` and `pab-terminal.slot` (512KiB each when allocated); `retention-sentinel` (token plus NUL/ff/80/CR/LF).
- `identity-before` and `identity-after` (each checked_program combines stdout/stderr into one capped file). A file may be absent, empty or partial depending on dispatch/failure.
- `pytest.stdout`, `pytest.stderr`; `waits.jsonl`; `pab-p-one-node-observer/records` and any other census members created by the unchanged test/observer.

Initialization/memory failure can precede slots and sentinel. Source failure can precede identity-before; finally may nevertheless attempt identity-after. Entry imports/collection can fail before observer records. Publication failure may leave PENDING/torn/missing terminal. Full archive and independently generated all-regular map are required before selecting evidence. Missing success markers are diagnostic observations, not reasons to discard an otherwise valid failure archive.

## Minimal recovery proposal — not executed

**Use a separate private recovery root; preserve the old root and held lane unchanged.** Do not call launcher CLI/main, make_root with the old invocation, acquire, run, retention, cleanup or finish. Rehydrating Lifecycle on the old root makes journal overwrite intent.slot and later publication collide with exclusive existing files; finish could rewrite final.slot/release the lane. retention additionally demands TERMINAL/pytest1/sentinel before starting hash, so it is unsuitable for a failed writer. No existing CLI exposes failure recovery. Narrow reuse of unchanged run_container, create_argv, call and parsers is sufficient in a one-shot parent-owned snippet; no source patch/new framework is needed.

Concrete proposed command envelope, only after parent authorizes this recovery scope: immutable candidate cwd; `PYTHONDONTWRITEBYTECODE=1 python3.14 -B -` with the bounded control flow below supplied on stdin. Import only reviewed launcher/finite-driver via importlib; no product imports. Fresh root `/private/tmp/pab-r34-recovery-<fresh-uuid>`0700, exclusive0600 receipts and slots. Store parent recovery authority/review hashes plus old-root/final-slot/owner/inode bindings in a separate recovery-intent receipt. Preserve old owner unchanged for existing volume/container labels and token; recovery identity is additional host provenance, not a fabricated new product invocation.

Exact existing callable sequence (L is imported launcher, F=L.F; root is the fresh recovery root; owner/objects are decoded from verified old final.slot):

```python
F.slot_create(root / 'intent.slot')
F.slot_create(root / 'final.slot')
x = L.Lifecycle(root, L.LANE, IMAGE, old_final['owner'], L.Real())
x.budget.charge(2 * F.SLOT)
x.objects = {'volume': old_final['objects']['volume'],
             'writer': old_final['objects']['writer']}
# No acquire(), no writer run, no old-root journal/publication.
# Preflight below must complete by t0+20; x.phase_end=t0+20.
x.run_container('archive', t0 + 80)
# Proceed only if archive COMPLETE/zero and full-ID stopped checks pass.
# Require clock<=t0+80 before beginning the second reader.
x.run_container('hash', t0 + 140)
audit = L.parse_archive(root / 'archive-start.stdout',
    root / 'archive-inventory.jsonl', x.budget,
    clock=x.a.clock, deadline=t0 + 170)
F.require(L.parse_map(root / 'hash-start.stdout') == audit['hashes'],
          'STOP recovery archive/map mismatch')
# Publish recovery-only integrity receipt, then inventory/final recovery slot.
# Inspect terminal/sentinel/identity/pytest as evidence, not success admission.
```

Here t0 is the fresh monotonic start, IMAGE is the exact full image above, and deadline/pressure checks precede every phase. Preflight uses x.dispatch/inspect under the20s aggregate deadline, retaining binary capped captures: exact `docker image inspect IMAGE` (Id/Linux/amd64/startup), `docker volume inspect VOLUME` plus L.volume_binding(old owner), and `docker inspect WRITER_FULL_ID` plus L.container_binding(...,'writer') equality to the full ID and L.stopped. Recheck nonOOM/empty Error and old observed exit1 without demanding writer0; reject changed bindings/state. Host free admission>=6GiB; current lane no symlink, exact inode and byte/JSON owner equality; old receipt hashes unchanged. Never release/take over the held lane. Reader census retains existing daemon pressure>=2GiB; no volume initialization runs.

For each mode, run_container expands the **existing exact** L.create_argv(name,VOLUME,IMAGE,mode,owner): names `pab-p-one-f78d3a20-d739-414c-b77e-ab1388470850-archive` and `...-hash`, same owner labels/image, numeric10001, CPU2/RAM2GiB/PID256/networknone/read-only/capdropALL/no-new-privileges/restartno/lognone, sole volume /tmp **readonly**. Command is `python /app/diagnostics/p-one-node/finite-driver.py archive|hash TOKEN INVOCATION`. No exec/cp/new image/bind/helper program inside container. Each create is once-only after exact-name absence; any collision stops without adoption/retry. Full64-hex create acknowledgement is captured, prestart inspect must match that exact ID/config/RO mount and created state, then exact `docker start --attach FULL_ID`, separate `docker wait FULL_ID`, full stopped inspect. Existing lifecycle_status retains zero attach/CLI/wait0/workload0/nonOOM/empty Error gates for readers. Readers execute sequentially only after stop is proved; their independent remote alarm is60s, while host whole reader phase<=60s includes protected12s wait/inspection reserve (attach therefore shorter). Unknown start/timeout means STOP, no second reader, no remote-stop inference.

Schedule: preflight<=20s, archive whole phase<=60s, hash whole phase<=60s, parse/hash/diagnostic inventory<=30s, final publication/local reap<=10s; total<=180s. Existing mutation reconciliation, if triggered by a create failure, is at most one observation<=30s and consumes the remaining total budget; it never authorizes another mutation. Stop at that failure. Protect the total with an outer invocation of existing L.call in a separate fresh0700 capture directory: `L.call(outer_root, "recovery", ["python3.14", "-B", "-c", recovery_snippet], L.Budget(L.SUPERVISOR_RESERVE), L.Real(), deadline=outer_start+178, cap=F.META, cwd=str(candidate), completed_nonzero=True)`. Here recovery_snippet is the one-shot control flow above, supplied as an argv string, not a new executable helper file; it prints only a bounded receipt pointer. Existing call timeout kill/reap has a1s reap allowance, leaving the final total margin up to180s; reserve the outer metadata capture separately inside the4GiB total. Do not use the existing1770s supervise default as if it were180. Kill/reap only owned local process/session on timeout, preserve spawn receipts and classify remote UNKNOWN; never stop/kill/remove remote helpers. A blocked kernel filesystem syscall still limits absolute wall-time guarantees as in the reviewed mechanism.

Caps remain unchanged: each file<=64MiB, total logical<=1GiB, census<=200000 entries with cooperative0.5s scan, archive<=2GiB, hash/metadata/stderr<=16MiB, host cumulative charged writes<=4GiB with16MiB protected publication reserve. Charge reserve against that cap, check2GiB host pressure during capture, retain capped prefixes and hashes on any failure. Archive parser validates member names/types/UID/GID/count/aggregate, computes all regular hashes and writes inventory; map parser rejects duplicate/invalid digests and equality checks every regular file. Do not extract paths onto host or relax caps to recover a nonconforming archive. Hash the binary archive and map themselves. Retain both new helpers, all old objects/images/tags/volumes/proof, old lane and historical slots; no cleanup even after equality.

Recovery receipt must distinguish `ARCHIVE-MAP-VERIFIED` from product terminal validity, list exact helper IDs/prestart/stopped/capture states, retain terminal raw bytes/checksum outcome and per-file diagnostic findings, record missing files and preserve unresolved objects on failure. A STOP terminal or missing observer is not product acceptance, but does not invalidate independently verified failure-file transport. Finalize only the new recovery proof; leave r33 final.slot untouched. Parent acceptance of this partial receipt and explicit recovery authority are the next operational gate. A fresh actual review of the recovered bytes can then resume concrete product diagnosis immediately; no another dummy/writer/adviser campaign is proposed.

No actual identity/ENTRY/47-field/offset/type/owner/enrollment/raw18944/empty-phase/EAGAIN/exception/guard/call-teardown/unreached-assertion success is established here. Historical PID229 cause remains UNKNOWN. No H/S/fullsuite/native/T2/platform/ext4/durability/installed/product/release/activation gate advances.
