# Launcher source freeze recipe — NOT execution authority

Run only source/text/Git operations for this preparation. Fresh independent review of this exact launcher is REQUIRED before any ONE execution. No Docker or launcher command was executed here.

The source-manifest pins 193 copied regular files from observer documentation base 8d5aba5fc702719843d055b7d3f693c407deb0e9, including exact observer/entry/guard. Launcher copies only that allowlist, never the checkout wholesale. Future execution uses one `docker exec -w /app <owned-name> python /app/diagnostics/p-one-node/entry.py --one-node`, PYTHONPATH=/app/diagnostics/p-one-node and PYTHONDONTWRITEBYTECODE=1. No separate collection. The Dockerfile is generated as FROM the newly owned inspected local base tag, WORKDIR /app, COPY source/ /app/ only; build uses --pull=false --network=none. No install or pull.

Proof root /tmp/pab-p-one-node-launcher-20261023 must be absent; umask077, root0700, files0600. Source manifest is verified before candidate imports and after test; runtime identity is checked pre/post, including all42 lock versions, Linux image pins, SQLite source/compile options and linked-library digest. Identity stdout must match pre/post. Existing inner deadlines and accelerated test clock remain byte-identical; outer monotonic budget1800s, test subprocess1200s with reserved retention time. All commands have finite timeout bounded by remaining outer budget.

Lane mkdir is atomic at /tmp/pab-worker-w1-docker-lane.lock, owner PID/token/base pin recorded; owner equality required for release, never foreign removal. Host disk reserve6GiB and in-container available memory/disk4GiB admission; tmpfs free900MiB before test. CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/cap-dropALL/no-new-privileges/tmpfs1GiB; inspect rejects binds/devices/ports/mounts. No sockets, secrets, audio, services or host mounts.

Before the test, binary tar sentinel includes NUL/FF/80 and is compared byte-for-byte. After the sole test, direct binary docker exec tar --hard-dereference exports ALL live /tmp before stop; independent LIVE regular-file hashes must equal archive hashes. Safe unique relative member names; symlinks metadata only; no extraction, DB open or docker cp. Export failure leaves LIVE-PROOF-HOLD with owned container/lane; no retry, no destructive cleanup. Successful retention permits exact owned stop/inspect/rm without -v, exact-name absence and owner-bound lane release receipts. Any later cleanup failure preserves remaining resources and evidence for parent handling; never foreign cleanup.

Records are bounded47-integer rows,512bytes/row,8rows/process,16KiB total; required child1 and owner2/3/4 set binds enrolled PID/raw wait. Structural/type/phase mismatch STOP. Even structurally valid records deliberately STOP for independent actual caller/runner offset and separate guard/report binding review; process-local numeric IDs cannot independently name unknown classes. Expected RED remains RED. No automatic cause/acceptance, test retry or tuning.

## Static freeze hashes

- launcher.py: `ea419c21e81966861f79cb6af81c724bad5ebf78618de4c31fd56f554bdf0ee6`
- runtime-identity.py: `81acbdbb70df0a9c2a5db1f9daf86373cb22ea3b92371d7dddd924420a60237c`
- source-manifest.json: `1d8280803813f6e550bbdde1028eb7c315a397c0157f6a4f0fcec2f67841984e`

## r16 successor pre-HOST freeze (2026-10-23)

This section supersedes the old tmpfs/sleep/exec/quota recipe above.
Ordinary fresh Docker local persistent volume; logical1GiB/inventory200000/export2GiB/host4GiB are software thresholds, not physical quotas. No Docker execution authorized. Matrix fixed at32 cases,9s/case,300s total before first run. Original inputs remain preserved.

- launcher.py: `df84f4202c3c61b1fb32eabd35c829e10b5e30928bc0b415c59491b93e281564`
- finite-driver.py: `5d07779e0ae6604fb78ccdb4d3877a9b8482aa6c818ef342b54661ae9cefe930`
- host-check.py: `25fd3334393040d581014968a896a2b50e3f510e0c18ff37fccc6513778d7bbf`
- source-manifest.json: `c40e785e9da917a303f52cf23d5c75fe2d75e42449f83c7053144084dae7f71d`

### Failed-case repair freeze

Original RED retained at `/private/tmp/pab-p-one-r16-HOST-1hn1pii1/HOST-receipt.json`. Fix: portable real bounded slot writes; fixture GID uses actual created directory. Rerun fixed scope:9 failed cases plus18/20 (previous wrong-GID early rejections) and29 (new derivative-byte binding). No other passing scenario rerun.

- launcher.py: `df84f4202c3c61b1fb32eabd35c829e10b5e30928bc0b415c59491b93e281564`
- finite-driver.py: `d5996ead1cd702a7ab7c95dbfa48bcd115bc781dc78a566aceccd86aa40efaea`
- host-check.py: `74a761472f387657b0655ead2ba49f279c12184d439097fc71cbadd4daeabc77`
- source-manifest.json: `5a8e3e1f3a180329700a29427cd2b56ba546a84bf31ac2c5b3f05fbf630446e2`

### Protected-phase and coverage repair freeze

Following the scoped green receipt, source inspection identified that early completion could allow retention to consume more than180s. Fixed local phase ceilings: writer create/binding before120; retention<=180, each helper lifecycle<=60; reconciliation<=120; finalization<=30. Added separate host supervisor (1770 worker cutoff preserves final30), local CLI spawn inventory, actual archive/hash helper HOST seams, lane collision and full hash mismatch fixtures. Rerun only affected prior failed cases plus changed17/18/20/21/23-29/31/source binding; no real fork in HOST.

- launcher.py: `575867084bffa9507fb9c5a014bd5750661082c989bc80e0db1103cf014d77b2`
- finite-driver.py: `03a76ee0a9776ffcbf529cadd632ea4a15638b9799b62ba2f8e0a25b38324fdf`
- host-check.py: `55b41a99b4d8f171ac41b4860bd76c2fe347a0fd9e56876658dc8033b6953c25`
- source-manifest.json: `7b447a9c4cf6958c11819eed6fe55e2177156bf6ce9d19c0a9a50d71080bead2`

## Current successor contract — source candidate, NOT runtime GO

Successor worktree: `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r16-20261023`, detached at documentation `f93b7fef6cc461b756cd9e13ff14e88e243359a4`; reconstructed executable `e941c7aee7f53348905659904b8299d5500d1250`. No new commit, push, root integration or historical mutation. The content hashes in the last repair freeze above identify the executable candidate; documentation appends do not advance those bytes.

Adopted parent policy: full `P1B23-STORAGE-PROPORTIONALITY-ASTRA-r16.md`. It explicitly supersedes the metadata-inclusive physical1GiB/inode quota/backend UUID prerequisite in the historical lifecycle review. Ordinary fresh Docker `local`, empty Options, persistent `/tmp`, volume name/creation/labels recorded honestly. No volume name is described as immutable. No fallback bind, setup container, root runtime initializer, install or daemon change. The historical FREEZE/HANDOFF text above is retained as history and is superseded wherever it describes tmpfs, sleeping PID1, writer exec, LIVE export or quota admission.

Source build proposal: exact retained base `sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb`; numeric COPY ownership for source and seed; only build-time `/tmp` ownership/mode metadata operation. Host source copies remain0700/0600. `/tmp` is prepared10001:10001/0700 with fixed binary0600 seed. No package/network operation. `--prepare` writes a context/proposal only; it never builds. Actual image availability, Docker's base-reference resolution, COPY destination traversal and empty-volume population remain unverified. Parent must freeze a usable exact recipe/image during the separate qualification assignment; no automatic repair is allowed on failed initialization.

Runtime proposal requires exact independently reviewed derivative image ID plus nonempty independent review and parent authorization files. Their hashes bind provenance; the launcher does not decide that arbitrary text confers authority. No Docker build/tag or writer exec is dispatched by the actual launcher. The remote PID1 invokes unchanged entry exactly once with fixed singleton/argv/env; assertion-enabled admission is explicit. Source/runtime checks precede product import and follow test phase. Exact Linux3.14.8/42 locked versions/SQLite3.40.1/package/source/linked-library digest retained. Entry/observer/runtime helper remain byte-identical.

Logical thresholds:1GiB aggregate regular data including1MiB lifecycle slots,64MiB/regular leaf,200000 entries, pytest streams16MiB each, map/inventory metadata16MiB, tar2GiB and all owned host logical writes4GiB. Counters refuse before driver writes; copied context and preallocated slots are included. Product L4194304 is unchanged and separate. These are acceptance/monitor/export limits, NOT a physical-byte, inode, metadata, deleted-open-file or allocator quota. Host and daemon admission each6GiB are observations, not reservations. Unknown topology is shared:4GiB host +1GiB volume with nominal1GiB admission margin. Image layers/cache/backend metadata are separate overhead, not a hard5GiB envelope. Source proposal and actual invocation have distinct retained roots/budgets; preexisting external review/auth files are referenced by hash, not silently copied outside accounting.

Remote schedule<=1350s:60 prechecks +1200 unchanged child allowance +30 post-test descendant termination/reap +60 source/identity/publication. Host phase schedule remains120/60/1200/30/60/180/120/30 =1800. Admission create/binding finishes before120; retention total<=180 and each helper lifecycle<=60; reconciliation<=120; publication<=30. Parent supervisor stops its worker by1770 to protect the last30 for an honest UNKNOWN receipt. This conservative cutoff may reject a late otherwise-completing worker; it never extends product time or borrows cleanup budget. The remote watchdog is independent of host attachment. No `waitpid(-1)` competes with the product during its test phase. Afterward PID1 inventories its private namespace, TERM/KILL escalates within30, records waits and rejects residual workload. OOM/missing terminal/unresolved daemon work stays UNKNOWN.

Monitor: separate thread outside protected route, no-follow census bounded to0.5s with0.5s inter-scan wait (<=1s cooperative cadence), admission and active checks; census error/timeout/logical breach or daemon free<2GiB stops work. Host supervisor and CLI streaming loop observe host pressure at<=0.5s/0.25s. Kernel/daemon IO stalls, uninterruptible processes, deleted-open bytes and changes between observations defeat an absolute physical/wall-clock guarantee. Failures preserve volume/lane; they do not claim completed cancellation. A supervisor-killed local worker can leave CLI sessions unresolved: bounded spawn receipts identify them; no remote completion or local descendant cleanup is inferred. This remains an explicit source/actual review concern, not HOST-qualified process behavior.

Mutation state before dispatch: NOT-SENT then SENT-UNKNOWN, acknowledged CONFIRMED, conclusively reconciled TERMINAL. UUID/random192-bit token bind every owned object. Unknown response gets one bounded exact discovery observation, never mutation retry; a negative observation cannot prove an in-flight operation absent. Full64-hex container IDs bind name/token/image/command/security/mounts before start and cleanup. All destructive container mutations use full ID, no force or `-v`. No volume/image/tag/host-proof deletion exists. Failed export or uncertainty retains stopped containers too. Named volumes are intentionally retained after successful cleanup.

Export only after exact stopped-writer/no-remote-workload receipt. At most two sequential same-image read-only helpers, archive then independent quiescent-storage map; no reader follows uncertain helper state. ALL `/tmp`, binary sentinel, lifecycle records and regular-file hashes included; no extraction/DB reopen/docker cp. USTAR-only parser rejects extension headers before payload allocation, normalized duplicates, escape, unsupported objects, UID mismatch, count/size/stream overflow, malformed/truncated streams and map mismatch. Link metadata is recorded without following links. Parser bounds can reject a legitimate long-path export; that is STOP with original volume retained, never permission to omit it.

Terminal completion requires valid final slot, complete hash inventory/handback, raw host success and `supervisor-complete.json` COMPLETE/worker_exit0. PENDING/torn/missing/finalization failure is not success. Hash inventory declares post-inventory publication exclusions. Lane release additionally requires every sent operation terminal, verified retention, full-ID absence and exact inode/full owner equality. Failed final publication after release remains UNKNOWN with release-intent and residual inventory; no new run is implied.

### Preserved product evidence boundary

All193 original manifest payloads match unchanged. Entry, observer, runtime identity and the entire product/guard/tests/locks/dependencies have no successor delta. Original manifest `files` and observer commit remain identical; `derivative_files` separately binds driver/runtime helper without self-hash cycle. The current parent root has later product bytes; this successor deliberately uses the historical pinned tree rather than mixing them.

The selected node is `tests/service/test_recovery.py::test_t1_real_blocked_pipe_backpressure_deadline`. Four47-integer rows and exact owner/child/enrollment/raw18944/empty-phase relations are structurally checked. Actual offsets/types/guard/call-teardown/EAGAIN/unreached-assertion binding requires independent ACTUAL review. Expected pytest exit1 remains RED; no52-record S construction acceptance is relevant. Historical PID229 active-final-exception remains UNKNOWN.

### Lane-specific writer inventory

- One source/identity precheck and one postcheck process each, bounded60s phase and16MiB output; no packages or user-selected script.
- One unchanged entry/pytest process; one selected test's tracked maintenance fork. `_t1_pipe_model` intentionally models storage/session and disables that test's limit installer, as frozen source already did; this is not a claim of actual L4MiB enforcement by this diagnostic. Its operation returns a fixed900000-character value, writes result pipe in65536 chunks against unchanged30s absolute deadline/100x test clock, and writes only the fixed `actual-EAGAIN` marker on actual kernel refusal. No DB fixture/campaign is added.
- Pytest's fresh temp root, one `maintenance-guard` fixture root, singleton tmp_path, test guard's bounded4096-byte atomic metadata/guard event records, bounded selected-node wait log, observer's fresh0700 root and0600 numeric records (at most8 rows/process,512/row,16KiB acceptance total). No repeated fixture campaign or prior-run namespace is reused.
- Fixed seed/sentinel, two512KiB slots, pre/post identity output, two16MiB pytest captures and one terminal update. Remote writer captures enforce before each write. Runtime namespace cleanup follows test phase only.
- No reader writes the volume. Two finite RO helpers emit one tar and one map; host captures/journals/inventory are bounded, no second archive copy. Host command ceiling64; one mutation per operation key; no export retry.

This short inventory identifies declared writers, not arbitrary filesystem metadata growth. Monitoring/threshold failure stops and preserves; no truncation or cleanup for space is selected.


## r18 F1/F2 executable content freeze — 2026-10-23

Fresh detached successor `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r18-20261023` reconstructed at documentation `87cec9a3ff9d2b829c04aa096e930d021b4fae83`, containing executable `8364308c8f035bf65362247b201514a060c06730`. No commit created. Only the existing six-path allowlist changed. The following content hashes, also frozen before and verified after the final HOST run, identify the candidate; all later changes are FREEZE/HANDOFF documentation only.

- launcher.py: `f28c8785df17854ee120bffb4ff08bf5555e21e7dac316d56434d733ed640c80`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- host-check.py: `dec48fd7274e887f223f898c8e66a628fbdec7ff246a545217606e681141339c`
- source-manifest.json: `e0a228c16b83f9acbc1f640582bd240d3a701d2128b4047812aa261314a66035`

F1: child deadline is min(actual child start+1200, driver start+1260). After quiescence the complete postcheck/publication phase is min(phase start+60, driver start+1350), including monitor join, source verification, identity, readback and terminal slot write; checks before and after publication reject overrun. Existing independent1350 alarm and min(now+30,start+1290) quiescence remain. Cooperative checks cannot interrupt a blocked IO syscall; the independent overall watchdog remains the ultimate bound. A publication crossing the deadline returns failure even if its slot bytes were written; host stopped-state/exit0 admission is still mandatory.

F2: host takes both returned pipe objects immediately after spawn before spawn-receipt publication, selector construction or registration. Remote owns both pipes through ExitStack callbacks before selector setup, now inside the kill/reap exception boundary. Every injected setup failure attempts exactly one owned group SIGKILL even with an exited leader, bounded wait and explicit closure of both pipe objects before fixture rescue. Host receipt latches UNKNOWN/remote_completion=false. Remote rethrows the original failure after kill/reap; writer subsequently records failure, never daemon completion. Remote capture has no spawn-receipt publication seam; that injection applies to host only.

Invocation accounting now partitions the unchanged4GiB logical-write cap: worker cap=4GiB-16MiB and supervisor publication reserve=16MiB, whose own writer charges actual encoded bytes before write. The reservation is conservative, not a second4GiB budget or physical quota. Evidence hash receipt records reserve/cap explicitly; repeated slot charges remain conservative. All policy/product limits unchanged; ordinary persistent local storage is still not a physical byte/inode/metadata quota.

Final HOST receipt `/private/tmp/pab-p-one-r16-HOST-5xxexl3z/HOST-receipt.json`, SHA256 `fa8ef7b0ee9c2e85845bda858f7ce5739394199aab8dd9e83fc9d6312fd3461d`, PASS7/7, elapsed0.589028s, fixed32 scenario groups/9s per group/300s total. Scope fixed in harness before execution:09,23,26,28,29,31,32. No other unchanged passing scenario rerun. It binds the four hashes above. Raw final invocation stdout `/private/tmp/pab-r18-host-final.stdout` SHA256 `ff1f3f4de61dd06ea20d03380a374718981c9dff8705976aba48424d068d33eb`; stderr `/private/tmp/pab-r18-host-final.stderr` SHA256 `0365ec4efccdc43fa8b90cab4adf3b5322414bdfaba5989d2b01aa74bee318df` (intentional injected final-publication error). Both0600. Raw per-command captures, pending/spawn/completion receipts remain under the HOST root0700 with0600 leaves.

Initial r18 scoped PASS7/7 `/private/tmp/pab-p-one-r16-HOST-xgbech_5/HOST-receipt.json` is retained. A subsequent source inspection corrected the fallback publication guard and made the pre-write deadline unconditional; the scheduling fixture now requires zero terminal publication attempts after expired source/identity postchecks. Also exposed supervisor reserve/cap in the evidence receipt. These changed seams justify the final same seven-group rerun; no RED or prior receipt was overwritten. Original32 RED, scoped12 PASS and14 PASS roots/hashes remain in the historical handoff.

All193 original payload hashes and both derivative hashes verified; original files mapping/observer commit unchanged. AST parsing and git diff --check pass. No executable or manifest edits after this receipt. No real Docker/Git mutation/product import/SQLite/pytest/network/native campaign in HOST; worktree creation was the sole Git metadata operation outside HOST. No push, history rewrite, commit, old-proof cleanup or delegation. Fresh independent GLOBAL source+HOST review is REQUIRED; no self-GO or runtime authority. Dummy qualification and actual diagnosis remain unrun/NO-GO pending their gates.


## Tiny non-product qualification pre-HOST freeze — 2026-10-23

SOURCE+HOST ONLY at documentation fd0bc585d897818c66f3ae62d38c6a8eeda769b3 / executable c3b966beafbb9cb861428e59a81b56adb3ccd127. Existing executable/manifest and193+2 payloads unchanged. Fixed matrix12 cases/9s each/120s total, stop first unexpected failure. No actual Docker or product invocation; fresh independent Sol GLOBAL review remains mandatory. Frozen before HOST:

- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`
- qualify.py: `4cc05bbdca86b9908fe392315d631b93631ad001d9ad06e2cea828337003dc55`
- qualification-host-check.py: `8241c1cbd52289f1f295c685a52207ce2ad08bd475c07fef4cea945e89a071c7`
- QUALIFICATION.md: `392c8114fd9be0bcaab3dfe4e2aca74d7d15b663989f45472c3fdaa0523a2e9f`


### Tiny dummy final HOST receipt (documentation-only append)

Sole frozen matrix PASS12/12, 0.5057634850090835s, CPython3.14.7/debugTrue/optimize0. Receipt `/private/tmp/pab-dummy-HOST-vxvsd480/HOST-receipt.json` SHA256 `5011f482967e4bcece25b182cdfe2c9f96e2dcf05f79f9d1dd088b45eda08360`. All pre-HOST executable hashes remain unchanged;193 original payloads plus2 original derivative hashes reverified, source pin c3b966be/doc fd0bc585 preserved. No actual Docker/product dispatch. Full handback and remaining independent GLOBAL/ACTUAL gates are appended to HANDOFF.
