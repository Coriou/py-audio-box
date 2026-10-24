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


## r20 cohesive recipe repair successor — 2026-10-23

SOURCE + HOST implementation only; NO self-GO. Fresh T3 worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-cc139bcf`, branch `t3/repair-dummy-build-recipe`, reconstructed by resetting only this owned branch from starting `03937811454881681ab989329e9649038db44079` to candidate `7df2dab65c710730dd87c1e36cae310fb82fb211`. Executable commit `a9c604aff77cce87cefd81baa5787fbcc6d41a5e`. Parent/root/history and old candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r18-20261023` preserved; old candidate rechecked clean at exact7df2dab. No old lock access or release.

Read complete parent r20 DUMMY-BUILD-ACTUAL-AUDIT, persistent dummy GLOBAL-SOL-r19 and STORAGE-PROPORTIONALITY-ASTRA-r16 before repair (focused recovery of combined output). No applicable AGENTS found. Lightweight memory registry search supplied no relevant evidence. Scope is five executable/manifest files and append-only FREEZE/HANDOFF/QUALIFICATION. Repository manifest's193 payload entries and both existing derivative entries are identical to7df2dab and every current payload hash was checked. finite-driver.py and dummy-writer.py are byte-identical. All other tracked bytes outside this scope are unchanged. No policy/deadline/quota/gate/product assertion expansion.

Shared launcher helpers `base_reference`, `base_steps`, `base_recipe`, `validate_recipe` serve product preparation and dummy preparation/validation. FROM is deterministic `pab-local-base/<invocation-UUID>:<48hex-owner-token>`, a local repository/tag; full raw BASE remains a separate expected inspect ID, never a FROM or registry manifest digest. Both proposals disclose exact pinned-base inspect, fresh reference-filtered tag absence query, exact-ID tag argv, tag inspect argv, retained reference and expected ID. Dummy plan separately freezes existing build labels/argv and lifecycle commands. Existing legacy SOURCE/EXECUTABLE/DOCUMENTATION constants identify historical reviewed inputs; current actual program bytes are bound by updated PINS and plan inputs, never misrepresented as the new commit.

Later authorized adapter order under the existing lane: base inspect/ID/Linux/amd64/startup -> exact fresh tag absence -> journal NOT-SENT/SENT-UNKNOWN -> tag pinned full BASE -> acknowledged CONFIRMED -> tag inspect/ID/Linux/amd64/startup -> TERMINAL tag operation with retained base_tag object -> recipe/steps revalidation -> immediate second tag inspect with same identity/startup admission -> sole `--pull=false --network=none` build. Collision refuses before tagging; no overwrite path, retry, tag removal, floating base or pull/install added. Lost tag response remains SENT-UNKNOWN with exact argv retained and no build/downstream dispatch; it is handed back for exact reconciliation, never retried. This assumes the existing trusted daemon/lane boundary, without claiming atomic compare-and-set against a malicious concurrent tag writer. Tag/image/cache bytes remain excluded backend overhead. Product launcher still only proposes build work; its proposal now discloses all prebuild tag obligations too.

Complete fatal BuildKit response is classified only when capture is COMPLETE, exit1, leader reaped, descriptors closed, no group kill, captured stderr byte count matches, newline-terminated final stderr line explicitly starts `ERROR: failed to build: failed to solve: ` and contains a diagnostic. Then build journal is TERMINAL/result FAILED. Generic exit1, partial/nonterminal error, lost spawn/response, wait timeout, overflow or other UNKNOWN capture cannot qualify; one existing label observation does not establish absence/completion and never authorizes retry. Failure still returns false and preserves state/lane; this repair does not auto-release a failed run. Original r19 actual SENT-UNKNOWN journal, slots, once-marker, captures and reports are untouched; r20's separate terminal interpretation remains historical evidence, not rewritten success.

Focused HOST command actually run: `PYTHONDONTWRITEBYTECODE=1 python3.14 -B diagnostics/p-one-node/qualification-host-check.py --recipe-repair`. CPython3.14.7, assertion enabled. Fixed affected selection3 groups /13 available groups,9s/group,120s total; final PASS3/3 in 3.034564911999041s. It executes actual product preparation/recipe validation, actual corrected host-check group29 (replacing the flawed raw-FROM green assertion), actual dummy preparation/full context/plan validation, rawFROM/foreign-tag/plan-step/build-argv rejection, actual Qualification adapter through eleven finite failure subscenarios: collision, wrong tag ID, unsafe startup, lost tag ack, complete fatal solve, generic exit1, truncated terminal, wait timeout, wrong immediate-reverify ID, unsafe immediate-reverify startup and post-tag recipe tamper. All assert no downstream volume/create/start/readers and at most one build; UNKNOWN and terminal states are checked. Affected successful build/tag lifecycle traverses existing writer/two-reader retention/cleanup/release and asserts retained tag/full pinned ID. Finite exact fake argv queue has no fallback; subprocess/Real.spawn/system/killpg/fork are denied; real private files/pipes/capture/parsers/journal/publication run. Native PID1 identity remains synthetic. No full unchanged lifecycle campaign was rerun; unchanged meaningful tests remain in the harness and old receipts keep their evidence boundaries.

Final receipt `/private/tmp/pab-dummy-HOST-970m2p0t/HOST-receipt.json` SHA256 `ef2972bb751250473fc995c0e8ea28f4a8f2ad7d3cd8c17e5125a24653019a69`. It binds all seven current executable/manifest bytes listed below. Private roots retained0700/leaves0600, no runtime plan/authority or actual preparation CLI created outside HOST. First receipt `/private/tmp/pab-dummy-HOST-s5v99rnr/HOST-receipt.json` SHA256 `002d555f131d58a5cfa6ffcf2f3af0da5f2a5ebf7ee6a55ccf27fea68b18b349` is RED: intended timeout fixture's accelerated fake clock exhausted admission before build, leaving unconsumed commands. Earlier branches passed. Corrected to fake wait timeout at actual capture seam; this is HOST fixture repair, no real operational retry. Intermediate PASS receipts retained: ju_09cdp (corrected fixture),9unh8s_d (added affected success path),s50ggq8n (tightened final-line terminal classification and manifest contract),uj57v58i (added immediate reverify/tamper coverage). Final970m2p0t adds execution of corrected product HOST29. Each rerun followed an explicit changed fixture/assertion/source concern; no RED erased or production attempt repeated.

Exact executable/source hashes:

- `launcher.py`: `0ded93a96a7136112c163d546cb269d6fdcbff2b06e8e87fecc5db5e30616c29`
- `qualify.py`: `9908fd5109a717edfce011b72f037edd62f16aeef581035fdc1510a71fc6af23`
- `host-check.py`: `a78b2b0ba03dc6bc64bf42be5c1494b4dd94d38bb1aa37bc0c996cbc986f582d`
- `qualification-host-check.py`: `89e8d3c127625c24701333a84128665d96b1af1ffd3046ceaf2cc89bf082a19f`
- `source-manifest.json`: `1d97c4f899de7a7cafe2d2137c917d7af6c9a17c037022ee93a478bd19ce219a`
- `finite-driver.py`: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- `dummy-writer.py`: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`

AST parse and git diff --check passed. No real Docker queries/tags/builds/volumes/containers, product imports/entry/runtime identity, SQLite/DB, pytest, native crash campaign, network, actual dummy, execution plan, delegation, cleanup or old lock mutation. Only stdlib HOST temporary IO and authorized Git/source/document writes. No new runtime resource obligation. Local-tag native builder usability, actual initialization/stopped persistence/two-reader equality remain unqualified. P47 expected RED, historical PID229 UNKNOWN, all H/S/native/T2/platform/ext4/durability/release/activation gates unchanged.

Handback: return this exact executable plus documentation successor for fresh independent Sol GLOBAL SOURCE + retained HOST review. No implementer GO or actual authority. After that review parent controls any separately authorized bounded new dummy and actual-receipt gate before product. Old failed build reconciliation/lock release belongs exclusively to parent and is not performed here.


## r26 source and affected HOST pre-run freeze

Reconstructed exact ce70817437b4b24e0d35711e6450bf680f32e2d4 in owned t3-dc329cf6 only. Original immutable candidate retained. SOURCE+HOST only; independent global review and actual authority remain pending. Narrow extension: build-time image construction hygiene replaces only dedicated image /tmp (ordinary/hidden inherited entries and symlink) before seed COPY; no mounted-volume cleanup, packages, network, pull or runtime initializer. Strict initialized and finite PID1/dummy/193 payload bytes unchanged. BASE unchanged; no 3.14.7 runtime qualification.

Frozen affected selection: --r26, initialization, both context recipes/product HOST29, product+dummy lifecycle failure matrix, complete success, lost-start acknowledgement, bounded real-pipe capture failures. Existing fake adapter, maximum14 groups/9s each/120s total; no new framework. Stop first unexpected failure and retain RED; changed fixture/source reason required for rerun.

- launcher.py: `7d8abee3375440400f5db034b3b19fecc59894df94525d414e444a76399bdcd6`
- qualify.py: `df175ce081c1278e8d8e830c13a320f5345dc4163ecd7759a5518c1cbadbcdb6`
- host-check.py: `d2d92d63e36cabc0450adfb2bc36464bc7bbd3da2c464d27a40e47a6bd4d6952`
- qualification-host-check.py: `23123dc8239fe0b47ca35891c641d55a7573428f4e66259cc9e4f13b981c94e6`
- source-manifest.json: `63773d57d7f1f255b3dee55be1687f5975a4bedcff872bb47dd14cb8a6fa2c94`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`


## r26 source and affected HOST pre-run freeze

Reconstructed exact ce70817437b4b24e0d35711e6450bf680f32e2d4 in owned t3-dc329cf6 only. Original immutable candidate retained. SOURCE+HOST only; independent global review and actual authority remain pending. Narrow extension: build-time image construction hygiene replaces only dedicated image /tmp (ordinary/hidden inherited entries and symlink) before seed COPY; no mounted-volume cleanup, packages, network, pull or runtime initializer. Strict initialized and finite PID1/dummy/193 payload bytes unchanged. BASE unchanged; no 3.14.7 runtime qualification.

Frozen affected selection: --r26, initialization, both context recipes/product HOST29, product+dummy lifecycle failure matrix, complete success, lost-start acknowledgement, bounded real-pipe capture failures. Existing fake adapter, maximum14 groups/9s each/120s total; no new framework. Stop first unexpected failure and retain RED; changed fixture/source reason required for rerun.

- launcher.py: `3f03da9fee11c3156d6d921b4afbbc216514425fba9ba767c949cc2c6ba8021c`
- qualify.py: `398b78ff44c0b61c91e45b09bfea1e208c5df6b705e0d1d4480fa43b0058a378`
- host-check.py: `95fab428e795eaedcf49b110d8684bb804932d74da482f1f28250de6b1bb1fcf`
- qualification-host-check.py: `2bb03a515a1df77e483dddecfe789fbd264ac8bff48d29e69bcf088c6a9c9213`
- source-manifest.json: `63773d57d7f1f255b3dee55be1687f5975a4bedcff872bb47dd14cb8a6fa2c94`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`


## r26 source and affected HOST pre-run freeze

Reconstructed exact ce70817437b4b24e0d35711e6450bf680f32e2d4 in owned t3-dc329cf6 only. Original immutable candidate retained. SOURCE+HOST only; independent global review and actual authority remain pending. Narrow extension: build-time image construction hygiene replaces only dedicated image /tmp (ordinary/hidden inherited entries and symlink) before seed COPY; no mounted-volume cleanup, packages, network, pull or runtime initializer. Strict initialized and finite PID1/dummy/193 payload bytes unchanged. BASE unchanged; no 3.14.7 runtime qualification.

Frozen affected selection: --r26, initialization, both context recipes/product HOST29, product+dummy lifecycle failure matrix, complete success, lost-start acknowledgement, bounded real-pipe capture failures. Existing fake adapter, maximum14 groups/9s each/120s total; no new framework. Stop first unexpected failure and retain RED; changed fixture/source reason required for rerun.

- launcher.py: `3f03da9fee11c3156d6d921b4afbbc216514425fba9ba767c949cc2c6ba8021c`
- qualify.py: `398b78ff44c0b61c91e45b09bfea1e208c5df6b705e0d1d4480fa43b0058a378`
- host-check.py: `95fab428e795eaedcf49b110d8684bb804932d74da482f1f28250de6b1bb1fcf`
- qualification-host-check.py: `6ed07bcfbb16c8ecec7cd491e37c29416743b2b353af6b7376705449253b3df3`
- source-manifest.json: `63773d57d7f1f255b3dee55be1687f5975a4bedcff872bb47dd14cb8a6fa2c94`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`


## r26 source and affected HOST pre-run freeze

Reconstructed exact ce70817437b4b24e0d35711e6450bf680f32e2d4 in owned t3-dc329cf6 only. Original immutable candidate retained. SOURCE+HOST only; independent global review and actual authority remain pending. Narrow extension: build-time image construction hygiene replaces only dedicated image /tmp (ordinary/hidden inherited entries and symlink) before seed COPY; no mounted-volume cleanup, packages, network, pull or runtime initializer. Strict initialized and finite PID1/dummy/193 payload bytes unchanged. BASE unchanged; no 3.14.7 runtime qualification.

Frozen affected selection: --r26, initialization, both context recipes/product HOST29, product+dummy lifecycle failure matrix, complete success, lost-start acknowledgement, bounded real-pipe capture failures. Existing fake adapter, maximum14 groups/9s each/120s total; no new framework. Stop first unexpected failure and retain RED; changed fixture/source reason required for rerun.

- launcher.py: `dbc5d32afb11bfc4254deaa6945f94e8da7b5e6f65b20d0b4b36eaac2267eb7d`
- qualify.py: `6999a3e4aaeece2fdd9abd3a4f23d8fcd71dc5e6773a0ae69be20b59dce91626`
- host-check.py: `1b80c3d500dd58fdc0892aaaffc58eb570915965062ce0b0aa17e85f78c0fff0`
- qualification-host-check.py: `48c32c03cdf0c51a4a0394d350120bed284f9c4d12ec257243766e4b015053ec`
- source-manifest.json: `63773d57d7f1f255b3dee55be1687f5975a4bedcff872bb47dd14cb8a6fa2c94`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`


## r26 cohesive hygiene and observability successor — 2026-10-23

Implementation handback for independent Sol GLOBAL review; no self-GO or actual execution authority. Executable commit `84477f38b4b6de52994e21e6cf6f3bafec801db6`, parent exact `ce70817437b4b24e0d35711e6450bf680f32e2d4`. Fresh owned T3 checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-dc329cf6`, branch `t3/diagnostic-capture-host-fix`; reset only that initially clean branch from3683745 to the immutable candidate. Original t3-cc139bcf independently rechecked clean at ce708174. Old candidates, objects, lane and all historical/private evidence preserved; no old lock access/release. Read full controlling root r26, r23, r21 and Astra r16; no applicable ancestor/scoped AGENTS. Memory registry used only for evidence-tier discipline, current source/receipts supply facts.

Shared base_recipe used identically by both prepare paths now removes/recreates ONLY dedicated image /tmp before seed COPY, including inherited ordinary/hidden entries and a symlink at that path. New directory0700 numeric10001:10001; seed COPY numeric10001:10001/0600. Isolated assertion-enabled build Python asserts real directory, exact seed-only names, regular seed, exact modes/owner and exact seed bytes. This explicitly extends the OLD metadata-only contract with narrow image construction hygiene. No packages, runtime root initializer, mounted-volume repair/cleanup, junk allowlist, install, network or pull. Strict initialized/census, finite-driver, dummy body/counters and193 payload mappings/bytes plus2 original derivative mappings/bytes remain unchanged and rehashed.

Dummy and product/reader lifecycle callers use bounded start --attach capture with exact binary stdout/stderr, per-stream lengths and incremental SHA256 in complete receipts. Default call still rejects completed nonzero commands; only explicit lifecycle attach/wait callers defer that gate. Completed attach (including nonzero) is followed by separate bounded wait and one reserved stopped inspect BEFORE CLI/workload status acceptance. Full inspect is captured; acknowledged full ID, image/owner/name/config/mount binding and stopped state are checked. STOPPED objects retain full inspect including ExitCode/OOMKilled/Error; start becomes TERMINAL only after proven stop. Nonzero attach, wait CLI, wait workload, OOM or daemon Error still STOP before readers/product/cleanup. Missing OOM/Error cannot pass acceptance. The stopped predicate establishes quiescence separately from failure acceptance.

Original phase/global limits stay fixed. Within each lifecycle phase,12s is reserved beyond attach for wait/inspection/local reap; wait<=4s ends before protected last6s; inspection ends within original phase. Dummy attach<=35s while independent PID1 hard alarm remains30s; product finite alarm/dispatch schedule unchanged, reader phase still<=60s. Budget exhaustion refuses further dispatch. Attach timeout/overflow/lost acknowledgement kills/reaps only local CLI when owned, preserves capped prefixes and UNKNOWN remote state, with no retry/adoption/reconciliation or reader/cleanup. Inspection failure or exhaustion never fabricates STOPPED. Existing non-start mutation reconciliation remains unchanged. No daemon logs enabled.

Affected command: `PYTHONDONTWRITEBYTECODE=1 python3.14 -B diagnostics/p-one-node/qualification-host-check.py --r26`. Host interpreter `3.14.7`, assertion enabled. Final PASS7/7 selected groups from15 available,9s/group/120s total. Final selection supersedes earlier pre-run prose saying14 available. Groups cover inherited ordinary/hidden/symlink clean-seed model and residual rejection; both actual prepare contexts and product HOST29;24 product+dummy failure subcases (completed attach nonzero/exact streams+hashes, wait CLI0/workload1, wait CLI nonzero, inspect failure, timeout, overflow, lost ack, pre-attach and pre-inspect exhaustion, OOM, daemon Error, wrong full ID); existing affected product real-pipe transport04/05/06/07/09 and full lifecycle23; full dummy success; lost-start HOLD; binary/overflow capture. Real private filesystem/pipe/capture/parser/journal/publication code runs against exact finite fake argv, guarded against real subprocess/Real.spawn/system/killpg/fork. These are HOST models, not actual Docker build/copy-up/remote timing qualification; cleanup assertions execute only against fake commands and owned HOST fixture trees.

Retained progression:

- `/private/tmp/pab-dummy-HOST-y0fu40dn/HOST-receipt.json`: PASS 6 groups/4.747253s; SHA256 `b2914117e7bfcf7b32e4c0844ea07f623109e6f5ccedf5941ee2a48cb0a88000`.
- `/private/tmp/pab-dummy-HOST-l8jfrzrg/HOST-receipt.json`: RED 3 groups/2.434604s; SHA256 `7c4fab015fc69556a755b8bfdd4b583c094f3ec884034273f2fe482d911e00be`.
- `/private/tmp/pab-dummy-HOST-e_rzh_w4/HOST-receipt.json`: PASS 7 groups/6.572917s; SHA256 `2b995609ce5b88d44e0fae3dffb559ab6ccb9cdb7e514e946695defeac89f1a6`.
- `/private/tmp/pab-dummy-HOST-wxtoq4sw/HOST-receipt.json`: PASS 7 groups/7.063593s; SHA256 `aa6c9953ab5403f94058b83e028036e52bd774548f28082d5175f457432153f2`.

Initial6-group PASS predates added incremental hashing/OOM/error/deadline/product transport coverage. Expanded RED l8jfrzrg stopped first at lifecycle_failures: inspection-deadline fixture truncated expected commands AFTER Fake copied the queue, leaving an unconsumed inspect. Moved truncation before Fake construction; no production behavior weakened. e_rzh_w4 then PASS7. Final wxtoq4sw follows explicit full-ID equality and required OOM/Error acceptance checks with wrong-ID fixture, PASS7. All receipts retained, no RED erased; final receipt hashes match current executable bytes. Each source/fixture revision had a pre-run freeze entry. No unchanged full campaign rerun. Original r20/r22/r25 receipts remain historical and untouched.

Final SHA256 pins:

- `launcher.py`: `dbc5d32afb11bfc4254deaa6945f94e8da7b5e6f65b20d0b4b36eaac2267eb7d`
- `qualify.py`: `6999a3e4aaeece2fdd9abd3a4f23d8fcd71dc5e6773a0ae69be20b59dce91626`
- `host-check.py`: `1b80c3d500dd58fdc0892aaaffc58eb570915965062ce0b0aa17e85f78c0fff0`
- `qualification-host-check.py`: `48c32c03cdf0c51a4a0394d350120bed284f9c4d12ec257243766e4b015053ec`
- `source-manifest.json`: `63773d57d7f1f255b3dee55be1687f5975a4bedcff872bb47dd14cb8a6fa2c94`
- `finite-driver.py`: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- `dummy-writer.py`: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`

AST parsing and git diff --check passed.193+2 mappings and actual bytes verified unchanged against ce708174, finite-driver/dummy byte equality checked. All four new HOST proof trees independently checked0700 directories/0600 regular leaves/no symlinks; final complete success evidence uses fixture identity only. Executable change is precisely five allowed source/manifest paths; documentation successor changes only FREEZE/HANDOFF/QUALIFICATION.

Runtime BASE remains `sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb` pending parallel Astra explicit disposition. No3.14.7 container qualification is claimed; historical3.14.8 diagnostic identity is not that gate. No actual Docker query/build/tag/container/volume, native product import, DB/SQLite, pytest, pull/provisioning, old object cleanup/lock release, network, actual plan/authority creation or delegation occurred. No runtime resources acquired. P47 expected RED, historical PID229 UNKNOWN and all native/T2/platform/ext4/durability/release/activation gates remain unchanged. Parent owns independent Sol global review, runtime disposition and any separately authorized future execution; this handback advances no operational gate.

## r36 cohesive SOURCE + affected HOST pre-run freeze — 2026-10-24

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-48db44e1`, reconstructed from `c06e2f49b48264b51d65b0a9d1506b973ee4f839`. Old t3-dc329cf6 and every old proof/lane/object remain untouched. Controlling full parent r36, r35 actual, r28 global and Astra r27 read. Selected existing Linux CPython3.14.8, unchanged42 locks; no 3.14.7 admission requirement. No actual authority.

Fixed invocation: `python3.14 -B diagnostics/p-one-node/qualification-host-check.py --r36`; five affected groups in order: entry_bootstrap_and_selection, semantic_identity, failed_writer_retention, r36_product_contract, lifecycle_failures. Available19 groups; <=9s/group, <=120s total, first unexpected RED stops. No full/dummy/native/product campaign. External process, spawn, fork and signals are guarded; stdlib synthetic packages only. Expected pytest1/driver0 remains tested by original actual-writer scheduling seam and original scripted lifecycle. Receipts retained under fresh0700 HOST roots with0600 files; never overwrite RED. Changes after RED require new pins and a stated affected rerun.

- `entry.py`: `f4c4862b000fc458e901b8f31f7c3e081550e35765e7d1cd04a19615a73443b9`
- `runtime-identity.py`: `0cb0d4a3c4fea5463ed675cdbcfb4af66f85ee03e91218fc68bf745be0967dd5`
- `identity_semantics.py`: `cd8655eed473792b12beb96f250cd22f3938ade45151509a2af4092ad93f1824`
- `finite-driver.py`: `10dd30e393bfe79553d9ef3170c31662ab2774847f1bab4028dc4657e07e7e0f`
- `launcher.py`: `3e2c7997ce9e0d9bbd1398176b6c0c4ea1b29f46153b1991f7366a29050a17df`
- `host-check.py`: `88514eee4d48b7a89adc6627bf28814a6df1b312a05a01c659ac684691d585d9`
- `qualification-host-check.py`: `4c236132a9127ddb592eaccea92eee0c7071e65a563f65ff4dedee69c49e2c65`
- `source-manifest.json`: `d6b089f35b2477ace3ec54560bc393afb83dfbe2058546a139f91a3c99832f8e`

### r36 first RED and changed-only repair pre-run freeze

Retain `/private/tmp/pab-dummy-HOST-c_efnjkg/HOST-receipt.json`: bootstrap/identity PASS; failed_writer_retention RED due cap fake queue expecting commands after correctly refused overflowing capture. Fix only that queue; add identity malformed-field/type and actual checked-command function fixtures, strict command scalar types and failed-parser raw diagnostics. No bootstrap rerun. Fixed `--r36-repair`: semantic_identity, failed_writer_retention, r36_product_contract, lifecycle_failures, <=9s/group <=120s total first RED stop; no other campaigns.

- `entry.py`: `f4c4862b000fc458e901b8f31f7c3e081550e35765e7d1cd04a19615a73443b9`
- `runtime-identity.py`: `5639171dbba9a8c9affabc1cc80605c30d6b5909a52f5dd4fcfe7f790578c861`
- `identity_semantics.py`: `fd2d1f6d40699aa258d9016996037e8f61875f9302b21f8a6700cac935f707de`
- `finite-driver.py`: `10dd30e393bfe79553d9ef3170c31662ab2774847f1bab4028dc4657e07e7e0f`
- `launcher.py`: `3e2c7997ce9e0d9bbd1398176b6c0c4ea1b29f46153b1991f7366a29050a17df`
- `host-check.py`: `88514eee4d48b7a89adc6627bf28814a6df1b312a05a01c659ac684691d585d9`
- `qualification-host-check.py`: `a0f793bddd04e476f9a6b677ccaaef8868310a6c68cb6b5a646f24aff9d8f613`
- `source-manifest.json`: `176e3b0b5b48cbdd818ac66591dd912e5db20cfc32731e9ab9c7ad7c1d19dc1c`

### r36 second RED and final affected pre-run freeze

Retain `/private/tmp/pab-dummy-HOST-9dzin5_g/HOST-receipt.json`: semantic identity, all seven failure-transport cases and product contract PASS; lifecycle_failures RED before commands because its old dummy preparation correctly rejects stale qualify.py pins. qualify.py is outside assignment and remains unchanged. Parameterize existing lifecycle fixture to run only affected product12 cases, without dummy preparation; no production pin bypass. Add explicit failed-writer cleanup refusal and preserve non-ASCII failed-command bytes before decoding. Fixed `--r36-final`: semantic_identity, failed_writer_retention, r36_product_contract, r36_lifecycle_failures; <=9s/group <=120s total first RED stop. Bootstrap earlier PASS unchanged. No full/dummy campaign.

- `entry.py`: `f4c4862b000fc458e901b8f31f7c3e081550e35765e7d1cd04a19615a73443b9`
- `runtime-identity.py`: `eb5d3a7dbdf717a219163bfeaaa9bb8616a0ad7c7b2ec315aa0eb8b24504c034`
- `identity_semantics.py`: `fd2d1f6d40699aa258d9016996037e8f61875f9302b21f8a6700cac935f707de`
- `finite-driver.py`: `10dd30e393bfe79553d9ef3170c31662ab2774847f1bab4028dc4657e07e7e0f`
- `launcher.py`: `348cc1097b267a734a763f8a56a940732abb8cb141c2f767744a91fcf835984c`
- `host-check.py`: `88514eee4d48b7a89adc6627bf28814a6df1b312a05a01c659ac684691d585d9`
- `qualification-host-check.py`: `98ec74e369d2f15f3aab628d4d3b5ce974f88dbfdf9e56fdf8c419f5b4482dad`
- `source-manifest.json`: `23cc4d3da39f9ec783efe4ed20ecf7ce5f015170f388351f286dfc0036412d43`

## r36 final executable freeze

Executable commit `f60176302113e21212d525526f81e27b1395d045`. No executable edits after final PASS. Full receipt history, hashes, final pins, scope and remaining gates are in the current r36 HANDOFF section. Documentation commit follows separately.

- `entry.py`: `f4c4862b000fc458e901b8f31f7c3e081550e35765e7d1cd04a19615a73443b9`
- `runtime-identity.py`: `eb5d3a7dbdf717a219163bfeaaa9bb8616a0ad7c7b2ec315aa0eb8b24504c034`
- `identity_semantics.py`: `fd2d1f6d40699aa258d9016996037e8f61875f9302b21f8a6700cac935f707de`
- `finite-driver.py`: `10dd30e393bfe79553d9ef3170c31662ab2774847f1bab4028dc4657e07e7e0f`
- `launcher.py`: `348cc1097b267a734a763f8a56a940732abb8cb141c2f767744a91fcf835984c`
- `host-check.py`: `88514eee4d48b7a89adc6627bf28814a6df1b312a05a01c659ac684691d585d9`
- `qualification-host-check.py`: `98ec74e369d2f15f3aab628d4d3b5ce974f88dbfdf9e56fdf8c419f5b4482dad`
- `source-manifest.json`: `23cc4d3da39f9ec783efe4ed20ecf7ce5f015170f388351f286dfc0036412d43`


## r53 B packaging/control runner SOURCE + stdlib HOST freeze — 2026-10-24

Reconstruction base `5d2fe5e1e7c40bc479f02f616f03da76ca8d886a` (exec `4e36ed22bf8edac67bde8affd0a0ca339212a6c6`), own t3-1521cb66 / prep/b-controls-runner. Parent accepts independent r52 combined SOURCE GO; actual remains HOLD. No delegation or applicable AGENTS. Scope is refreshed guard/control mapped hashes plus one new copied runner derivative, new runner/HOST/PLAN, and appends to the three EXISTING diagnostics/p-one-node handoffs (the requested docs/p-one-node path does not exist). No frozen source/proof/launcher changes.

Final affected HOST selection frozen before execution: source/inventory, separate campaign command/security binding, exact pre-fixture selection, ordered100/first-red/partial retention/exhausted deadline/residual-descendant refusal, unchanged finite capture binary/16MiB overflow/timeout against fake Popen/selector. Stdlib only; subprocess/run/system/fork/kill/killpg denied except explicit finite fake Popen boundary. Hard30s total; stop first unexpected failure. No runtime admission, pytest import/collection, product import or DB.

Earlier retained PASS receipts: `/private/tmp/pab-B-controls-HOST-lbdac641/HOST-receipt.json` (initial four groups) and `/private/tmp/pab-B-controls-HOST-fcxmscaz/HOST-receipt.json` (five groups after separate command binding/partial hash/residual check). Final run justified by added private umask, exhausted campaign deadline and residual-descendant HOST cases. No RED was erased and no actual was retried.

- `diagnostics/b-controls/runner.py`: `2191bf4807433404946551ae9959ac6351137029cd392a0e6adb94ab46f33cf7`
- `diagnostics/b-controls/HOST.py`: `5b1d709ca3e4aaf4a0a368c7ec764686fca183e61209182cdb59595de6b64862`
- `diagnostics/p-one-node/source-manifest.json`: `da8ec4f9e19e9d13c472978b7ceb1cf80978cbfb50a2af1364f6010cc5208e57`


### r53 final executable and HOST result

Executable `2a266f3f496351e98a2a410071a135e0a935d707`; no executable edits after final HOST PASS5/5. Final receipt `/private/tmp/pab-B-controls-HOST-glpw6zw3/HOST-receipt.json` SHA256 `fbe5be54a4aadb2c37b1c3d3ee5c2e47ad1f6817cf6bf90c00aa926b490d323c`. Source/private-inventory verification `/private/tmp/pab-B-controls-source-verification-3jewu83e/verification.json` SHA256 `2e2b102ce53b3b43f70a947ddcc308db6de6c8a1fe9d9f7bd9e138637d1cc545`. Full exact source pins/coverage/retention and limitations are in current r53 HANDOFF and PLAN. No pytest/runtime/Docker/product/DB/native work or old proof/lock cleanup. All100/P211/preservation and actual74 remain pending. Following docs-only commit does not self-approve this preparation.

## Explicit original-positive profile — pre-run HOST freeze (2026-10-24)

Reconstructed this own checkout from immutable candidate docs
228a54691eacac7b7837b030332080a456f48dfd (executable
2a266f3f496351e98a2a410071a135e0a935d707). Read full root Sol r54 and
Astra r44 before implementation. Scope: launcher, finite-driver, host-check,
manifest and these three handoffs only. No guard, target, model, assertions,
conftest, observer, entry, identity, lock or B100 runner change.

Before first run, freeze `host-check.py --profile-scope`: six existing/added
groups 23/26/29/30/32/33, <=33 available groups, <=300 seconds total,
<=9 seconds/group. Group32 executes actual writer orchestration with fake
external boundaries: diagnostic1 succeeds, diagnostic0 refuses, positive0
succeeds, positive1 refuses, and existing deadline failures refuse. Group33
covers closed profile/expected/command/terminal tampering, exact container
binding, unchanged readers, actual74 versus9, wrong node/owner/child,
missing/duplicate B tokens, waits/PASS absence, guard failure/owned state and
real private tar/hash/slot retention with fake reader commands. Groups23/26
retain legacy lifecycle success/failure,29 source copy without build,30 legacy
47-field parser. No actual pytest, collection, product imports, DB, Docker,
network, native or runtime campaign. Retain first RED and any motivated fix
receipt without retrying an operational stage. No new full campaign/framework.

First affected HOST receipt: `/private/tmp/pab-p-one-r16-HOST-1uo4ogqu/HOST-receipt.json`,
PASS six groups, no RED. Source inspection then found final host classification
still used the historical RED label for positive success; corrected before
handback. Added direct finite CLI dispatch/refusal and legacy requested-exit
binding checks. Same six-group frozen scope/caps will run once for these changes;
this is source/HOST verification, not an operational retry.

Second affected receipt: `/private/tmp/pab-p-one-r16-HOST-ycnqy5ds/HOST-receipt.json`,
PASS six groups. Final completeness check adds positive intent PENDING binding
and executes actual host final publication in group33 to verify the new final
classification and acceptance=false. Same bounded affected matrix is the final
verification run; both earlier PASS receipts remain retained, no RED occurred.

Final affected receipt: `/private/tmp/pab-p-one-r16-HOST-tcuxvmuo/HOST-receipt.json`,
PASS6/6,0.614528387s; SHA256
2574ce0df777eb04ed8313b46867af2fa097523b3470576d7d2b917e23078c5b.
Executable commit0c058776d641503434de1c5eccb05ecb7ca51fcd. No further
executable changes or HOST runs after this frozen final receipt. Independent
SOURCE/HOST review and every operational gate remain parent owned.
