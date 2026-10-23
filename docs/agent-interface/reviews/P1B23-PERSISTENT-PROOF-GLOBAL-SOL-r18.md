# P1B23 persistent proof — independent GLOBAL source + retained HOST review r18

2026-10-23. **Bounded SOURCE + HOST GO for readiness to prepare the r16 finite dummy qualification plan.** r17 F1/F2 are closed in the inspected source and affected retained HOST proof. No concrete blocking path was found in the complete successor lifecycle. This is not actual Docker qualification, permission to invoke the product writer, or product acceptance. A separately frozen non-product dummy writer, exact argv and usable reviewed recipe/image remain necessary before qualification.

## Authority, scope and exact freeze

Independent Sol low READONLY review. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r18-20261023` is clean at documentation `fd0bc585d897818c66f3ae62d38c6a8eeda769b3`, executable `c3b966beafbb9cb861428e59a81b56adb3ccd127`. Compared against executable `8364308c8f035bf65362247b201514a060c06730` / documentation `87cec9a3ff9d2b829c04aa096e930d021b4fae83`. Executable-to-doc delta is FREEZE/HANDOFF only. Sole write is this root report; no commit. Unrelated untracked root STT evidence is untouched. Checked applicable ancestry/scoped AGENTS inventory; none found. Memory registry search supplied no relevant evidence and supports no finding.

Read complete r17 GLOBAL F1/F2, controlling STORAGE-PROPORTIONALITY-ASTRA-r16, original brief and responses in HANDOFF/FREEZE, historical L1–L4 source/lifecycle reports and r13 global driver lessons; complete launcher, finite driver, harness, entry, observer and runtime helper. Inspected source/manifest and retained HOST receipts, including original RED and scoped successors. Used text/Git, stdlib AST, JSON/stat and hashes only. No candidate import, harness rerun, pytest, Docker, SQLite/DB open, runtime/product execution, fix, delegation, push, extraction or cleanup. No runtime resource was acquired.

| Frozen file | SHA256 |
|---|---|
| launcher.py | `f28c8785df17854ee120bffb4ff08bf5555e21e7dac316d56434d733ed640c80` |
| finite-driver.py | `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732` |
| host-check.py | `dec48fd7274e887f223f898c8e66a628fbdec7ff246a545217606e681141339c` |
| source-manifest.json | `e0a228c16b83f9acbc1f640582bd240d3a701d2128b4047812aa261314a66035` |

These independently computed hashes equal the final retained HOST receipt and r18 freeze. FREEZE SHA256 is `14525d6aa83427f08abc84ffde774b61e1707d751e4a1957c7fc1242e69364bc`; HANDOFF `08b9c66ba027fcfd2a04cea4c2ada31756172cc73c701cbb28960c23b2f417fc`. Historical prose saying “no commit created” describes preparation; the supplied final Git pins above govern this review. No executable/manifest delta occurs in the documentation commit.

Exactly six allowed paths differ from original e941c7a: launcher, finite-driver, host-check, manifest, FREEZE and HANDOFF. Independently hashed all193 original manifest payloads: zero mismatches; original `files` mapping equals e941c7a. Both derivative hashes match. Entry/observer/runtime identity/guard/product/tests/locks remain unchanged. The new driver is separately bound; neither manifest nor launcher hashes itself into a cycle. AST parsing of all six Python files and Git diff --check passed without executing candidate code.

## F1 — closed, with actual scheduling control flow exercised

`finite-driver.py:387–433` computes the child deadline immediately before dispatch as `min(time.monotonic()+1200, start+1260)`. Early prechecks cannot lend the unused60s to the child. Quiescence retains `min(start+1290, now+30)`. After confirmed quiescence, `post_deadline=min(now+60,start+1350)` covers monitor join, source verification, identity process, identity readback and terminal publication. Checks after source verification and before/after terminal write prevent late successful return. Failure publication uses the same deadline; an expired source/identity phase cannot receive a fresh60s. Independent1350s alarm remains unchanged.

Group32 now calls the actual `F.writer`, rather than supplying only a fake terminal to host routing. Its controlled clock and external/environment seams exercise five paths recorded in the receipt: early precheck at5 yields child deadline1205; child reaches that deadline and fails; early child finish at6 yields post deadline66; slow identity and slow source verification fail with no terminal publication attempt; publication taking61s returns failure; normal path returns0. Quiesce bounds are asserted. This closes the r17 missing scheduling evidence. Capture/identity IO, admission, monitor thread, signals and slot IO are fixtures in this group; it proves scheduling/control flow, not native timing or PID1 execution. Actual capture/private slot IO is separately exercised elsewhere.

Cooperative deadline checks do not interrupt blocked filesystem syscalls. The overall alarm remains the ultimate source bound; kernel/daemon stalls remain honest residual uncertainty. A terminal slot written across the post deadline does not authorize retention success: writer returns nonzero and host requires exact stopped writer exit0 before readers. No product deadline/clock/assertion was changed.

## F2 — closed across both capture implementations

`launcher.py:95–160` takes both original pipe objects immediately after spawn, before spawn receipt, selector construction, set_blocking or registration. Setup remains inside the same exception/finally lifecycle. Selector-close failure cannot bypass stream closure. Failure attempts one owned group SIGKILL independently of leader liveness, then bounded leader wait; completion is UNKNOWN with `remote_completion=false`. Both pipes and captures are explicitly closed.

`finite-driver.py:282–314` registers ownership callbacks for both pipes before selector setup, which now sits inside the kill/reap exception boundary. Constructor/first-registration/set_blocking failure attempts one group kill and0.1s wait, rethrows the original exception and exits the stack closing both original pipes. Writer failure/quiescence remains separate; no local signal result means daemon completion. Remote capture has no spawn-publication seam.

Retained group09 includes original select failure plus host spawn-publication/selector-constructor/first-register/set_blocking and remote constructor/first-register/set_blocking failures. Fake leaders are already exited with open pipes. Every setup fixture checks the two original objects closed before rescue; host asserts exactly one kill and bounded reap/UNKNOWN; remote records exactly one kill and wait `[0.1]`. The receipt contains these pre-rescue observations. These are real pipe/selector/private-file paths with fake process control, not destructor-based closure or a fake closure oracle.

## Retained HOST continuity and global coverage

Final receipt `/private/tmp/pab-p-one-r16-HOST-5xxexl3z/HOST-receipt.json` independently hashes to `fa8ef7b0ee9c2e85845bda858f7ce5739394199aab8dd9e83fc9d6312fd3461d`: PASS7/7,0.589027851s, groups09/23/26/28/29/31/32. It binds all four current hashes. Fixed limits remain32 groups,9s/group,300s total; CPython3.14.7/debug true/optimize0. This is HOST-ONLY / FAKE EXTERNAL COMMANDS / NO PRODUCT CLAIM.

Independently rechecked retained history:

| Root suffix | Receipt SHA256 | Result |
|---|---|---|
| 1hn1pii1 | `127154636fdda662792c7b0a51f9530a3d5e5f061ac8fa457d16468280b88a70` | initial32, RED03/16/17/23–28 |
| lxq71z25 | `cda48d5d7f509c1f1570d946aa276bb2fdd8aa9a69a05139811233eb998046d7` | scoped12/12 PASS |
| nhrnd5lv | `1294fb4a203855063d5b5b4b1de0201ce56fbef9a475c9c7649b0449c69e30b7` | scoped14/14 PASS |
| xgbech_5 | `5fa60b344bcbd14efdee07f23d320750a1c64b093b16dc83311423b3e8083a40` | first r18 scoped7/7 PASS |

All five roots are0700; inspected regular leaves are0600. Final invocation stdout/stderr hashes equal FREEZE: `ff1f3f4de61dd06ea20d03380a374718981c9dff8705976aba48424d068d33eb` / `0365ec4efccdc43fa8b90cab4adf3b5322414bdfaba5989d2b01aa74bee318df`, both0600. Injected final-publication failure is expected in stderr. No receipt was regenerated or overwritten by this reviewer.

Causal reuse is valid within its scope: portable slot writes/GID repair explain12 scoped cases; original18/20 were wrong early GID rejections and corrected intended-error fixtures are retained; protected retention/helper deadlines and global additions explain14; F1/F2/reserved supervisor accounting explain latest7. The first r18 green precedes the tightened expired-publication guard and stronger scheduling assertion, justifying the final rerun. This is a reviewed union of historical and affected-path results, not a current unchanged32/32 execution.

Complete harness coverage includes private IO/umask/stable reads/slots; real binary pipes/caps/exited leader/spawn/read/select/write/close/reap failures; optimization/startup/resource/volume/UID admission; host space/logical/export/census failure and pressure; actual archive/hash readers and malformed/duplicate/escape/link/UID/truncated/map-mismatch inputs; complete synthetic handback, lost create/start acknowledgements, stopped-writer failed export, remove uncertainty and final-publication failure; actual source copying, structural47 rows, namespace cleanup and supervisor pressure. Latest full success/failure/source-copy paths execute changed code. Unchanged case bodies and corresponding unchanged policy seams support earlier reuse; earlier hashes are not represented as current hashes.

Fake.spawn is a finite exact argv/cwd/stdio/session/close_fds queue without fallback. Matrix guards real subprocess entry points/os.system/killpg. Real IO/parsers/lifecycle functions run; fake Docker terminal/47-field rows remain synthetic, and command expectations partly share production helpers. Thus routing and refusal evidence is meaningful, while Docker CLI syntax, image behavior, native fork/signals and product truth remain unqualified.

## Cohesive L1–L4 and policy assessment

L1 source closure remains numeric COPY ownership of private source/seed plus build-time `/tmp` metadata preparation for10001:10001/0700. No runtime root initializer, host bind or privileged repair. Actual FROM resolution, source traversal, image build and empty-volume seed population still require dummy receipts.

L2 uses one ordinary fresh `local` persistent volume, empty Options, honest name/creation/labels identity; no immutable-volume-ID or quota claim. Volume outlives finite writer stop. No volume/image deletion, restart, auto-remove, TTL or sleeping keeper. Export failure preserves original volume and stopped containers.

L3 uses finite PID1, unchanged entry once, independent watchdog/monitor and post-test private-namespace cleanup with no competing product reaper. Exact stopped exit0/terminal binding precede sequential same-image RO archive/hash helpers. Helper lifecycle<=60s, retention<=180, reconciliation<=120, final publication<=30; host supervisor cutoff1770 protects final30 within1800. F1/F2 now complete the reviewed phase/capture closure. A stopped/killed local worker or CLI remains UNKNOWN; detached sessions/remote work are not claimed reaped by a local kill receipt.

Logical1GiB includes two512KiB slots,64MiB leaves/200000 entries and bounded driver streams/metadata. Census0.5s plus0.5s wait, monitor failure/pressure stops, host/daemon6GiB admission and2GiB pressure checks remain. Export2GiB and invocation host4GiB accounting are preserved. Worker now receives4GiB minus16MiB; supervisor publication receives the reserved16MiB and charges encoded bytes before writing. This closes r17's separate supervisor accounting seam conservatively. No second4GiB allowance is added. Image/cache/backend overhead, deleted-open bytes, rapid growth, concurrent use, allocation/metadata and blocked IO remain physical residuals; no hard physical/inode quota or durability is claimed. No new quota gate is warranted.

L4 retains token192/UUID intent and NOT-SENT/SENT-UNKNOWN/CONFIRMED/TERMINAL journal, one bounded discovery after lost response without mutation retry or negative-absence inference, full-ID config binding/removal and foreign/collision refusal. Partial acquisition/uncertainty blocks the lane. Release requires retention, terminal sent operations, exact full-ID absence and owner/inode equality. Pending/torn/missing final state, failed publication or worker failure cannot become success; separate supervisor COMPLETE/exit0 is required. Successful retained volume is intentional evidence, not authorization for accumulation.

Archive is binary, bounded, safe/unique and compared for every regular hash against an independent quiescent RO map; no extraction/DB reopen/docker cp. Source checks, assertion admission, runtime identity and original47-field/expected-RED requirements remain intact. No concrete additional blocker was found when considering success and failure paths together.

## Handback and remaining execution gates

Parent may now prepare the concrete bounded dummy qualification under r16: separately freeze a tiny **non-product** finite UID10001 writer, exact argv, recipe/image and authority; at most one source/metadata build<=300s and one writer followed by two sequential RO readers, whole qualification<=600s. Verify actual initialization/readability, security/mount identity, detached-wait independence, stopped persistence, binary archive/map equality and exact retained inventory. Stop on first failure without repair/retry. HOST already covers export failure; do not deliberately strand another actual volume to repeat it.

The current driver offers product writer/archive/hash modes only. Its `writer` launches the product entry and **cannot serve as dummy qualification**. This review does not invent an approved dummy executable or product authorization. Review the actual dummy receipt independently before any separately authorized ONE product diagnosis.

P47 remains expected RED, requiring independent actual caller/runner offsets, exact type/original-exception, owner/enrollment/raw18944/empty-phase, EAGAIN, guard/call-teardown and unreached-assertion binding. Historical PID229 remains UNKNOWN. No actual Docker/dummy/product result, H/S construction, native/installed/platform/ext4/durability, T2, release or activation gate advances. Candidate/history/private evidence unchanged; sole write this report, no commit and no acquired cleanup obligation.
