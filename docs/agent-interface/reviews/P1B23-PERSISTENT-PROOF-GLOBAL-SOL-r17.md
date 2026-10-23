# P1B23 persistent proof — independent GLOBAL source + retained HOST review r17

2026-10-23. **NO-GO for dummy lifecycle qualification on this freeze.** Two concrete source defects remain in the finite phase/capture lifecycle. Repair them together within the existing six-file boundary, preserve the retained RED history, and obtain fresh independent review of the resulting source and affected HOST proof. No new physical-quota prerequisite, backend investigation, product invocation or H/S gate is proposed.

## Scope and pins

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r16-20261023`, clean at documentation `87cec9a3ff9d2b829c04aa096e930d021b4fae83`; executable `8364308c8f035bf65362247b201514a060c06730`. Sole write is this root report; no commit. Root contains unrelated untracked STT evidence, left untouched. No applicable AGENTS file was found in the candidate inventory. Memory lookup provided no task-specific evidence and supports no finding.

Read the entire controlling STORAGE-PROPORTIONALITY-ASTRA-r16 policy, historical BLOCKED-PIPE source L1–L4/lifecycle decisions and S-DRIVER-GLOBAL-ASTRA-r13 lessons; complete six candidate files, entry, observer and runtime helper. Reviewed actual cohesive functions and fake boundary, rather than treating handoff claims as proof. Used text/Git, stdlib AST and hash/JSON inspection only. No candidate import, harness rerun, pytest, Docker, SQLite/DB open, product/runtime execution, fix, delegation, push, extraction or cleanup. No runtime resource was acquired.

Executable/current/last14-case receipt hashes agree:

| File | SHA256 |
|---|---|
| launcher.py | `575867084bffa9507fb9c5a014bd5750661082c989bc80e0db1103cf014d77b2` |
| finite-driver.py | `03a76ee0a9776ffcbf529cadd632ea4a15638b9799b62ba2f8e0a25b38324fdf` |
| host-check.py | `55b41a99b4d8f171ac41b4860bd76c2fe347a0fd9e56876658dc8033b6953c25` |
| source-manifest.json | `7b447a9c4cf6958c11819eed6fe55e2177156bf6ce9d19c0a9a50d71080bead2` |

The documentation commit changes FREEZE/HANDOFF only relative to the executable commit. Current FREEZE SHA256 `4ba121ad583c77bf797ffe2f490fc5a7e5d9f89fc369b35c6babdcfb7b14bb7c`; HANDOFF `816d50d5ed450c76e3dd9d7978f095cc1151d80fbe1eaf42d8c5c05c187cbfca`. Base-to-candidate delta is exactly the six allowed paths. Independently hashed all193 original manifest payloads: zero mismatches, original `files` mapping identical to e941c7a. Thus entry/observer/identity/guard/test/product payload preservation is established for this pinned tree, not inferred from the newer parent tree. Both derivative entries match; finite-driver bytes are separately bound, with no manifest self-hash cycle. AST parsing passed without executing candidate code.

## F1 — phase maxima are cumulative boundaries, not individual allowances

**Blocking source contract violation:** `finite-driver.py:343–346,365,377–385,395–404`.

The policy preserves an entry allowance of1200s, remote prechecks60s, cleanup30s and postchecks60s, within1350s. The child capture instead receives `start + 1260`, regardless of its actual launch time. If prechecks finish at start+5, the entry can run approximately1255s. Nothing in capture supplies the missing `min(child_start + 1200, start + 1260)`. The1350s watchdog bounds the whole driver; it does not enforce the1200s child maximum. This is the same early-completion lending error already repaired for host retention, still present remotely.

Likewise, after an early child completion/quiescence, identity-after receives `start + 1350`, potentially lending it most of the unused test phase instead of at most60s. Source verification before that call is also outside a phase-local60s check. Quiesce already correctly uses `min(start + 1290, now + 30)`; apply that principle to the child and complete postcheck/publication phase, preserving cumulative boundaries and all existing product deadlines/assertions. Do not solve this by changing the child allowance or loosening the policy.

HOST case32 exercises only writer's non-PID1 refusal. The full scripted success fakes writer output/terminal state and never traverses writer's successful phase scheduling. No retained receipt tests early prechecks plus a child near1200s, or early child completion plus slow postchecks. Add bounded clock/external-boundary fixtures of the actual scheduling path; no real product, sleep campaign, fork or Docker is required. This is a source-derived counterexample, not an observed overrun.

## F2 — post-spawn setup failures lose ownership of capture descriptors

**Blocking all-path capture closure gap:** `launcher.py:93–101,139–153`; related remote seam `finite-driver.py:282–289`.

Host call spawns successfully, then publishes its spawn receipt and constructs the selector **before** adding either process pipe to `streams`. A failed spawn-receipt publication or selector constructor enters the exception handler, attempts the owned group kill/reap, then closes only capture files: both process pipe streams are absent from the cleanup list. Failure in first-stream set_blocking/register similarly leaves stderr outside that list. `descriptors_closed = all(...)` can report true because it checks only the incomplete list. Object destruction is not explicit, verified pre-rescue closure; the retained fake keeps the process object alive and would expose this path if injected.

Remote capture has the analogous acquisition ordering: selector construction and stream setup precede its try/except; callbacks are added one stream at a time. Selector construction or first registration failure can leave pipes unregistered for cleanup and skip capture's bounded group-kill/reap branch. Writer's later namespace quiesce is useful, but does not make this capture API all-path closed or test its fd ownership.

Acquire ownership of both returned pipes immediately after spawn, before any fallible publication/selector setup, and encompass setup in the same bounded failure lifecycle. Preserve one group-kill attempt even with exited leader, bounded leader reap, pending/UNKNOWN semantics and no inference of remote completion. Extend the existing capture scenario group with spawn-publication, selector-constructor and first-registration failures, checking both original pipe objects before fake rescue. Do not add a generic framework or broaden runtime scope.

Cases07–12 currently cover spawn, second capture-open, read, selector.select, capture-write/close and wait failures. They do not cover these post-spawn acquisition windows. Their green fd observations are valid for exercised paths only; they cannot close F2.

## Retained HOST evidence and coverage

Independently read/hash-checked the three HOST-receipt.json files under `/private/tmp/pab-p-one-r16-HOST-…`; roots0700 and regular leaves0600. No receipt was regenerated:

| Root suffix | Receipt SHA256 | Result |
|---|---|---|
| 1hn1pii1 | `127154636fdda662792c7b0a51f9530a3d5e5f061ac8fa457d16468280b88a70` | initial32, RED03/16/17/23–28 |
| lxq71z25 | `cda48d5d7f509c1f1570d946aa276bb2fdd8aa9a69a05139811233eb998046d7` | scoped12/12 PASS |
| nhrnd5lv | `1294fb4a203855063d5b5b4b1de0201ce56fbef9a475c9c7649b0449c69e30b7` | scoped14/14 PASS |

Recorded elapsed0.4413/0.6054/0.8182s; fixed<=32 scenario groups,9s/case,300s total. Final14 bind current four executable/manifest hashes. Earlier greens belong to earlier hashes; source inspection supports reuse only where unaffected. Initial18/20 were early GID rejections, not intended cap/monitor coverage; repaired receipts explicitly exercise intended errors. The portable slot repair and final phase/reader changes explain the scoped reruns. This is not a final unchanged32/32 run.

Fake.spawn uses a finite exact argv queue, exact cwd/stdio/session/close_fds checks, no real fallback; real subprocess entry points are forbidden during matrix execution. Actual private IO, pipes/selectors, capture, slot, context copy, archive/map parser, lifecycle dispatch and handback functions execute in HOST. Full success uses synthetic terminal/47-field rows; command construction partly shares production helpers, so it is routing evidence rather than independent Docker syntax truth. All recorded pre-rescue pipe observations are closed (12/7/10 observation groups respectively). F2 explains the untested limitation.

Coverage groups inspected: private APIs/umask/stable reads/slots (01–03); capture binary/caps/exited leader/error/reap (04–12); optimize/startup/resource/volume options/UID (13–16,27,32); host space/logical/export/census error and pressure (17–20); actual readers and binary safe tar/map malformed/duplicate/escape/UID/truncation/mismatch (21–22,26); full synthetic success/lost volume-create or writer-start ack/stopped-writer failed export/remove uncertainty/final publication (23–28); actual source context (29), structural47 rows (30), namespace cleanup and supervisor pressure fixtures (31). These are meaningful global HOST seams, with the F1/F2 gaps above. No product acceptance follows from fixture-generated writer completion.

## Cohesive policy and lifecycle assessment

The controlling r16 policy replaces the unsupported physical-quota gate. This candidate honestly selects ordinary local persistence, empty options and name/creation/labels identity; no immutable-volume-ID or physical/inode/metadata quota claim. Numeric COPY ownership and build-time `/tmp` metadata operations address L1 in source. Actual base resolution, image build/COPY traversal and empty-volume UID10001 seed population remain unqualified; no root/privileged runtime initializer or host bind is selected. Those are required dummy qualification observations, not a reason to invent another backend gate.

L2 is materially addressed: one retained named volume outlives finite writer stop; no volume/image deletion path, restart, auto-remove, writer exec, TTL or sleeping keeper. Export failure retains stopped containers and volume. L3 has finite PID1, one unchanged entry, independent alarm/monitor, private namespace post-test reap, no competing product reaper, stopped-state checks and sequential RO helpers; F1/F2 prevent full closure. Killing a local CLI/worker remains explicitly UNKNOWN, never remote completion. The real fork/signal/daemon behavior is source-only, not qualified by fixtures.

Logical1GiB includes two512KiB slots;64MiB leaves/200000 census entries,0.5s census plus0.5s wait, bounded driver writes and16MiB captures/maps are explicit. Export2GiB and host4GiB write counters include preparation/captures/slots/journals; image/cache/backend overhead is disclosed separately. Host/daemon6GiB admission and2GiB pressure stops are observations; unknown topology is shared conservatively. No hard physical ceiling is established against deleted-open bytes, concurrent use, rapid growth, IO stalls or metadata. The separately budgeted supervisor publication is a small additional accounting seam that should be folded into the invocation budget when touching this lifecycle; current repeated slot charging is conservative but is not a formal shared counter with the supervisor.

L4 is materially addressed: token192/UUID intent precedes Docker mutations; NOT-SENT/SENT-UNKNOWN/CONFIRMED/TERMINAL journal, one bounded discovery after lost response without retry/negative-absence inference, full-ID binding and removal, foreign/collision refusal and exact lane owner/inode checks. Uncertainty holds the lane; no name cleanup or destructive reconciliation. Successful release requires retained archive/map, terminal operations and inventory. Final slot/readback plus raw worker success and separate supervisor completion prevent accepting a pending/failed publication as success. Partial acquisition stays blocked rather than being adopted. These are source controls, not current daemon facts.

Readers run only after writer stopped confirmation, are same-image RO and sequential, and stop on helper uncertainty. Tar is binary/stream bounded, safe/unique and all-regular hashes must equal an independent quiescent map; no DB open/extraction/docker cp. Retention cannot establish physical durability or power-loss recovery. API inspection finds the old Path.open(opener) defect removed, explicit optimization guards present, exited-leader group kill retained, and old stale-success final-state defect materially addressed. F2 is a distinct remaining acquisition window, not grounds to discard the valid earlier HOST results.

## Handback and remaining gates

Repair F1/F2 cohesively and exercise their affected seams within the existing bounded scenario groups, preserving original receipts/source pins and product payloads. Re-freeze exact executable hashes after HOST; docs-only commit must preserve those bytes. No harness rerun or repair was performed by this reviewer. Do not spend a dummy invocation to rediscover these static defects.

After fresh independent source/HOST GO, parent still needs a separately frozen tiny non-product writer/argv and usable reviewed derivative recipe/image for the r16 single finite dummy lifecycle qualification (one metadata/source build<=300s, total<=600s, one writer/two sequential RO readers). This launcher has only product writer/archive/hash modes; its writer must not substitute for the dummy. Actual init, CLI/Docker semantics, detached wait, server terminal/quiescence, persistence after stop, retention and exact ownership need actual receipts and independent review before ONE product diagnosis.

P47 remains expected RED and requires independent actual offsets/type/original-exception/owner/enrollment/raw wait/EAGAIN/guard/call-teardown/unreached-assertion binding. Historical PID229 remains UNKNOWN. No H/S construction, product, native, installed, platform, T2, durability, release or activation gate advances. Candidate/history/private proof unchanged; sole write this report, no commit or acquired cleanup obligation.
