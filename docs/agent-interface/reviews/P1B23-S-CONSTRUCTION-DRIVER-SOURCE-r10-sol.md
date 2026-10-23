# P1B23 S construction driver — independent SOURCE review r10

2026-10-23 · Independent Sol low · **Bounded SOURCE GO for readiness to consider ONE separately parent-authorized finite construction. D1–D4 CLOSED at source; no new blocking source finding.** This does not authorize execution, replenish the consumed r6 attempt, establish construction success, accept S/all255 or advance H or a product gate.

## Authority and provenance

ABSREADONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-driver-d1-d4-r10-20261023`; executable freeze `0dfbd74116a5bfbf17ad490d21713a24101a68a6`; documentation HEAD `e02de53038f221b14c32fa75bb9e1d870d6f4129`; comparison driver `5dd7b5b0b584f533f582ed9466dda5b32723d02f`; application/recipe pin `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`. Independently verified HEAD and empty candidate porcelain. Compared with r9, only driver.py and RECOVERY-P1B2 change; executable-to-HEAD is documentation only. Application-pin-to-HEAD adds only driver.py/Dockerfile and changes RECOVERY-P1B2. Every preexisting tracked application, recipe/list/script, guard/control, test and dependency mode/blob is preserved.

Read FULL r9 driver review and r8 recipe review/exact gates, complete new D1–D4 handoff, driver, Dockerfile, recipe and construction script. Consulted original S brief/contracts, original/prior findings and implementation responses carried in S-WITNESS-SOURCE-r3-sol, including focused original handoff, r2 assignment and r3 assignment/response sections; read r6 copy-failure and closed-carrier source assessments. Large combined historical reads truncated; focused reads recovered the controlling requirements. This is a fresh driver-path review, not a claim to have re-audited every historical appendix/all255 path or historical runtime receipt. Prior requirements and unresolved product gates are not waived.

Only host text/JSON/hash, read-only Git and reviewer-owned stdlib AST parsing/comparison were used. No candidate import, definition/helper execution, recipe/source-copy execution, pytest/collection, Docker, SQLite/DB opening, runtime experiment, fix, delegation, commit, push or cleanup. Sole write is this root report. No runtime resource was acquired. Checked ancestry and tracked/scoped inventory supplied no applicable AGENTS.md. Memory registry search supplied no relevant S fact and supports no verdict claim. Existing root STT/unrelated artifacts remain untouched; root is not globally clean.

All driver line references below refer to the exact executable freeze.

## D1 CLOSED — independent reader and launched bytes

`regular` (:20–41) opens root and each directory component with O_DIRECTORY|O_NOFOLLOW using dir_fd, and the leaf with O_NOFOLLOW|O_NONBLOCK. The same leaf descriptor supplies before/after fstat and read. It requires regular type, explicit cap, admitted permission mode where specified, equality of device/inode/FULL mode/size/mtime_ns/ctime_ns, and read length equal to stable size. `checked` (:191–195) admits only Git regular blob100644/100755, validates canonical relative paths and compares bytes against exact binary cat-file output. Recipe/list admission, manifest input generation, copied payload readback and repeated hostchecks use this independent path. Separate valid opens prove bytes/modes, not cross-open inode identity.

The actual recipe code is bound at :196–224: admitted bytes are written to an exclusive private snapshot; writer closes; mode0400/readback is verified; read-only/no-follow/nonblocking fd is opened and snapshot unlinked. Before launch, that actual inherited object is checked regular0400/nlink0, bounded, same-fd metadata-stable and exactly equal to admitted recipe bytes/read length. Offset is reset. Only the read-only fd is inherited via pass_fds. Literal bootstrap reads it and compiles those bytes with original filename, __main__/__file__/argv/package fields and script-directory sys.path[0]; cwd is private checkout C. No executable recipe pathname is subsequently reopened to load code. The recipe's tracked SELF read is data admission, not a replacement executable load. The fd closes in finally; earlier snapshot/preparation failures retain private partials and STOP.

This meets the concrete stable-reader/launched-input gate. It is not a filesystem snapshot, arbitrary same-user adversary guarantee, universal Python-startup equivalence claim or descriptor-interruption qualification.

## D2 CLOSED — outer Git capture and finite command accounting

No check_output remains in the driver. Worktree-add calls the bounded wrapper directly; rev-parse/status/ls-tree/cat-file use unique git-NNNNN labels (:119–124). Each has stdout16MiB/stderr1MiB caps, deadline at most60 seconds, pending argv/cwd/caps/deadline/reap receipt and completed argv/status/timeout/overflow/failure/retained-byte/elapsed receipt. Binary blobs remain bytes throughout comparison. `call` (:73–117) accounts for output-open/spawn/read/write/select/wait/kill/reap/close failures; publication failure propagates STOP, with stderr fallback and pending records where publication succeeded.

The unchanged recipe still contains internal check_output calls. As explicitly scoped in r9, those are enclosed by the one bounded recipe process/aggregate output-time envelope; this closure does not claim individual inner-Git receipts or individual internal allocation caps. No application recipe change is smuggled into this driver correction.

Construction stdout is65536, every captured stderr1MiB, ordinary stdout16MiB, archive stdout1100MiB. Effective command budget is min(request,60), including written build300/archive180 requests, plus up to5 seconds host reap. Host filesystem hashing/traversal/JSON/tar parsing remain cooperative; this is not a comprehensive outer wall-clock supervisor or guarantee through uninterruptible kernel I/O. Killing Docker CLI is not proof of remote completion. Timeout/cap/capture failure latches Docker command uncertainty, preventing later audit/destructive cleanup/lane release. These disclosed limits do not reopen the concrete r9 external-capture finding.

## D3 CLOSED — full container ID and uncertainty

Create dispatch is marked before call (:264). Successful stdout must be canonical64 lowercase hex, followed by inspect binding exact full ID/name/token/image (:126–133,265–270). Start, every exec including sentinel/archive/live, later inspections, stop and rm use container_id. Bound inspections repeat ID/name/token/image equality. Name filtering after rm corroborates absence and supplies no mutation authority. rm has no -v.

If create failed, returned malformed output or initial binding failed before container_id assignment, finally makes ONE read-only exact-name reconciliation (:340–345). It admits only canonical full inspect ID plus exact name/owner token/image. Failure/mismatch/unknown ownership enters OWNER-UNCERTAIN with lane retained, never foreign cleanup. Reconciliation never retries create/start or clears failed/uncertain state. A fully bound but unretained object likewise receives finite HOLD, not stop/rm. No unsafe name-based mutation remains.

## D4 CLOSED — protected acquisition/publication and handback

All acquisition states and flags exist before evidence/lane mkdir (:43–50). Evidence preparation, checkout dispatch, context, lane wait/mkdir and both owner publications are inside protected accounting (:170 onward). Lane state distinguishes pending/acquired/release-pending/released; separate owner states distinguish pending/published. Exclusive mkdir cannot convert preexisting lane into ownership. An occupied lane waits at most60 seconds and STOPs without stealing.

Failure after mkdir/stat/partial owner publication enters the latched handler and bounded state/owner stderr handback; owned evidence permits STOP/HOLD/final-state receipts. Partial publication never meets release conditions. Release requires acquired lane, both publications, no command uncertainty/pending tag/build, completed owned removal where applicable, matching directory device/inode/type and exact owner document (:364–369). Preexisting/mismatched lane is never deleted. Release/publication failure remains STOP; released versus release-pending reflects operations already completed. Final inventory/publication failure cannot yield successful exit.

Failure-path audit covered evidence acquisition, Git/recipe/copy validation, occupied/colliding lane, partial publication, tag/build uncertainty, create/start/binding, construction/output/assertion, first audit/export failure, pre-stop/stop/stopped/rm/absence and final receipt/hash failures. On cleanup-block failure, OWNER-UNCERTAIN carries remaining state; no later mutation follows that exception. No repeated failed audit or execution retry exists. Finite handback does not imply proof survives indefinitely in volatile tmpfs. Remaining lifetime is an elapsed-from-start-dispatch estimate under ordinary daemon/kernel operation; None means no start dispatch recorded. PID1 sleep1800 is not a remote-completion receipt.

## Preserved inputs and construction gates

Independently read all263 selected current leaves and Git blobs, checked modes/nonsymlink ancestors, parsed selected Python ASTs and recalculated compact sorted-key UTF-8 manifest: **263 sorted unique paths,29390 bytes, SHA256 `66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`**. All match application pin. Schema and lock remain explicit; no suffix-discovery or omission change. Driver/recipe/script and constant bootstrap/LIVE/identity/decoder programs parse as AST without execution. Git diff --check r9-to-HEAD passed.

| Input | Independently calculated SHA256 |
|---|---|
| Driver | 48365949744d01120bdde7ebef537a8adb6b91b7327c637e1137dd39159a27b6 |
| Dockerfile | d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 |
| Recipe | 1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74 |
| Literal list | 5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6 |
| Construction script | 3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4 |

Private detached checkout HEAD/clean porcelain must equal bd34b74, not documentation HEAD. Driver independently verifies recipe/list, selected bytes/modes, exact generated manifest/freeze and exact copied entry set before image execution (:183–235). Inner script verifies manifest before candidate import/helper execution. Unchanged source/data closure includes contracts' literal schema and local service/worker graph; installed dependencies remain separate environment obligations. Acquisition-aware decoder restoration, nested independent runner/runner_code_S release and namespace clearing remain unchanged.

The entire outer52-record audit block is independently byte-identical to r9. Raw construction exit0, exact counts/order, nine fixed dimensions/expected True then eight False/repairs,36 projections, actual fixed cache/offset/singleton values, foreign checks, restoration sentinel, initial/final actual equality and final exact controls/completion remain mandatory. Inner emitter still bounds64 records/4096 ASCII bytes including newline/65536 total. None were executed or passed here.

Resource declarations/inspect assertions remain CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/capdropALL/no-new-privileges/private1GiB tmpfs, no bind/volume/device/port inheritance; host disk6GiB, Docker memory4GiB, Linux MemAvailable4GiB and tmpfs900MiB reserves. Linux assertion-enabled CPython3.14.8/all42 locked versions and three decoder hashes are required pre/post. Dockerfile uses exact local owned base tag, verified base ID, COPY --chown and source-only --pull=false/--network=none build; no install/pull.

Binary sentinel retains token+NUL/FF/80/CR/LF and UID10001/mode0600/exact tar bytes. ALL-/tmp binary archive precedes stop; audit rejects unsafe/duplicate normalized paths, links/unsupported types and wrong regular UID, bounds regular size64MiB/aggregate1GiB/member count<200000, hashes every regular payload without extraction and requires entire equality with independent LIVE regular map including sentinel. retained is set only after successful audit. Failed proof retains resource/lane with explicit finite volatile handback. This outer sentinel construction has no product DB fixture and does not acquire blocked-pipe/product database lifecycle requirements. No archive/LIVE or cleanup result was observed here.

## Prospective vectors and remaining authority — NOT RUN

```text
python3 DRIVER_CHECKOUT/docs/agent-interface/evidence/s-construction-driver.py --repository REPOSITORY --evidence ABSENT_PRIVATE_EVIDENCE --driver-sha256 48365949744d01120bdde7ebef537a8adb6b91b7327c637e1137dd39159a27b6 --dockerfile-sha256 d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 --source-review INDEPENDENT_DRIVER_REVIEW --authorization PARENT_ONE_AUTHORIZATION
```

Recipe argv is HOST_PYTHON -c literal-bootstrap FD C/RECIPE --checkout C --freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521 --destination CONTEXT/source, cwd C; the exact bootstrap bytes are frozen at driver:222. Sole construction vector, cwd /s-closed-candidate:

```text
docker exec -w /s-closed-candidate OWNED_FULL_ID timeout 60s python -u docs/agent-interface/evidence/s-exit-context-construction.py --manifest manifest.json --manifest-sha256 66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46 --executable-freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521
```

Complete external command families are finite: worktree-add; numbered Git rev-parse/status/ls-tree/cat-file including repeated four hostchecks; one fd-bound recipe Python; Docker info/base inspect/tag absence/tag/tag inspect/build/image inspect; anchored container absence/create/binding/start/inspect; exec reserve/manifest/identity/decoder/sentinel-create/sentinel tar/tmp-capacity; ONE construction; ONE archive/live audit; post manifest/identity/decoder; pre-stop/stop/stopped/rm/absence; optional single uncertain-create inspection. All outer dispatches use call; literal programs and input vectors are frozen in driver bytes, expanded argv/cwd/caps are prospective per-command receipts. No shell interpolation, automatic import preflight, layout search, alternate compiler, maintenance or retry is selected. Review/authorization contents are bounded/retained; their semantic authority remains the parent's decision.

Parent may now consider ONE separately authorized attempt on these exact pins. Future independent actual review must check actual copied inputs before imports, complete environment/output/resource/command receipts, raw construction exit and all52 records, ALL-/tmp binary/LIVE equality, complete private evidence hashes, exact-ID stop/removal/absence and lane ownership/release. Any missing/uncertain proof is STOP with remaining resource/lifetime, not completion. This report authorizes none of those operations.

All original S requirements remain: False8/True9 same-active-object/attributes/native-inner exact ordered routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion. Historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. Projection qualifies decoded predicates only; actual decode qualifies static recognition/cache mapping only. Synthetic transaction-return/original-inode controls retain their limits. No native unwinding/FULL/S/all255 or H acceptance follows. Late genuine completion FULL, H/P/overlap/grouped/integration/maintenance, blocked-pipe/storage, other-family/native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation remain separate and open. No runtime resource acquired or cleaned; no commit.
