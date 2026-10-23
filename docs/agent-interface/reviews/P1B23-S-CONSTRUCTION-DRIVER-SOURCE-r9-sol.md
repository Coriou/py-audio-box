# P1B23 S construction driver — independent SOURCE review r9

2026-10-23 · Independent Sol low · **SOURCE NO-GO for ONE future construction readiness at this driver freeze.** Four concrete driver gates remain open below. Do not execute this freeze to investigate them. The reviewed application/recipe remain unchanged; this verdict grants no runtime result, construction acceptance, S/all255 acceptance or product gate.

## Authority and provenance

ABSREADONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-construction-driver-source-20261023`; driver freeze `5dd7b5b0b584f533f582ed9466dda5b32723d02f`; documentation HEAD `c5df9a3efe650abe82be50c9ffb7304e21439b0e`; application pin `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`. HEAD and empty candidate porcelain independently checked. Driver freeze parent is `116404c9a86302dbf2cffca7524b0f99995570e4`. Application-pin-to-driver changes only the two new driver files and RECOVERY-P1B2; driver-to-documentation changes only RECOVERY-P1B2. Thus all preexisting application/recipe/list/script/test/dependency Git blobs and modes are unchanged.

Read FULL owning S-SOURCE-RECIPE-SOURCE-r8-sol, including its exact before-execution gates, and FULL S-COPY-FAILURE-ACTUAL-r6-sol. Read the complete new handoff, driver, Dockerfile, reviewed recipe and construction script. Consulted original S brief/prior findings/responses carried in S-WITNESS-SOURCE-r3-sol and the relevant blocked-pipe launcher L1–L4 source findings. Large historical combined output truncated; focused reads supplied the original contract and relevant findings. This is not a new exhaustive review of every historical appendix or all255 implementation paths. No historical objection is waived.

Only host text/JSON/hash, Git and stdlib AST parsing were used. No candidate import/helper execution, recipe/source-copy execution, pytest/collection, Docker, SQLite/DB opening, runtime experiment, fix, delegation, commit, push or cleanup. Sole write is this root report. No runtime resource acquired. Checked ancestry and tracked scoped inventories supplied no applicable AGENTS.md; no global filesystem absence claim. Memory registry search supplied no relevant S fact and supports no verdict claim. Existing root STT/unrelated untracked artifacts remain untouched; root is not globally clean.

Paths/lines below refer to `docs/agent-interface/evidence/s-construction-driver.py` at the driver freeze unless otherwise stated.

## Blocking findings

### D1 — independent admission/readback does not implement the required no-follow stable same-descriptor read

**Blocking input-integrity gate; :116–123,130,135; also :18–20,136.** `checked` first lstat-checks directory and leaf pathnames, then performs a separate `leaf.read_bytes()`. That open follows symlinks, has no O_NONBLOCK, and has no before/after fstat observations on its descriptor. An intervening leaf replacement can therefore change the object/type opened after the lstat; replacing a leaf with a FIFO can block the host before its Git comparison. Same bytes can also pass despite the earlier metadata observation referring to a different object. Parent pathname checks likewise do not bind traversal to opened no-follow directory descriptors. This is the exact local mechanism omitted, not a demand for security against arbitrary hostile Python.

This reader admits the recipe/list before invoking the recipe and independently checks the actual copied payloads. The reviewed recipe's own correct read_regular (:44–69) cannot retrospectively supply this independent driver's missing descriptor observation. In particular the recipe program is reopened by pathname at :130; its self-hash calculation occurs after interpreter loading. The copied payload comparison at :135 also uses checked. Current reviewer observations of regular leaves do not prove this future read contract.

Minimal correction: use an independently reviewed directory-relative no-follow/nonblocking regular reader with Git mode admission, stable same-descriptor device/inode/full-mode/size/mtime_ns/ctime_ns and exact length before returning admitted bytes. Bind the recipe actually launched to the reviewed admitted bytes; a later self-description is not that binding. Preserve the r8 distinction: equality of separate validated reads proves bytes/modes, not cross-open inode identity. Do not execute/import the recipe as a preflight or change the application recipe to mask the driver gap.

### D2 — external Git commands bypass finite output capture and complete command receipts

**Blocking bounded-command/proof gate; :77–83,110–121.** Several `subprocess.check_output` calls run directly: rev-parse, status, ls-tree and repeated cat-file. They have timeout60, but stdout is accumulated without a byte cap, stderr is inherited without the driver's 1MiB cap, and no per-command argv/status/timeout/byte-count receipt is published. This contradicts the complete all-command bounded output/command-manifest gate. The timeout alone is not an output bound or a receipt. The reviewed recipe's internal Git subprocesses are enclosed by call('reviewed-recipe'), so their aggregate host output/time is separately capped; that does not cover these direct outer-driver calls.

Minimal correction: route these commands through bounded recorded capture with explicit stdout/stderr caps and failure accounting, retaining their actual argv and exits. Preserve exact binary Git blob comparison. Do not infer success from a summarized hostcheck file. This finding needs no Docker or product DB test and does not require impossible guarantees against uninterruptible kernel I/O. The call wrapper itself allows up to five additional seconds for host reap; describe that allowance rather than claiming every entire call is strictly <=60s.

### D3 — destructive cleanup uses mutable names rather than the exact owned full container ID

**Blocking exact-owned-ID gate; :159–176,227–233.** create stdout and inspect records may contain an ID, but the driver never captures/validates a canonical full owned ID for subsequent operations. pre-stop checks only the owner label on inspection by name; stop, stopped-inspect and rm all use `name`. There is no equality assertion tying the inspected full ID to create's returned ID or to later destructive argv. A name can be rebound between the ownership observation and mutation; the code does not express the required exact-ID scope. The random name and owner token are useful admission checks, not an immutable resource identifier.

Minimal correction: capture the full ID after successful creation, reconcile uncertain creation by exact name plus owner token and a validated full inspect ID, record that binding, and use that ID for stop/inspect/rm. Preserve rm without -v and anchored-name absence as corroboration. Unknown/mismatched ownership remains STOP with retained lane; never delete a foreign resource to reconcile uncertainty. This is also the still-applicable exact-ID portion of prior L4, not a new broad resource framework requirement.

### D4 — lane acquisition/publication failures are outside the accounting handler

**Blocking acquisition-failure handback gate; :141–145.** `lane.mkdir()` and both owner writes execute before the try and before created/failed/started_at are initialized. A failure after successful mkdir (for example owner-file write failure) exits without entering the STOP/OWNER-UNCERTAIN handler or final inventory. The lane may remain empty/partially published, with no structured owner/remaining-resource handback. Atomic mkdir correctly avoids stealing an occupied lane, but does not close this acquired-resource window. This is the concrete acquisition window already identified in prior L4.

Minimal correction: establish local acquisition state and place acquisition/publication inside the protected accounting scope. Retain a bounded failure receipt identifying whether this invocation acquired the lane and whether owner publication completed. Never remove a preexisting/mismatched lane. Initialization/publication failure must not claim lane release or completion. No retry or cleanup by this reviewer is authorized.

## Independently verified preserved inputs

Reviewer code independently read Git blobs and current regular leaves/modes, checked nonsymlink ancestors, parsed all selected Python ASTs without imports, and recalculated compact sorted-key UTF-8 manifest bytes. Exactly263 sorted unique paths,29390 manifest bytes, SHA256 `66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`. Every selected current byte and mode matches application pin. Schema and lock are explicitly present; no new omission or suffix-discovery selection. The unchanged literal list retains the r8-reviewed old262-plus-schema inventory; this review does not manufacture a new copy receipt.

| Input | Independently calculated SHA256 |
|---|---|
| Driver | ae9771fa0c1e299a07cab4b3865ece336f960b3d43c0e888ea9bd52b0e1e7d62 |
| Dockerfile | d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 |
| Recipe | 1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74 |
| Literal list | 5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6 |
| Construction script | 3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4 |
| contracts/v1/schema.json | f6f3ee5e207053e7542e1699e9ccc0b3a809c8a6697c7cf849d0f12f7f038bc3 |
| tests/release/requirements.lock | 1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad |

Git diff --check application-pin-to-HEAD passed. Driver, recipe, script and selected Python files parse as AST. No candidate code or helper definition was executed. These are current source observations, not actual copied/image/runtime identity results.

The unchanged recipe supplies no-follow traversal and stable same-fd reads, exclusive destination, Git-derived modes, copy readback, final source HEAD/porcelain and exact driver-recipe-freeze output. Driver independently calculates manifest, checks actual output manifest bytes, recipe-freeze schema/value, selected entry set and bytes/modes before image execution (:123–139), subject to D1. Future private detached checkout HEAD must equal application pin; documentation HEAD cannot substitute. Git worktree add changes administrative metadata but does not reset/clean/edit historical worktree files or HEAD. No archive extraction is selected.

The retained r8 source-data closure still applies to the unchanged service/contracts/worker graph, package initializers, local/conditional imports and literal schema read by contracts.validate at import; lock is an explicit identity input. Third-party installed files remain an environment obligation. Acquisition-aware decoder restoration remains unchanged, as do nested independent runner/runner_code_S release and namespace clearing. No product DB fixture or maintenance call is introduced.

## Exact prospective vectors and command inventory — NOT RUN

Host proposal, with explicit path substitutions, remains:

```text
python3 DRIVER_CHECKOUT/docs/agent-interface/evidence/s-construction-driver.py --repository REPOSITORY --evidence ABSENT_PRIVATE_EVIDENCE --driver-sha256 ae9771fa0c1e299a07cab4b3865ece336f960b3d43c0e888ea9bd52b0e1e7d62 --dockerfile-sha256 d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 --source-review INDEPENDENT_DRIVER_REVIEW --authorization PARENT_ONE_AUTHORIZATION
```

Recipe vector (:130):

```text
HOST_PYTHON SOURCE/docs/agent-interface/evidence/s-construction-source-recipe.py --checkout SOURCE --freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521 --destination CONTEXT/source
```

Sole construction vector (:137,190), cwd /s-closed-candidate:

```text
docker exec -w /s-closed-candidate OWNED_CONTAINER timeout 60s python -u docs/agent-interface/evidence/s-exit-context-construction.py --manifest manifest.json --manifest-sha256 66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46 --executable-freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521
```

Complete external command families in the source: Git worktree add; direct Git rev-parse/status/ls-tree/cat-file; recipe host Python; Docker info; image inspect/base tag absence/tag/tag-inspect; build/image-inspect; anchored container absence/create/start/inspect; exec memory reserve/manifest-before/identity/decoder/sentinel-create/sentinel tar/tmp-capacity; sole construction; all-/tmp tar and independent live hashes; manifest-after/identity-final/decoder-final; pre-stop inspect/stop/stopped inspect/rm/absence. hostcheck repeats Git/byte/mode/headroom checks initial/before/after/final. All call-managed argv are recorded in label.json; D2 identifies missing command receipts. No retry, product DB, pytest/collection or automatic import preflight is selected. Review/authorization file contents are bounded and retained, but their semantic authority is a parent decision, not verified by nonemptiness.

## Positive source observations and evidence limits

Dockerfile FROM is the literal owned local tag `pab-s-construction-driver-owned-base:20261023`; driver requires absence then tags and verifies base `sha256:cc251f26cead1f99a309e44e3b763e524b619fcbe02e6d675c9379d09c75f68f`. COPY --chown=10001:10001 plus workdir and --pull=false/--network=none builds are source-only, without install/pull. This addresses prior L1's root-owned restrictive copied source problem at source; actual image readability remains untested.

Container declarations/inspect assertions cover CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/capdropALL/no-new-privileges/private1GiB tmpfs. No binds, inherited volumes, devices, sockets, secrets, ports or audio operations are selected. Host disk reserve6GiB, Docker memory4GiB, Linux MemAvailable4GiB and tmpfs free900MiB are asserted. Linux CPython3.14.8/assertions, all42 lock identities and three decoder digests are required pre/post. Actual environment/headroom remains unverified.

call drains binary pipes with construction stdout65536, stderr1MiB, other stdout16MiB, archive stdout1100MiB. Effective command deadline is min(request,60), even for archive/build calls written with180/300; killpg and bounded additional reap follow. Killing a host Docker CLI does not prove remote exec completion. PID1's sleep1800 supplies a finite server lifetime under ordinary kernel/daemon operation, not a guarantee through daemon/kernel failure. No indefinite volatile HOLD is promised. This sentinel-only source construction does not inherit the blocked-pipe DB diagnostic's persistent-volume requirement. Prior L2/L3 are relevant for honest finite handback/output/remote-work limits, not a basis to demand DB storage here.

Binary sentinel token+NUL/FF/80/CR/LF requires UID10001/mode0600 and exact tar bytes. audit exports ALL /tmp binary before stop, rejects unsafe/duplicate normalized names, nonregular/nondirectory members (including symlinks), wrong regular UID, >64MiB file, >1GiB aggregate and >=200000 members, hashes every regular payload without extraction, and compares the complete map with separately collected live regular hashes including sentinel. Accepted inventory retains name/type/mode/UID/link metadata. No symlink is followed by extraction. Limits on member count are checked after getmembers; host parsing/hashing are cooperative and not a comprehensive wall-clock supervisor. No archive or live equality was observed by this review.

created=True precedes create dispatch, correctly accounting for possible allocation despite failed/lost response. On failed inspection/export/cleanup, owner uncertainty is retained with remaining finite lifetime and incomplete proof; failed audit does not trigger another audit attempt. retained is set only after export/map/sentinel success. Successful nominal cleanup requires owner check, stop10s, stopped exit0/not OOM/not Running, rm without-v, absence and exact matching lane-owner release. D3/D4 prevent full ownership certification. No destructive unknown-resource reconciliation is authorized. Eventual tmpfs loss after PID1 expiry is explicit; retained stopped metadata is not durable proof of unexported bytes.

Construction success requires raw construction exit0, exact52 records and ordering: pins1, actual-operands2, actual-results2, projections36, primary-complete9, controls-complete1, construction-complete1. Nine ordered dimensions require one True followed by eight False expected predicates and repairs; projection foreign results, actual cache/offset/acceptance values, restoration sentinel, exact final controls/completion, and initial/final actual equality are checked (:197–218). Unchanged inner assertions carry detailed predicates and source restoration/release obligations. Bounds remain64/4096 ASCII bytes per record/65536 total; outer checks do not replace the script. Wrapper reporting/DONE exit0 cannot replace construction exit0; caught failure now ends sys.exit(1). r6 remains consumed failure after pins, not a successful construction.

## Handback and remaining gates

D1–D4 need a new frozen source response and independent review before readiness for considering ONE separately parent-authorized construction. No retry, fix, full255 or H gate is authorized by this report. Future actual review must independently verify actual copied bytes/modes/manifest/recipe freeze before imports, source/runtime pins, complete command/output/resource receipts, all52 records and actual raw exit, binary ALL-/tmp retention/live hashes, complete private evidence hashes and exact-ID ownership/cleanup/absence/lane receipts. Partial/uncertain proof is STOP with explicit remaining resource/lifetime, never false completion.

All product gates remain open. Historical4 passes/1 failure/250 unrun, Sa;Ss; without Sf; RED remain unchanged. False8/True9 ordered same-active-object/attributes/native-inner routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion remain mandatory. Projection qualifies decoded predicates only; actual decode qualifies static recognition/cache mapping only. Neither proves native unwinding/FULL/S acceptance. Synthetic transaction-return/original-inode controls retain their limits. Late genuine completion FULL, H/P/overlap/grouped/integration/maintenance, blocked-pipe/storage, other-family/native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation remain separate and unqualified. No H qualification transfers. No runtime resources were acquired or cleaned by this review; no commit.
