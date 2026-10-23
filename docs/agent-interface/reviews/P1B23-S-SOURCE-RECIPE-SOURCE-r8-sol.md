# P1B23 S source recipe — independent SOURCE/RECIPE review r8

2026-10-23 · Independent Sol low · **Bounded SOURCE GO for readiness to consider ONE separately authorized future finite construction. F1 CLOSED at source; no new blocking finding.** This does not authorize execution, replenish the consumed r6 invocation, qualify construction success, accept S/all255, or authorize maintenance, integration or release. The exact outer driver/Dockerfile/input and publication gates below remain required before execution.

## Authority and provenance

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-source-recipe-f1-r8-20261023`; executable/recipe freeze `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`; documentation HEAD `116404c9a86302dbf2cffca7524b0f99995570e4`. Independently checked HEAD, clean candidate porcelain and executable parent exactly `babbc81a27fd82436d94703a7e26deef4527acf7`, whose reviewed executable is `c181818b3dd6768751e6eaa916260339a3cc051a`. The executable commit versus babbc81a changes ONLY the recipe (10 insertions, one deletion); freeze-to-HEAD changes ONLY RECOVERY-P1B2. Relative to c181818b the additional changed path is that accumulated documentation, not another executable input.

Read FULL owning S-SOURCE-RECIPE-SOURCE-r7-sol and S-COPY-FAILURE-ACTUAL-r6-sol, the new r8 handoff and complete recipe/construction script. Consulted original S brief, revised/shared contract and prior responses carried in S-WITNESS-SOURCE-r3-sol, and the prior source findings/closures including S-CLOSED-CARRIER-SOURCE-r6-sol. Original requirements and historical results retain their attributed scope; this F1-only review is not a fresh exhaustive all255 or historical runtime audit.

Only read-only Git/text/JSON/hash and host stdlib AST parsing/comparison were used. No candidate import, candidate definition/helper execution, recipe execution/source-copy construction, pytest/collection, Docker, SQLite/DB opening, runtime/race experiment, fix, delegation, commit, push or cleanup. Sole write is this root report. No applicable AGENTS.md was found in checked ancestry or candidate tracked inventory. Memory registry lookup supplied no relevant S fact; no memory-derived fact supports the verdict. Existing root STT/unrelated artifacts and historical proof were untouched; root is not globally clean. No runtime resource was acquired.

## F1 CLOSED — stable metadata on the SAME opened descriptor

`docs/agent-interface/evidence/s-construction-source-recipe.py:54–66` retains the opened leaf as one stream. It obtains `info = os.fstat(stream.fileno())`, reads into data, then obtains `after = os.fstat(stream.fileno())` BEFORE returning. Both observations independently require regular type and permission bits equal to the admitted Git mode. Explicit tuples compare exactly device, inode, full mode, size, mtime_ns and ctime_ns. `len(data) == info.st_size` binds the read length to the stable size. No pathname stat or second open supplies the post-read observation.

All relevant paths use this reader: selected payloads via tracked (:90–101), the literal list and both separately frozen driver inputs (:96,102), copied-file readback (:138), and generated manifest/driver-recipe-freeze.json readback (:147–148). Git tree admission remains regular blob mode100644/100755, with exact cat-file byte equality before hashing. Thus a metadata/length mismatch fails before a read result is admitted by the source contract. This is static enforcement reasoning, not an executed race test, a filesystem snapshot or a guarantee against arbitrary hostile Python/filesystem behavior.

The later tracked(name) call is another independently validated read. Its equal bytes/mode establish frozen content, **not cross-open inode identity**. The explicit new comment at :139 and handoff correctly state that same-content regular replacement between opens is not rejected as an identity change. Source and destination inode equality is neither required nor claimed. This resolves the r7 objection without confusing equal bytes with identity.

No new file descriptor or candidate owning reference is introduced by the correction: only local bytes and stat results are added. The same stream context closes the leaf on normal return or read/fstat/assertion failure; the existing finally closes the directory duplicate. The changed block catches/suppresses no primary error and adds no restoration assertion. Ordinary successful close lets the original read/validation failure propagate; an actual close failure remains failure with Python exception context, not successful cleanup. This is not an interruption/leak qualification for every preexisting acquisition edge or a general filesystem framework review. No such new framework or assertion waiver is added or needed for F1.

## Preservation, literal inventory and source/data closure

Independent AST comparison finds only read_regular structurally changed; all other top-level ASTs are identical to c181818b. Complete Git changed-path inspection against babbc81a proves every other tracked mode/blob is unchanged. All263 selected source files compare byte-for-byte to c181818b and to the current freeze; the two driver inputs match their current Git blobs and modes. Current selected leaves are regular, checked ancestor components are not symlinks, and permission bits match Git. These current source checks are not future copy receipts.

Independently parsed the retained old r6 manifest and current literal list: exactly263 sorted unique paths, EXACT set equality old262 UNION {contracts/v1/schema.json}. All old paths remain; only the previously authorized construction-script digest differs from r6. No suffix discovery, arbitrary omission or new bound exists.

Directory traversal remains directory-relative O_DIRECTORY|O_NOFOLLOW, including absolute root components; leaf reads retain O_NOFOLLOW|O_NONBLOCK. Destination mkdir remains absent-only; output leaves retain O_CREAT|O_EXCL|O_NOFOLLOW, Git-derived fchmod and exact readback byte comparison. Source reread/Git comparison and final HEAD/porcelain checks remain. Failure leaves partial destination evidence. There is no archive extraction, import preflight, candidate launcher or cleanup in the recipe.

The unchanged conservative repository graph remains explicitly selected: contracts/{__init__,validate}.py; service/{__init__,admission,core,dispatch_store,persistence,persistent_init,reconciliation,recovery,resource_store,worker_transport}.py; worker/{__init__,runtime_journal,transport}.py. Local/conditional imports stay within that graph; recovery's dynamic __import__ names hashlib/collections. Package initializers are empty/docstring-only. Module/class initialization creates constants, regex/lock/list/PID/view/dataclass surfaces rather than invoking product constructors/operations. contracts/validate.py:11 reads SCHEMA from literal v1/schema.json at import, explicitly selected and pinned. Reconciliation's local schema read names the same file and is dormant here; DB/journal/wave/process operations lie in dormant product functions. JSON loading does not fetch external schema files. Release lock remains an explicit required data input. Installed stdlib/third-party files remain a separate environment obligation.

The unchanged construction script verifies the complete manifest before candidate import/helper execution and requires schema/lock alongside source inputs. Its only extracted guard methods remain exit_intervals_S/exit_site_S; its seven literal _s_exit_* helpers retain their stdlib/local-helper closure. Actual runner is inspected, never invoked; no alternate runner is compiled. Acquisition-aware restoration remains `real_decode is None or dis.get_instructions is real_decode`: before capture there is no acquired decoder obligation; after capture exact identity is mandatory. Nested independent runner/runner_code_S release and local/namespace clearing remain byte-identical. Early primary import failure cannot reach construction-complete; genuine restoration/cleanup failure remains STOP. No successful restoration receipt is inferred.

Prior F1/F2/F4 and owning-reference F3 closures retain their bounded source meaning. Closed carrier inventory/exact stored-field types, literal deltas, fixed actual offset/cache/singleton checks, ordered first failures/repairs, foreign passthrough/rejection, sentinel and output-before-assert obligations are unchanged. No recognizer, guard, production, recovery target, controls, conftest, dependency, process/reaper/transport/receipt or resource policy changes exist in this correction.

## Independently calculated successor pins

Expected manifest is UTF-8 compact sorted-key JSON without newline, with exactly executable_freeze/files. It has **263 entries, 29390 bytes**, SHA256 **`66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`**. It satisfies <=512/262144 and matches the construction parser's schema/argv. Selected bytes are unchanged, but executable_freeze changes this digest; historical e942dd92 is not successor authority. Calculation used reviewer JSON/hash logic over frozen bytes, never candidate code or recipe invocation. No destination, generated manifest or source-copy receipt was created.

| Input | SHA256 |
|---|---|
| s-construction-source-recipe.py | `1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74` |
| s-construction-source-paths.json | `5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6` |
| s-exit-context-construction.py | `3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4` |
| contracts/v1/schema.json | `f6f3ee5e207053e7542e1699e9ccc0b3a809c8a6697c7cf849d0f12f7f038bc3` |
| tests/release/requirements.lock | `1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad` |

All selected Python sources parsed as AST without import/execution. Git diff --check c181818b-to-HEAD passed. The unchanged script requires51 records before final completion and52 afterward: pins1 + initial/final actual-operands/results4 + projection-results36 + primary-complete9 + controls-complete1 + construction-complete1. The36 are27 baseline/model/repair,2 non-CALL,2 ambiguity,4 owner/NULL,1 sentinel. Nine literal dimensions and caps64 records/4096 ASCII bytes including newline each/65536 total remain. **All52 are finite source obligations, not executed or passing checks.**

## Full prospective argv and remaining execution gates — NOT RUN

SOURCE must be a separately prepared clean checkout with HEAD EXACTLY bd34b74cc0634c6e3b65cabb373ecbad3ccc2521. Current candidate documentation HEAD116404c does not satisfy the recipe's HEAD==freeze check. DEST must be a fresh private absent path. Exact prospective preparation vector:

```text
python3 SOURCE/docs/agent-interface/evidence/s-construction-source-recipe.py --checkout SOURCE --freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521 --destination DEST
```

Exact prospective inner vector, working directory /s-closed-candidate:

```text
timeout 60s python -u docs/agent-interface/evidence/s-exit-context-construction.py --manifest manifest.json --manifest-sha256 66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46 --executable-freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521
```

These vectors are compatible source proposals, not an approved complete outer driver. BEFORE ANY execution, parent must separately authorize ONE future attempt and freeze/review exact outer driver, Dockerfile and all input bytes/argv. Driver must bind these reviewed recipe/list hashes, selected source/data hashes/modes, generated manifest and driver-recipe-freeze.json; independently prove actual copied bytes/modes and parser-compatible manifest against this freeze before candidate execution. Historical driver or H copy evidence cannot substitute. Preserve assertion-enabled execution and complete private source preparation/identity/input/stdout/stderr/resource receipts.

That exact driver must also bind Linux CPython3.14.8/all42 locked identities and decoder pins, no pull/install, source-copy-only construction,60-second bounds,CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/private tmpfs1GiB, serialized exact ownership/admission and host headroom. Preserve stdout65536/stderr1MiB driver caps and complete failure status. Binary LIVE export and independent equality of all retained hashes must precede exact-owned cleanup; retain exact owner/lane/stop/removal/absence receipts and publish the complete private evidence bundle and its hashes for independent actual review. Retention failure preserves the hold and requires explicit bounded handback; volatile LIVE storage is not an indefinite retention guarantee. No cleanup was performed by this review.

Unexpected layout/stage/outcome, missing/extra record, bound, restoration or release failure means STOP with one bounded report, no retry/layout search. Independent actual-receipt review and parent decision must precede any separately authorized S maintenance gate. The r6 consumed attempt remains consumed; this report supplies only bounded source readiness for considering a new separately authorized attempt.

Projection qualifies decoded predicates only; unpatched actual decode qualifies static recognition/cache mapping only. Neither proves live f_lasti/unwinding/native FULL or S acceptance. All255 remain required; historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. False8/True9 ordered same-active-object/attributes/native-inner routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion remain mandatory. Synthetic transaction-return/original-inode controls retain their limits. Late genuine completion FULL, H/P/overlap/grouped/integration, blocked-pipe/storage, other-family/native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation remain separate and open. No H qualification transfers.
