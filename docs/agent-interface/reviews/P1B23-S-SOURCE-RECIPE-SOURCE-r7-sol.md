# P1B23 S source recipe — independent SOURCE/RECIPE review r7

2026-10-23 · Independent Sol low · **SOURCE/RECIPE NO-GO for future ONE construction readiness.** One blocking recipe finding: the required before/after descriptor metadata validation is absent. Literal selection, repository source/data closure and acquisition-aware restoration pass the bounded static review. No construction, S acceptance, all255 acceptance, maintenance, integration or release authority follows.

## Authority and exact provenance

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-source-recipe-r7-20261023`; executable/recipe freeze `c181818b3dd6768751e6eaa916260339a3cc051a`; documentation HEAD `babbc81a27fd82436d94703a7e26deef4527acf7`. Independently checked clean candidate porcelain, HEAD parent equal executable, executable parent exactly `7e3cf1253daca04a8fdaa01c47ca37cd8af5fa49`. Comparison executable is `4fc4089b06ed4e8250334d4930ce0cbf446cbcf2`.

Read FULL S-COPY-FAILURE-ACTUAL-r6-sol and S-CLOSED-CARRIER-SOURCE-r6-sol, the new handoff and entire recipe/construction script; consulted original S/revised S1/shared requirements and prior findings/responses carried in S-WITNESS-SOURCE-r3-sol, r4/r5 source findings and Astra H-EXACT-CONTEXT-DESIGN. Historical receipts and closures remain attributed prior evidence; this is not a new exhaustive historical-proof audit or new runtime review.

Only read-only Git/text/JSON/hash, host stdlib AST parsing and compile(AST) without execution were used. No candidate imports, candidate/helper definitions executed, recipe invocation, source-copy construction, pytest/collection, Docker, SQLite/DB opening, runtime probe, fix, delegation, commit, push or cleanup. No runtime resource was acquired. Sole write is this root report. No applicable AGENTS.md found in checked ancestry and scoped source/report inventory. Memory registry search supplied no relevant task fact; no memory-derived fact supports this verdict. Existing root STT/unrelated artifacts and historical proof remain untouched; root is not globally clean.

## F1 OPEN — missing post-read fstat and stable metadata proof

**Blocking for the explicitly required recipe contract.** `docs/agent-interface/evidence/s-construction-source-recipe.py:44–58` opens each leaf without following links, performs one fstat at line56, checks regular type and Git permission bits at line57, then returns stream.read() directly at line58. There is no second fstat, no comparison of before/after device/inode/mode/size/mtime/ctime, and no read-length versus stable size assertion. This same reader admits source payloads, the two driver inputs and copied outputs.

The later `tracked(name)` call at line131 is another independent open with the same one-sided check. It compares bytes/mode with the earlier payload; it does not establish stability across either read or bind the two opens to the same inode. Likewise Git blob equality and final HEAD/porcelain checks prove selected content/provenance, not the requested per-read metadata invariant. For example, bytes can remain equal to the frozen blob while mode or metadata changes after the pre-read fstat; no post-read check rejects that observation. A source pathname can be replaced by another regular same-content inode between opens and still satisfy the present comparisons. This is static failure-to-enforce reasoning, not an observed race or a claim that arbitrary wrong bytes bypass Git comparison.

Minimal correction: retain the opened leaf descriptor, read into a local byte value, fstat that SAME descriptor again before returning, require regular/Git-mode predicates on both observations, require a stable explicit metadata tuple (device, inode, mode, size and nanosecond modification/change times), and require byte length equal to the stable size. If cross-open source/copy identity stability is claimed, retain and compare the appropriate identity separately rather than treating equal bytes as inode identity; source and destination inodes themselves must of course be distinct. Review output creation/readback metadata checks under the same contract. Preserve directory-relative O_NOFOLLOW traversal, exclusive creation, frozen blob equality and exact byte/hash comparisons. This does not require a generic filesystem security framework or an executed race experiment.

Freeze the bounded correction and obtain fresh independent SOURCE/RECIPE review. Do not execute this recipe or use the present manifest hash as successor authority after changing recipe or selected script bytes. No retry or replenished invocation is granted here.

## Literal selection, Git identity and bounds — verified at source

Independently parsed the READONLY old `/tmp/pab-s-closed-carrier-construction-r6-20261023/manifest.json` and the literal new JSON. Old count is262; new count263, sorted and unique. Exact set equality is **old262 UNION {contracts/v1/schema.json}**, with no omitted old path or suffix-derived replacement. The sole old selected digest change is the authorized S construction script. The new recipe and path-list files are separately frozen driver inputs, not imported construction modules, and need not be invented as additional members of this import manifest.

For every selected file and both driver-input files, checked current leaf regular type, every ancestor against symlinks, Git100644/100755 blob type/mode, actual permission bits and exact bytes against the executable freeze's Git blob. These are current read-only source checks, not proof that the future recipe implements F1 or an actual copy receipt.

Independently reconstructed the future manifest as UTF-8 JSON, sorted keys, compact separators, no newline: top-level keys exactly executable_freeze/files; files maps literal paths to SHA256 strings. Count **263**, size **29390 bytes**, SHA256 **`e942dd92d93804af6c52301a0f236a78a51d5965bccba7729dedb9d249ffa4f4`**. Both <=512 entries and <=262144 bytes hold. Recipe serialization matches the construction parser's schema and arguments. Recipe also emits separate driver-recipe-freeze.json binding executable, manifest hash/size/count and both recipe-input hashes. Expected outputs are calculations only; nothing was copied or generated by invoking candidate code.

| Frozen input | SHA256 |
|---|---|
| s-construction-source-paths.json | `5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6` |
| s-construction-source-recipe.py | `a2a7da0a9bea184d02f769d6a4f894c7681a7e2758f5fd5ad3882ab9c8491106` |
| s-exit-context-construction.py | `3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4` |
| contracts/v1/schema.json | `f6f3ee5e207053e7542e1699e9ccc0b3a809c8a6697c7cf849d0f12f7f038bc3` |
| tests/release/requirements.lock | `1128749ee3509236a0dbd614e03869c001daab61a1f7fad3959b4c9b5a6b18ad` |
| tests/service/test_child_wait_guard.py | `18d27bfcf692a9c1459158344027fcdd421f0d9586d1d80acf1a24bccf4fad9d` |
| tests/_maintenance_wait_guard.py | `cf58a559189912c39711ad2cd4b871919bf5a4bb18c85c85a69392678e774ca7` |
| tests/service/test_recovery.py | `5174830dcfdd64f717bb15df6b9fdc1f27ef0e8e55b9919f4a4289e2108d29cb` |
| service/recovery.py | `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29` |
| tests/conftest.py | `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3` |

The recipe's no-follow design otherwise has the expected narrow shape: absolute source/destination-parent components are opened using O_DIRECTORY|O_NOFOLLOW; selected parents use directory-relative equivalent flags; source leaves use O_NOFOLLOW|O_NONBLOCK; Git tree admits only regular blobs. Destination creation is absent-only mkdir, followed by no-follow directory opening; copied parents are no-follow directories and leaves use O_CREAT|O_EXCL|O_NOFOLLOW. Copy bytes are read back and compared, modes are set from Git, and sources are reread against frozen blobs. There is no recursive archive extraction, link dereference, external path selection, import preflight or runtime launch. Failures leave the partial destination rather than deleting evidence. These strengths do not close F1.

## Explicit static repository source/data closure

Conservatively inspecting local imports including function-local/conditional imports gives the closed repository graph:

- `contracts/{__init__,validate}.py`;
- `service/{__init__,admission,core,dispatch_store,persistence,persistent_init,reconciliation,recovery,resource_store,worker_transport}.py`;
- `worker/{__init__,runtime_journal,transport}.py`.

Every path is in the literal manifest. Package initializers are empty/docstring-only. The direct recovery imports and transitive cycles stay in this graph. Recovery's dynamic __import__ calls name stdlib hashlib/collections; no repository dynamic loader is needed. Module/class surfaces create constants, regex, lock/list/PID/view objects, properties and dataclasses; property lambda bodies and product constructors are dormant. Guarded __main__ product entry points are not entered by import. No Core, Journal, Admission or recovery operation is invoked by this construction.

`contracts/validate.py:11` loads SCHEMA from its literal ROOT/v1/schema.json at import. That exact data file is now explicitly required alongside the release lock before helper execution/import, then included in the complete pre-import hash verification. json.loads does not fetch schema references. Reconciliation's function-local schema read names the same selected file; it is dormant here. Other DB/journal/wave/filesystem reads occur inside dormant product functions. Third-party/stdlib installed files remain the separate environment identity obligation, not repository-manifest closure.

Construction reads its manifest and selected guard/control/recovery sources, compiles only two extracted guard methods (exit_intervals_S/exit_site_S) plus the seven literal _s_exit_* helpers in the separately authorized future run. Static inspection of that path requires only stdlib dis/contextlib/collections and those helpers, no new recipe-module import or product file resource. Actual runner is captured for inspection, never invoked; no alternate runner is compiled. The source/data omission identified in r6 is closed at source. This establishes bounded repository closure, not actual import success or third-party file closure.

## Acquisition-aware restoration and preservation

Only script changes versus4fc4089b are the explicit schema/lock required entries and `real_decode is None or dis.get_instructions is real_decode`, with its explanatory comment. Recovery import still precedes decoder capture and controls. Before capture, None denotes no acquired restoration obligation; successful cleanup therefore propagates the original early import error without the old false restoration AssertionError. After capture, exact identity remains required and an actual mismatch still fails. No completion record is reachable after the propagating error/mismatch. There is no unconditional restored=True, swallowed exception or changed capture boundary.

Nested independent witness.runner and witness.runner_code_S release and further local/namespace clearing remain intact. A genuine cleanup/setter failure remains failure, with Python's unsuppressed exception context retaining a primary failure; no successful restoration is inferred from it. The local projection sentinel and final unpatched actual checks are byte-preserved.

Complete Git mode/blob map comparison against4fc4089b changes exactly four paths: the new literal JSON, new recipe, construction script and RECOVERY-P1B2 appendix. All other entries, including guard/recognizer, controls/helpers, production, recovery target, conftest, dependencies, process/reaper/transport/receipt machinery and resource policy are identical. Prior document is an exact byte prefix of the successor; git diff --check passes. AST parsing of selected Python sources and parse/compile-only of the two reviewed scripts passed without execution.

Prior F1/F2/F4 and owning-reference F3 closures retain their bounded source meaning. Unchanged successful inventory remains52: pins1 + initial/final actual-operands/results4 + projections36 + primary-complete9 + controls-complete1 + construction-complete1. Projections36 are primary baseline/model/repair27, non-CALL2, ambiguity2, owner/NULL4, sentinel1. Script requires51 before completion and52 after; caps remain64 records/4096 ASCII bytes including newline each/65536 total. Nine literal dimensions, actual fixed offsets/cache/singleton scan, exact carrier identity/stored fields, foreign passthrough/rejection, delta/first-failure/repair and output-before-assert obligations are preserved. These are source counts and obligations, not measured successful output.

## Frozen future argv — NOT AUTHORIZED OR RUN

These are the complete current recipe and inner-script argument vectors, retained for review; SOURCE must be a separately prepared clean checkout at the executable freeze, not the current documentation HEAD, and DEST a new private absent path. F1 must be corrected and successor pins reviewed before use:

```text
python3 SOURCE/docs/agent-interface/evidence/s-construction-source-recipe.py --checkout SOURCE --freeze c181818b3dd6768751e6eaa916260339a3cc051a --destination DEST
```

```text
timeout 60s python -u docs/agent-interface/evidence/s-exit-context-construction.py --manifest manifest.json --manifest-sha256 e942dd92d93804af6c52301a0f236a78a51d5965bccba7729dedb9d249ffa4f4 --executable-freeze c181818b3dd6768751e6eaa916260339a3cc051a
```

Inner working directory is /s-closed-candidate. This builder is not the complete outer orchestration driver. Before any future ONE attempt, freeze/review its exact driver/Dockerfile/argv, separate recipe hashes, actual copied modes/bytes and manifest, Linux CPython3.14.8/all42 locked identities, no pull/install,60-second bounds,CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/private tmpfs1GiB, serialized ownership/admission/headroom and complete private input/stdout/stderr/identity/resource receipts. Binary LIVE retention export and independent hash equality precede exact-owned cleanup; failed retention preserves the hold. Missing/extra record, wrong layout/stage/outcome, restoration/release or bound failure means STOP, no retry. Fresh independent actual-receipt review and parent decision precede any separately authorized maintenance gate. The consumed r6 invocation remains consumed.

All255 remain a complete future S gate; historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. False8/True9 ordered same-object/attributes/native-inner routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion remain required. Synthetic transaction-return/original-inode controls retain their limits. Projection qualifies decoded predicates only; unpatched actual decode qualifies static recognition/cache mapping only. Neither proves live f_lasti/unwinding/native FULL or S acceptance. No H qualification transfers. Late genuine completion FULL, H/P/overlap/grouped/integration, blocked-pipe/storage, other-family/native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation gates remain separate and open.
