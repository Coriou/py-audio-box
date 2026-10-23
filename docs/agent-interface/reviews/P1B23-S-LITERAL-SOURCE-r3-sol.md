# P1B23 S literal — independent SOURCE review r3

2026-10-23 · Sol low · **SOURCE NO-GO for finite construction readiness ONLY.** Four blocking source findings remain. The narrow recognizer delta is within the authorized scope and preserves the other S requirements; its construction evidence is not ready. No construction, maintenance, all255 gate, S acceptance, integration or release approval follows.

## Pins, authority and evidence

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-exit-context-source-20261023`, executable `8254bede688ccb69afeac1bb6a90dc386bacf2ae`, documentation HEAD `c68f13c9fa762ddf0add1185a8e9a8c7845e6586`. HEAD and empty porcelain independently verified. Comparison is NONobserver S `98fbff2a45b18bce2ca53a60a945ce9bc1e52319`, executable `fc8fba40ab4202de3103926581136b459c4e56d4`; executable parent is 98fbff2a. Executable-to-HEAD changes only the appended RECOVERY-P1B2 handoff. Sole write is this root report; no commit.

Consulted the new complete appended handoff and owning brief, S-FINAL-ACTUAL-r2-sol, S-WITNESS-SOURCE-r3-sol and its reproduced original brief/findings/responses/contracts, H-EXACT-CONTEXT-DESIGN-ASTRA, and fresh H-PROJECTION-SOURCE-r6-sol. The H report was a list of pitfalls to independently test against S, not qualification transferred to S. Original F1 transient-frame accounting, F2 direct-hardener/control prerequisites, and F3 transaction-return/original-inode source closures retain their prior bounded meaning. Wrong-order still proves skipped prior configuration plus late AUTH, not executed-SQL configuration-order coverage. The latter two F3 controls remain synthetic, not native successful-transaction/alias receipts.

Accepted S diagnosis remains attributed retained evidence: actual decision3/detail1, zero LOAD_CONST74 candidates, executing LOAD_SMALL_INT74 at2574, CALL1 at2576, next non-cache2584, actual caller2576. Later False8/event/native-inner comparisons matched observationally after the original rejection; they were not original acceptance. Sa;Ss; without Sf; is RED. Historical PID49's first failure, True/fresh positives and counterfactual uninstrumented timing remain unproved. This review did not re-audit the private runtime archive.

Only host text/Git/hash/stdlib AST parsing and byte comparisons were performed. No candidate imports or definitions executed, pytest/collection, runner compilation, Docker, SQLite/database opening, runtime experiment, fix, delegation, push or cleanup. No applicable AGENTS.md found in checked ancestry/scoped trees. Memory keyword search supplied no relevant task fact. Root preexisting STT artifacts were preserved; root is not claimed globally clean. No runtime resource acquired.

## Blocking findings

### F1 — actual operand flags are recorded but not validated

`tests/service/test_child_wait_guard.py:1093–1115`.

S improves on H by reading actual raw bytes, checking CALL1 and immediate SMALL_INT74, and resolving the two selected name operands with `>>1`. However, the low bits are only emitted as owner_flag/attribute_flag; no assertion validates them against the supported explicit-PUSH_NULL representation. The selected LOAD_GLOBAL/LOAD_ATTR/PUSH_NULL raw opcodes are not explicitly compared to their expected opcode values, nor are decoded full operands retained and compared with the selected raw operands/resolved indexes. The three-field projection records intentionally omit Instruction.arg, so they cannot supply that actual operand check. The no-EXTENDED_ARG assertion checks the chosen instruction itself; adjacency excludes an intervening prefix, but a prefix immediately before the owner is not explicitly excluded or compared through the decoded full operand.

Independently read matching decoder `/tmp/pab-h-r6-decoder-i3g3e0f_/dis.py`, SHA256 `17d4d9a9195c9ee5aa27a96f3e0871f9363fd501e3a64ef20cb1cc876d7ad553`. Its ArgResolver uses `arg//2` for both selected names and `arg&1` for the NULL / NULL|self flag; SMALL_INT resolves to its operand itself, while CONST indexes co_consts. `_unpack_opargs` uses two-byte units and accumulates EXTENDED_ARG. Ordinary get_instructions does not produce separate CACHE records merely because show_caches=True. No host opcode interpretation or Docker acquisition was used.

Required source correction: retain bounded actual decoded operands separately from the immutable three-field projection model; validate selected raw opcode/operand/full decoded operand/name index/flag relationships and unsupported-prefix rejection, and emit the observed values before assertions. Explicitly validate the supported flag contract rather than accepting either bit by omission. Preserve immediate SMALL_INT versus constant-table CONST semantics. No actual flag value, compiler cause or S layout pass is invented here. The handoff's claim that low flags are validated is currently too strong.

### F2 — actual and projected failures lose result diagnostics before assertions

`test_child_wait_guard.py:1116–1125,1197–1204,1206–1231,1244–1249`.

The actual interval result is asserted before actual-result is emitted. Each legal cache-site result and the argument/next rejection are evaluated inside assertions; the actual CALL scan is asserted before its result record. If the first actual interval assertion fails, no actual recognition result or ordered clause attribution is retained. actual-raw-input and actual-input contain inputs, not those observed results. The final actual path repeats the same problem.

Projected primary inputs are emitted only after the unaltered baseline probe and delta assertion. Baseline, primary, repair, foreign and ambiguity guard results are asserted without emitting their observed result values; primary-complete is only reached after success. `_s_exit_first` supplies an expected first-clause comparison but no emitted observed clause vector/result on failure. This is weaker than the adopted output-before-assert requirement and can leave the next decisive failed premise unattributed.

Required correction: capture bounded observed intervals/site results, cache/adjacent results and singleton scan, and emit before asserting them; include bounded ordered predicate diagnostics where required. Apply this to the projected baseline/primary/repair/foreign and uniqueness paths as well. Keep fixed expectations and exercise the real guard; a copied diagnostic expression is not an acceptance oracle. Maintain the64/4096/65536 output caps and independently freeze the revised finite record inventory.

### F3 — finally restores patches but does not release fixture references on all failed paths

`test_child_wait_guard.py:1129–1161,1178–1269`; script `:86–97`.

The installed decoder patch and runner_code_S are restored in finally, including the intentional sentinel. This is not evidence of an installed-patch leak. But carrier/foreign/records/dispatch/check references are never cleared in the projection owning scope. Assertions during clone-field comparisons precede its try, and failed foreign/primary/repair assertions retain these objects through traceback frames/closures. The controls scope likewise retains model/repaired/isolated/ambiguous streams and closures without an enclosing reference-release finally.

The script does clear witness.runner and runner_code_S on the ordinary failure path, an improvement over H. It nevertheless asserts decoder restoration BEFORE that clearing, so a restoration failure skips even those assignments. Its runner/actual/witness references remain in the failed main frame. Invocation-local references can survive through a retained traceback; locality alone is not the required ALL-path release.

Required correction: initialize owned references and place cleanup in finally covering the complete owning scopes, including clone construction/comparison, controls, and the actual witness. Perform guaranteed clearing even when restoration checks fail, then report/assert restoration. Retain actual strongly only for the intended invocation duration. Do not broaden production cleanup or change guard authority.

### F4 — exact field-delta expectations are derived from the actual baseline

`test_child_wait_guard.py:1209–1215,1233–1257`.

`expected_delta` repeats the projection's comparison against actual records. It proves the projection follows changes, but does not independently require the fixed challenged field inventory. For example, if either decoy slot already equals a requested projected value, both delta and expected_delta omit that field and construction can continue. The contract instead requires the explicitly frozen decoy changes, with a coincident baseline causing STOP rather than silently reducing the challenge. Isolated non-CALL merely emits its claimed delta; owner/null and ambiguity have no independent complete delta assertion.

Required correction: independently declare the exact typed changed-field inventory for each fixed row/supplementary model, require the baseline permits that inventory, and assert preservation of offsets/count and all other fields. Retain the decoy pair during repair and CONST during boolean repair. This is a bounded control-proof correction, not permission to add dimensions or generate a campaign.

## Recognizer, scope and preservation assessment

The new exit_intervals_S requires exact captured code identity, runner globals os identity, code<=16384/stream<=4096, exact-int bounded increasing offsets, and adjacent non-CACHE slots LOAD_GLOBAL os / LOAD_ATTR _exit / PUSH_NULL / admitted literal / CALL. Literal is exact int74, excluding bool,73 and dynamic LOAD_FAST; CALL is exact-int arity1. It returns each CALL-to-next-non-cache interval. exit_site_S requires exactly one candidate and caller-lasti containment. at_exit_S calls that helper after unchanged same-active-object/attribute/burn checks. There is no arbitrary forward CALL search, generic whitelist, name fallback or ambiguous-site acceptance. The explicit shape is a prospective STOP contract, not an already corroborated S raw layout. No separate callable-binding bypass was found in this narrow frozen-code/test-owned context.

Independent byte checks establish:

- Removing only the two new S methods and restoring only the old final-recognition block reconstructs the ENTIRE98fbff2a guard byte-for-byte. Thus strict complete False8/True9 ordered code/frame/exception-event tuples, native inner seam, same object/type/args/code/name, burn, configuration/hardener/continuous authorizer, private/session/snapshot/intent bindings and every other guard are preserved.
- ENTIRE original child-control file is the exact64052-byte prefix. All original78×2+11 declarations/assertions remain. Appended205 lines contain only the five construction helpers. No arbitrary unrelated helper mutation found.
- Recovery tests, production recovery.py and conftest are byte-exact to98fbff2a. The complete changed path set is only S guard, appended controls, authorized new construction script and append-only handoff. No observation plugin, production, bootstrap, aggregate, dependency, token/cap/policy, process/reaper/transport or recovery assertion change.
- Script AST selects the unique source catch-all os._exit(74) call, captures the loaded runner, and compiles only extracted helper definitions in its prospective execution. It does not rebuild, patch, assign or invoke runner code. Host review parsed source only. git diff --check passed; candidate remained clean.

Frozen SHA256 independently recomputed:

- guard `cf58a559189912c39711ad2cd4b871919bf5a4bb18c85c85a69392678e774ca7`
- child controls `eef7b76c3d28e16cffc9a401f4f8e81e9ca092f0c8b3ccf1dcd3205c8e75d944`
- construction script `5cd25018582852c722b0ca063759da6b58afdf41e14a97264fd963019475a031`
- recovery tests `5174830dcfdd64f717bb15df6b9fdc1f27ef0e8e55b9919f4a4289e2108d29cb`
- production recovery `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`
- conftest `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3`

## Finite construction inventory and evidence limits

The projection model is accepted as a design: immutable namedtuple(offset,opname,argval), exact carrier `is` dispatch, co_name-only inspection clone with other co_ fields compared, foreign and actual decoding delegated to the captured real function. No projected stream is executable-bytecode or live-unwind evidence. Carrier hit checks exercise real S methods; `_s_exit_first` is diagnostic only. Finally restoration and sentinel exist, subject to F3's distinct release gap.

Nine literal primary rows are fixed:74 True;73/False/True/other-attribute/earlier74+73/earlier74+dynamic/arity0/arity2 False. Each has baseline, primary, repair; ordered first failure implies the preceding predicates passed. Boolean repair retains CONST; decoy repair retains the consumed earlier pair. Supplementary non-CALL, ambiguous CALL, owner and NULL controls/repairs and per-projection foreign identity checks exist. Ambiguity requires two intervals and final unique-site rejection. Bounds and actual/foreign passthrough checks remain prospective assertions.

Static successful inventory is36 projection invocations:27 primary baseline/model/repair, two non-CALL, two ambiguity, four owner/NULL, one sentinel. Two unpatched actual passes each check four legal offsets2576/2578/2580/2582, reject2574/2584, and scan all actual CALLs for singleton2576. Straight-line successful output is31 records: pins1, actual3×2, primary2×9, supplementary4, controls-complete1, construction-complete1. These are source counts, not executed/collected/passing results. Emitter enforces<=64 records, <=4096 ASCII bytes including newline each, <=65536 total before printing. Code/stream bounds and fixed local cache interval are finite. Bounds fail closed; they do not establish adequate diagnostics, target flags or actual completion. F2 corrections must fit a newly reviewed inventory.

Manifest schema/freeze/digest and required four paths are checked before candidate import and after controls; owner must separately bind ALL imported source entries to Git freeze. Script pins Linux/CPython3.14.8. The60-second outer deadline, CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/private tmpfs1GiB, source-copy/no-pull/install, lane ownership/headroom, private retention, LIVE binary export/hash and exact-owned cleanup are proposed driver obligations, not enforced or observed by this source-only review. No driver execution or resource receipt is qualified here.

Correct/freeze/review source first. Any subsequent ONE finite construction still needs separate parent authorization, exact source/environment/manifest and retained proof, independent actual-receipt review, then parent decision. No automatic S maintenance launch, retry, lowering search or all255 substitution. Unpatched decode can establish only static actual recognition/cache mapping; projections can establish only decoded predicate-boundary controls. Neither proves native FULL, final active exception propagation or S acceptance.

ALL original obligations remain open: native False/True/fresh positives with strict4096/61/0/effective61, once-only native PRAGMA1 before unchanged hardener/policy, continuous authorizer/no preprepared workload, exact first private INSERT OperationalError13/prefixzero, no receipts/scan/original completion, both burn routes, complete exact8/9 frame/code/event-offset/native-inner same-object/attributes final route; every original parent refusal/trap/unaccepted/source/count assertion and all directed controls/restoration/publication/ownership receipts. All255 remain a complete future gate; historical four passes/one failure/250 unrun are not a reduced gate or255 successes.

Late genuine engine FULL at completion after receipts/scan remains UNPROVEN; early S/A/B/R do not fill it. H/P acceptance/overlap/assembly/grouped, R15/R5/J/T/C/declared73, owner/transport/close/reap/old-core fresh resume, four-B2/conditional service-release/T2/C1295/final global Astra, mandatory C1 launcher stub b2_pinned_launcher_context_unavailable/B3/C1/C2, native Darwin/A1, all19 Linux/ext4/VFS/WAL/FULL/SHM/peer-credential/installed durability, Apple Silicon/hardware/browser/acoustic/activation remain separate and open. Historical trees, observer/proof and STT artifacts untouched. No resource cleanup performed or needed in this review.
