# P1B23 S checkout mode — independent ACTUAL/source audit r11

2026-10-23 · Independent Sol low · **Bounded diagnosis GO. Construction/readiness remains NO-GO on the current driver: the consumed attempt stopped at checkout mode admission. No product acceptance or further execution authority.**

## Scope and provenance

Sole write is this root report; no commit. Read FULL S-CANONICAL-DRIVER-CONSTRUCTION-r11-20261023, S-HOST-LANE-SOURCE-r11-sol and S-CONSTRUCTION-DRIVER-SOURCE-r10-sol, original construction brief at candidate docs/agent-interface/phase-1/RECOVERY-P1B2.md:2830, complete current driver, source recipe and Dockerfile. No applicable ancestor AGENTS.md was found. A quick memory registry search yielded no relevant S fact; no memory fact supports this verdict.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-driver-host-path-r11-20261023` is clean at documentation HEAD `384af0b5adcc3829ee2f555e1602455f900c1893`; executable `c445017a4800f613dd1a28641890934aa9df3654` to HEAD changes only RECOVERY-P1B2. Driver SHA256 `9df294cfeaab023aedd1d2bd9530307f20caab67584b39232203254a2f7065a5`; Dockerfile `d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2` independently match retained/source bytes.

Only reviewer-owned stdlib text/JSON/hash/stat and read-only Git inspection were used. No candidate import or definition execution, recipe, pytest/collection, Docker, SQLite, runtime experiment, chmod, fix, retry, delegation, push or cleanup. Root has preexisting unrelated STT artifacts; they were untouched. This audit neither creates a worktree nor changes historical trees.

## Actual receipt audit — confirmed early STOP

Evidence `/private/tmp/pab-s-canonical-driver-construction-r11-20261023`; sibling authorization/captures `/private/tmp/pab-s-canonical-driver-authorization-r11-20261023`.

Independently reread and verified ALL22 entries in evidence-hashes.json, including final-state, against retained bytes. Inventory intentionally excludes checkout and cannot hash itself. Independently hashed all five sibling leaves; every size/hash matches the canonical actual report: host-argv712, host.exit2, host.stdout5, host.stderr1647, parent-authorization2376 bytes. All sibling leaves are regular0600, both private parents0700, observed UID501. Retained driver, source review and authorization equal their respective source/original bytes. Host argv agrees with invocation.json and the canonical reported vector; raw stdout is exactly STOP plus newline, raw exit is1 plus newline. Complete stderr agrees with STOP traceback and state.

Exactly FOUR attempted external dispatches are retained: worktree add, HEAD, all-untracked porcelain, full ls-tree. Each pending/completed argv pair agrees, all completed exit0 without timeout/overflow/failure, and both stream sizes agree with receipts. No cat-file dispatch was reached in the attempt (reviewer read-only cat-file checks are separate). No unexplained additional command receipt appears in the verified inventory.

Private registered detached checkout HEAD independently equals `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`; all-untracked porcelain is empty, corroborating retained HEAD/empty status. Git status does not establish full permission equality. Recipe and literal list are each observed0600 although retained Git tree says100644 and checked() requires0644. Both independently equal exact Git blob bytes:

- recipe SHA256 `1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74`;
- list SHA256 `5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6`.

Traceback driver:199 → checked:197 → regular:30 identifies the FIRST recipe mode assertion. Thus ONE host invocation, host exit1, FOUR Git dispatches, ZERO recipe/Docker/inner construction invocations, inner exit UNREACHED, records0/52. No manifest was generated/admitted, copied-source proof or runtime qualification was reached. Expected manifest remains `66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`, 263 files/29390 bytes, not an actual result.

## Cause and complete remaining host preparation mode audit

Driver:15 sets process umask077 before driver:183 worktree-add. The subprocess inherits it; observed0600 regular files are consistent with creation of non-executable files under that mask. This is the source-consistent causal diagnosis, distinct from the directly observed modes/bytes/traceback. No controlled Git experiment or Git implementation tracing was performed; no claim about other checkout file modes is inferred from these two observations.

Audited ALL remaining driver host preparation paths and the complete recipe for the same mismatch:

- Existing driver/Dockerfile are admitted0644 before checkout; they are read, not created under this mask. External source-review/authorization have no imposed0644 admission. Retained review, authorization, driver, invocation, command streams/receipts, input/execution freeze, host checks, STOP/final inventory and owner documents are private writes under077. Readers requiring0600 therefore agree. No new0644 admission is imposed on the retained driver copy.
- Evidence/context directories explicitly0700; lane default mkdir under077 is0700, owner documents0600. Checkout remains behind evidence0700 even if newly created checkout directories become0755 under a future scoped022. No historical permission change is needed.
- Recipe snapshot begins private, then explicit same-fd fchmod0400; readback and inherited unlinked fd require0400. It does not depend on ambient mask for its final admitted mode.
- Recipe destination root explicitly0700. Intermediate mkdir0755 under inherited077 yields private directories; no code requires their exact0755 mode. Every payload leaf begins0600 and is explicitly fchmod'd to its Git0644/0755 mode before readback. Both manifest.json and driver-recipe-freeze.json use the same writer with100644 and explicit fchmod0644. Therefore restoring077 before recipe launch does NOT reproduce the checkout leaf mismatch in output/manifests. Repeated hostchecks use the same exact Git-mode reader on checkout/copied leaves; their requirements must stay intact.
- Context Dockerfile is written0600, with no later0644 assertion. Build context is read by the host owner. Dockerfile COPY --chown=10001:10001 assigns copied source ownership to the declared UID/GID10001; private source directories consequently remain traversable by that user. Host context/source ancestors are not runtime bind mounts. Container verification checks copied regular type/Git modes/bytes; Docker COPY ownership is explicit. No demonstrated additional directory or Dockerfile mode blocker follows from077. This is source reasoning, not an observed Docker build/access receipt.
- Remaining host tar/JSON/hash readers, source traversal and final inventory impose no conflicting newly-created0644 requirement. Container tmpfs1777/sentinel0600 are explicit independent runtime policies, not host checkout permissions.

No second same-class preparation stop was found. This does not predict runtime success or waive any later admission.

## Smallest correction and static closure before a future attempt

Precise proposed executable allowlist: ONLY docs/agent-interface/evidence/s-construction-driver.py, surrounding the ONE `call('prepare-checkout', ...)` at checkout creation; append the existing RECOVERY-P1B2 handoff for provenance. No recipe, Dockerfile, list, application pin, expected manifest, mode admission, command wrapper, output audit, resource policy or historical tree change is needed.

Keep global/private umask077. Immediately before the worktree-add call, save the current mask while setting022; put that exact call/assertion inside try; restore the saved077 in finally, before any later Git/read/recipe/receipt path. In outline: `checkout_mask=os.umask(0o022); try: [existing call/assert]; finally: os.umask(checkout_mask)`. This is lexical scope for Git checkout only, including restoration on nonzero/exception/publication failure. call() also publishes its prepare-checkout receipt/streams inside that scope: their temporary default0644 is contained by the existing0700 evidence parent, and their readers do not require0600. If leaf0600 for those particular captures is retained as a required policy, make only those wrapper output creations explicitly0600 within this same narrow correction; do not silently claim022 preserves leaf0600. All other evidence stays under077. Parent privacy remains0700 throughout.

Static closure must verify the complete diff/AST: exactly one scope, restoration in finally on every returning/raising path, no other command/recipe runs under022, evidence mkdir0700 precedes it, no global022 or dropped Git-mode check, no chmod of historical checkout, and no changed recipe/list/Dockerfile/application/manifest/audit bytes. Recheck complete host-preparation mode paths above against final proposed source and explicitly account for prepare-checkout captures. Freeze the successor driver/hash and documentation provenance for independent source review before any separately authorized fresh attempt. This report performs neither implementation nor retry and replenishes no invocation.

Alternative: an explicitly authorized future owned-worktree creation policy may establish Git-compatible modes at creation while retaining the0700 private boundary. It must be a reviewed source change with finite scope, not an ad hoc chmod of retained evidence or weakened admission. The current scoped umask correction is smaller.

## Ownership, handback and verdict limits

Final-state independently confirms failed=true; evidence acquired; checkout command-returned; context/recipe/lane/owner/base-tag/build/create/start/stop/remove unattempted; container_id null, uncertain_commands empty, remaining_seconds null. Proposed resource names are declarations, not acquired objects. Source control flow and four receipts support ZERO runtime resources acquired by this attempt; no Docker census was run or needed for this bounded conclusion. The retained registered checkout/private evidence/auth captures are the actual residuals, intentionally untouched. Private/tmp retention is volatile, not indefinite durability.

Bounded diagnosis GO does not mean construction success. Historical4 passes/1 failure/250 unrun, Sa;Ss; without Sf; RED, exact False8/True9/native-inner/both-burn/authorizer/INSERT13/prefixzero requirements, all255/native unwinding/FULL and other product/platform/release gates remain unchanged. H qualification does not transfer. No product acceptance, cleanup, source fix or execution authority is supplied.
