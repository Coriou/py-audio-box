# P1B23 S host-path actual audit r10

2026-10-23 · Independent Sol low · **Bounded GO for the early-STOP diagnosis and canonical caller-input proposal only. NO-GO for declaring the whole unchanged command host-compatible: one concrete hard-coded lane read remains incompatible. No execution, construction success or S acceptance is approved.**

## Scope and provenance

Sole write is this root report; no commit. Read FULL S-REVIEWED-DRIVER-CONSTRUCTION-r10-20261023 and S-CONSTRUCTION-DRIVER-SOURCE-r10-sol. Inspected the complete frozen driver, owning construction brief at RECOVERY-P1B2:2830, original S handoff/brief and revised/shared final-exception requirements, r8 recipe review and r10 D1–D4 response. Historical combined output truncation was recovered with focused reads for controlling sections; this is not a fresh all255 or historical-runtime audit. No applicable AGENTS.md appeared in checked ancestors/scoped inventory. Memory search supplied no relevant S evidence.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-driver-d1-d4-r10-20261023` has HEAD `e02de53038f221b14c32fa75bb9e1d870d6f4129` and empty all-untracked porcelain. Current driver/Dockerfile bytes independently equal Git executable freeze `0dfbd74116a5bfbf17ad490d21713a24101a68a6`:

- Driver SHA256 `48365949744d01120bdde7ebef537a8adb6b91b7327c637e1137dd39159a27b6`.
- Dockerfile SHA256 `d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2`.

Only reviewer-owned host filesystem/text/hash/JSON and read-only Git inspection was used. No candidate import/helper execution, recipe, runtime experiment, Docker query, pytest, SQLite, fix, retry, delegation, push or cleanup. Existing root STT artifacts remain untouched.

## Actual partial receipt independently verified

`/tmp/pab-s-reviewed-driver-construction-r10-20261023` and `/private/tmp/pab-s-reviewed-driver-construction-r10-20261023` currently identify the same directory device/inode, mode0700. The latter spelling was used to read the existing object; no relocation/copy/new attempt occurred. Its entire inventory is exactly two regular mode0600 files:

| File | Bytes | SHA256 |
|---|---:|---|
| STOP.json | 1370 | `7ef56d6947dc56dc574ed6055d69d1257dbef323f2ef0a24b582b26505519ace` |
| independent-source-review.txt | 15871 | `7a07d358d490c393428bba616122baa85a18a462964e71d2d270a79162341381` |

The copied review equals the current root source review byte-for-byte. Retained authorization SHA256 is `545e8312c6e5f1174ecf19bd5505677851e01e3cbe416529d6fc54abafcec4f5`, leaf0600 in parent0700; its contents are not reproduced.

STOP.json independently confirms evidence=acquired; lane, lane_owner, evidence_owner, checkout, context, recipe, base_tag, build, create, start, stop and remove all=unattempted. Proposed owner resource names are declarations, not acquisitions. The primary retained traceback is driver:177 → regular:26, `NotADirectoryError: [Errno 20] Not a directory: 'tmp'`, during authorization admission after source-review copying and before the first external dispatch at :183.

The construction report records host exit1/stdout STOP, container_id=null, uncertain_commands=[], remaining_seconds=null and secondary final-inventory failure at :381 through the same `/tmp` spelling. These latter stderr fields are attributed to that report: STOP.json contains error/state/owner only. Complete raw host stderr was not retained in this two-file bundle, so I cannot independently rehash/replay the secondary traceback or those stderr-only fields. Missing final-state.json/evidence-hashes.json is confirmed; the two independent hashes above do not substitute for them. No fresh daemon census was needed or performed.

**ONE host attempt consumed; external dispatches0; inner construction invocations0; construction exit UNREACHED; records0/52.** The observed failure precedes checkout, recipe/source copy, lane acquisition and Docker dispatch. No actual environment, imports, 263-file copy, decoder, nine-dimensional result, repair/restoration, archive/LIVE equality or cleanup/release gate was reached. No product failure or pass is inferred.

## Host components and minimal caller-input proposal

Host lstat/readlink confirms `/tmp` is a symlink storing `private/tmp`. Reviewer directory-relative O_DIRECTORY|O_NOFOLLOW traversal from `/` independently admitted every ancestor of canonical `/private/tmp`, repository, root source-review, driver, existing evidence and authorization paths; leaves were inspected without following symlinks. `/private/tmp` is a real directory mode1777; canonical evidence/auth parents are0700. Thus the observed failure is the intentional no-follow rejection of the alias, not absent authorization content. These are current component checks, not future race guarantees.

Canonicalizing an actual caller input once in the parent, before freezing argv, and checking its resulting real components is compatible with the existing reader without code changes. Use canonical existing parents plus fresh absent leaf names for future evidence; do not resolve by creating evidence, reuse the consumed directory or weaken O_NOFOLLOW. SELF/Dockerfile already have canonical checked ancestors; repository/source-review remain their checked `/Users/...` paths. Canonical evidence propagates to C=R/checkout, context, snapshot, selected source/data and final inventory, including recipe argv/cwd. Canonical authorization removes the observed :177 failure. Parent must supply fresh authorization explicitly binding any future attempt.

Exact proposed caller vector, **NOT RUN and not approved as a complete successful unchanged-driver route**, with fresh names proposed but not created or reserved:

```text
python3 /Users/ben/.txcode/worktrees/py-audio-box/s-driver-d1-d4-r10-20261023/docs/agent-interface/evidence/s-construction-driver.py --repository /Users/ben/Projects/py-audio-box --evidence /private/tmp/pab-s-reviewed-driver-construction-r10-canonical-20261023 --driver-sha256 48365949744d01120bdde7ebef537a8adb6b91b7327c637e1137dd39159a27b6 --dockerfile-sha256 d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 --source-review /Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-S-CONSTRUCTION-DRIVER-SOURCE-r10-sol.md --authorization /private/tmp/pab-s-reviewed-driver-authorization-r10-canonical-20261023/parent-authorization.txt
```

## Concrete residual gap: lane owner no-follow read

**Driver:61 fixes `lane=Path('/tmp/pab-worker-w1-docker-lane.lock')`; :366 calls `regular(lane/'owner.json',0o600)`.** Lane acquisition uses exists/mkdir/lstat and ordinary owner-file open, which traverse the host alias. However, successful release requires the later independent no-follow reader. That reader starts at `/` and rejects `tmp` before it can compare owner contents. This occurs before release-pending/unlink/rmdir; the handler latches STOP and retains the lane. It is not merely a mkdir path and is not corrected by canonicalizing any argv field. Parent cwd, canonical authorization/evidence/repository and Python script spelling cannot change this absolute constant.

Other residual host `/tmp` uses are disk_usage (:143,146), which follows the alias and measures the same filesystem. Container `/tmp` uses—LIVE walk/archive, tmpfs declaration/inspection, wait log, sentinel and capacity—belong to the container namespace and are not host no-follow inputs; substituting host `/private/tmp` there would alter the reviewed contract. No such substitution is proposed.

Therefore caller-only canonicalization receives bounded source-compatibility GO for its actual input scope, but **cannot receive whole-command compatibility GO for this frozen driver on this host**. This is a concrete gap, not grounds for a broad new implementation campaign or general permission framework. If the parent requires a success-capable host route, the smallest identified source issue is the host lane spelling reaching :366; any correction must preserve the same lane object, exclusive ownership checks and no-follow safety. No correction or additional review round is performed or authorized here. An unchanged-driver attempt would knowingly retain this later STOP risk and cannot be represented as repaired by argv alone.

## Gates retained

The consumed attempt remains consumed. No execution authority is replenished. Linux assertion-enabled CPython3.14.8/all42 locked identities/decoder pins, resource and reserve gates, complete raw construction exit0/all52 exact records, copy-before-import, binary all-/tmp/LIVE retention and exact-owner stop/removal/absence/lane release remain unproved. No S construction or all255 acceptance follows.

Historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. False8/True9 ordered same-active-object/attributes/native-inner routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion remain required. Projection/static decoding and synthetic controls retain their stated limits. Late genuine completion, native unwinding/FULL, maintenance, H/P/overlap/grouped/integration/broad, blocked-pipe/storage, native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation gates remain open. No H qualification transfers.
