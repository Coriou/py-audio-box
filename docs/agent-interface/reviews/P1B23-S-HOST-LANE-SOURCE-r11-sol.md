# P1B23 S host lane — independent SOURCE review r11

2026-10-23 · Independent Sol low · **Bounded SOURCE GO: the r10 host-lane residual gap is CLOSED for readiness to consider ONE separately parent-authorized finite construction on this host and these exact bytes. No execution authority, retry authority, construction success or product acceptance is supplied.**

## Scope and provenance

Sole write: this root report; no commit. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/s-driver-host-path-r11-20261023` inspected READONLY. HEAD is documentation `384af0b5adcc3829ee2f555e1602455f900c1893`; executable commit is `c445017a4800f613dd1a28641890934aa9df3654`; all-untracked porcelain is empty. Compared against executable `0dfbd74116a5bfbf17ad490d21713a24101a68a6` and documentation `e02de53038f221b14c32fa75bb9e1d870d6f4129`. Executable-to-documentation changes only RECOVERY-P1B2. The parent's frozen uncommitted delivery was not altered by this reviewer. The inherited r11 handoff describes that earlier uncommitted delivery; its provenance wording is historical, not the current clean committed candidate state supplied in this assignment. This does not change the verified bytes or confer authority from a commit.

Read FULL root S-HOST-PATH-ACTUAL-r10-sol and S-CONSTRUCTION-DRIVER-SOURCE-r10-sol, complete r11 handoff, inherited D1–D4 response, original construction brief at RECOVERY-P1B2:2830 and associated original/prior requirements carried in those reviews. Read complete current driver and Dockerfile. This is a narrow host-lane preservation review, not a fresh all255/historical-runtime audit. Checked root/candidate ancestors supplied no applicable AGENTS.md. Memory registry search returned no relevant S evidence.

Only reviewer-owned host text/hash/JSON, lstat/readlink/stat and read-only Git inspection were used. No candidate import/helper, recipe/source-copy, pytest/collection, Docker, SQLite, runtime experiment, fix, delegation, push, cleanup or resource acquisition. Existing root STT artifacts were untouched. No claim that the root is globally clean.

## Exact minimal change and preservation

Independently required current driver bytes to equal the complete old driver after exactly one literal replacement plus three specified comments. The only changed executable statement is:

```text
lane=Path('/tmp/pab-worker-w1-docker-lane.lock')
→ lane=Path('/private/tmp/pab-worker-w1-docker-lane.lock')
```

Everything else in the executable is byte-identical, including regular() O_DIRECTORY/O_NOFOLLOW/O_NONBLOCK traversal, same-fd type/mode/cap/metadata/length checks; exclusive lane mkdir and owner publication; acquisition states; captured directory device/inode and exact owner-document release checks; full-ID binding and uncertainty handling; bounded commands/receipts; retention and closed source admission. The full52-record audit block is therefore byte-identical. D1–D4 source closures remain preserved, with their r10 evidence limits. No generalized framework or lock selection is introduced.

Relative to e02de530 only driver and appended RECOVERY-P1B2 differ; inherited handoff is an exact byte prefix. All preexisting tracked application-pin entries retain mode/blob except the allowed documentation file. Dockerfile/recipe/list/construction script equal 0dfbd741. Independently read all263 selected current leaves and compared their bytes to bd34b74 Git blobs; recalculated compact sorted-key manifest:263 entries,29390 bytes, SHA256 `66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`. Git diff --check passes. These are source preservation checks, not copied-input/runtime receipts.

| Input | Exact value |
|---|---|
| New driver SHA256 | `9df294cfeaab023aedd1d2bd9530307f20caab67584b39232203254a2f7065a5` |
| Unchanged Dockerfile SHA256 | `d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2` |
| Application/recipe EXEC | `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521` |
| Recipe SHA256 | `1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74` |
| Literal list SHA256 | `5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6` |
| Construction script SHA256 | `3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4` |

## Current host identity and residual-path audit

lstat(`/tmp`) identifies a symlink; readlink is exactly `private/tmp`. Independently lstat-checked `/`, `/private`, `/private/tmp` are directories, no symlink: respective device/inode16777223/2,16777223/908098721,16777223/1000678450; modes0755,0755,1777. stat(`/tmp`) and stat(`/private/tmp`) agree on device/inode16777223/1000678450. Both `/tmp/pab-worker-w1-docker-lane.lock` and `/private/tmp/pab-worker-w1-docker-lane.lock` are absent by lstat now. No existing lock inode was available to compare, and none was created/acquired. Identical basename under the same physical parent proves the same global serialization address under the observed mapping; this is not a second lane. Existing future lane/owner symlinks must never be resolved into ownership.

The r10 release reader now reaches the canonical real host components before reading owner.json, closing its concrete alias rejection. Host `/tmp` residuals are only the two disk_usage calls, permitted and same-filesystem under the checked mapping. All other `/tmp` uses belong to container programs/argv: LIVE/archive, tmpfs/inspection, wait log, sentinel and capacity. They are unchanged and must remain container `/tmp`.

Existing ancestors of driver, Dockerfile and root source-review location were lstat-checked as real directories, with inspected existing leaves regular. Proposed evidence `/private/tmp/pab-s-driver-host-path-r11-20261023` and authorization parent `/private/tmp/pab-s-driver-host-path-auth-r11-20261023` are absent now, not reserved. These point-in-time checks do not guarantee future absence, ownership or race freedom.

## Exact prospective host argv — NOT RUN

```text
python3 /Users/ben/.txcode/worktrees/py-audio-box/s-driver-host-path-r11-20261023/docs/agent-interface/evidence/s-construction-driver.py --repository /Users/ben/Projects/py-audio-box --evidence /private/tmp/pab-s-driver-host-path-r11-20261023 --driver-sha256 9df294cfeaab023aedd1d2bd9530307f20caab67584b39232203254a2f7065a5 --dockerfile-sha256 d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 --source-review /Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-S-HOST-LANE-SOURCE-r11-sol.md --authorization /private/tmp/pab-s-driver-host-path-auth-r11-20261023/parent-authorization.txt
```

The canonical evidence path propagates to checkout/context/snapshot/copied inputs/final inventory; canonical authorization avoids the original admission rejection. No `/tmp` caller alias is admitted. The two r10 reviews are supporting review context; this r11 report is the new driver review input. No authorization file is supplied by this review.

Before execution, parent must perform the following finite checks and freeze their result with the ONE authorization:

1. Recheck exact candidate executable/documentation identities, clean candidate, driver/Dockerfile hashes above and application pin availability. Recheck this completed review is nonempty, regular, bounded262144 bytes and admits through canonical no-follow components. Freeze actual host Python executable, assertion-enabled startup and argv; the driver uses sys.executable for its one recipe command. No import/collection preflight is authorized here.
2. Recheck `/tmp` symlink text and physical parent identity against canonical `/private/tmp`; every canonical component of all actual argv paths must be a real directory, no symlink. If mapping/components changed, STOP; no alternate lock. Check the existing lane if present by lstat under both spellings for same device/inode/type; symlink/mismatch/unknown ownership is STOP, never stealing/resolving/deleting it. This review found no lane. The existing exclusive acquisition and <=60s occupied-lane wait remain authoritative.
3. Recheck evidence leaf absent immediately before acquisition; keep canonical existing parent and fresh absent leaf. Parent may separately prepare the fresh authorization parent0700 and regular private authorization leaf0600, with no-follow components and explicit ONE authority binding these hashes, paths and finite scope. Do not reuse consumed r10 evidence/authorization. If a proposed name has become occupied, STOP and obtain newly frozen authorized values; no reuse or automatic retry.
4. Reconfirm finite resource/source/environment obligations: host disk6GiB; Docker memory4GiB and exact preexisting base ID `sha256:cc251f26cead1f99a309e44e3b763e524b619fcbe02e6d675c9379d09c75f68f`; no pull/install; owned base-tag absence; Linux assertion-enabled CPython3.14.8/all42 locked identities/three decoder pins; exact263 copied bytes/modes/manifest before candidate imports; CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/capdropALL/no-new-privileges/private1GiB tmpfs/no inherited mounts/devices/ports, Linux available memory4GiB and tmpfs900MiB. Dynamic checks stay in the frozen finite driver sequence and must pass before its ONE construction dispatch; this review ran none.
5. Preserve all per-command pending/completed receipts and caps: <=60s command budget plus <=5s host reap, construction stdout65536, stderr1MiB, ordinary stdout16MiB, archive1100MiB. Host filesystem/tar/JSON work remains cooperative; CLI termination is not remote completion. No failed command/audit/construction retry, alternate recipe or automatic maintenance is approved. Complete binary ALL-/tmp archive and independent LIVE equality/sentinel proof must precede owned stop/removal; uncertainty retains resource/lane with finite handback. Exact full-ID owner binding, exact-owner lane release and final private receipt/hash inventory remain mandatory.

Inner ONE construction stays `docker exec -w /s-closed-candidate OWNED_FULL_ID timeout 60s python -u docs/agent-interface/evidence/s-exit-context-construction.py --manifest manifest.json --manifest-sha256 66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46 --executable-freeze bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`. Raw exit0 and all52 exact ordered records remain required. Future independent actual-receipt review and parent acceptance must assess copied inputs, environment, raw output, retention, command uncertainty, exact-ID cleanup/absence and lane release. Nothing here claims those outcomes.

## Retained boundaries

Old r10 attempt remains consumed STOP: external dispatches0, inner construction0, records0/52. It is neither a candidate construction failure nor retry authority. This SOURCE GO only removes the identified host-lane source obstruction; it replenishes no invocation.

Historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. False8/True9 same-active-object/attributes/native-inner ordered routes, both burns, strict4096/61/0/effective61, once-only PRAGMA1/continuous authorizer, exact first private INSERT OperationalError13/prefixzero/no receipts/scan/completion and every original assertion remain required. Projection/static mapping/synthetic controls retain their evidence limits. No S/all255/native unwinding/FULL, H transfer, late genuine completion, H/P/overlap/grouped/integration/maintenance, blocked-pipe/storage, native/platform/installed durability, T2/full release/global Astra/hardware/browser/acoustic/activation gate advances. These are separate product gates, not additions to the finite host-path correction.
