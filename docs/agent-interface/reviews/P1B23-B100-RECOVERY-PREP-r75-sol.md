# P1B23 B100 recovery preparation r75 Sol

2026-10-24. **PREPARED FOR INDEPENDENT SOL ADVERSARIAL REVIEW; acceptance=false. No actual recovery authority or runtime GO.** Concrete private reader, physical/logical parser, independent hash reader mode, capture, orchestration and supervisor are implemented and frozen. No Docker command, daemon query, volume access, product/guard/pytest import, campaign, commit or delegation occurred. Held objects, lane and old proof remain untouched. Parent must accept independent review before separately authorizing ONE operation.

## Identity and preserved premises

Read full r74 audit/proposal and r73 failure; inspected existing finite-driver census/reader, launcher parser/map/capture/lifecycle/supervisor and controlling successor bounds, and current B runner's ordered/receipt/identity/terminal behavior. Memory-registry search supplied no relevant premise. No applicable repository AGENTS.md found.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-65b1e6b7` remains clean for tracked files at documentation `7a53112d1c6c757fea51b20721b9fc6200dd9bc0`; executable `f5bc92a1098266c4b3f0565198727b6e38fd61a5` remains its ancestor. All197 AUTH mapped hashes reverified without product imports. No candidate source edits.

Accepted r74 findings remain:104 inventoried leaves,17 command receipts/34 streams,017 call FAIL, USTAR100-byte linkname gap. No reinterpretation of the old failure. Exact image remains `sha256:14b0cf52e7337b449938701c0256fe1f32fe846fa25d62f8d18413a190edbbcb`; volume `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce`; stopped writer `82ccb71b8e178e06458c894849ed185db6f6ddb5da2d555ff5897fb1ab8dae93`; stopped archive `ba057c4428408f5367201d7da751c2e4378dfcaa4069b98a2e9a833653864eaa`. These are retained historical bindings, not fresh daemon observations.

Both original roots remain readonly/preserved: `/private/tmp/pab-r73-operational` and `/private/tmp/pab-p-one-r16-e49b684f-65aa-4deb-b350-39767210c3ce`. Preparation rehashed all104 inventoried files plus the inventory itself, checked exact path set, modes/UID/GID/single links, inode/device/size/mtime/ctime, and directory identities before and after tests. Failed16896-byte archive prefix remains unchanged, SHA256 `2ea52cca81c3cf72e8ffc548d1ec881a4f7b40c5ccaadb5ba38eaf2d0ce52e4f`.

Existing owner `/private/tmp/pab-worker-w1-docker-lane.lock/owner.json` was read privately for exact equality and concrete argv freezing; token was never printed. Current bytes exactly equal retained final handback owner. AUTH's earlier owner has a different PID and lacks final authorization/wrapper fields; stable token/invocation/source/documentation fields match. Requiring full AUTH/current equality was an initial preparation error, corrected in a retained successor. No owner mutation/acquire/release/replacement. Operation code observes current inode at its own start and requires equality throughout that operation; it makes no historical inode-continuity claim.

## Single selected transport: PAX successor

Selected **PAX only**, extending the existing bounded no-follow archive reader and independent regular-file hash approach. No LP1 implementation, wire-format selection flag or fallback. Frozen r74 LP1 source/report bytes were not modified. Earlier private drafts/manifests are retained preparation history; only `frozen-READY.json` selects operational inputs.

Exact local stdlib implementations of `create_pax_header`, `_create_pax_generic_header`, `_proc_pax`, `_apply_pax_info`, `frombuf` and `nts` were retained in `tarfile-audit-source.txt`. PAX represents long names/linknames using local extended headers. With UTF-8/surrogateescape, tarfile detects surrogate characters, emits `hdrcharset=BINARY`, and preserves their original bytes in path/linkpath values. This applies to long absolute links and newline/non-UTF8 metadata; link targets are never normalized or traversed. Directory headers conventionally carry a slash; the logical tar reader strips that convention. Raw component tuples and the metadata row remain authoritative.

Each logical member has exactly one local PAX `x` header containing canonical ASCII `pab.meta` JSON. This records base64 raw relative path, kind, literal link target where applicable, original stat size/mode/UID/GID/mtime_ns/ctime_ns/nlink. The following ordinary header carries the existing file/dir/symlink type, permissions and UID/GID; header mtime remains0 as in the old transport. Files carry exact bytes; original directory/link stat sizes and nanosecond metadata live losslessly in `pab.meta`. Regular hardlinks are conservatively rejected via nlink==1; no admission was weakened.

Host first audits **physical**512-byte records, checks checksums and bounded PAX lengths before allocation, parses decimal length-prefixed PAX records, and rejects duplicate/unknown keys, globals, GNU longname/longlink, sparse/special/hardlink types and extra extension chains. Allowed PAX keys are only `pab.meta`, generated `path`/`linkpath`, and generated `hdrcharset`. It regenerates the selected writer's entire physical PAX-plus-member header chain and requires exact byte equality. This binds header numeric fields, generated extension keys/values, type, zero padding and metadata without relying on tarfile's permissive extension handling.

Then a separate standard host `tarfile` logical pass checks raw surrogateescape round trips, names/linknames, type/UID/GID/mode/size/mtime, metadata and payload offsets; it independently rehashes file bytes. Logical iteration hides PAX headers; physical audit counts/bounds them separately. Raw component tuples must be unique, strictly sorted, relative, NUL-free, with no empty/dot/dotdot components; root `.` is the only exception and must be a0700 directory. Every nonroot parent must already be an admitted directory. Complete terminator, zero tail and10240-byte blocking are required. No extraction, filesystem reconstruction or symlink following.

The hash mode opens and reads regular files independently, emits canonical bounded JSONL metadata/hash rows for **all** entries, and compares the complete path/type/link/stat metadata set plus every regular hash with the exported bytes. Both modes use dirfd-relative O_NOFOLLOW traversal, O_NONBLOCK regular opens, pre/post file signatures, census bounds and final whole-census stability checks. A failed reader can leave a syntactically complete tar prefix; acceptance additionally requires successful captured CLI, remote wait, stopped exit0 and full binding. A tar terminator alone is never sufficient.

This strict canonical contract may reject a different Linux stdlib serialization; it cannot silently accept a variant. Local audit is not execution of Linux3.14.8's tarfile. Independent review must assess that version boundary; an actual mismatch is STOP, without another format or retry.

## Private frozen files and execution entry

Fresh private root: `/private/tmp/pab-r75-recovery-prep-20261024`. All final directories0700 and regular leaves0600/current UID/single-link verified. Files were created exclusively; selected generation/testing used umask077. One early nonsecret preparation successor initially inherited the shell's default file mode and was immediately tightened before final verification. New synthetic symlinks are explicitly inventoried as inert fixtures; they are not old proof or extraction outputs.

SHA256 values:

| Selected artifact | SHA256 |
| --- | --- |
| `frozen-READY.json` | `279e71b734283b56466631938e386bcfab4833a400393ea3d9f7b43340048181` |
| `entry.py` protected host supervisor | `8a48bb82f90ce3acef4900cb43769f23d5b3d06b98291af8f548d24f6b5b1508` |
| `worker.py` orchestration/recovery/publication | `691a683e60c2e0be754e28eaaa3bc6ed77902f58378135543e0afb090d20ab68` |
| `reader.py` literal isolated PID1 source | `4d0a170c2c1a93bd682dde18bdfc197b93bdd9b56c1e937d110a3447e9608215` |
| `codec_v2.py` selected codec/physical and logical host parser | `a49a9aac842890de192696fe41726e1b76b4952c4447f0f2e036eb89b5434c9b` |
| `capture.py` extracted bounded binary capture | `86ece75d0716e9d64c945d6f641c3176c15b58f744cd77c57275b8b76765f7dc` |
| `argv-export.json` concrete token-bearing create argv | `1171b3fc4e8df5e1e950dc25ba03a127df1fb6f0752939c38d09e528853845e2` |
| `argv-hash.json` concrete token-bearing create argv | `0bd4e4f77a244af337f62db4833985bbea1a3627dc5e2133f45623fb055b3eb7` |

Manifest also freezes `privateio.py`, copied stdlib-only `identity.py`, `checks.py`, old-proof baseline, retained bindings/AUTH, source provenance, local tarfile source audit and exact final test source/argv/spawn/completion/stdout/stderr/results. `reader.py` is byte-for-byte the extracted testable reader subset of the selected codec, checked at freeze. It imports stdlib only and its main requires assertions, Linux, PID1, UID/GID10001 before opening `/tmp`.

Future execution entry, **not executed here**:

```text
/usr/local/bin/python3 -I -S -B /private/tmp/pab-r75-recovery-prep-20261024/entry.py --execute-reviewed PARENT_PRIVATE_REVIEWED_AUTHORIZATION_JSON
```

Authorization JSON must set `authorize_one_recovery:true`, `independent_review_accepted:true`, `acceptance:false`, and `frozen_sha256` to the selected manifest digest above. These flags record parent authority; they do not manufacture independent review. Parent must retain the actual independent Sol report and its acceptance before supplying this receipt. Manifest/provenance transitively bind r73/r74, original AUTH/owner/argv/wrapper/inventory and candidate source hashes. No authorization receipt was manufactured during preparation.

Entry and worker verify frozen inputs. Worker captures exact candidate HEAD/clean tracked-status checks and rechecks all197 mapped files. Fresh fixed host roots are `/private/tmp/pab-r75-recovery-supervisor-20261024` and `/private/tmp/pab-r75-recovery-operation-20261024`; either existing prefix prevents dispatch/retry. No such operational root was created here.

## Implemented eventual operation

Two sequential fresh helpers have fixed names ending `-r74-export` and `-r74-hash`; both absence checks occur at admission and each is checked again before creation. Exact frozen create argv uses the admitted immutable image, existing private owner/invocation labels, CPU2/2GiB/PID256, networknone, readonly root, User10001:10001, capdropALL/no-new-privileges, restartno/logsnone, workdir/app, and sole old volume mounted readonly at/tmp. No env override, bind, stdin script, entrypoint override, rebuild or injected helper file. Config.Cmd is exactly `python -I -S -B -c SCRIPT MODE`.

Admission, each prestart and postread/final observations recheck current owner, exact full retained image/volume observations and all old container command/config/HostConfig/mount/state bindings. All volume attachers are enumerated using full IDs and individually inspected. Exact set must equal the two old stopped containers plus acknowledged new helpers; unresolved/foreign/active/paused/restarting attachment stops admission. Full helper inspections enforce literal command, inherited exact image Env, security/startup/resources, sole RO mount and created/stopped states. The operation retains owner inode equality observed for this operation only. Parent-controlled lane exclusivity remains necessary; discovery is not an atomic daemon lock.

Each mutation has a unique SENT-UNKNOWN journal before dispatch, pending exact argv, spawn/PID/session receipt, raw binary stdout/stderr, completion/exit/byte counts/SHA256, local CLI leader reap and descriptor closure. Capture adds fsync to the existing capture path. No create/start retry or name suffix. Unknown create acknowledgement remains UNKNOWN; acknowledged full IDs are retained. Attach occurs once, then wait<=4s/full stopped inspection with reserved time. Exit0/nonOOM/empty Error and binding equality are mandatory. Failed/uncertain/malformed export prevents the hash reader. No cleanup or lane action exists.

Worker decodes export before second reader; after hash equality it stores the full decoded metadata/hash/offset map and recovers root intent/terminal, sentinel/seed, strict identity semantics,000–017 selection/result/checksummed receipts and stdout/stderr hash reconciliation.000–016 require original exact PASS receipts/results;017 requires the enclosing call FAIL and actual failure output/traceback. Root completed list is not fabricated. All nested proofs remain included in raw export/map;018–099 absence is checked against the complete inventory. Identity comparison uses the existing strict semantic implementation, allowing only its documented ldd-address observation difference. Any content discrepancy remains failure/pending rather than repair.

`transport_verified` is separate from content verification and from historical `old_retained:false`/`old_test_failure:true`; acceptance and lane_released remain false. Final observations, full old-proof rehash, summary, checksummed512KiB slot, reread inventory, byte accounting and directory fsync are implemented. Supervisor captures/reaps the worker, retains OS outcome, rehashes old proof, writes its own slot and inventories both fresh roots with explicit inventory-self exclusion. All objects/lane/old and new proof are retained on every outcome. Prefix/finalization failure is UNKNOWN; fsync is not hardware durability evidence.

Whole supervisor<=360s; worker watchdog/cutoff330s, outer capture<=345s with remaining publication reserve. Worker admission<=60s; each reader lifecycle<=75s, attach<=65s with10s wait/inspect reserve; export decode<=45s and remaining decode/reconciliation before protected final30. Each reader independently alarms60s and cooperatively checks55s. Software limits: archive2GiB; map16MiB; each command stderr16MiB; logical regular total1GiB; leaf64MiB; entries200000; logical and physical PAX metadata each16MiB; raw path/target4096 bytes. Combined supervisor/worker host evidence<=4GiB, with16MiB excluded from worker allocation for supervisor and another worker final16MiB reserve. Initial host headroom6GiB; host and reader statvfs pressure checks require2GiB. These are observational/software bounds, not filesystem quotas.

Hard IO stalls or a killed local CLI do not prove remote cancellation. Watchdog/publication failure can leave incomplete receipts and UNKNOWN state; no next reader or retry is authorized. Independent review should scrutinize supervisor reserve/reaping, phase cutoffs, exact inspect predicates, PAX cross-version behavior and publication accounting. Actual orchestration success/failure paths were not exercised against Docker or a simulated daemon here.

## HOST-only evidence and limitations

Final frozen-source suite:26 bounded stdlib HOST checks, OS exit0, binary raw stdout/stderr retained with pending/spawn/completion and source binding. `ready-host-suite.stdout` SHA256 `8e590d1111c8df939db7240f37f5c8fb6b87797fad5a6575e561c94b5e9de569`; completion SHA256 `beddbb6a0c3997f03ef014c80f64912ebccf9780c1d8b5ddc373ce56c844c926`.

Coverage includes >100-byte filename and absolute link target, newline names, no-follow outside/loop links, lossless nonUTF8/newline raw PAX path/link round trips, truncated/malformed/PAX-overflow/trailing streams, duplicate/order/path traversal/absolute path rejection, special files/unknown PAX keys, raw path/link/leaf/count bounds, duplicate/truncated maps, hash and link-metadata mismatches, actual binary capture with NUL/FF/80/CR/LF plus exit7/stderr, retained-prefix capture overflow, slot checksum/size corruption and old-proof unchanged checks. Exact production reader main runs on HOST only to verify immediate PID1/platform rejection with empty stdout; actual reader functions are extracted for the synthetic codec fixture.

Darwin filesystem rejected the attempted raw nonUTF8 fixture filename (errno92); therefore nonUTF8 representability is exercised using in-memory PAX records through the exact selected encoder and both host parser passes, not a claimed POSIX/Linux directory traversal test. Initial fixture GID mismatch was corrected to actual created-directory GID for HOST fixtures only; production UID/GID10001 admission is unchanged. Failed drafts/partial fixtures and preparation history remain private. No Linux readiness, PID1 success, Docker, image Python, actual volume, runtime restoration, product assertion or nativeT2 qualification is inferred.

Final preparation inventory:232 regular leaves,22 directories,18 inert synthetic symlink fixtures,1662345 regular bytes including inventory. Every regular leaf0600/single-link/current UID; all directories0700. Paths/targets are base64 in inventory; symlinks are never hashed by following them. `preparation-inventory.json` SHA256 `d01786879d6a55c8a2fb76a30541d7929f54caefa1c364a989facebf1a8bba75`; checksummed `preparation-final.slot` SHA256 `312cc20d202efe9ed90e39c29f2fb78ee7ec302cc1fe2ba3cfe3bb547552c479`. Inventory excludes itself and this authorized root report; it includes draft history. Selected operational manifest remains separate from historical drafts.

Only this root report was written. Preexisting unrelated root untracked artifacts were preserved. No actual operations, second campaign, held-object change or self-acceptance. Next gate is independent Sol source/HOST-evidence adversarial review of this exact selected manifest, then parent decision on ONE recovery operation and a separate independent ACTUAL-receipt audit.
