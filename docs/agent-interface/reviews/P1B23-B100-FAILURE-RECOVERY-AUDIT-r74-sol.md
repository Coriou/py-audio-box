# P1B23 B100 failure recovery audit r74 Sol

2026-10-24. **STOP / UNKNOWN HOLD remains. No critical discrepancy found in r73's bounded failure handback. Recommend parent approval of ONE finite stopped-volume evidence recovery operation, followed by independent actual-receipt audit. This report grants no execution, retry, patch, cleanup or campaign acceptance.**

## Reviewed scope and identity

Read full r73 and r72 reports, current candidate B PLAN/runner, controlling FREEZE successor/lifecycle/archive source, private wrapper/dispatch and retained receipts. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-65b1e6b7` is clean at documentation `7a53112d1c6c757fea51b20721b9fc6200dd9bc0`; executable `f5bc92a1098266c4b3f0565198727b6e38fd61a5` is its ancestor. Independently rehashed all197 AUTH mappings against candidate bytes without mismatch; independently reconstructed all100 ordered node names from source AST and matched AUTH. Runner `6a4274955504dae791e5562761601adcc8ca4ae27b6bd7cf487696eba0de791e`; manifest `5c4cbda64403d0d6b12f7102eae912af1318c5716e0b1c1153549caf10d22cba`. Admitted image `sha256:14b0cf52e7337b449938701c0256fe1f32fe846fa25d62f8d18413a190edbbcb` is the sole proposed recovery image.

No applicable repository AGENTS.md found. Lightweight memory-registry search yielded no relevant premise. Source and current retained bytes supply this audit; retained Docker inspections are historical receipts, not a fresh daemon observation.

## Independent receipt reconciliation

Roots: `/private/tmp/pab-r73-operational` and `/private/tmp/pab-p-one-r16-e49b684f-65aa-4deb-b350-39767210c3ce`.

- Final inventory independently hashes to `cef8bb93c442c15c48ad97cda334b23e22f945063fbf021c3b88a61750e86c9f`: exactly104 regular, single-link, current-UID leaves with matching lengths/hashes/recorded UID/GID and0600; directories0700. Current path set equals inventory entries plus inventory itself, with no symlinks. Inventory/report/outside-script exclusions are explicit; this is not proof that excluded originals were inspected.
- All17 command completions independently equal their pending argv; all34 binary streams match recorded sizes/SHA256. All COMPLETE, local leader reaped, descriptors closed, no group kill. Only writer-start, archive-start and outer campaign CLI exit1; wait CLI exit0 reporting workload1 is distinct. Supervisor COMPLETE/worker_exit1. No runtime success follows from local CLI completion.
- AUTH SHA256 `9ca2f83ffa53494fa9417af8139cfa467648d1fc393eb5bf3041b2cc55e0cbaa`, wrapper `170d3d6fa2507b3334a74bde073280e80c8b18c9d9a7af75b541cec5c80d3ca0`, frozen cwd/argv, outer pending and exclusive once marker reconcile. Dispatch source contains one outer call; retained operations contain one writer create/start and one archive create/start, no hash create/start. This supports once within the retained authorized lane, not a universal daemon-history assertion.
- Both512KiB host slots independently validate body SHA256/newline/zero padding. Final owner equals intent owner and **current** `/private/tmp/pab-worker-w1-docker-lane.lock/owner.json`; token remains private. Final STOP-OR-UNKNOWN-HOLD, retained=false, acceptance=false, lane_released=false. The once marker remains SENT-UNKNOWN by design and operation creates remain CONFIRMED; neither is silently rewritten as TERMINAL. Completed command receipts and stopped bindings resolve the observed dispatch outcome, not every journal label. Earlier FINALIZATION-PENDING handback is not success.
- Retained stopped writer `82ccb71b8e178e06458c894849ed185db6f6ddb5da2d555ff5897fb1ab8dae93` and stopped archive `ba057c4428408f5367201d7da751c2e4378dfcaa4069b98a2e9a833653864eaa` match exact image/name/owner labels/commands. Both exited1/Pid0, nonrunning/nonpaused/nonrestarting/nondead, nonOOM, empty Error. User10001:10001,2CPU/2GiB/PID256, networknone, readonly root, capdropALL, no-new-privileges, restartno/logsnone; no extra startup/exposure/mounts. Writer has sole RW `/tmp` volume; archive same volume RO.
- Volume receipt: `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce`, local/Scope local/Options null/CreatedAt `2026-10-24T03:19:21+02:00`, exact labels. A name/creation timestamp is an observed binding, not immutable storage identity. Current owner equality is not historical lane-inode continuity.

## Failure and evidence boundary

Independently parsed the raw USTAR prefix in memory:16896 bytes, SHA256 `2ea52cca81c3cf72e8ffc548d1ec881a4f7b40c5ccaadb5ba38eaf2d0ce52e4f`,15 complete members, EOF before another header, no complete archive terminator. Node017 result independently says exact `tests/service/test_child_wait_guard.py::test_B_directed_isolated_seams[B-S-original74-real-observer]`, pytest_exit1, collected=true, setup PASS / call FAIL / teardown PASS, accepted=false. Nested proof34/35 success steps cannot establish the enclosing assertion or override FAIL. Its stdout/stderr/receipt and exact traceback remain unavailable in this export.

`ordered()` refuses first nonzero before advancing and checks exact selection/result/deadline before earlier PASS. Reaching017 therefore supports000–016 only by source path; their retained receipts are still UNVERIFIED.018–099 remain82 UNRUN. Root `completed` assignment occurs only after ordered() returns; an absent completed list is not evidence that no prior node ran. Do not demand a fabricated17-element root success list. The root terminal, descendants census, pre/post semantic equality, sentinel and complete file map remain pending recovery. Post identity alone (Linux3.14.8/42 pins/SQLite3.40.1) is not full runtime acceptance; outer17.82s is not target timing.

Raw archive stderr identifies `finite-driver.py:242`, symlink `tf.addfile(info)`, `tarfile.create_ustar_header`, `ValueError: linkname is too long`. Source uses USTAR while census admits links up to4096 encoded bytes. USTAR linkname is100 bytes; its filename prefix field does not extend linkname. Local stdlib source corroborates that format limitation, without claiming execution of Linux's library. Nested pytest basetemp growth plausibly produced a long absolute `*current` target; **the exact offending path/target is UNKNOWN until a new RO inventory**. Last exported directory is node017 proof/pytest. Do not derive the missing link from traversal order or present path growth as the call-failure cause.

The unchanged reader correctly failed closed under its frozen contract. This is a concrete transport representability gap, not permission to skip links, truncate names or relax existing parser acceptance. No critical false success claim found in r73. r72 remains bounded evidence for its separate unchanged original-positive only; it supplies no B100 PASS, timing waiver, restoration-completeness, nativeT2/platform/hardware/durability/release credit.

## Proposed single bounded recovery operation — parent approval required

Prefer a new length-prefixed protocol over changing the old USTAR parser or image. Combine the no-follow inventory and all regular-file bytes in **one fresh export reader**, then one fresh independent quiescent hash reader; no preliminary inventory-only microgate, old-reader restart, test repeat, writer restart or product execution. This is an explicit new recovery exception to FREEZE's old at-most-two/USTAR-only procedure (one old reader already ran); it requires separate parent authority. Neither old source nor old archive is altered. PAX would also represent long links, but is not the selected proposal.

### Admission and frozen command

Parent's recovery receipt must bind r73/r74 hashes, original AUTH/frozen argv/once/wrapper/inventory hashes, exact existing owner/token/invocation, image, volume and two full stopped IDs above. Use current lane owner bytes for equality; do not acquire/release/replace the held lane or invent an old acquisition inode. Capture its current lstat identity for this operation only. Rehash the old104 leaves and USTAR prefix before and after recovery; preserve both roots unchanged. All new proofs go to a fresh0700 sibling recovery root, exclusive0600 leaves, with explicit final-inventory self/publication exclusions.

Before creating any helper, use bounded read-only daemon inspection to confirm exact image ID/config/User/security startup, full old container bindings/stopped states, and volume name/Driver/Scope/Options/CreatedAt/Mountpoint/labels equal retained observations. Enumerate all containers using that volume and inspect them: no active/restarting/paused workload or foreign attachment may be unresolved. Both historical containers stay stopped. Require parent-controlled lane exclusivity throughout; a read-only mount alone cannot prevent another RW attacher. Check current owner and stopped/no-other-writer state again before each start and after both readers. Unknown inspection/CLI/discovery or binding drift => STOP; no reader starts. No DB opens, extraction, docker cp, network, product imports, service helpers or namespace killing.

Fresh fixed names are `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce-r74-export` and `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce-r74-hash`. Require exact name absence; collision => STOP, not suffix retry. Retain newly acknowledged full IDs. Exact create argv schema below is frozen by substitution of the **existing private token** and the literal SCRIPT below; serialize and hash both resulting argv arrays before parent dispatch approval. Do not publish the token in this report.

```text
["docker","create","--name",NAME,
 "--label","pab.owner="+EXISTING_PRIVATE_TOKEN,
 "--label","pab.invocation=e49b684f-65aa-4deb-b350-39767210c3ce",
 "--cpus","2","--memory","2g","--pids-limit","256",
 "--network","none","--read-only","--user","10001:10001",
 "--cap-drop","ALL","--security-opt","no-new-privileges",
 "--restart","no","--log-driver","none","--workdir","/app",
 "--mount","type=volume,src=pab-b-e49b684f-65aa-4deb-b350-39767210c3ce,dst=/tmp,readonly",
 "sha256:14b0cf52e7337b449938701c0256fe1f32fe846fa25d62f8d18413a190edbbcb",
 "python","-I","-S","-B","-c",SCRIPT,MODE]
```

MODE is respectively export/hash. No helper file bind, image rebuild, entrypoint override, env source path or stdin injection is needed: script bytes live in exact Config.Cmd. `-I -S -B` isolates startup and prevents site/product bootstrap and bytecode writes. Inspect the created full ID before start: exact literal command/script hash, labels, image/User/workdir, security/resource settings above, one RO volume at/tmp, no image entrypoint/healthcheck/onbuild/declared volumes, no privileged/capadd/device/bind/tmpfs/hostPID/auto-remove/extra exposure. Inherited image Env must equal the inspected immutable image Env, without injected startup overrides. Do not pass this through the old command-binding predicate and pretend unchanged approval.

For each helper once: full-ID start --attach, wait, full stopped inspect; no stdin, logs or daemon log harvesting. Wait result must equal stopped ExitCode0; Pid0/exited/nonrunning/nonOOM/empty Error, exact configuration/mounts again. A local COMPLETE receipt alone is insufficient. No second reader follows failed/uncertain export. Never retry SENT-UNKNOWN create/start; preserve partial captures and report the unresolved full ID/discovery facts.

### Finite bounds, protocol and reconciliation

Whole host operation <=360s: admission<=60; export lifecycle<=75; hash lifecycle<=75; bounded decode/reconciliation<=90; protected publication<=30. Each reader: independent PID1 alarm60s, cooperative work deadline55s; attach<=65s with last10 reserved for wait<=4 and stopped inspect<=6. Ordinary command timeout=min(remaining phase, available reserve); no blind timeout enlargement. Local CLI spawn/capture uses exact binary streaming caps, pending/spawn/completion records, lengths/hashes, local leader reap/descriptors closure and no retry. Hard IO stalls/daemon uncertainty cannot be described as guaranteed remote cancellation.

Regular logical aggregate<=1GiB, each leaf<=64MiB, entries<=200000, path and symlink target<=4096 **raw bytes**, metadata<=16MiB, export stdout<=2GiB, hash stdout<=16MiB, each stderr<=16MiB. New host evidence<=4GiB including all streams/publications with16MiB supervisor/final reserve; old roots separately retained and counted as preexisting evidence. Observe host/daemon free>=2GiB during work and require sufficient initial headroom for bounded new captures. These are software/observational limits, not physical quotas or inode guarantees. Breach stops with prefixes retained, never truncates a logical member into accepted proof.

Protocol LP1:8-byte unsigned big-endian JSON length (<=32768), canonical ASCII JSON with sorted keys/compact separators; entry record followed by exactly size raw bytes only for export regular files, then digest frame. Hash mode omits regular payload but reopens/re-reads every regular file independently. Raw POSIX path/target bytes are base64, so arbitrary filenames/newlines/non-UTF8 and long absolute symlink targets survive losslessly; absolute links are inert metadata, never traversed or normalized. Entry names are relative component tuples, no empty/dot/dotdot/slash/NUL component, unique canonical order, root '.' only. Accept only directory/regular/symlink, numeric UID/GID10001; record permissions, size, mtime/ctime nanoseconds, nlink and full literal link target. Device/inode are checked for within-operation stability, not used as a historical storage-identity premise. Unsupported special files, unreadable file, source drift, race, count/size/deadline/output overflow, short write/read or malformed frame => uncaught failure/no valid END.

Host decoder must independently bound lengths **before allocation**, count entries/logical/meta/streams, reject duplicate/unsorted/unsafe paths, unsupported kinds/schema/extra records/invalid base64, missing digest/END/trailing bytes, partial frames/payload or checksum mismatch. END binds counts/logical/metadata and SHA256 plus length of every preceding wire byte; accept it only with complete binary capture and remote exit0/stopped binding. Hash reader must produce the same semantic metadata/path/link set and every regular count/hash as decoded export; protocol headers differ by mode. Independently compute hashes from exported payload rather than trusting exported digest frames. No archive extraction or filesystem recreation is involved.

Reconcile ALL files including seed, exact token sentinel ending NUL/FF/80/CR/LF, root intent/terminal slots, both identity documents, all000–017 selection/result/receipt/stdout/stderr and real Observer/nested proofs. Parse slots independently with exact512KiB length/body checksum/newline/zero padding. Compare raw stream lengths/hashes against each recovered node receipt and exact AUTH node/caps/owner/root identity. Recover017 enclosing stdout/stderr and call traceback before diagnosing; incomplete/torn terminal or absent receipt remains explicitly pending even if transport succeeds. Check absence of018–099 by complete inventory plus runner source, not nested success. A complete export may be labeled RECOVERY-TRANSPORT-VERIFIED only after independent hash equality; original retained=false/archive failure and campaign FAIL remain historical facts. Any different observed count/status is a discrepancy to investigate, not auto-repair.

Final publication must retain raw binary streams unmodified (no text conversion/base64 clipping/tool-output substitution), exact script/argv/provenance hashes, all local and daemon receipts, decoded canonical inventory/hash map, summary and checksummed final slot. Hash/inventory all new leaves, fsync completed files/publications, declare exclusions, independently reread/check final slot and inventory. If publication cannot finish within reserve, report UNKNOWN with all prefixes and objects held. Never claim durable hardware storage from fsync. Retain old writer/archive, new stopped readers, volume/image/tags, all old/new proof and lane on every outcome. No cleanup/release is part of recovery.

### Frozen proposed reader source (not executed or runtime-qualified)

SCRIPT is exactly the UTF-8 bytes inside the following fence, including one LF after the last line. SHA256: `9798721881ea50f8d6577665628e41f28914531b92a58f0fc4a1655e6fa22476`. Source was AST-parsed and hashed only. Parent must obtain independent source/host-capture review of this proposal and freeze its concrete argv hashes before approving the one operation; this audit is not self-approval of newly proposed code. Exception policy is deliberately fail closed; there is no skip/continue/format fallback. Host parser/capture implementation remains to be prepared and independently reviewed within parent authority; no operational readiness claim is made for an absent implementation.

```python
import os, sys, stat, signal, time, json, hashlib, base64, struct
assert __debug__ and sys.platform == "linux" and os.getpid() == 1
assert os.getuid() == os.getgid() == 10001
mode = sys.argv[1]
assert mode in ("export", "hash") and len(sys.argv) == 2
signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
signal.setitimer(signal.ITIMER_REAL, 60)
end = time.monotonic() + 55
LEAF, TOTAL, ENTRIES, META = 64*1024**2, 1024**3, 200000, 16*1024**2
cap = 2*1024**3 if mode == "export" else META
out, wire, used, meta = sys.stdout.buffer, hashlib.sha256(), 0, 0
def check(ok):
    if not ok: raise RuntimeError("STOP recovery bound/type/stability")
def tick(): check(time.monotonic() < end)
def b64(b): return base64.b64encode(b).decode("ascii")
def encode(v): return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
def send(b, hashed=True):
    global used
    tick(); check(used + len(b) <= cap)
    check(out.write(b) == len(b)); used += len(b)
    if hashed: wire.update(b)
def frame(v, hashed=True):
    b = encode(v); check(len(b) <= 32768)
    send(struct.pack(">Q", len(b)) + b, hashed)
def signature(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
root = os.open(b"/tmp", os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
check(stat.S_IMODE(os.fstat(root).st_mode) == 0o700)
def directory(parts):
    fd = os.dup(root)
    try:
        for p in parts:
            tick()
            nxt = os.open(p, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise
pending, rows, logical, metabytes = [()], [], 0, 0
while pending:
    tick(); parts = pending.pop(); path = b"/".join(parts) if parts else b"."
    check(len(path) <= 4096 and all(p not in (b"",b".",b"..") and b"/" not in p and b"\0" not in p for p in parts))
    fd = directory(parts[:-1])
    try:
        s = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False) if parts else os.fstat(root)
        check(s.st_uid == s.st_gid == 10001)
        kind = "file" if stat.S_ISREG(s.st_mode) else "dir" if stat.S_ISDIR(s.st_mode) else "link" if stat.S_ISLNK(s.st_mode) else None
        check(kind is not None)
        row = dict(path=b64(path), kind=kind, size=s.st_size, mode=stat.S_IMODE(s.st_mode), uid=s.st_uid, gid=s.st_gid, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, nlink=s.st_nlink)
        if kind == "file":
            check(0 <= s.st_size <= LEAF); logical += s.st_size; check(logical <= TOTAL)
        elif kind == "link":
            target = os.readlink(parts[-1], dir_fd=fd)
            check(len(target) <= 4096); row["target"] = b64(target)
            check(signature(s) == signature(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)))
        else:
            childfd = directory(parts)
            try:
                check(signature(s) == signature(os.fstat(childfd)))
                names = []
                with os.scandir(childfd) as it:
                    for child in it:
                        tick(); names.append(os.fsencode(child.name))
                        check(len(rows)+len(pending)+len(names)+1 <= ENTRIES)
                check(signature(s) == signature(os.fstat(childfd)))
                pending.extend(parts+(n,) for n in sorted(names, reverse=True))
            finally: os.close(childfd)
        metabytes += len(encode(row)); check(metabytes <= META)
        rows.append((parts,row,signature(s))); check(len(rows) <= ENTRIES)
    finally: os.close(fd)
rows.sort(key=lambda r:r[0])
frame(dict(protocol="PAB-R74-LP1", mode=mode, entries=len(rows), logical=logical))
for parts,row,sig in rows:
    tick(); fd = directory(parts[:-1])
    try:
        now = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False) if parts else os.fstat(root)
        check(signature(now) == sig)
        if row["kind"] == "link": check(b64(os.readlink(parts[-1],dir_fd=fd)) == row["target"])
        meta += len(encode(row)); check(meta <= META)
        frame(dict(record="entry", **row))
        if row["kind"] == "file":
            f = os.open(parts[-1], os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=fd)
            try:
                check(signature(os.fstat(f)) == sig)
                digest, count = hashlib.sha256(), 0
                while True:
                    tick(); b = os.read(f, min(65536, row["size"]-count+1))
                    if not b: break
                    count += len(b); check(count <= row["size"]); digest.update(b)
                    if mode == "export": send(b)
                check(count == row["size"] and signature(os.fstat(f)) == sig)
                frame(dict(record="digest",path=row["path"],bytes=count,sha256=digest.hexdigest()))
            finally: os.close(f)
    finally: os.close(fd)
os.close(root)
frame(dict(record="END",entries=len(rows),logical=logical,metadata_bytes=meta,preceding_bytes=used,sha256=wire.hexdigest(),acceptance=False), False)
out.flush()
```

## Concrete next decision and exit conditions

Parent may authorize preparation/review of this exact recovery transport and then ONE bounded two-reader operation on the held stopped volume. No user question is required from this sub-agent. Preparation must preserve the frozen script or return any corrected successor for fresh review; do not silently substitute bytes under this digest. Do not call an unimplemented host decoder ready for execution.

Successful exit: complete lossless export + independent re-read map equality + full stopped bindings + final proof transport/inventory, all objects/lane held, acceptance=false, then a fresh independent ACTUAL recovery audit determines what017 failed on and whether000–016 receipts can be promoted. Failure/uncertainty: immediate STOP/UNKNOWN, preserve partial new proof and objects, no second reader after uncertain first, no retry/repair/cleanup. No product/timing/runtime acceptance follows in either case; parent STOP remains until a separately scoped decision.

Only this root report was written. No Docker commands, test/pytest/runtime/product imports, helper execution, commits, source/private-proof edits, delegation, retry, cleanup or release. Read-only stdlib JSON/hash/tar/AST and Git/text inspection only. No memory-derived technical fact was used.
