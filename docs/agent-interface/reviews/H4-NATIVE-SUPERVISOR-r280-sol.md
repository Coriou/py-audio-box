# Native supervisor and child entry — r280 Sol

2026-10-25. **Implemented bounded SOURCE stage for parent independent review.
PARTIAL Lot A; NO-GO for installed/native readiness or release qualification.**
This is an implementation report, not independent acceptance. NativeRuntime,
Dispatcher/Core integration and the complete r277 acceptance 1 remain NEXT.

## Pins and authority

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-a5d44039`, branch
`t3/native-supervisor-child`. Entry root was exactly
`326c4d28ecbb935b96cb3b50e35342b076a66ac4`.

Requested historical partial source
`49dbb509d24040c0d446a5b66ea519962d4f9900` was cherry-picked as
`5e626a9`; historical report
`f8f7a63ff2da8fcf681c914e3395452f5f4e11d9` as `5da25c6`.
Those original commits and the historical `t3-05c63b46` candidate were not
edited. r278 remains byte-identical to the original report commit; its historical
claims were not rewritten as current acceptance.

New source commit: **`de6c07be80d702c19f7dcabe113e16b02e7dcae4`**.
This report is a separate subsequent commit.

Read full r277 and r278 during this turn. r277 SHA256 remains
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
Repository/ancestor AGENTS discovery found none. Both test conftests were read;
no pytest discovery/collection was used. A memory-registry keyword search found
no relevant r277/r278/r280 entries and supplied no technical authority.
No delegation or independent review was performed by this implementation agent.

Entry free space: 3,464,632 KiB, below 6 GiB. Only tiny inert source fixtures and
existing archive tests were used. Production prepare independently refuses below
6 GiB plus declared attempt disk demand; tests inject capacity, not reduce that
actual-work gate. No ML, hardware probe, Docker, network, download, installation,
real listener, native subprocess, process crash experiment, operational cleanup
or HELD action occurred. Test fixture teardown only removed its own tiny temporary
fixtures through existing unittest cleanup.

## Concrete implementation

Only five files belong to this new source commit:

- `worker/native_supervisor.py`: actual NativeSupervisor, native-only closed
  journal, all nine finite operation handlers, spool accounting, launch fence,
  direct owned-process lifecycle, listener production function and explicit CLI.
- `worker/native_child.py`: inherited-socket exec entry, independent native
  descriptor/Authority checks, bounded canonical config/result frames, token and
  binding response, one proposal and separately chunked WAV, EOF verification.
- `worker/native_protocol.py`: closed response variants and tuple/nonce echo,
  native frozen descriptor/voice/settings/reference/budget relation checks and
  full-descriptor provenance hash. Existing TLS/request/receipt codecs retained.
- `tests/worker/test_native_supervisor.py`: injected process/channel/clock/storage
  tests entering the actual supervisor, child inference and encoder functions.
- `headless-dist-manifest.json`: explicit supervisor source membership.

The operator supplies a bounded trusted registration map of exact four-element
snapshots, typed execution, sealed bundle mappings and enrolled inventory. Wire
prepare cannot register itself or select a validator. Prepare recomputes the
Authority binding, validates frozen plan/profile/admission, active attempt and
allowed asset relations, native tuple and hardware freshness. Start repeats
registration/inventory and full sealed input verification. This exact-snapshot
registration is intentionally narrower than the future principal/profile/version
NativeQwenContexts integration; it is not a new public registration operation.

Native requests bind deployment/configuration/lane, original allocation boot,
runtime/binding and request_id. Boot is the session nonce in the r277 envelope;
responses echo the request tuple and request_id. Historical collect/observe/ACK
use original boot; other old-boot commands refuse. TLS handlers require TLS 1.3
and the exact client certificate digest, with bounded canonical frames. The
listener uses an explicit numeric address, backlog eight, four bounded connection
workers and separate lifecycle/transfer command slots. CLI checks private trust
files; no installed-policy/enrollment rule was changed or bypassed.

The journal uses native-journal-v1, checksum, exclusive flock, 1 MiB ceiling,
O_EXCL temporary file, file fsync, atomic replacement and directory fsync. It keeps
at most 64 lifetime records/eight unacked, with no eviction. Unrecognized root
entries, retained temporary journal or missing allocation directory refuse startup.
Prepared/launch-intent records on restart are UNKNOWN, never PID-adopted or
relaunched. Completed output/proposal/receipt is revalidated and remains readable
and ACKable under its original identity. ACK persists a tombstone and retains
transfer files: this stage performs no automatic transfer cleanup.

Private 0700 allocation directories and 0600 no-follow/non-hardlinked files use
supervisor-derived names. Input checks exact digest/size/PCM, offsets, byte-equal
replay and final fsync; no partial input can start. Reserved worst case is twice
input plus output, globally capped at 64 MiB across retained records. Output is
one bounded PCM16 WAV with exact hash, frames, sample rate, provenance, manifest
and worker-message relations. Public frames expose bytes/digests, not spool paths.
Native receipts contain no engine_uid and cannot mint a Linux file-owner receipt.

Launch intent is durable before the sole injected/production Popen factory.
The command uses an absolute configured managed interpreter, -I/-B, the fixed
archived native_child.py path, one inherited socket FD, close_fds, DEVNULL streams,
fixed offline environment and start_new_session. No shell, PATH lookup, listener
FD, TLS key or nested model process is passed. The handle and spawn token remain
owned in memory; optional start identity is checked before group signals. Default
start identity is null, permitted only under the r277 exact-handle/token premise.

The independent watchdog does not acquire the transfer/hash lock. Cancel/cutoff
requests TERM, then KILL after 0.2 seconds while the held leader remains live;
failed bounded escalation retains UNKNOWN. Terminal requires the exact handle
wait, a joined receiver, verified parent FD closure and durable journal write.
Popen/wait/receiver/storage uncertainty quarantines without replacing the handle
or issuing replay launch. A zero exit without a complete result is not timely.
Result without exit is not terminal. Nonempty clone conditioning is retained and
blocks terminal; only an exact empty owned directory may be removed.

The child enters the existing actual infer function, NativeQwenBackend and WAV
encoder, with the permanent Python accidental-spawn guard before ML imports.
The guard is not protection against malicious native extensions/direct syscalls.
No Linux PID1, auth, supervisor, receipt, public schema, DDL, install policy,
service wiring or parent-owned foundation file was edited.

## Executed validation

All invocations used `/private/tmp/pab-0c-venv/bin/python -I -B` and explicit
unittest names, inserting `.` on sys.path. No subprocess-based test was selected.

Final focused command:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path.insert(0,"."); unittest.main(module=None,argv=["unittest","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=1)'
```

**44 PASS.** Covers native handlers, TLS fake handler/pin refusal, closed fields,
canonical framing/base64, tuple/nonce bindings, old boot/config/lane refusal,
stale/future inventory, wrong descriptor/topology, input gap/conflict/replay/hash,
spool symlink/hardlink refusal, disk/RAM/quota gates, output hash/PCM/cap/provenance,
launch intent before fake Popen, repeated start, TERM/KILL/identity failure,
bounded escalation UNKNOWN, wait/receiver/journal uncertainty, old-boot UNKNOWN,
completed collect/ACK replay and new registered work after completed restart.

One source fixture enters actual native child run_socket → infer → in-process
fake ML backend → existing encoder → bounded parent receiver → fake exact wait →
durable receipt → output bytes → journal reopen/collect/ACK. Tests also enter
complete loader with fake upstream through the existing native-backend tests.
No NativeRuntime or Core publication is part of this fixture. Qualified/installed
profile values in fixture metadata are explicitly injected source declarations,
not observations of installed readiness.

Separate clean-interpreter import/archive command:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path.insert(0,"."); unittest.main(module=None,argv=["unittest","tests.worker.test_qwen_import_seam","tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change"],verbosity=0)'
```

**6 PASS**, including complete native loader root/extracted archive and unchanged
legacy import seams. `git diff --check`: PASS.

Intermediate failures were corrected, not hidden: the first new child-entry test
hit the existing Linux generated_source decoder's MPS refusal, so native provenance
now uses the same full-descriptor hash algorithm through its own native validator.
A combined import test run had three lazy-import assertions fail because the new
inference test had already imported NumPy; the existing import tests passed in a
fresh interpreter. Two later new fixtures initially had an artificial monotonic
clock mismatch and a temporarily schema-invalid clone plan before its final
mutation; both fixture issues were corrected. Final commands above pass.

`lib/tts.py` remains unchanged, SHA256
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Historical legacy digest RED is not waived or regenerated.

## Remaining parent-owned gates and actual limits

This smaller stage does not close all r277 acceptance 2–7 combinations or full
Lot A. It supplies source coverage relevant to each, with no installed/Linux
receipt-consumer integration claim. Parent separate Sol review remains required.

NEXT source lot must supply NativeRuntime TLS client, detached immutable transfer
handoff, local control-owned spool/import UID, Dispatcher/Core publication,
shared finite receipt selection for live/history/recovery, and exact cancellation,
publication-intent/storage-fault and reservation/ACK ordering. NativeQwenContexts,
native demand and resolution/admission integration remain parent-owned. The local
native_execution relation helper must be reconciled with the parent's common
closed-lane extraction; existing runtime_protocol was deliberately untouched.

The production CLI/listener/exec/wait/signal paths are source implementations;
they were not bound, launched or kernel-qualified. Fake socket TLS tests do not
establish actual certificate enrollment, bridge/firewall reachability or denial
behavior. Factory verification/launch/journal system calls are trusted local
operations, not hard realtime preemptible syscalls; command replies retain their
original two-second communication cutoff and watchdog does not wait on transfer
hashing. Fault-injection coverage is focused, not an exhaustive crash/fsync matrix.

Available RAM uses a conservative sysconf free-page check; unsupported platforms
return zero and refuse. Darwin availability/accuracy, interpreter arm64/dependency
closure, OS process-start identity availability, memory/headroom and MPS behavior
remain packaging/qualification gates. No memory-containment claim is made.

Clone interruption with nonempty conditioning deliberately remains UNKNOWN.
Arbitrary cancellation-point clone readiness is unresolved. ACK retains bounded
spool bytes; capacity exhaustion requires parent recovery/retention authority,
not an automatic sweep or launch replay. No full public HTTP/CLI/MCP journey,
clean onboarding, real hardware speech, acoustic/performance/restart result,
CPU/CUDA release qualification or complete Lot A is claimed. Native Metal remains
UNAVAILABLE / UNQUALIFIED pending later authorized package and hardware lots.
