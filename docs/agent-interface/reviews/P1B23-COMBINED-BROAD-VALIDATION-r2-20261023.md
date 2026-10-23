# P1B23 combined staged broad validation r2 — 2026-10-23

**STOP / broad validation NO-GO** for exact HEAD `6eb6cec51e51ec0142bbbee2ab665a6ed168ee2c`, executable freeze `413b465bc8927f59444d99980b5235caa7d4e752`. Stage 1 (four B2 files) stopped at its first unexpected result: **811 passed, 1 failed, 1 teardown error in 366.81s**, pytest exit 1. Stage 2 (full service/release) **NOT RUN**. No retry, source fix, assertion/skip change, commit, delegation, integration or release GO.

## Authority and identity

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-7cbc0fbe`; sole repository write is this report in the parent checkout. Read full selector review, original COMBINED-BROAD-VALIDATION-20261023, COMBINED-OWNERSHIP-GUARD-CAP-r1-sol, complete appended candidate handoff, orchestration protocol and pertinent ledger chronology through resumed authorization. No applicable AGENTS.md found in checked ancestry/scoped source. Memory registry search supplied no acceptance evidence.

Parent accepted the narrow selector correction and independently reviewed five passes; that is prior evidence, not this run. Original broad stage1 **18 passed** (original17 + capB) was retained without repetition. Original four-B2 **254 passed/1 failed KeyError proof** remains immutable. The reviewed selector correction is not a waiver of that failure. Current executable diff against combined `bd29d757995c98b8f87b596be9cffe80074dc997` is only the reviewed test_recovery.py selector/control change (25 insertions,1 deletion); production remains accepted O. No T2/C imports or source changes.

Candidate HEAD, clean `--porcelain -uall`, freeze-to-HEAD handoff-only delta and all191 host manifest hashes checked initially, before tests, after retention and finally. Container manifest verified all191 files before runtime identity/imports and after execution/retention, against its embedded manifest; host retained manifest matches the selector image's reviewed source-COPY manifest. Existing immutable exact image `sha256:a506dbfb13a8944afde9ffbc6dfed811cce9737eb23656a6f34e9dcba07bb252` reused; no build/pull/install/alternate version. Host Python performed static/hash/archive/Docker orchestration only; no Darwin pytest/product imports/native GC/allocator/causal work.

## First unexpected result

Node: `tests/service/test_recovery.py::test_lotp_late_refusal_retains_unaccepted_plan[rehearsal]` (`tests/service/test_recovery.py:4054`, modeled rehearsal branch at4058–4060).

```text
maintenance terminal wait pid=161 status=18944: exit 74 without semantic ValueError phase witness
1 failed, 811 passed, 1 error in 366.81s (0:06:06)
```

The failure and teardown error report the same guard rejection. Actual retained final waits.jsonl row binds node above, **owner34/PID161/raw18944/phase empty**. No native signal is established. The test deliberately injects `sqlite3.OperationalError('modeled private capacity failure')` into `_write_core_completion` and expects the parent preparation to refuse with ValueError. A declared negative test does not grant generic exit74 authority: the raw receipt lacks the required semantic witness and remains rejected. This report establishes that boundary failure; it does not assign production/native causality or prove the underlying exception path. No diagnostic test, alternate runtime or correction was run.

Exactly one pytest invocation:

```sh
docker exec -w /combined pab-combined-broad-r2-fourb2-20261023 python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs -x --basetemp=/tmp/broad-fourb2
```

Wrapper elapsed **367.657532s**, exit1, timeout=false; stderr empty. No skips or xfails reported in the executed prefix, hence no skip reasons to list. `-x` plus the guard latch stopped execution; “stopping after 2 failures” is pytest's failure+teardown accounting for this one node. Only the recovery prefix executed; later recovery cases, packaged_lifecycle/admission/dispatcher and full service/release remain pending. No count extrapolation. The authorized second scope would have been `tests/service tests/release -q -rs -x -p no:cacheprovider` with a new basetemp/container; it was not started.

## Runtime, resources and raw retention

Fresh recorded CPython3.14.8, all42 installed lock versions, SQLite3.40.1, source_id `2022-12-28 14:03:47 df5c253c0b3dd24916e4ec7cf77d3db5294cc9fd45ae7b9c5e82ad8197f3alt1`, Debian libsqlite3-0 `3.40.1-2+deb12u2 amd64`. Compile options, extension path, ldd and package identity retained. Linked library SHA256 `2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8`. Linux comparison identity does not establish Darwin equivalence or binary-source authentication.

Exclusive atomic lane `/tmp/pab-worker-w1-docker-lane.lock` used exact owned PID/token/pin marker. One owned container `pab-combined-broad-r2-fourb2-20261023`, ID `02e2c9f2fb2a` (full ID retained), with inspected CPU2/memory2g/PID256/networknone/read-only/user10001:10001/capdropALL/no-new-privileges/private tmpfs1g. No host mounts/socket/secrets/ports/services/models/audio. Finite PID1 6000s, pytest75min, aggregate3h. Reserve>=4GiB and export>=5GiB checked; final free12,288,528,384 bytes before report. Known historical148MiB prefix was not treated as full peak bound. Last failed-stage tmpfs reading559084KiB/1GiB; posttest memory550.4MiB/2GiB/PID1 only. No timeout/OOM/resource-cap enlargement or fixture deletion.

Before tests, harmless mounted/tmp sentinel was binary-exported with live docker exec tar, host-verified token/uid10001/mode0600. After failure, entire mounted/tmp was binary-exported while alive. Initial safe audit correctly rejected **17 tar hardlink entries** produced from deliberate fixture hardlinks; archive preserved unchanged, container/lane held. This was a retention-procedure issue, not a second test execution. Authorized export-only correction used `docker exec NAME tar --hard-dereference -C /tmp -cf - .`, retaining all names and bytes as regular files while preserving symlinks as metadata. No docker cp/extraction/SQLite open.

Corrected archive: **26,849 members /18,564 regular /560,585,972 regular bytes**. Unique bounded relative names, no traversal/hardlink/special entry; per-file64MiB/aggregate1GiB/uid10001 checks pass. **Every archived regular hash equals separately collected LIVE hashes before removal**. Independent audit also matches all initial regular bytes and each initial hardlink's logical bytes to corrected archive; initial rejected archive is not silently discarded. Symlink inventory retained without following links. All raw fixtures/SQLite bytes, wait JSONL and helper metadata retained.

Independent receipt audit: **15 complete actual launch/ack/done triplets**, closed schemas, bounded private metadata, PID/node/32hex-token and parent binding, eight owner-bound child events with enroll/terminal equality; no nested helpers or crafted helper negatives in this prefix. **56 unique actual(owner,PID) terminal waits:29 exit0,2 raw18688/exit73 with D witness,24 admitted raw18944/exit74 with semantic witnesses,1 rejected raw18944 with empty phase.** Zero synthetic wait rows and no observed native signal. Positive helper triplets are distinct from terminal wait counts. Parent guard.directory=None correctly has raw wait JSONL and no required parent numeric event files. Direct test forks outside guard have no manufactured raw receipts. Failing fixture inventory retains17 regular files, including245760-byte core.sqlite. CapA direct OS/EFBIG+SQLite append and capB explicit-checkpoint witnesses were reached; their passing assertions and retained fixture/helper proof do not establish printed counts hidden by pytest capture. No all19 Linux claim; peer-credential/installed-host mechanisms were not reached/qualified by this prefix.

## Cleanup, preservation and remaining gates

After verified exports, independent receipt audit and source checks, exact full container ID matched the create receipt; graceful owned stop --time10 returned0. Inspected ExitCode0/OOMKilledfalse/Runningfalse/Error empty, exact rm without-v and exact-name absence verified. Exact owner marker rechecked before lane release; lane absent. Historical LIVE-PROOF-HOLD.json remains as chronology, with LIVE-PROOF-HOLD-resolved.json recording export-only resolution. Images and both archives retained. Driver exit0 means orchestration ended, not pytest PASS; pytest exit1 is authoritative. Initial STOP trace includes retention rejection and is preserved.

All124 selector,340 combined and187 original broad indexed evidence files rehashed unchanged. All1008 historical/STT hashes verified before/after; historical worktree HEAD/status inventories unchanged. No reset/clean/prune or historical/root STT write. Final candidate exact and clean. Evidence at `/tmp/pab-combined-broad-r2-20261023`: scripts, exact argv/stdout/stderr/status/timing, create/inspect/stats/runtime identity/manifests, both tar streams, live maps/inventories, receipt audit, failure fixtures and cleanup records.

Parent receipt review and adjudication of this STOP precede any new campaign/integration decision. T2/C original1295 matrix, final Astra, mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2/ext4/SHM/peer credentials/installed durability, native Darwin V1/V2/V3/fresh-helper/collection causes, A1 unexplained drift, Apple-Silicon hardware/browser/acoustic and host/public activation remain pending. Preserve L4MiB, result1MiB distinct from snapshot1MiB, docs64KiB, inspection2048/16MiB, H/charges, both locks, strict source/history joins, accepted/no-attempt authority, once-only capture/consumption and finite64-round exact-owned single-reaper cleanup/kernel-latency limits. No self-integration/release GO.

Selected raw SHA256:

- `manifest.json`: `41fa571f67543b5d84ce657f8febe54f3593b9c948ccd4b365d1623c688f2ead`.
- `fourb2-pytest.stdout`: `95482be34f4e4c5f4ed1bbb0e256d82a5da05a8180ec7fb738228fdd0966382f`.
- `fourb2-pytest.stderr`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `fourb2-pytest.json`: `c23ab5eb5297fddfe7e063535236ae48eba20ec22940139e59340f0572ad287d`.
- `fourb2-sentinel.stdout`: `c24ee6cb300ad3dce87bd926cbf1bf96833cad63e75728f72197be5013ac3d1a`.
- `fourb2-archive.stdout`: `47996f4d82af1f04e14c99170e50b6718b8fb6abe6812463432b8e688fe0dda6`.
- `fourb2-retained-archive.stdout`: `c1f8812d62bdb2565b6302b373558a5b3d2fa64496e615dbb897b88a190e1fa6`.
- `fourb2-retained-live-hashes.stdout`: `3b045bf06434a57d00ec887950d1f9c2a6ba7835d1b1a1e1ca9fac6d100bb597`.
- `receipt-audit.json`: `fb274a02b01dd63198f9f2e5151c7179940a01719ba380b4b8d1188e2c5c38bb`.
- `cleanup-stopped.stdout`: `a5c264e5a909ae4c441b07433e531e92a8f85244c4b4ceba46c10a4c3eace088`.
- `preservation-final.json`: `337135fa21b7f683b14a3117037f6e0b4b5b098945dc6f65c014ae40de7fda72`.
