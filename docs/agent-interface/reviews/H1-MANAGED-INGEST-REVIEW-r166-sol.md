# H1 managed LOCAL ingest — independent r166 Sol review

2026-10-24. **NO-GO for bounded H1A source adoption: two P2 media-admission defects.** The authorized local unittest runs pass (13 ingest + 20 voice-store). This is SOURCE/local-unittest review, not installed durability, operational activation or full-release acceptance.

## Pins and boundary

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f62009a3`, branch `t3/managed-local-ingest`, base `e72f08278bc72871a4239a3f1d7d0874734edd7d`, source `b1dcda182716e48fcb0750d8d772dc813e77647c`, report/HEAD `34f840156c41843df3aa69893ef5cbb36373d133`. Candidate was clean before review and after the tests. Base-to-source changes exactly five paths: `service/ingest.py`, `service/core.py`, `service/voice_store.py`, `tests/service/test_ingest.py`, `tests/service/test_voice_store.py`; source-to-HEAD adds only r163. Core changes are confined to asset verification. No API, dispatcher, contracts, persistence, worker, recovery or guard change.

Read full r163 and r162, the current plan's H0 closure/H1 scope/H2-H3 handoffs and parallel-lane boundaries, all five changed files and their diff, both conftests, and the existing Core transaction boundary. Inspected the authorized interpreter's actual `wave` parser source to resolve its size/bit-depth behavior. Sole authored write is this report in the parent checkout. No code/test edits, commits, delegation, Docker, model, network, native-fault campaign, standalone proof/user-data access, lane operation, operational cleanup or activation. Test cleanup was limited to the tests' own temporary DB/media roots.

## Blocking findings

### R166-1 — P2: partial declared PCM frames are admitted

`service/ingest.py:51–64` trusts `wav.getnframes()` and verifies only that many decoded bytes. In the inspected interpreter, `wave.Wave_read.initfp` computes `data.chunksize // framesize`; the remainder is discarded. The loop requests exactly the rounded-down frame count and never validates the raw data chunk length or probes its remainder.

Concrete source-derived counterexample: a mono PCM16 WAV with data chunk length 3, containing one complete two-byte sample plus one extra sample byte (and proper RIFF padding), is admitted with `frames=1`. A second variant declares those same 3 bytes but physically ends after the two-byte sample, with the enclosing RIFF length including the missing byte/padding. It also supplies every byte the ingest loop asks for and passes. `expected <= length` compares audio payload to the entire encoded file length, so it does not detect either case. Thus a malformed or truncated data chunk is published successfully, and frame metadata describes only its rounded prefix. The existing `audio()[:-3]` test catches loss of full frames, not this case.

Minimal fix: validate raw RIFF/data chunk extents against actual encoded length, including chunk padding, and require declared PCM data size to be divisible by `channels * sample_width` before deriving frames. Retain bounded reads/seeks and the 8192-frame decoder loop; do not allocate declared chunk lengths. Add rejection cases for a complete misaligned chunk and the missing-remainder variant, with unchanged files/DB after refusal. This counterexample was reasoned from source, not executed as an extra unapproved test invocation.

### R166-2 — P2: unsupported bit depths become false PCM16/24/32 metadata

`service/ingest.py:51–55,65` checks `getsampwidth()` and maps byte width directly to a PCM subtype. The inspected `wave._read_fmt_chunk` computes sample width as `(sampwidth + 7) // 8`, rather than requiring bits-per-sample to be 16, 24 or 32. It also parses but does not validate `wBlockAlign` or average byte rate.

Concrete source-derived counterexample: change the bits-per-sample field of an otherwise valid mono PCM16 WAV from 16 to 15. The parser still reports width 2; all current checks pass and the published metadata says `PCM_16`. Likewise 17-bit PCM maps to width 3 and `PCM_24`, and 25-bit maps to width 4 and `PCM_32`. These are outside the explicitly supported media set and can mislead later consumers of frozen reference metadata. This is a validation defect at construction, not a request for new media support.

Minimal fix: validate the actual PCM format fields before accepting the decoder result: exact supported bit depth and consistent block alignment (and byte rate); for any accepted extensible PCM, explicitly validate its container/valid-bit policy rather than silently rounding. A small bounded RIFF validation helper can address both findings without exposing paths or changing contracts. Add unsupported 15/17/25-bit and inconsistent alignment fixtures, preserving valid PCM16/24/32 mono/stereo cases. These counterexamples are source conclusions; no custom reproduction run is claimed.

## Other reviewed behavior

The trusted LOCAL boundary accepts bytes or blocking `read(size)` streams, with no caller path, URL, ID or ownership/provenance metadata. Principal syntax is validated; authentication/derivation remains the transport owner's responsibility. Reads request at most 64 KiB with a one-byte over-limit probe, reject over-return/non-bytes and cap originals at 24 MiB. Duration/rate/channel limits and bounded frame reads are present. Float/compressed paths refuse before sample decoding; integer 24/32-bit samples are finite without a floating-point/model decoder. Huge declared lengths do not cause payload-sized allocation on the staged seekable file. A trusted blocking reader can still block; H2 needs independent transport time/byte limits.

Original bytes and their complete SHA/provenance are retained; random IDs prevent cross-principal content adoption. Private current-uid no-follow directory FDs are held through publication/cleanup. O_EXCL 0600 staging, partial-write handling, file fsync, no-overwrite hard link, staging unlink to one link, and directory fsync precede DB admission. The precommit cleanup targets only this call's random names. Failure after COMMIT is attempted retains complete bytes and propagates the error through the existing transaction boundary; it does not report false success. No additional concrete data-loss or wrong-success defect was found within this trusted private-namespace model. Cleanup errors may leave files and mask an earlier exception; r163 discloses propagation/retention rather than promising infallible cleanup. Hostile same-uid namespace mutation and power loss were not qualified here.

Core.asset preserves trusted DB membership/schema/ID checks and legacy 1 MiB behavior, conditionally granting 24 MiB only to directly owned upload-shaped WAV metadata. It hashes in bounded chunks, checks overflow, regular/current-uid/single-link status, and compares inode/size/timestamps before/after with actual length/hash. Directory and leaf opens are no-follow. It verifies stored bytes, not the truth of arbitrary caller metadata: future transports must not expose raw asset-row or principal construction. Existing P1/owner/transaction fences are unchanged.

Compact summary/version pages are principal-filtered, capped at 20 and avoid full get/record decoding; versions now uses SELECT 1 for existence. Direct archived reads remain possible. Counts/queries can scale with row count; legacy full reads/history/idempotency storage remain unbounded as disclosed. New registration checks canonical metadata <=64 KiB and each string <=16 KiB inside apply, after prior idempotency lookup, without truncating authored identity or changing old reads. H0 schema identity/refusal, immutable versions, principal ownership, aliases/archive and idempotency logic remain intact. No new concrete blocker was found in these changes.

## Executed verification and limits

After inspecting both conftests, ran each authorized command once in the pinned candidate:

- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_ingest.py -v` — **13 PASS, exit 0, 0.427 s**.
- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_voice_store.py -v` — **20 PASS, exit 0, 0.752 s**.

No unexpected native fatal, failure or retry. Actual Core/SQLite and media ran only in test-owned temporary roots. `git diff --check e72f082 HEAD` passed. No pytest/conftest guard, full suite, affected existing persistence/Core regression module or native fault campaign ran.

The tests include a real 60-second 48 kHz stereo PCM32 maximum supported payload, original bytes/hash/restart, cross-principal duplicate separation, short reads, tamper, symlinks, ordinary failure cleanup and actual >1 MiB ingest→voice registration→restart. Low disk is injected fstatvfs, not a filled filesystem. Insert/COMMIT denial is SQLite authorizer injection, not disk failure or power loss; COMMIT denial verifies conservative complete-orphan retention/nonadoption. There is no race/crash/power-loss qualification. The PCM24/32 samples are zeros, not acoustic quality evidence. The new metadata test exercises 16 KiB text refusal/acceptance and a new valid replay; it does not directly construct a preexisting over-limit replay or independently exercise the aggregate 64 KiB limit. Source placement supports those policies, but these are test coverage limits. Existing passing tests do not close R166-1/2.

## Parent handoff

Fix the two bounded parser defects and review the resulting source/tests before H1A adoption. H2 may progress in parallel against compact interfaces, with upload Dispatcher commands, bounded error translation and shared contracts coordinated by the parent. No arbitrary getattr or client-supplied principal/asset metadata should become an adapter shortcut. Remote acquisition, engine jobs, preparation/candidate selection, generation and stitch remain later H1/H4 work; this slice does not satisfy the complete H1 journey.

H3 owns installed crash/upgrade/restore/orphan policy and durability qualification. Later actual CPU, CUDA and packaged Apple Silicon native Metal evidence/listening remain required. UI stays deferred. Historical UNKNOWN/HELD lanes and resources remain unchanged; no operational continuation or release GO follows from this review.
