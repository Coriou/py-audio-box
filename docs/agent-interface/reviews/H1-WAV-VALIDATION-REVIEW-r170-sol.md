# H1 WAV validation — independent r170 Sol review

2026-10-24. **GO for R166-1/2 closure and bounded whole-H1A source adoption readiness.** No new blocking finding in the authorized ingest/test delta. Carry forward r166's nonblocking Core/store conclusions and evidence limits. Parent integration/adoption remains pending; this is not installed crash, hardware, activation or release GO, and does not complete the whole H1 journey.

## Pins and review boundary

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-26d84130`, branch `t3/riff-wav-validation`, clean HEAD/report **`12ff83857d1c987c1b33d7b4c33df4e91238c801`**, fix **`502f83174e56485162bd194a72ee62cab809b6ed`**, base **`34f840156c41843df3aa69893ef5cbb36373d133`**. Base-to-HEAD changes only `service/ingest.py`, `tests/service/test_ingest.py`, and r168. Core/store compact changes are inherited unchanged from r163/r166; no renewed broad source review of those files is claimed.

Read full r168, r166 and r163, complete candidate ingest/test files and their delta, and both candidate conftests before running unittest. Sole authored write is this report in the parent checkout. No source/test edit, commit, delegation, Docker, model/network invocation, pytest guard/full suite, native fault, lane/resource operation or activation. Executed tests use their own isolated temporary Core/SQLite/media roots and normal owned unittest cleanup. Historical proof/STT and UNKNOWN/HELD resources remain unchanged.

## R166 closure and adversarial source findings

**R166-1 closed.** `_riff_pcm` validates raw structure before `wave` metadata. The 12-byte header must exist, be RIFF/WAVE, and declare exactly actual length minus eight. A small/underflow-like RIFF declaration cannot pass the actual-length/header constraints. Python integer arithmetic does not wrap. Each cursor begins at 12 and advances by eight plus unsigned payload size plus odd padding; even an empty chunk makes progress. Remaining header space and the full padded extent are checked before reading format bytes or seeking onward. The final cursor must equal actual length. This rejects truncated headers/payload/padding, inconsistent RIFF size and trailing bytes. Unknown payloads are skipped by absolute seek, never read or allocated according to declared size. Total chunk count is at most 4096, including fmt/data; chunk 4097 refuses before its header is read.

Exactly one fmt must precede exactly one data. State uses `None`, so a zero-sized data chunk still counts as present and cannot hide a second data chunk; zero frames subsequently refuse. Duplicate fmt/data, data-before-fmt, missing fmt/data and malformed post-data chunks refuse. Ancillary chunks before fmt, between fmt/data or after data remain allowed. An apparent embedded chunk header inside an ancillary/data payload is payload under RIFF chunk semantics; the scanner does not reinterpret sample bytes. Mandatory odd padding is included in extents without requiring a particular padding value. No cursor/seek, arithmetic or hidden second top-level data acceptance defect was found.

**R166-2 closed.** The raw format admits only tag 1, exact 16/24/32 bits, one/two channels and 8000–48000 Hz inclusive. Alignment must equal channels times exact byte width, byte rate must equal rate times alignment, and data size must divide exactly by alignment. Thus unsupported bits cannot round into false PCM subtype metadata and partial frames cannot be rounded away. fmt size is exactly 16 or 18 with cbSize zero. All extensible formats, float/compressed tags, RF64 and RIFX refuse. This is an explicit conservative classic-PCM policy documented in r168 and the helper; r163 promised PCM widths, not extensible support. No claim that every valid PCM WAV container variant is accepted is warranted. Future format conversion/support belongs to later work.

`wave` channels/width/rate/frame count must equal the raw tuple. Existing <=8192-frame reads must consume exactly the validated full data payload, with nonempty frame-aligned reads and final exact byte count. Nonempty duration <=60 seconds, 24 MiB encoded input, and exact subtype mapping remain. The staged file is controlled by this invocation; raw validation and decoding operate on that same file. No unsupported width, prefix-only payload acceptance or mapping error was found.

The only ingest behavior changes are raw media rejection and decoder cross-checking before publication. Input reads, complete original-byte SHA/provenance, principal/opaque ID handling, private staging, fsync/hard-link publication, DB admission and known-error versus ambiguous-COMMIT cleanup logic are unchanged from r163. No normalization or sample conversion was introduced. No concrete bug requiring a minimal fix remains in this reviewed delta.

## Executed evidence and coverage limits

Each authorized module was run once in the pinned candidate using `/private/tmp/pab-0c-venv/bin/python`:

- `-m unittest discover -s tests/service -p test_ingest.py -v`: **16 PASS, exit 0, 0.762 s**.
- `-m unittest discover -s tests/service -p test_voice_store.py -v`: **20 PASS, exit 0, 0.855 s**.

No unexpected fatal, failure or retry. `git diff --check 34f840156c41843df3aa69893ef5cbb36373d133 HEAD` passed; candidate remained clean after the runs.

The seven regression fixtures are concrete executable cases: complete three-byte mono PCM16 data plus padding; its missing-remainder/padding variant retaining the original RIFF declaration; 15/17/25-bit headers; wrong block alignment; wrong byte rate. r168 reports an actual pre-fix run of 14 tests with seven expected assertion failures (`ValueError not raised`), rather than source-only counterexamples. That historical run is implementation-reported evidence, not independently replayed here; this review did not reset/edit source or execute a separate pre-fix harness. This review directly observed all seven refuse post-fix, with empty staging/assets and unchanged empty assets DB after each case.

Other directly passing tests cover trailing/truncated RIFF, huge declared unknown chunk sizes before/after data, absent odd padding, duplicate fmt/data, ordering, chunk-count overflow, float/extensible refusal, all three supported widths in mono/stereo with ancillary chunks in all permitted positions, original bytes/hash, and a real 60-second 48 kHz stereo PCM32 payload plus ancillary bytes to exactly **24 MiB**. One extra byte refuses while retaining seven previously admitted rows/files and empty staging. Existing restart, principal separation, short/oversized/failing reads, tamper, ordinary publication/DB failure, conservative COMMIT-orphan retention/nonadoption and >1 MiB ingest→voice registration→restart tests pass.

Not every source branch has a dedicated fixture: exact 4096-chunk acceptance, fmt18 cbSize-zero acceptance/nonzero refusal, nonzero padding, isolated RIFF-size underflow and each short-header position are source-reviewed, not separately executed cases here. These are coverage limits, not identified acceptance defects. Width samples remain zeros; no listening, acoustic quality, consent or clone-suitability claim follows.

Carry r166 limits unchanged: compact count/query work and legacy histories/full reads are not globally bounded; aggregate 64 KiB registration budget and preexisting over-limit replay are supported by source placement rather than direct fixtures. Trusted blocking input can block, so transport byte/time bounds remain necessary. Low disk is fstatvfs injection; SQLite insert/COMMIT denial is not native storage failure or power loss. No full suite, existing Core/persistence regression module, race/concurrency campaign, installed crash/upgrade/restore, provider, production or hardware inference qualification occurred.

## Parent handoff

R166's two media blockers are closed; carry its other source/local-test findings without an additional blocker. Parent may adopt the pinned bounded H1A change after its integration gate. H2's parallel contract fix remains disjoint; no H2 source was edited or copied. Next parent work is ingest and API integration, then the upload control plane with authenticated principal derivation, explicit owning-Dispatcher methods, shared contracts, bounded transport reads/time and error translation. Never accept caller asset metadata/principal or expose arbitrary getattr.

H3 retains installed durability/crash/orphan reconciliation ownership. Remote acquisition, preparation/candidate selection, generation/stitch/export and the complete H1 journey remain later work. Actual CPU, CUDA and packaged Apple Silicon native Metal qualification/listening remain later gates. UI stays deferred. Historical proof/STT and HELD/UNKNOWN resources and their authority are unchanged.
