# H1 WAV validation fix — r168 Sol implementation

2026-10-24. Bounded R166-1/2 implementation complete; focused isolated unittest PASS. Fresh independent Sol review required before parent H1 adoption. No release GO, activation or installed crash qualification.

## Pins and scope

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-26d84130`, branch `t3/riff-wav-validation`, initially clean at `9c51ce3aaeb09f7867e797f39d023de02a85134b`. Reset only this own branch/worktree, as authorized, to r163 documentation `34f840156c41843df3aa69893ef5cbb36373d133`, containing source `b1dcda182716e48fcb0750d8d772dc813e77647c`. Full r163 and r166 read, plus ingest/test source and both conftests. No AGENTS.md found under the worktree parent. Historical `t3-f62009a3` untouched.

Fix commit **`502f83174e56485162bd194a72ee62cab809b6ed`** changes exactly `service/ingest.py` and `tests/service/test_ingest.py`. This report is committed separately on the own branch; only this report is copied to root. No Core/store/API/contracts/recovery/guard/worker edits, Docker, model, network, lane, delegation, deployment, pytest/full-suite/native-fault execution or HELD-path operation. Test cleanup only covers owned temporary DB/media roots.

## Implemented policy

A bounded raw RIFF scan now precedes all wave-derived metadata. Declared RIFF size must equal actual encoded length minus eight; every chunk header, payload extent and required odd-size padding byte must fit inside that exact boundary. No trailing bytes are allowed. Padding values are unspecified and are preserved. Exactly one fmt chunk must precede exactly one data chunk. Duplicate fmt/data and data-before-fmt refuse; ancillary chunks may precede fmt, lie between fmt/data, or follow data. Unknown ancillary payloads are skipped by seek, never allocated from declared lengths. At most 4096 total chunks are admitted, bounding header reads/seeks independently of the existing 24 MiB byte budget. The raw scan reads 12 bytes, eight per admitted chunk, and at most 18 format bytes.

Only classic format tag 1 integer PCM with exact 16/24/32 bits, mono/stereo, 8000–48000 Hz is admitted. Format size is 16, or 18 with zero extension length. Other format sizes and all extensible PCM are explicitly unsupported: no valid-bit/container/channel-mask/GUID combination is admitted. IEEE float (including 32-bit float), compressed WAV, RF64 and RIFX remain unsupported. Block alignment must equal channels times exact byte width; byte rate must equal sample rate times alignment. Data size must divide exactly by alignment, so no sample byte or truncated remainder is rounded away.

The existing wave decoder remains in use with its 8192-frame reads. Its channels/width/rate/frame count must equal the raw validated fields; actual decoded byte count must match the full declared data size. Existing positive duration/60-second, encoded byte, staging/publication and rollback behavior remains. Original bytes and whole-file hashes are preserved without normalization/transcoding. Ancillary payload semantics and acoustic suitability are not validated.

## Evidence

Before changing implementation, added seven source-derived regression cases and ran the authorized ingest unittest command. Result: 14 tests, seven expected assertion failures (`ValueError not raised`) in the new regression method; all existing tests passed. The complete three-byte mono PCM16 data chunk with padding, physically missing remainder/padding with unchanged RIFF declaration, 15/17/25-bit headers, wrong alignment and wrong byte rate were all accepted by the old implementation. These were deliberate bug demonstrations, with no unexpected fatal. The pre-fix failing assertions precede cleanup assertions, so unchanged-state assertions are evidence from the passing post-fix run only. No fixture correction was needed.

After the fix:

- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_ingest.py -v`: **16 PASS, exit 0, 0.678 s**.
- `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_voice_store.py -v`: **20 PASS, exit 0, 0.742 s**; this module is unchanged.
- `git diff --check`: PASS before fix commit.

New refusal tests assert empty staging/assets and unchanged empty assets DB after each malformed input. Coverage includes exact counterexamples, trailing bytes, RIFF truncation, huge declared chunk size before/after data without allocating that size, missing odd padding, duplicate chunks, ordering, chunk-count overflow, float/extensible tags. Valid tests cover PCM16/24/32 mono/stereo with odd ancillary chunks before/after audio and LIST metadata, original bytes/hash, and a real 60-second 48 kHz stereo PCM32 payload plus ancillary padding to exactly 24 MiB. One additional byte refuses without changing the seven previously admitted assets/rows or leaving staging files. Existing original-ingest/restart, bounded/short reads, >60-second refusal, tamper, ordinary publication/DB failures, conservative COMMIT orphan retention and ingest→voice registration→restart tests pass.

No independent review, full-suite, installed process, crash/power loss, concurrency, provider, production, hardware inference or listening evidence is claimed. Runtime evidence here is only actual Core/SQLite and generated integer PCM bytes in test-owned temporary roots.

## Handoff

Parent owns fresh Sol source review before H1 adoption. H2 continues on its disjoint source lane; no H2 changes were copied or edited. Installed durability/orphan reconciliation remains H3-owned. Actual CPU, CUDA and packaged Apple Silicon native Metal qualification/listening remain required later; UI remains deferred. Historical UNKNOWN/HELD gates remain unchanged.
