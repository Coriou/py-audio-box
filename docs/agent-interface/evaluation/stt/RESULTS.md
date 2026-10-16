# CPU STT probe results — 2026-10-15

Historical first-pass results. See the [hands-on debugging follow-up](DEBUGGING.md) for subsequent controlled tests and local setup repairs; the original receipts below are preserved.

**The tested CPU profiles remain unqualified as a usable starter.** Both
faster-whisper attempts reached their deadlines. Small completed no clip; turbo
completed two English clips. The different-runtime whisper.cpp diagnostic never
ran because its image pull stalled. This does not establish that Intel CPUs,
Whisper, French speech or whisper.cpp are generally unsuitable. No default was
changed and no further experiments were attempted.

## Scope and terminal outcomes

Actual runtime: Intel i7-8700B, AVX2/FMA/F16C visible inside Linux x86_64
**OrbStack**, 12 guest vCPUs and about 15.7 GiB VM RAM. Each inference container
had four CPU quota, four CT2 threads, one worker, batch one, 6 GiB memory and no
network. Image: `sha256:fd86a754708e099477a8e624babe5e9880df365a100f18cdc784f55f917a038e`.
faster-whisper 1.2.1 / CT2 4.8.3 requested INT8 and reported `int8_float32`.
[Host/VM](docker-host.json), [guest ISA](results/guest-cpuinfo.txt),
[image](results/image.json), [packages](environment-freeze.txt).

Six pinned public FLEURS human read-speech clips (three FR, three EN) total
51.10 seconds. Known language was forced on these clips. Additional controls
were silence, seeded noise, a constructed FR→EN concatenation, low amplitude and
VAD ablations. Beam five, temperature zero and **128 maximum new tokens** were
fixed bounds. No context prompt, word alignment or diarization was tested.
This is a small mechanics probe, not representative arbitrary media or natural
code switching. [Methodology](README.md), [provenance/hashes](corpus-and-models.json).

| Configuration | Model load | Completed output | Terminal result |
| --- | ---: | --- | --- |
| multilingual small INT8 | 3.17 s, one load | None; first clip was 8.94 s EN | Exit 124 at 295.34 s; OOMKilled=false |
| multilingual large-v3-turbo INT8 | 12.82 s, one load | Two EN clips, 14.82 s audio | Exit 124 at 295.20 s; OOMKilled=false |
| whisper.cpp multilingual tiny Q5_1 | No load/inference | Model verified; image unavailable | Pull stopped; no STT result |

The two inference windows totaled about 590.54 seconds (~9 m 51 s), below the
~15-minute allowance, serialized without competing inference. TERM at 295 seconds
returned exit 124. All task inference containers were inspected and removed;
none remain running. [Small state](results/small.container.json),
[turbo state](results/turbo.container.json), [summary](results/summary.json).

## Completed English subset only

| Turbo clip | Audio | Decode | RTF | Normalized word edits |
| --- | ---: | ---: | ---: | ---: |
| en-1771 | 8.94 s | 135.22 s, first decode | 15.12 | 0/29 |
| en-1914 | 5.88 s | 99.67 s, reused model | 16.95 | 1/16 |

Weighted over **these two clips only**: raw WER 2/45 = **4.44%**; normalized WER
1/45 = **2.22%**; raw CER 8/268 = **2.99%**; normalized CER 7/264 = **2.65%**.
Aggregate decode RTF is 234.89/14.82 = **15.85**, excluding load.
[Raw records and exact text](results/turbo.jsonl).

The normalized edit is **Twentieth → 20th**; the other raw edit is punctuation.
This is not a wrong numeric meaning. The normalizer preserves digits/accents and
does not expand number words. These favorable tiny-subset scores are not a
general accuracy claim. The second clip proves reuse, not a matched warm speedup:
content differs and the planned same-clip repeat was never reached.
Forced-language probability=1 is not an auto-language-detection score.

**No FR, mixed, negative, low-amplitude, VAD-ablation or same-clip-repeat result
completed.** Missing work is a timeout, not zero error or successful silence
handling. Small has no WER/CER. Python finally did not run on termination, so
final internal memory summaries are missing; external observations survived.

## Resource evidence and limits

| Worker | Highest external memory.current sample | Largest observed kernel memory.peak |
| --- | ---: | ---: |
| small | 714,252,288 bytes (~681 MiB) | 1,400,434,688 bytes (~1.30 GiB) |
| turbo | 3,749,769,216 bytes (~3.49 GiB) | 3,899,023,360 bytes (~3.63 GiB) |

These are dedicated-container cgroup figures **including page cache**. Observed
kernel peaks may increase after the last observation. Sampling began after small
started, so sampled current misses its startup peak. Turbo's completed records
reported process max RSS 2,076,340 KiB; this is not final whole-container/host
memory. [External trace](results/turbo.external-memory.log).

cpu.max was 400000/100000 (four CPUs). Small used near one core; saved throttling
was only 2,504 microseconds, versus 64,750 for turbo's snapshot. Missing AVX2 and
quota throttling therefore do not explain the observed severe latency. The
backend/thread-utilization cause remains unverified. Unrelated services had low
but nonzero activity and were left untouched; this was not a laboratory-isolated
host. [Small counters](results/small-cpu-quota-stat.txt),
[turbo counters](results/turbo-cpu-quota-stat.txt).

## Third diagnostic: blocked pull and watchdog defect

The authorized different-runtime diagnostic changed model size/quantization and
used auto language detection with FR-first ordering; it was not a matched rank.
The unexecuted [runner](run-cpp.sh) uses a 270-second deadline plus five-second
kill margin. It was never invoked. The image's bundled English-only model was
never selected.

Pinned amd64 image manifest `sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647`
declared 504,943,686 compressed layer bytes. Multilingual ggml-tiny-q5_1.bin,
revision `5359861c739e955e79d9a303bcbc70fb988958b1`, was downloaded and verified:
32,152,673 bytes; SHA256 `818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7`.

Public host registry metadata worked; the isolated anonymous Docker pull produced
no progress log and no ready image. **The intended 120-second pull cap failed:**
the alarm inherited through exec docker did not terminate the client. I manually
killed its exact verified PID. File receipts show 14:41:25Z–14:43:52Z,
**147 seconds**, exit 137: a **27-second cap overrun**, separate from the correctly
bounded inference runs. Future pulls need an external parent watchdog that owns
and terminates/reaps the process group; do not reuse the inherited-alarm pattern.
No retry, alternate image/build or fourth configuration followed.
[Pull disposition](results/whisper-cpp-pull-attempt.json),
[image absence](results/whisper-cpp-image-availability.txt).

Corpus/initial-model payloads totaled 2,747,590,602 bytes; the tiny model adds
32,152,673. Existing local test-image layers were reused. The derivative adds
357,601,762 uncompressed bytes; adding the third image's declared 504,943,686
compressed bytes gives an intended artifact payload of about 3.64 GB, below
4 GiB. **Actual registry wire/retry/partial bytes were not measured.** An empty
progress log does not prove zero network transfer.

## Next prerequisite

Resolve the bounded public image-pull path and external-watchdog handling, then
run the pinned tiny-Q5_1 FR-first diagnostic once in a fresh receipt directory.
Only after useful CPU output should a matched small-profile comparison select
the starter. CT2 utilization deserves a separate bounded investigation.
Natural code switching, FR accuracy, context biasing, long/noisy media and VAD
harms remain unqualified. See [runtime research](../../research-stt-runtimes.md).
