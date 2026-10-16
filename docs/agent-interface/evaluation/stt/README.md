# Bounded public-data CPU STT probe

Date: 2026-10-15. This is a small runtime/contract probe, not an accuracy ranking,
release qualification or proof of natural code-switch recognition.
Results and limitations are summarized in `RESULTS.md` after the run.

The host is an Intel i7-8700B (6 physical/12 logical cores, AVX2/FMA/F16C), with
64 GiB physical RAM. The Linux x86_64 OrbStack Docker VM exposes 12 vCPUs and
16,843,739,136 bytes RAM. See `host-cpu.txt` and `docker-host.json`.

## Corpus and model provenance

- Public speech: [Google FLEURS](https://huggingface.co/datasets/google/fleurs),
  revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0.
  Attribution: Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech* (2022), Google.
- Deterministic selection: first three distinct sentence IDs from each of the
  French `fr_fr` and English `en_us` test TSVs. These are public human read speech,
  six samples total; neither random nor representative of arbitrary media.
- Negative controls: four seconds each of digital silence and seeded Gaussian
  noise (`numpy.default_rng(1729)`, standard deviation 0.02), PCM16 mono 16 kHz.
- Constructed mixed control: selected French + one-second silence + selected
  English. It is **not natural code switching**. It uses automatic language
  detection; natural single-language clips receive their known language.
- Small conversion: [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small),
  revision `536b0662742c02347bc0e980a01041f333bce120`, MIT.
- Larger conversion: [mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo),
  redirecting to [dropbox-dash/faster-whisper-large-v3-turbo](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo),
  revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, MIT.

`corpus-and-models.json` records source URLs, revisions, SHA256 hashes, TSV IDs,
exact authored/source-normalized text, durations and derived sample identities.
The JSON source metadata receipts retain canonical repository IDs and license
labels. The downloaded assets are only under `work/stt-evaluation`; no legacy
voices/cache or user audio is mounted. Public source/model downloads total
2,747,590,602 bytes, plus the new dependency image layer. The existing ~1.55 GB
test image was reused, not downloaded afresh.

## Runtime and supervision

`Dockerfile` extends the existing local test image only for this probe, adding
faster-whisper 1.2.1 and CTranslate2 4.8.3. `base-image-id.txt`,
`results/image.json` and `environment-freeze.txt` identify the actual base,
derivative and installed transitive versions. These are evidence of this run;
rebuilding from a mutable base tag without restoring its exact identity is not
bit-for-bit image reproduction. No model is baked into the evaluation image.

`run.sh` runs small then turbo **sequentially**, with four CPU threads, four Docker
CPU quota, one model worker, batch size one, 6 GiB container memory, no network,
beam five, temperature zero and `max_new_tokens=128`. The token ceiling is a
probe bound, not an unconstrained accuracy setting. Model weights are requested
as CPU INT8; each receipt records the effective type exposed by CT2. Word
timestamps, glossary/context prompts and diarization are disabled in this probe.

Each container uses `timeout`: TERM at 295 seconds, KILL after five more seconds.
The host runner retains exit status, Docker `State`/OOMKilled and declared limits
even if Python's `finally` never runs. Containers are removed only after that
receipt is saved. The initial plan contained two faster-whisper configurations.
A later authorized whisper.cpp tiny diagnostic was blocked before inference;
see RESULTS.md for its pull receipt and watchdog defect. No automatic retry,
repair loop or fourth configuration followed.

Each completed configuration attempts sixteen decodes: nine primary logical
samples, one same-clip warm repeat, VAD-on ablation on one natural clip per
language plus both negatives, and a 0.02-gain version of the first natural clip
with VAD off/on. Low amplitude and synthetic negatives are mechanics probes,
not representative microphone/noise coverage.

Python records model initialization separately from first decode and warm decode.
The first decode is labeled cold-first-decode; the OS page cache is not forcibly
cleared. One warm repetition does not estimate stable production throughput.
`ru_maxrss` is process peak RSS in KiB; sampled cgroup `memory.current` runs at
200 ms and can miss spikes. Kernel `memory.peak`, when available, includes the
whole dedicated container and page cache. None of these is total host memory.

Raw WER tokenizes on whitespace with case/punctuation intact. Normalized WER/CER
uses NFC + casefold + Unicode word tokens, retaining digits and accents without
number expansion or semantic rewriting. Empty-reference rates are null and
insertion counts are reported. Corpus weighted WER is total edits / total
reference words, not the average of per-clip percentages.

## Reproduce intentionally

Run from repository root, in an exclusive inference slot, after inspecting host
limits and keeping the exact base image. These commands download public assets
and perform real CPU inference; do not run them merely to read this report.

```sh
docker build -f docs/agent-interface/evaluation/stt/Dockerfile \
  -t py-audio-box:stt-evaluation-20261015 docs/agent-interface/evaluation/stt
docker run --rm --cpus 2 --memory 1g \
  -v "$PWD/docs/agent-interface/evaluation/stt:/evaluation:ro" \
  -v "$PWD/work/stt-evaluation:/assets" \
  py-audio-box:stt-evaluation-20261015 python /evaluation/provision.py
bash docs/agent-interface/evaluation/stt/run.sh
docker run --rm --network none --cpus 1 --memory 512m \
  -v "$PWD/docs/agent-interface/evaluation/stt:/evaluation:ro" \
  -v "$PWD/docs/agent-interface/evaluation/stt/results:/results" \
  py-audio-box:stt-evaluation-20261015 python /evaluation/summarize.py
```

The runner refuses existing result files/containers. Use a fresh owned run
directory and unique names for an intentional new experiment; never append into
this receipt or overwrite its assets/results.
