# CPU TTS benchmark and listening protocol

This is a comparison harness, not a production-engine recommendation. Qwen3 remains
unchanged as the default. Optional Kokoro and Pocket TTS candidates require separately
provisioned, pinned assets. No human listening acceptance is claimed by this harness.

The synthetic corpus in `benchmarks/tts/corpus.json` contains 30 French sentences,
15 English sentences, six paragraphs (three per language), and two longer scripts
(one per language). It covers amounts, dates, percentages, units, acronyms, names,
negation, dialogue and parenthetical clauses. `critical_literals` are assertions
against the original script and mandatory listening checks; they are not expected
literal ASR spellings. Number expansion or pronunciation lexicons need their own
approved exact-value assertions. No private voice reference is included.

Run inside the candidate's Linux Docker image, with model assets already downloaded:

```sh
python benchmarks/tts/run.py --backend kokoro \
  --assets /models --preset-fr ff_siwis --preset-en af_heart --threads 4 \
  --ids fr-01 fr-03 en-01 en-02 --repeat 2 --timeout-sec 600 \
  --image-digest sha256:ACTUAL_IMAGE_DIGEST --out /work/tts-pilot
```

Use explicit language-compatible presets (`--preset-fr` / `--preset-en`) or a
single `--preset` override for a bounded single-language run; the CLI must reject
unsupported requests rather than quietly select an English voice for French. A
French failure is a recorded result, not permission to alter the approved script.
For cloning use `--backend pocket-tts --reference /work/approved-reference.wav`.
References must be approved local recordings, with consent and identical bytes for
all comparable clone attempts. Record their hash and relevant recording conditions.
Use different output directories for each run; existing runs are never overwritten.
Omit `--ids` for the full corpus. A long-script rejection is retained in the results;
it does not establish successful long-form narration. Provisioning or phonemizer
network downloads are not part of offline inference acceptance: perform the
benchmark in a separate container with networking disabled after provisioning.

Each attempt launches a new CLI process. Every repeat is a **cold process**; operating
system filesystem cache state is uncontrolled. This harness makes no warm-model
performance claim. Add a separately labelled persistent-worker benchmark before
comparing warm latency. `--seed` sets the shuffle seed and is forwarded only to Pocket TTS; Kokoro
receives no seed and records null. Backend capability/metadata determines whether
seeded sampling is supported. Temperature is null unless the backend reports it. Do not infer
unsupported controls from a CLI argument alone.

`results.json` and `results.csv` retain every attempt, including errors and timeouts.
The harness records measured WAV duration, rate and channel count; `rtf_wall` is
elapsed process seconds / generated audio seconds. Lower RTF is faster; RTF 0.1
means 10 times real time. Backend timing metadata remains separate; compare
synthesis-only RTF only when its timing boundaries are documented consistently.
Timeouts terminate the owned subprocess group, then kill and reap it before the
next attempt. Raw stdout/stderr stay in local per-attempt files for diagnosis.

Environment evidence includes architecture, CPU model/logical count, cgroup CPU/RAM
limits, cpuset, Python and installed package versions. Image digest is an explicitly
supplied operator value, or `not_supplied`; the harness does not invent one. The
corpus SHA-256 is included. Save the asset manifest/checkpoint revision and image
build/runtime receipt with the run. RAM peaks, CPU utilization, model-load duration
and first-audio latency are not measured by the parent harness; report them only
when obtained from a separate instrumented measurement. Platform observations from
a developer Mac do not validate the target Intel Linux Docker environment.

Open local `listen.html` to compare successful attempts. A deterministic shuffle
produces neutral sample names and neutral audio filenames. Keep `blind-map.json`
hidden from listeners until ratings are complete. Files are local; nothing is
uploaded. The listening view is a review aid, not an automatic quality verdict.

For each sample record naturalness (1–5), stability of voice identity (1–5), exact
text correctness, every listed critical name/quantity, omissions/additions/repeats,
pause timing, clipping, background noise and join artifacts. Use headphones and the
same playback level. Review long scripts end to end for drift/fatigue. Keep notes
and listener identity separate from machine results. ASR WER/CER can flag problems,
but cannot replace listening, especially for names and quantities. Select a final
take only after human review; no engine promotion follows automatically from a
faster runtime or low WER.

## Bounded scratch pilot, 14 October 2026

Eight Kokoro attempts completed: `fr-01`, `fr-03`, `en-01`, `en-02`, each twice,
with `ff_siwis` / `af_heart`, two threads and ONNX Runtime CPUExecutionProvider.
The scratch Linux x86_64 container reported an Intel Core i7-8700B, 12 logical CPUs,
unlimited cgroup CPU/RAM, and Python 3.12.13. Assets were Kokoro-82M-v1.0 ONNX FP32;
exact hashes and installed runtime versions are retained in the local receipts.

All eight attempts produced nonempty 24 kHz mono WAVs. Cold-process wall times were
7.030–12.983 seconds; generated durations were 2.581–6.357 seconds; wall RTF was
1.753–3.025. These ranges describe this bounded pilot only. Operating-system cache
state was uncontrolled. Scratch image digest was not supplied. This run did not
verify network isolation, measure peak RAM, run ASR, or obtain human listening
ratings. It does not qualify the full corpus, long-form narration or production.
Local receipts and the blinded listening view are in `/tmp/pab-kokoro-pilot`.
