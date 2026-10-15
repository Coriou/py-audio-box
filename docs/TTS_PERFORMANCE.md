# Measure cold versus resident CPU synthesis

`benchmarks/tts/warm.py` compares fresh Kokoro CLI processes with a homogeneous
batch that retains one model worker between successful jobs. It runs the same
original scripts, French/English presets, FP32 asset bundle and thread count in
both modes. Threads 1, 2 and 4 run sequentially by default. This measures process
reuse; it does not imply better pronunciation or naturalness.

Run inside the candidate image after asset provisioning, with network disabled:

```sh
python benchmarks/tts/warm.py \
  --assets /models/kokoro --ids fr-01 fr-03 en-01 en-02 \
  --threads 1 2 4 --repeat 2 --timeout-sec 600 \
  --preset-fr ff_siwis --preset-en af_heart \
  --image-digest sha256:ACTUAL_TESTED_IMAGE_ID --out /work/kokoro-warm-comparison
```

The asset manifest must declare a Kokoro FP32 checkpoint. The report retains its
full bytes hash and file identity; the adapter independently verifies asset files.
The precision label comes from the pinned manifest, rather than a speed estimate.
Use an unused output directory; the harness never replaces listening takes. Keep
CPU quotas, architecture, package versions, scripts, preset and assets identical
between comparisons. The environment receipt records CPU/cgroup constraints and
operator-supplied image identity. It cannot inspect a Docker daemon from inside
the container to establish that image identity independently.

For each language/thread/repetition group, cold attempts run first, then a warm
batch. The first batch job includes worker creation and model initialization;
later successful jobs reuse the loaded model. Filesystem/page-cache state is
uncontrolled, so “cold” means a fresh process, rather than an empty OS disk cache.
A failed or timed-out batch worker is discarded; with continue-on-error the next
job creates a new worker and is explicitly marked cold. The batch report retains
this distinction. Do not average restarted jobs as if all were warm.

`results.json` records every requested attempt, process exit/timeout, raw batch
report and measured WAV frames/rate/channels/duration. Per-job metadata preserves
reported initialization, generation and export timings. Initialization can include
imports, asset hash verification and ONNX session loading: individual phase times
are unavailable unless the CLI explicitly instruments those phases. Missing
measurements remain missing; there is no invented phase allocation.

Group wall speedup is the sum of measured cold process wall times divided by the
whole measured batch process wall time, including its first model load. It is
reported only for successful groups. Warm process wall per requested job is an
amortized batch measure, rather than that job's measured latency. Backend
`generation_seconds` and job latency remain separate. RTF always means elapsed
seconds divided by generated audio duration; a lower value is faster.

The harness samples `/proc` RSS for its owned subprocess group every approximately
50 ms. It reports the largest observed **sum of process RSS**, in bytes. Shared
pages can be counted more than once across the CLI parent/worker; brief peaks can
be missed. This is a sampled lower bound within that process-group accounting,
not model-only memory, proportional-set-size, cgroup peak, host RAM, or evidence
of a memory improvement. The benchmark observer and other containers are excluded.
On systems without `/proc`, the value is null. Compare only the same scope and
sampling policy; inspect the report before claiming lower memory.

Cold/warm WAVs are compared numerically: finite decoded samples, identical sample
rate/array shape, exact PCM equality, maximum absolute sample difference, WAV hash
and decoded PCM hash. Numerical identity is useful evidence that worker reuse did
not alter those samples; it cannot establish correct spoken text, naturalness,
pronunciation or listening acceptance. Review representative short and long WAVs
in French and English separately, using the listening protocol in
`docs/TTS_BENCHMARK.md`.

The historical `benchmarks/tts/run.py` remains a cold-process harness. Preserve its
baseline receipts/image alongside this new report. A historical image without a
batch command cannot be the warm arm: run baseline cold tests separately and do
not attribute runtime/model changes to model reuse. No measured performance
result is claimed by this document before a corresponding run exists.

## Measured six-phrase campaign, 14 October 2026

The completed campaign in `work/tts-performance-20261014/cold-vs-batch/results.json`
contains 36 successful attempts: 18 cold-process takes and 18 batch takes. It
compares French `fr-01`, `fr-03`, `fr-07` with `ff_siwis`, and English `en-01`,
`en-02`, `en-03` with `af_heart`, at 1/2/4 threads, one repetition. Each group
contains three phrases. Partial earlier runs are excluded from these figures.

This was a frozen pre-guard-patch source snapshot in image
`sha256:b7155f3b5736a21e8449b7e435b571014dbc249f036d76a30a31e2f1676b486c`.
The receipt records benchmark source hash
`bcbd49889ccf3bd0906ab2f30b747180bb762a0cc267bb1ff3aa0808298c6aa0` and
voice-cpu source hash
`6b45090f75b9d0d84b1eb1bfa8aeb0115d61d9fbe6b82807e161387998aeffca`.
The run used an absolute output directory, Kokoro presets with no reference WAV,
and finite 180-second per-job deadlines. Those inputs do not exercise the later
relative-output and reference/deadline guard corrections. These results remain
attributed to the frozen source/image; a final-source recheck is separate evidence.

The Linux x86_64 container reported an Intel Core i7-8700B at 3.20 GHz, 12 logical
CPUs, a four-CPU cgroup quota and a 4 GiB memory limit. The runtime used Python
3.11.17, NumPy 2.4.6, kokoro-onnx 0.6.1 and ONNX Runtime 1.31.0 with the pinned
`Kokoro-82M-v1.0-onnx-fp32` bundle. The actual runtime ONNX version is 1.31.0;
the asset manifest's provision-time version entry is 1.29.0. The report retains
both provenance records rather than treating them as the same measurement.

| Language | Threads | Sum of three cold process walls (s) | Whole three-job batch wall (s) | Process wall speedup |
|---|---:|---:|---:|---:|
| French | 1 | 24.603 | 17.212 | 1.43 |
| English | 1 | 25.666 | 18.699 | 1.37 |
| French | 2 | 18.227 | 11.642 | 1.57 |
| English | 2 | 20.330 | 14.462 | 1.41 |
| French | 4 | 21.395 | 10.554 | 2.03 |
| English | 4 | 19.993 | 10.993 | 1.82 |

The table rounds values; the machine report retains full precision.

All 18 paired cold/batch comparisons have exactly equal decoded PCM, matching
rate/shape and finite samples. That supports numerical preservation for these
six scripts and this frozen configuration. It provides no acoustic or exact
spoken-text acceptance, and no guarantee for longer scripts, other voices or
other platforms. One repetition and uncontrolled page-cache state limit the
performance conclusion; the four-thread figures describe these three-phrase
batches, rather than a universal speed multiplier. Sampled process-group RSS is
retained in the machine report; no memory improvement is claimed.


## Final-source recheck

The final source image `sha256:24a40ae526151bc6d8c375068dfb282abf7620d19a83053e59f264f74c001220` passed an additional paired four-thread recheck after all reviewer fixes. All 12 attempts succeeded and all six PCM comparisons are exactly equal.

| Language | Three separate cold CLI processes (s) | One three-job batch, first load included (s) | Speedup |
|---|---:|---:|---:|
| French | 18.167 | 9.553 | 1.90 |
| English | 18.630 | 11.265 | 1.65 |

For these two three-phrase groups, total process wall was 36.796 s for six separate commands versus 20.817 s for two homogeneous batches: 43.4% less time. Each batch includes its initial model load. This is one repetition on the same four-CPU/4-GiB Linux x86_64 setup, not a universal speed promise or an acoustic rating. Full receipts: `work/tts-performance-20261014/final-source-check/results.json`.

The final no-network, no-source-mount unit image passes 195 tests (one opt-in/package-dependent skip). Astra accepts the bounded implementation after three performance review rounds. Raw WAVs, scripts and listening metadata are in `work/tts-listening-20261014/enriched/`; the downloadable archive is `work/tts-listening-20261014/audios-complets-a-ecouter.zip`. No human listening acceptance is claimed.
