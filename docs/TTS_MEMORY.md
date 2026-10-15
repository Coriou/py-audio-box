# Resident TTS memory

The Kokoro CPU adapter disables ONNX Runtime's CPU memory arena. The dedicated
CPU images set `MALLOC_MMAP_THRESHOLD_=131072` before Python starts, keeping glibc's
large allocations returnable on free instead of raising that threshold dynamically.
The combination is important: disabling the ONNX arena alone did not consistently
reduce transient memory. It trades some allocation overhead for less resident RAM.

Weights stay FP32. Graph optimizations, tensor reuse, thread counts, presets and
synthesis parameters remain unchanged. Take metadata records `cpu_mem_arena: false`.
No quantization, precision change, shorter text or output caching is involved.

CPU batch and Qwen workers also release each completed result after export/send,
before waiting for another job or allocating its audio. The model remains resident.
Weak-reference regressions check that lifetime and unchanged samples. Qwen savings
are not inferred from the Kokoro measurements. The allocator environment applies
to the dedicated CPU Docker images; the general Qwen toolbox image is unchanged.
Pocket shares the CPU image configuration, but real Pocket memory measurements
remain unavailable because model provisioning requires owner access.

The startup policy is documented by [glibc's allocation tunables](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html).
ONNX documents the default arena in its [Python session options](https://onnxruntime.ai/docs/api/python/api_summary).
For custom/direct deployments, set the allocator variable **before starting
Python**; setting it after runtime imports is not equivalent. The setting is not
a hard memory cap. Large inputs can still require substantial transient memory.

## Reproduction and scope

`benchmarks/tts/memory.py` measures fresh sequential CLI batches, initial model
load included. Supply JSON `items` with unique `id`, `language`, and exact `text`.
Use frozen images, the same verified bundle/scripts, identical CPU/memory limits,
and disabled network:

```sh
python benchmarks/tts/memory.py --assets /models --scripts /work/memory-scripts.json \
  --language fr --threads 4 --repeat 1 --timeout-sec 300 \
  --image-digest sha256:ACTUAL_IMAGE_ID --out /work/unused-output-directory
```

Each requested take, failure, process log, WAV and batch report is retained.
Reports record asset/source/helper/script hashes, allocator environment, original
text, frames/rate/channels, finite samples and decoded PCM hashes. PCM equality
establishes numerical preservation for tested inputs, not acoustic acceptance.

The observer reads owned CLI group `smaps_rollup` roughly every 50 ms. RSS may
count shared pages twice; PSS apportions them; USS sums private clean and dirty
pages. Partial snapshots are discarded and missing counts remain explicit. Values
are observed maxima of sequential, non-atomic group sums, not guaranteed
simultaneous peaks, mathematical lower bounds, or container/host/GPU peaks.
The observer is outside the measured group. Short spikes may be missed; OS page
cache and background services are uncontrolled. One batch per language/variant
is a bounded experiment, not a universal memory or latency guarantee.

## Exploratory variants

Raw receipts are preserved under `work/tts-memory-20261015/`:

- `candidate-*`: no ONNX arena alone; English observed PSS maximum rose from
  1,506,784,256 to 1,649,173,504 bytes.
- `shrink-*`: arena shrink after every window; lower memory, but English whole
  batch wall rose from 60.162 to 89.037 seconds.
- `periodic-*`: shrink every fourth window; only about 3–4% lower observed
  maximum and English wall increased about 18%. This proxy was removed.
- `malloc-en-r1`: no arena with `MALLOC_ARENA_MAX=2`; no useful peak improvement.
- `mmap-en-r1`: selected policy's exploratory run, separate from frozen final runs.
- `probe-*`: direct-adapter exploratory retention and kernel process-RSS probes;
  different accounting from production CLI group observations.
- `interrupted-*`: interrupted engineering runs/builds, excluded. The second
  exploratory baseline French run briefly overlapped a stopped pilot startup;
  its timings are excluded. Baseline `*-r1` and final runs are the selected pair.

Early reports retain overly strong `lower bounds` wording; the accounting
definition above corrects it. The description-only harness change is recorded by
actual per-report source hashes. It did not change inference or sampling.

## Frozen final-source comparison, 15 October 2026

One batch per language and variant, five takes each (short, long narration, short, the same narration again, short). French uses ff_siwis; English uses af_heart. Same pinned Kokoro FP32 assets, scripts and four threads. Linux x86_64, four CPU quota, 4 GiB memory limit, network disabled. Both final batches include initial model loading. All 20 attempts succeeded; all ten baseline/final decoded PCM pairs are exactly equal.

| Language | Baseline observed PSS maximum (MiB) | Final observed PSS maximum (MiB) | Change | Baseline batch wall (s) | Final batch wall (s) | Wall change |
|---|---:|---:|---:|---:|---:|---:|
| fr | 1368.0 | 852.7 | -37.7% | 61.253 | 66.260 | +8.2% |
| en | 1437.0 | 873.9 | -39.2% | 60.162 | 73.654 | +22.4% |

This selected policy prioritizes RAM: it saves about 38–39% of the observed maximum on these batches, with about 8% longer French and 22% longer English process walls. It is not a simultaneous speed improvement. Previous throughput measurements describe the older allocation policy and remain historical receipts. These measurements do not guarantee the same tradeoff on other inputs or hosts.

Tested inference image: `sha256:eacc44d6c6338d131435ef6527e0d8301caac4afa929bb5d1d493ebb7085ff35`. Baseline: `sha256:24a40ae526151bc6d8c375068dfb282abf7620d19a83053e59f264f74c001220`. The baseline used the common memory observer mounted read-only; the final image embeds it. Source/helper hashes record the observer version; its only semantic wording correction and later allocator-environment provenance field do not alter sampling. Actual final allocator setting is recorded as 131072 in both reports.

Baseline French discarded one incomplete snapshot; baseline English and both final batches had no missing snapshots. Complete sample counts and all raw values remain in `work/tts-memory-20261015/comparison-final.json` and the underlying `baseline-*-r1` / `final-*-r1` reports. One full batch per arm remains a limited latency/memory experiment.

The packaged release differs from the tested image only in its CPU README; runtime source hashes and allocator environment match the workspace and measured image exactly (`release-source-hashes.json`, `final-source-hashes.json`). No numerical savings for Pocket/Qwen are claimed.

Final isolated unit image: **202 passed, one skipped** (two expected duplicate-ZIP
fixture warnings), network disabled and no source/voice/cache mounts. A separate
real five-inference check retains the first read-only output array through four
later calls and verifies unchanged finite samples. Environment/package/quota
parity is recorded in `environment-parity.json`. Runtime sources in both measured
and packaged image trees match; the sole source-tree difference is the updated
CPU README (`measured-source-tree.json`, `release-source-tree.json`).

Astra's final independent review gives **bounded GO with no blockers** and confirms
the raw comparisons, calculations, image source parity and final suite; eight
additional targeted tests passed independently. The reviewed release is active in
local `py-audio-box:kokoro-cpu` and `py-audio-box:pocket-cpu` tags. Prior CPU images
remain recoverable under their performance-final tags. Exact identities and the
review receipt are in `work/tts-memory-20261015/validation.json` and
`astra-review-r2.txt`. These measurements and reviews preceded Git publication; no deployment was performed.
