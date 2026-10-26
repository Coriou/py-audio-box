# Pocket / Kokoro CPU pilot — 26 October 2026

**Pocket remains an optional cloning lane. This pilot does not justify replacing
Kokoro for ordinary CPU preset narration, or replacing Qwen for cloning.** Qwen
was not measured; no verified local bundle was found in the bounded host/cache
scan or the sixteen retained Docker model volumes. One listener has now reviewed all twenty blind samples; findings below are
qualitative and do not establish general speaker fidelity.

The successful comparison contains five unchanged French corpus items (`fr-01`,
`fr-03`, `fr-07`, `fr-10`, `fr-paragraph-3`), each run twice per engine: **20/20
successful attempts** in the final configurations. The scripts cover a short
introduction, exact money, proper names, negation and a short paragraph. Kokoro
uses `ff_siwis`; Pocket uses one already acquired, eight-second public reference.
A preset and a reference-conditioned voice are different tasks: this comparison
measures narration costs and supplies listening material, not comparative cloning
quality. It does not qualify long-form narration, English or other hardware.

## Measured costs

| Metric | Kokoro | Pocket French |
| --- | ---: | ---: |
| Successful attempts | 10/10 | 10/10 |
| Median cold-process wall time | 5.68 s | 39.51 s |
| Cold wall-time range | 4.07–10.34 s | 29.72–101.13 s |
| Median cold wall RTF | 1.66 | 15.14 |
| Median sampled process-group peak RSS | 504 MiB | 1065 MiB |
| Maximum sampled process-group peak RSS | 641 MiB | 1067 MiB |

Lower RTF is faster: RTF 1 means one second of computation per second of produced
audio. These cold measurements include startup, imports, model verification/loading,
and, for Pocket, reference conditioning. Every take starts a fresh process;
filesystem cache state is uncontrolled. The groups contain the same scripts, but
produced durations differ, so wall-time and RTF comparisons answer different
questions. Successful WAV creation does not prove all words or values were spoken
correctly. Do not interpret these numbers as warm-model or first-audio latency.

Both runs used the same Intel Core i7-8700B host under Linux/amd64 OrbStack,
2 CPU quota, 2 requested backend threads and a 4 GiB container limit. Model runs
were serial, network-disabled, with read-only pinned source and model volumes.
The RSS sampler sums the owned CLI/worker process-group RSS; shared pages may
count twice, sampling can miss peaks, and this is not container or host peak RAM.
It is also distinct from the 4 GiB admission limit. Backend-specific loading,
conditioning and generation intervals are retained separately without assuming
identical timing boundaries. Both successful engine containers exited zero
without OOM.

The actual source was `f82e671`; the existing cold harness used the RSS sampler
from `benchmarks/tts/warm.py`. Exact image IDs, asset manifests, script coverage,
seeds (Pocket 1729/1730; Kokoro has no seed), measurements and audio hashes are in
[the sanitized results](agent-interface/evaluation/pocket-comparison-r408/results.json).
The reference hash is
`ca1021790719c2099a10cdb7ce9c2e94a8968fa137d95cb4f6997baea5d48e5b`.
No weights, credentials or recordings are committed.

## Retained initialization failures

Two earlier Pocket runs produced twenty initialization errors and no audio.
The non-root benchmark UID had no passwd entry; TorchInductor's default cache
selection called `getpass.getuser()` and failed. Setting `HOME` alone did not fix
it. The successful run supplied `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor` in
its bounded temporary filesystem. Original failures and the diagnostic traceback
remain retained separately; they receive no successful-generation credit and are
not counted as audible quality results. Kokoro did not require that Torch-specific
environment setting. This pilot does not modify the installed worker policy.

## Human listening findings

On 26 October, Ben reviewed the reference and all twenty neutral samples before
unblinding. [The sample-by-sample record](agent-interface/evaluation/pocket-comparison-r408/listening-review.json)
preserves the mapping and paraphrased observations, without invented scores.

- Pocket generally received the more natural/human voice descriptions. Samples
  001 and 009 were Pocket; 001 used the same reference conditioning as the other
  Pocket takes, which does not itself prove speaker identity.
- Both Pocket paragraph takes (004, 006) reportedly omitted `A-2047, version B.`
  Both money takes (013, 020) reportedly spoke the amount incorrectly and too fast.
  These are four takes with material text-fidelity failures out of ten Pocket
  takes in this small pilot, not an estimated general failure rate.
- Kokoro was often described as robotic or clearly AI. Both names-sentence takes
  (002, 011) sounded English-accented; 002 also had an incomplete `accompagne`.
  Both Kokoro amount takes (014, 018) were reported correct.
- Pocket sample010 had a perceived low-quality MP3-like artifact. The delivered
  file is PCM24 WAV; the observation does not establish an encoding fault.

The recorded requests contain the complete paragraph and amount. The embedded
player preserves the original WAV bytes. The cause of the missing/incorrect speech
remains undiagnosed. Successful generation and playback therefore cannot serve
as text-fidelity acceptance. Pocket must not be promoted as reliable exact-text
narration on this evidence; Kokoro also needs pronunciation/prosody improvement.
No comparison against Qwen was performed.

## Listening and decision boundary

The local kit contains twenty neutral, shuffled WAV filenames, `listen.html`, a
rating worksheet and a separate `blind-map.json`. The HTML embeds the original
WAV bytes so playback also works when an asset viewer serves only that single
file; it does not depend on sibling-file URLs. WAV copies were SHA-256 checked;
all are mono PCM24 at 24 kHz. The listening page contains no backend names or source
paths. Rate naturalness and exact spoken text before opening the mapping. Check
`1 250,50`, names, negation, omissions, additions and repetitions explicitly.
After unblinding, compare Pocket to its reference; speaker-fidelity scoring does
not apply to Kokoro's unrelated preset. A wrong amount or negation remains an
error even if the voice sounds pleasant.

Generate a kit from separately collected cold runs without loading any model:

```sh
python3 benchmarks/tts/compare.py \
  --run /absolute/results/kokoro \
  --run /absolute/results/pocket \
  --reference /absolute/approved-reference.wav \
  --out /absolute/fresh-listening-directory
```

The tool records missing/error attempts, requires matching supplied corpus hashes
and script identities, and refuses duplicate attempts, unsafe/symlink audio inputs
and existing output directories. Its coverage check uses the supplied union;
joint omissions require an external plan check. This pilot separately checked
all five planned IDs and both repeats for both engines. It does not elect a winner.

The evidence supports retaining Pocket as an explicit option for CPU cloning,
with its gated model acquisition and measured cost visible. The listener often preferred Pocket naturalness, but reported material fidelity
errors. This supports no claim of better cloning than Qwen or general performance
superiority.
The headless release's operational acceptance remains separate from this backend
pilot and from acoustic/text-fidelity acceptance, which this pilot has not passed.
