# CT2 CPU latency investigation — 2026-10-15

**A reproducible OpenMP placement profile restores useful CPU STT speed on this OrbStack host.** With the same image, small weights, INT8, four threads and beam five, explicitly setting `OMP_PROC_BIND=spread` and `OMP_PLACES=cores` completes the fixed FR/EN pair near real time. Packed MKL remains enabled. Disabling CT2 packed GEMM also avoids the severe stall, but successful packed-GEMM runs with explicit placement rule out a general claim that packed kernels are broken. This is a practical profile cure with evidence consistent with a thread-placement/scheduling interaction; its lowest-level cause is not proven.

Original results, application source/defaults and unrelated services were preserved. All changes and new receipts in this investigation are within this directory.

## Fixed controls and actual outcomes

Existing image `sha256:fd86a754708e099477a8e624babe5e9880df365a100f18cdc784f55f917a038e`, faster-whisper 1.2.1 / CTranslate2 4.8.3. Known language, beam five, temperature zero, VAD off, word timestamps off, max new tokens 128, one worker. Same complete pinned public FLEURS clips: FR `fr-1679` (6.00s), EN `en-1914` (5.88s), FR first. Each container: four CPU quota, 6GiB RAM, no network. No competing task inference. This investigation lane performed no model downloads or model copies. A separate task lane concurrently downloaded a whisper.cpp Docker image, adding background network/disk activity during these comparisons; unrelated host application load also remained nonzero.

| Small model profile | Load | FR decode / RTF | EN decode / RTF | Terminal result |
| --- | ---: | ---: | ---: | --- |
| Original four threads, packed MKL, no placement | 6.253s | Incomplete | Not reached | Killed after 121.994s container lifetime; no OOM |
| `CT2_PACKED_GEMM=0`, no placement | 7.597s | 6.673s / 1.112 | 9.060s / 1.541 | Exit 0 |
| Packed MKL + spread/cores | 3.773s | 5.172s / 0.862 | 5.915s / 1.006 | Exit 0 |
| Same spread/cores profile repeated | 3.982s | 6.403s / 1.067 | 5.918s / 1.006 | Exit 0 |

The first run did finish its encoder: 18.373s wall / 18.474s process CPU, then generation consumed at least 93.45s of observed time without completing. A live snapshot found all four active CT2 workers' last CPU was guest CPU 11 despite each having an allowed mask of 0–11. They shared about one core of process CPU. Broad masks do not guarantee scheduling distribution; a single last-CPU snapshot alone is not causal proof.

The unpacked run's FR encoder used 3.896s wall / 15.035s process CPU (~3.86 cores) and generation used 2.717s / 10.784s (~3.97 cores). Successful spread runs retained MKL packed=true and showed real distinct worker masks: `0-1`, `4-5`, `8-9`, `10-11`. First spread FR encoder used 2.596s wall / 9.287s CPU, generation 2.533s / 5.019s CPU. The different CPU/wall ratios by stage should not be reduced to an assertion that every stage uses all four cores.

[Summary with stages, CPU/memory counters and metrics](summary.json); raw per-run `stdout.log`, `stderr.log`, `host-load.log`, `container.json`, `supervisor.json` are retained in their named directories. The repeat additionally preserves exact diagnostic/supervisor source snapshots.

## One authorized turbo generalization

After the successful small-profile repeat, the same spread/cores profile was run once on the existing turbo weights, with only model selection changed. All five turbo model/config/tokenizer files match the original manifest before and after the run: [before](turbo-model-hash-verification.json), [after](turbo-model-hash-verification-after.json). No new weights were downloaded. All other fixed settings, clips and limits were retained; its parent TERM deadline was 110s.

| Turbo spread/cores | Load | FR decode / RTF | EN decode / RTF | Terminal result |
| --- | ---: | ---: | ---: | --- |
| One pair | 10.288s | 13.198s / 2.200 | 11.536s / 1.962 | Exit 0; 36.461s container lifetime; no OOM |

Turbo's FR hypothesis exactly matches this one reference, including **Aerosmith**. EN has the same numeric-normalization edit as small. FR encoding used 12.049s wall / 44.495s process CPU; EN encoding 10.569s / 38.903s, around 3.7 cores. This shows the practical placement profile also avoids the original turbo failure mode for this fixed pair. It is a different model with different memory/accuracy/speed behavior, not an isolated model-size ranking or a matched retrospective speedup against the original EN-first run. Cgroup kernel memory peak was 3,906,560,000 bytes (~3.64GiB), including page cache; this is not host memory or stable production residency.

## Accuracy remains a separate limitation

All three completed small-profile runs produced exactly the same hypotheses. FR reference is “Le groupe Aerosmith a annulé les derniers concerts de sa tournée.”; the hypothesis is “Le compagnie Rossmiss a annulé les derniers concerts de sa tournée.” Normalized word error is 2/11 = **18.18%**, including the proper name. The English normalized edit is “Twentieth” → “20th”, 1/16 = **6.25%** with the original normalizer that does not expand number words. It is the same numeric meaning. Good runtime mechanics do not establish acceptable French accuracy, broad quality, natural code switching, long media, VAD behavior or production concurrency. There is no same-clip warm repeat inside one resident model; the spread repeat starts a fresh container/model.

## Static evidence and integrity

- Original harness correctly exhausts the lazy generator and includes full decode. File fsync occurs after the decode timer and cannot explain the first stalled inference. The installed constructor forwards `cpu_threads` to CT2 `intra_threads`; four threads are requested correctly.
- Original audio hashes/rates/channels/durations were independently checked. The diagnostic records both fixed clip hashes per attempted sample. All four small model files match original manifest SHA256 values before the successful runs and after them; see [before](model-hash-verification.json) and [after](model-hash-verification-after.json). This verifies consistency with the original local receipts; it does not substitute for independent artifact-signing provenance.
- Whisper pads short clips to a 30-second encoder window. A short speech duration therefore does not imply proportionally short encoder work.
- `metadata2/` sees AVX2/FMA/F16C and 12 guest CPUs with four-CPU quota. Twenty NumPy 1024-square float32 matrix products used 0.496s wall / 1.974s process CPU (~four cores). This proves available multicore arithmetic in this VM/image during the tiny probe, not CT2 throughput.
- Initial metadata probe `metadata/` stopped on unavailable `/proc/pressure/cpu`; no model load/inference occurred. `metadata.py` was corrected to record unavailable pressure files. Both receipts remain intact.
- Exact [CT2 v4.8.3 backend source](https://github.com/OpenNMT/CTranslate2/blob/v4.8.3/src/cpu/backend.cc) selects MKL on Intel by default and reads **`CT2_PACKED_GEMM`**, default true. The installed binary contains that name. `CT2_USE_EXPERIMENTAL_PACKED_GEMM` is not the recognized name and was not used. Runtime verbose logs prove AVX2 / MKL / packed state / effective `int8_float32` per experiment.
- Late baseline process I/O counters were nearly unchanged while CPU increased; mounted-weight I/O is not supported as the explanation for its prolonged generation. No native-volume model-copy experiment was needed.

## Supervision and measured budget

The first supervisor had a cleanup defect: TERM at 120s did not finish the PID-1 Python process, KILL followed two seconds later, and process-group TERM then raised `PermissionError` before the client was explicitly reaped. The container was already stopped, inspected and removed; its Docker state is exited/137/OOMKilled=false, actual lifetime **121.994s**, supervisor elapsed **123.289s**, `client_exit: null`. This is an approximately two-second per-config cap overrun and an incomplete original client-reap receipt; it remains visible. Subsequent host process inspection found no old Docker client.

The helper was corrected to wait/reap immediately after Docker KILL, then use a process-group fallback if needed. If final inspection still shows Running, it now kills/waits and requires another stopped inspection before deletion. Cheap TERM-resistant fixtures in `watchdog-fixture/` and `watchdog-fixture2/` verify observed TERM→KILL, client reaping, stopped inspection and removal; the second took 4.459s including its two-second deadline/grace. Later inference runs use a 115s TERM deadline (turbo: 110s), leaving observed cleanup room below 120s.

Five inference invocations (three small profiles, one authorized repeat and one authorized turbo generalization) totaled **219.048s of actual container lifetime including model load and cleanup**, below the 360s investigation budget. No further inference was run. Every successful run has an exited/0/no-OOM receipt; the last two additionally record explicit terminal-receipt-before-removal and `client_reaped: true`. Final filtered Docker inventory is empty: [task containers](final-task-containers.txt). The cheap metadata and watchdog fixtures did no inference.

This helper provides host-owned supervision and bounded individual Docker calls. Sequential Docker cleanup call timeouts can exceed the nominal reserve if the Docker daemon stalls; it is **not a proven hard total-wall deadline under every daemon failure**. Actual completed-run lifetimes and cleanup receipts are the relevant evidence here. Process CPU, cgroup memory (including page cache), and max RSS have their own scopes; none is total host memory. Sampling may miss spikes.

## Use the working evaluation profile

`run-spread.sh` concretely selects the successful small profile with the exact existing image. It refuses existing output directories/containers through the supervisor. Launch only intentionally with an exclusive inference slot and a new owned receipt directory:

```sh
bash docs/agent-interface/evaluation/stt/debug-inference/run-spread.sh \
  pab-stt-debug-UNIQUE docs/agent-interface/evaluation/stt/debug-inference/UNIQUE
```

The runner is syntax-checked; its equivalent exact Docker arguments were exercised twice. No application default or general runtime recommendation has been changed. Retain this profile as a local CPU evaluation starting point, then test representative French/proper-name material before accuracy acceptance.

## Attribution limits

Host load changed during these non-laboratory runs. macOS `CPU_Speed_Limit` samples ranged from 60 to 73 during the small profiles; these are constraints, not measured instantaneous clock frequency. Guest-reported 2000 MHz is not an independent physical-frequency measurement. Host-load/thermal changes, page-cache state and lack of randomized interleaving prevent a precise isolated speedup attribution. The evidence supports the explicit placement profile working twice, not that thermal restriction, virtualization or every backend interaction has been eliminated. Original small began with another clip (8.94s EN), so current FR-first timing is not a matched retrospective speed ratio.
