# Hands-on failure investigation — 2026-10-15

Final [Astra review](debug-review-astra.md): GO for bounded local repairs and mechanics; no global/default or quality promotion.

The [original results](RESULTS.md) remain historical evidence. This follow-up separates inference configuration, pull transport, supervision defects and quality acceptance. We did not change application defaults or target unrelated services.

## CPU inference: a working local profile

Using the same small model/image, the fixed six-second French / 5.88-second English pair did not complete under the original four-thread configuration. Four active CT2 workers shared approximately one CPU; their observed last CPU was the same despite broad allowed masks. Explicit `OMP_PROC_BIND=spread` and `OMP_PLACES=cores` distributed workers and completed both clips twice, retaining AVX2, Intel MKL, packed GEMM, INT8 and beam five.

| Profile | French decode | English decode |
| --- | ---: | ---: |
| Original small configuration | Incomplete at deadline | Not reached |
| Small, explicit placement | 5.17 s | 5.91 s |
| Small, same placement repeated | 6.40 s | 5.92 s |
| Turbo, explicit placement | 13.20 s | 11.54 s |

Times exclude model load. The current profiles share clips and timer scope; they are not a controlled retrospective speed ratio against the original EN-first corpus run. Host load, power constraints, page-cache state and concurrent image downloading remained confounders. Disabling packed GEMM also worked, but successful packed runs with explicit placement show that packed kernels are not generally broken.

This is a demonstrated local workaround and evidence for a CT2/MKL/OpenMP/OrbStack placement interaction, not proof of its lowest-level cause or a universal CPU setting. It shows that this Mac can produce useful CPU STT output. Native-host versus VM isolation and other operating systems remain untested.

Small misrecognized the French proper name; turbo exactly matched this one French reference. English differed by equivalent numeral formatting and punctuation. Two short utterances do not qualify French, mixed-language, contextual dictation or long-media accuracy. No runtime/default is promoted.

The [inference report](debug-inference/REPORT.md), [machine-readable evidence](debug-inference/summary.json) and [working evaluation runner](debug-inference/run-spread.sh) preserve exact settings, model hashes, stage timings, worker masks, resource scopes and all failed/successful receipts. Five inference containers totaled 219.048 seconds including load and cleanup; none remain running.

## Supervision fixes

Docker survives the old inherited Perl alarm; a harmless Docker-versus-sleep comparison reproduces that defect. New parent-owned supervision explicitly terminates/reaps clients and, for inference, inspects a stopped named container before removal. TERM-resistant fixtures exercise escalation. A first new inference cleanup error and a first pull receipt error exposed macOS process-group `EPERM` behavior; failed receipts are retained and subsequent helpers handle/reconcile that behavior.

The inference helper has bounded individual Docker calls, but sequential cleanup timeouts are not a proven hard total deadline under a stalled daemon. Actual lifetimes and stopped/removal receipts establish what happened here. See the reports for the first approximately two-second inference cap overrun and incomplete initial client-reap receipt.

## Pull and alternate runtime

The pinned image finally downloaded in 234.448 seconds under a 300-second external deadline. The earlier 180/120-second diagnostic caps were too short at measured throughput, and canceled downloads did not retain the layers for the next attempt. Registry metadata/body access worked. This explains the new capped attempts; it does not establish the exact cause of the historical zero-output CLI pull. See the [pull report](debug-pull/REPORT.md).

Three new attempts observed 1,247,331,418 compressed-progress bytes in aggregate, separately from the image's 504,943,686-byte compressed manifest. Actual wire/retry bytes are unmeasured. The later authorized 1.6GiB provisional payload allowance supersedes the earlier local debugging limit; the historical first-pass artifact calculation is not a cumulative campaign total.

The alternate CLI exposed an unexecuted-runner defect: it rejects `--no-context` but returns zero while printing help. The supported `--max-context 0` reaches model loading, then the prebuilt binary crashes with exit 132. A signal handler captured SIGILL at `libggml-cpu.so.0+0x21f9d`; disassembly identifies an EVEX AVX-512 instruction unsupported by this AVX2 CPU. The image was built with `GGML_NATIVE=ON` / `-march=native`, despite its generic linux/amd64 image label. Explicit AVX-512 cache toggles do not override the native branch.

An AVX2-targeted rebuild of exact source commit `0c7be696371069e0baa2515ed7aa08d5a5473aa6` succeeded in 61.14 seconds. It disables native compilation and AVX-512/VNNI/AMX, explicitly targeting AVX2/FMA/F16C/BMI2. The verified tiny Q5_1 model then processed the 11.88-second FR/EN pair in 2.994 seconds of CLI-reported total time; host supervision took 3.435 seconds. The fresh JSON artifacts exist, the client exited zero and was reaped, and the exited/no-OOM container was inspected before removal. Aggregate pair time includes different model/runtime/language-detection settings from CT2, so this is a mechanics result, not a fair model/runtime speed ranking.

French had 10/11 normalized word edits on this single clip; English's 1/16 edit is equivalent numeric formatting. Tiny is not accepted as a bilingual starter. The [portable-build investigation](debug-cpp-build/REPORT.md) records build flags, hashes, exact fault capture, failed toolchain attempts and successful receipts. A missing C++ toolchain required an explicitly bounded task-only provisioning step; application images/defaults were not changed. Fresh artifact verification is required in addition to exit status. This is a proven image-portability defect and a demonstrated local repair; architecture labels alone are insufficient CPU qualification.

## Decision and next work

Use the explicit-placement CT2 runner for further local accuracy evaluation and the AVX2 build configuration for whisper.cpp portability work. Do not silently change existing TTS defaults or promote tiny from throughput alone. The next product qualification uses representative French, English and natural mixed speech, work terminology, context bias and false substitutions; streaming, long/noisy media and other hardware remain unqualified. Native-versus-VM isolation is still needed to identify the lowest-level CT2 interaction, even though a working local profile is demonstrated.

STT model/audio assets remain available for this continuing qualification. Build outputs are ignored local artifacts, while source/settings/hashes and terminal receipts remain in the research packet.

Final inventory contains no task containers. The unrelated running-container count changed from four to three during this shared-host investigation: previously listed `167635d74df2` no longer exists. Our commands did not target it; its removal's actor/reason is unknown. We did not recreate or restart it, and do not claim preservation of the entire unrelated runtime inventory.
