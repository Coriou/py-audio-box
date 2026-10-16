# Portable whisper.cpp repair — 2026-10-15

The immediate whisper.cpp failure was a **binary compatibility defect**, demonstrated by capturing its faulting instruction. A portable rebuild now completes a small CPU transcription probe. This result does not resolve the separate CTranslate2 performance issue or qualify French accuracy.

## Confirmed cause and repair

The pinned original image is `ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647`. Its embedded source commit is `0c7be696371069e0baa2515ed7aa08d5a5473aa6`.

Its saved CMake cache enables `GGML_NATIVE=ON`; actual CPU compilation uses `-march=native`. Although its individual AVX512 cache options say OFF, native compilation overrides that branch. The Intel i7-8700B guest exposes AVX2/FMA/F16C but no AVX512.

The bounded SIGILL capture reproduced exit 132 in 0.56 seconds. Signal 4, `si_code=2`, occurred in `libggml-cpu.so.0` at offset `0x21f9d`, bytes `62 f2 fd 08 7c c1`. Disassembly resolves the unsupported EVEX instruction `vpbroadcastq %rcx,%xmm0` in `ggml_backend_cpu_device_get_extra_buffers_type`. See [capture](sigill-run/stderr.log) and [disassembly/build evidence](original-image-static-evidence.txt).

The rebuilt same-source CLI uses `GGML_NATIVE=OFF`, AVX/AVX2/FMA/F16C ON, and AVX512/VNNI/AMX OFF. Actual CPU flags include `-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2`; tests/server were excluded. The network-disabled, four-CPU/6-GiB build finished in **61.14 seconds**, below its 300-second parent deadline. Clock-skew warnings remain in its log; the target completed and actual artifacts were hashed. See [flags](portable-v1/ggml-cpu-flags.make), [cache](portable-v1/CMakeCache.txt), and [hashes](portable-v1/artifact-hashes.json).

## Preserved setup failures and limits

The original diagnostic's `--no-context` was rejected while the CLI returned zero. Successful exit alone therefore does not validate a run. The corrected `--max-context 0` original-image attempt exposed the SIGILL.

The first rebuild failed immediately because the runtime image had GCC/CMake but no C++ compiler. The first ephemeral provisioning attempt fetched Ubuntu indexes, then failed on a script quoting error before installation. Corrected provisioning installed only g++ and its two dependencies in a task-owned derived image. Across both index fetches and packages, apt reported approximately **111.5 MB**, below the authorized 150-MB budget. The package plan contains exactly 13,526,736 bytes. An initially incorrect awk column reported zero; [independent budget recheck](toolchain-v2/budget-recheck.json) preserves and explains that defect. Future source uses the correct column; no further provisioning was run. Provisioning had a 150-second parent limit; its successful attempt took 16.99 seconds. Failed attempts and logs remain intact.

## Working probe and quality boundary

One fresh run reused the verified tiny Q5_1 model and hash-verified FR-1679/EN-1914 clips, auto language detection, four threads, beam five, no GPU, and zero retained context. The run used an 85-second inner timeout and 90-second parent deadline. It completed both clips: **11.88 seconds audio**, **2.994 seconds CLI aggregate time including initialization**, **3.435 seconds host-supervised wall time**. These are pair aggregates, not per-clip timings. Exit 0, OOMKilled=false; terminal inspection and removal receipts exist.

French normalized word edits were **10/11**; English **1/16**, the latter a number-format difference. This proves working CPU mechanics and useful execution speed for this tiny probe. It is not an accuracy ranking, matched CT2 comparison, or approval of tiny French quality. See [quality and exact text](pair-v1/quality-summary.json).

## Reproduction and cleanup

Run helpers from the repository root with Python 3: `build-portable.py`, then `run-pair.py`, in this directory. They intentionally refuse existing output directories; retain receipts and choose new owned paths/names for an intentional repeat. `build-command.sh` inside `portable-v1/` records the exact CMake commands. The task-owned local toolchain image is `py-audio-box:stt-cpp-avx2-toolchain-20261015`; no new download is needed while it remains available.

All containers launched for this repair reached terminal state and were removed. Unrelated containers/services and application defaults were untouched. Compiled build trees and the SIGILL helper binary are excluded by this directory's `.gitignore`.
