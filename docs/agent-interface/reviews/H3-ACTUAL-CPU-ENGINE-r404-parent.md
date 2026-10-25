# r404 actual CPU engine smoke

2026-10-25. **Actual engine smoke PASS after source correction; installed service and release remain unqualified.** Source correction commit `a534296`.

Built control `sha256:22d0156f8ccb4706e856c1685ebf20b3674eac4745178d59d3faaf20717c9071` and original export worker `sha256:e6123d99feac1c86a1ac4f41aec50a2d664bc9da5e9b8b5efdfbf652e4a26608` from the pinned Python base and hash-locked requirements. Fixed worker is `sha256:4a78552f515a53d71abad6340e2b0a2018a79cfed61b114f9a0f936eaacc44e6`. These are local build identities, not installed admission receipts.

Resolved Debian packages: ffmpeg `7:5.1.9-0+deb12u1`, libsndfile1 `1.2.0-1+deb12u1`, espeak-ng `1.51+dfsg-10+deb12u2`. Extracted FFmpeg executable SHA256 `0dafc1360bb07743f76abeb1e4ae16b0aaa1331e9041a0e8d6a8524111dc0cbb`; recipe verified it during build. Full converter runtime closure review remains outstanding.

Downloaded the existing pinned upstream Kokoro model (325,532,387 bytes) and voices (28,214,398 bytes), verified both existing SHA256 values. Canonical manifest SHA256 `b6cc7358a42da68b4cf599249eb2c3fafd280b20f2372146de1067f3ac01c0f2`. Source origins: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 ; runtime license https://github.com/thewh1teagle/kokoro-onnx ; model card https://huggingface.co/hexgrad/Kokoro-82M . Local qualification assets only; no distributed model bundle or installed profile promotion.

## Actual failure and correction

Original engine failed because phonemizer 3.4.0 copies eSpeak into temporary storage and dlopens it; `/tmp` is noexec. The product daemon uses the same noexec constraint. Preserved that mount protection. Linux-only adapter pins the expected phonemizer API, copies each library into a distinct bounded sealed memfd, and retains independent library state; finalization terminates/unloads the library and closes its descriptor. Mac behavior unchanged. No fallback provider or synthesized readiness receipt.

The initial preparation also failed because Docker refuses cp into an already-read-only rootfs. The final engine probe uses a derivative image containing the exact model bytes and probe script. It is explicitly **not** the named-volume installed topology.

## Fixed source measured result

Container `pab-cpu-export-r404-engine-fixed`: Linux/amd64 OrbStack, 2 CPU quota, 2 GiB memory limit, 128 PID limit, no network, read-only root, all capabilities dropped, no-new-privileges, 128 MiB noexec tmpfs. Separate fresh named result volume; no source bind mounts. Engine process runs root in this diagnostic, so dropped-UID installed containment is not proved.

- Exit 0, two live eSpeak wrapper handles verified distinct.
- Two actual French `ff_siwis` generations, 74,240 frames each at 24 kHz mono (~3.093 s).
- Cold model load 0.745 s; generation 2.234 s then 1.832 s. Single short sample, not general performance qualification.
- Both speech WAV SHA256 `924f8050f22e49084a3611eb4f91b6f97cf8118265ecd4e955bbf5fe0ecb8c41`.
- Peak process RSS 480,376 KiB (~469.1 MiB), not total container peak.
- Actual FFmpeg PCM24 export 222,822 bytes; product RIFF parser verified frames/rate/channels/subtype; SHA256 `0597d73aa2f3c7c7aa59e6dd8a3528739c3341173655b6ffa7e84f8a168ca56b`.
- Actual locked runtime: kokoro-onnx 0.6.1, onnxruntime 1.31.0, espeakng-loader 0.2.4, phonemizer 3.4.0.

Evidence and audio retained under `docs/agent-interface/evaluation/cpu-export-r404/`, especially `fixed-results/engine-smoke.json`, `fixed-results/speech-0.wav`, `fixed-results/export.wav`, fixed build logs and state/image JSON. Large model binaries are deliberately not staged in Git. No human listening assessment claimed.

Focused Kokoro tests: **8 passed**. A broader optional-backend run had 18 passed and 6 failures due to absent host torch/PyYAML in Pocket tests; no broad-suite green claim. `git diff --check` passed.

Remaining: independent correction review; actual converter complete closure; measured installed profile/configuration; fresh named-volume install and API/CLI/MCP generation/export/download; dropped-UID/process containment; cancellation, stop/reopen/replay. CUDA/Metal and Docker Desktop remain UNRUN. No release, push, deployment or cleanup of prior retained resources. Host free disk ended around 6.3 GiB, so remeasure and budget before another substantial allocation.
