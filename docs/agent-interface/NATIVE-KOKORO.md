# Native Apple CPU candidate: Kokoro prerequisites

This is a **no-weight prerequisite check**, not an installed speech capability.
Kokoro provides preset voices, **not cloning**. Qwen CustomVoice presets do not
qualify Qwen Base cloning either; native Base/MLX cloning remains separate work.
Qwen float32 remains experimental (one diagnostic 4 s WAV took about 74 s to
generate); the tested pretrained FP16 candidate failed sampling. Neither is an
installed 30 s request readiness claim.

From the source checkout, on macOS 14 or later in a native Apple arm64
CPython 3.11 shell (the selected NumPy/ONNX wheels require macOS 14; CI tests 15):

```sh
python3.11 -c 'import platform; assert platform.machine() == "arm64", "Use native Python, not Rosetta"'
python3.11 -m venv .venv-kokoro
.venv-kokoro/bin/python -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements/kokoro.lock
.venv-kokoro/bin/python -m pip check
.venv-kokoro/bin/python scripts/ci-native-kokoro.py --output kokoro-preflight
```

Use a new output directory for each check. No model is downloaded. The existing
`requirements/kokoro.lock` is retained byte-for-byte: Kokoro ONNX 0.6.1,
ONNX Runtime 1.31.0, eSpeakNG loader 0.2.4, phonemizer 3.4.0 and their existing
hash-pinned dependencies. Its original resolver target was Linux; actual
binary-only resolution and execution on Apple CI, including pip's selected
wheel URLs/hashes in `kokoro-install.json`, are the Mac closure evidence. No
blind upgrade or claim that a Linux receipt qualifies macOS.

The [official loader 0.2.4 release](https://pypi.org/project/espeakng-loader/0.2.4/)
provides `espeakng_loader-0.2.4-py3-none-macosx_11_0_arm64.whl`, SHA256
`d27cdca31112226e7299d8562e889d3e38a1e48055c9ee381b45d669072ee59f`.
Inspected wheel code uses package-local `libespeak-ng.dylib` and
`espeak-ng-data`; the wheel includes the 1.52.0 dylib and English dictionary.
The preflight validates those paths and loads that dylib before constructing
Kokoro's actual tokenizer with explicit paths. No Homebrew eSpeak/SoX or system
library fallback is provisioned. Missing/wrong dependencies and unusable dylibs
produce actionable errors with their original exception retained.

The [pinned Kokoro release](https://pypi.org/project/kokoro-onnx/0.6.1/) wheel
SHA256 is `50c8de4950d601df41428ee5462a48c8a78bef441bf671f2492e070ef44d8a32`.
Its tokenizer uses the loader and phonemizer; upstream's advertised M1 speed
is not this project's benchmark. The [pinned phonemizer release](https://github.com/bootphon/phonemizer/releases/tag/v3.4.0)
and [ONNX Runtime CPU installation docs](https://onnxruntime.ai/docs/install/#install-onnx-runtime-cpu)
are the primary references. The check creates a tiny, weight-free ONNX Add graph
(IR8/opset13, encoded according to the [official ONNX schema](https://github.com/onnx/onnx/blob/v1.17.0/onnx/onnx.proto))
and asserts `CPUExecutionProvider` only, 2 intra/1 inter threads, arena disabled,
then phonemizes/tokenizes real English through the pinned tokenizer. This checks
native binaries and phonemization, not the Kokoro graph, voices, audio, speed,
quality, or backend synthesis. The permanent product spawn guard is installed
before dependency imports. The existing owner enforces 60 s elapsed, sampled
6 GiB RSS, 0.5 s freshness, bounded logs and independent cleanup/reaping.

`.github/workflows/apple-kokoro.yml` is path-scoped and manually dispatchable;
it installs no weights, uses no model cache or secrets, and uploads compact
preflight/owner/install receipts. The native source archive includes the new
preflight module and this document; its existing `native-qwen-v1` admission is
unchanged. Dependency locks remain external installation inputs, as before.
AGPL-3.0-only project source remains separate from Kokoro adapter MIT,
Kokoro model Apache-2.0, phonemizer GPL-3.0 and eSpeak NG GPL-3.0-or-later;
see [upstream eSpeak licensing](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING).
No dependency binary or model is added to the source archive.

## Next bounded experiment (requires separate approval)

One public pinned model/voices closure from `requirements/tts/kokoro-fr.expected.json`
(353,746,785 bytes), explicit English preset checked against actual available
voices, `Hello. The test is complete.`, existing KokoroBackend/CPU provider,
2 threads. Proposed acquire120/load60/generate60/total180 s, 6 GiB sampled RSS,
existing logs/reap guards and <=12 minute free standard runner. No retries,
French, cloning or persistent architecture. Validate a playable nonempty WAV,
hash/duration/latency/RTF/resources and retain it for human ending assessment.
No model acquisition or synthesis is authorized by this prerequisite slice.

After backend proof, the smallest public-control-plane seam is a distinct typed
native CPU execution/context and child dispatch, reusing the existing immutable
Kokoro engine contract and bounded supervisor cancel/reap lifecycle. It must
advertise preset-only CPU capability, preserve separate load/request deadlines,
and test API/CLI/MCP download/cancel/restart. Do not widen `native-qwen-v1` or
reuse Linux PID1/cgroup containment receipts as Mac evidence. No CPU adapter,
installer activation, default hardware/fallback policy or clone promise is
implemented here. Combined independent product review remains required before
integration.
