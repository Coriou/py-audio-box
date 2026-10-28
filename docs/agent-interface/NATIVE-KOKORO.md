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

## Explicit opt-in backend experiment

`scripts/ci-native-kokoro-speech.py --run-en-backend --output kokoro-speech`
is a separate diagnostic backend harness, never installed admission. The
`en_backend` manual workflow input defaults false and permits only the development
branch or main. The actual speech environment reruns the no-weight guarded
preflight before acquisition. Ordinary push/PR jobs acquire no models.

The English descriptor `requirements/tts/kokoro-en-macos.expected.json` is exactly
the existing French descriptor with language `en`; its own SHA256 binds every
origin, size, revision and file hash. Both source archives carry that descriptor.
The two public files total **353,746,785 bytes**. Only GitHub and its
`release-assets.githubusercontent.com` redirect host are allowed; there is no
authentication, hub/model cache, retry or asset substitution. Partial files remain
inside the owned temporary tree and are removed after the first failure.

The existing Kokoro engine verifies the immutable English manifest and files,
then invokes the existing backend and PCM16 encoder. The only preset is
`af_heart`: absence fails, with no substitute. Text is `Hello. The test is complete.`;
CPUExecutionProvider only, 2 threads, FP32 floating inputs and audio, and the
pinned upstream default speed is asserted to be 1.0. No reference, instructions,
seed or clone mode is accepted. A diagnostic subclass observes phases and closes
through the existing engine; it does not alter synthesis options.

Acquisition has an independent owner cap of 120 s; load (including engine and
backend verification) 60 s; generation 60 s; encoding 15 s. A single 180 s cutoff
from acquisition start covers acquisition plus the backend child, so phase caps
cannot be added to extend total time. The existing independent owner gains only
a closed Kokoro phase profile; Qwen default30/completeness120 generation and
other limits remain unchanged. The same 6 GiB sampled RSS/0.5 s freshness/log
bounds/reaping apply; cleanup evidence is reported separately. The hosted job
is capped at 12 minutes. The child and parent use the same system-wide monotonic
clock on Darwin CPython 3.11, as the existing owner does.

Reports retain descriptor identity, options, device/provider/dtype, load and
generation host elapsed times, backend-only total, acquisition+backend total,
WAV hash/duration/RTF and owner RSS/exit/reap receipts. The engine rejects
nonfinite/out-of-range audio before encoding; the harness also checks PCM format,
size, nonempty/nonzero content and agreement with engine hash/frame receipts.
Human ending and quality remain unverified. No Mac speech capability is claimed
from a preflight pass or a future raw backend WAV alone. A model dispatch requires
separate explicit authorization and final exact-source gates; none follows from
running the prerequisite job.

After backend proof, the smallest public-control-plane seam is a distinct typed
native CPU execution/context and child dispatch, reusing the existing immutable
Kokoro engine contract and bounded supervisor cancel/reap lifecycle. It must
advertise preset-only CPU capability, preserve separate load/request deadlines,
and test API/CLI/MCP download/cancel/restart. Do not widen `native-qwen-v1` or
reuse Linux PID1/cgroup containment receipts as Mac evidence. No CPU adapter,
installer activation, default hardware/fallback policy or clone promise is
qualified here. Source now has a separate `native-kokoro-cpu-v1` protocol,
`native-kokoro-execution-v1` descriptor, receipt/journal family, guarded child,
CPU-only metadata provider and typed catalogue/control transport. It reuses the
existing Kokoro engine and native owner lifecycle. Source tests cover bounded
startup/request handshakes, CPU proposals, cancellation/reaping, completed journal
reopen and saved preset identity. The CPU source seam is not yet connected to an
installer/activation factory; no installed public journey or CPU default is
advertised from those tests. Combined independent product review remains required
before integration.

CPU product budgets are separate: 30 s startup, 30 s generation and one fixed
60 s job, compared with the diagnostic's observed 1.71 s load/1.66 s generation.
This is explicit headroom for the short EN preset candidate, not a latency
guarantee or benchmark for arbitrary text. Each job loads once; no persistent
model reuse is claimed. The owner observes a 2 GiB sampled RSS ceiling, 0.5 s
freshness and fixed cancellation/reaping bounds. Sampling never proves a
continuous memory cap. Darwin admission observes Mach free+inactive pages through
[Apple's host info ABI](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/host_info.h)
and [VM statistics ABI](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h);
this estimate is not guaranteed allocatable RAM. Observation errors supply zero
admission capacity. No Linux cgroup/PID1 or Qwen MPS receipt is accepted as CPU
native evidence.
