# Apple Silicon: evidence gates and native diagnostics

Qwen native MPS speech remains **experimental and unqualified for installed use**.
Its owner supports MPS/float32/SDPA only, without a CPU fallback. The separate
[Kokoro native CPU preview](NATIVE-KOKORO.md) has a recorded ordinary installed
EN preset journey on one hosted M1 running macOS 15.7.9. Docker
on macOS is a separate Linux CPU topology; amd64 images also require emulation.
Neither an import pass nor an MPS tensor pass qualifies installed speech.

## Executable prerequisite check

On a native Apple Silicon terminal, use CPython 3.11 (not Rosetta). These commands
install the pinned Python dependencies and check prerequisites; they do **not**
install or start the speech service, download weights, or create credentials.
The pinned 12 Hz native route does not use the SoX executable; no Homebrew
installation is required by these checks. Existing SoX is observed optionally.

```sh
uname -m                         # must print arm64
python3.11 -c 'import platform; assert platform.machine() == "arm64"'
python3.11 -m venv .venv-apple
.venv-apple/bin/python -m pip install --require-hashes --only-binary=:all: \
  --index-url https://pypi.org/simple -r requirements/qwen-macos-arm64.lock
.venv-apple/bin/python -m pip check
.venv-apple/bin/python scripts/ci-apple-silicon.py deps-check --report apple-ci/dependencies.json
.venv-apple/bin/python scripts/ci-apple-silicon.py mps-probe --report apple-ci/mps-probe.json
```

A wrong OS/architecture or Python version is refused. Missing optional SoX is
recorded as absent. A failed MPS operation
fails the check; it never silently switches to CPU. Preserve the JSON report.
See [native package lifecycle](NATIVE-PACKAGE.md) for the existing explicit
operator-managed source package. There is no qualified native service quickstart.

## Automatic free hosted checks

`.github/workflows/apple-silicon.yml` uses public-repository standard `macos-15`
arm64 runners, pinned action commits, read-only repository permissions and no
secrets. Dependencies and clients are separate jobs, each limited to 20 minutes.
Path filters limit this workflow. Normal Linux Unit push and PR coverage remains
enabled, including this development branch.

The dependencies job installs `requirements/qwen-macos-arm64.lock` with hashes
and binary-only wheels, runs pip check, real imports and a CPU operation, then
executes a real MPS arithmetic operation with fallback disabled before Torch
import. The MPS child has a 30-second deadline and exact wait/reap ownership.
The conservative default MPS watermark is retained. Reports identify source,
runner image/version, physical RAM, free disk, Python/dependency versions,
selected tensor device, MPS allocation/working-set values and peak child RSS.
Physical RAM and free disk are capacity observations, not available-RAM admission.

The clients job creates fresh CLI/service virtual environments from their hash
locks. It exercises actual CLI help, the temporary HTTP/Core/Dispatcher fixture
and MCP stdio initialization, tool listing and discovery parity. It loads no
engine and therefore does not establish real public speech generation.

Reports are explicit JSON artifact paths with seven-day retention. No workspace,
HOME, `.env`, credential, model tree or hub cache is uploaded. The native Qwen
lock retains torch/torchaudio 2.10.0, qwen-tts 0.1.1, transformers 4.57.3 and
tokenizers 0.22.2; newer upstream releases are not implicitly adopted.
The native lock uses the narrowly patched Qwen wheel described in
[native dependency provenance](NATIVE-DEPENDENCY.md). On every relevant Apple
push/PR run, a separate fresh interpreter exercises real locked 12 Hz imports under the
permanent native spawn guard without weights; passing that gate is not synthesis.

## Opt-in cold native inference prerequisite

Before merge, push **this exact branch** with `[apple-speech]` in the final commit
message to run the finite `native-speech` job. Ordinary pushes do not download
weights. Manual dispatch with `qwen_speech=true` accepts only
`refs/heads/feat/apple-silicon-native` or `refs/heads/main`; arbitrary branch/tag
refs are refused by the job condition. The same allowlist covers the manual
tensor probe. Once registered by a push, this workflow can be dispatched by
numeric workflow ID through the GitHub API/CLI before merge; the UI may require
the file on the default branch. See
[GitHub dispatch documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch).
The older manual tensor-only input remains
available, although automatic dependency checks now already execute that probe.
Do not rerun an expensive failure automatically.

The speech job needs successful dependency and client jobs. Guarded imports run
unconditionally in the dependency job and again in the fresh speech environment
before any model acquisition. For a deliberate qualification attempt, first
require successful no-weight Apple CI and normal Unit CI at the exact source
SHA, then dispatch the trusted ref and verify its resolved SHA has not moved.

The job installs the same lock and downloads only the public, ungated
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` snapshot at
`85e237c12c027371202489a0ec509ded67b5e4b5`. Its descriptor pins every file's
size and SHA256 (2,498,383,610 bytes total). No token, gated acceptance, persistent
weight cache or paid runner is involved. Acquisition checks at least 10 GiB free
disk and enforces a 180-second between-read deadline plus 20-second socket timeout.
The entire job has a 12-minute deadline.

`scripts/ci-apple-speech.py` calls the product `native_child.infer` verifier,
spawn guard, complete local loader, native backend and PCM encoder in an owned
subprocess. It tries one short English preset, then French only after English
succeeds. Each child gets a 180-second total bound: input verification up to 30 seconds,
cold model loading up to 120 seconds, short generation up to 30 seconds, and
encoding up to 15 seconds, all within the same total. An observational subclass
calls the actual native backend unchanged and persists phase timings plus the
upstream selected device. This direct probe does not relax the installed
supervisor's existing 30-second total deadline. It retains a sampled 6 GiB RSS ceiling. RSS sampling is an early-stop guard, **not** an OS memory
reservation or a bound on all Metal allocations. The default MPS allocator limit
remains enabled. Failure stops the sequence; there is no OOM retry or fallback.
The parent kills and reaps its child and removes only its private model tree.
A timeout establishes only a budget overrun, not hardware incompatibility.
Phase timestamps assume Darwin CPython 3.11's system-wide monotonic clock,
which is comparable across parent and child processes (since Python 3.10 on
macOS); they are not portable persisted timestamps. See
[Python's monotonic clock documentation](https://docs.python.org/3.11/library/time.html#time.monotonic).

Only small reports/logs and successful short WAV files are uploaded. Successful
WAV reports contain format/duration/hash, wall time, RTF and peak process RSS;
nonempty nonsilent PCM alone does not establish speech fidelity or the expected
ending. There is no paid STT. This is a direct native inference prerequisite,
**not** an installed HTTP/CLI/MCP synthesis test. It grants no native-ready claim.
CPU comparison is explicitly not run because the installed native owner requires
MPS. Public install/enrollment, synthesis/download/restart/cancel, cloning,
alias/default conventions, cache reuse and cold installer qualification remain
separate required gates. Do not relax trust, containment, memory or deadline
checks to pass them.

## Why this order

[GitHub's official runner table](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
lists the standard public arm64 macOS runner as M1, three CPUs and 7 GB RAM;
standard public runner compute is free. Artifact storage remains quota-dependent.
This hardware benchmark is not a guarantee for everyday Macs.

[PyTorch's MPS guide](https://docs.pytorch.org/docs/stable/notes/mps.html) requires
moving tensors/modules to the device; availability flags alone are insufficient.
[Its allocator documentation](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
explains CPU fallback and warns that disabling the high watermark can cause
system failure. This workflow disables fallback and preserves the watermark.

[Qwen's official implementation](https://github.com/QwenLM/Qwen3-TTS) demonstrates
explicit device/dtype/attention selection and reusable clone prompts. The
[CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
identifies Apache-2.0 weights, English/French support and Ryan as an English
speaker. Upstream CUDA examples do not prove Metal compatibility. The pinned
0.6B CustomVoice BPE layout is admitted explicitly; arbitrary checkpoint aliases,
remote code, unlisted config paths and malformed vocabulary remain refused.

[Kokoro upstream](https://github.com/hexgrad/kokoro) describes Apache-licensed
weights. The repository uses the separate
[kokoro-onnx implementation](https://github.com/thewh1teagle/kokoro-onnx), fixed to
CPUExecutionProvider, so a Torch MPS flag does not accelerate that adapter.
[Pocket upstream](https://github.com/kyutai-labs/pocket-tts) targets CPU and does
not officially support a GPU device argument; its current language/features
must not be projected onto this repository's older pinned model/package.
Pocket gated assets require users' own authorized terms acceptance.

Application source remains **AGPL-3.0-only**. Dependency and model licenses remain
separate; an upstream MIT/Apache license does not relicense this application.


## RSS sampler and diagnostic failure boundaries

The harness reads the owned PID in process: Darwin uses
[`proc_pid_rusage`](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c)
with Apple's 96-byte
[`rusage_info_v0`](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h).
`ri_resident_size` is bytes; `ri_proc_start_abstime` binds process identity and
`ri_proc_exit_abstime` detects exit. Linux uses
[`/proc/PID/stat`](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)
starttime and resident pages multiplied by the OS page size. Linux never loads
Darwin symbols. The parent retains PID ownership until first identity binding;
subsequent identity changes, permissions and malformed reads fail closed.

One reader thread samples every 0.1 seconds; it is never replaced on failure.
The owner independently enforces elapsed/phase deadlines, the existing 6 GiB
sampled RSS limit, bounded output and kill/wait cleanup. No fresh observation
within 0.5 seconds is a sampler failure. A blocked reader cannot block cleanup;
its stop status is retained. Reports distinguish sampler errors, last valid
sample and age, sampled peak, owner failure, requested termination/signals,
observed exit and reaping. A sampled peak is not an RSS bound, and lost sampling
never proves memory safety. RSS does not measure all Metal/unified allocations.
Python stdout is unbuffered; progress JSON is capped at 64 KiB. Phase timestamps
assume parent and child share the same system-wide monotonic clock (Darwin
CPython 3.11); they are not portable across hosts or clock namespaces.

The previous cold-load `ps` 0.5-second timeout was a sampler transport failure,
not evidence of Qwen incompatibility, Metal failure or OOM. Its original receipt
is retained separately; this source repair does not qualify inference.

### Proposed cache separation (not implemented)

Each current probe downloads into a new private tree and removes it during
cleanup, so a harness failure discards roughly 2.5 GB of verified weights.
A future diagnostic lane could cache only permitted public weights under an
exact checkpoint/revision/file-path/size/SHA256/policy key, without restore-key
fallbacks, tokens, user configuration or audio. Revalidate all files and reject
symlinks/traversal; atomically promote only a complete verified immutable tree.
Record cache hit/miss and validation separately from process cold-load timing;
OS page-cache state remains uncontrolled. Respect cache quotas and model terms.
Keep the final installer acceptance lane explicitly cache-disabled with a fresh
root. Diagnostic cache reuse would prove neither cold installation nor installed
product reuse. No cache or model acquisition is added by this sampler patch.


The diagnostic wrapper now records bounded host talker-forward counts and
first/last timestamps plus tokenizer-decode boundaries. These observe the pinned
model without changing generation arguments, tensor values or synchronization.
A completed host forward is not a first-token/audio measurement. Summaries update
at most once per second plus decode/cleanup boundaries and preserve the original
phase start. Decode/forward errors retain their original exceptions. No new
weights run or changed diagnostic deadline is implied by this instrumentation.


## Explicit EN completeness measurement

The opt-in speech workflow selects `--diagnostic-en-completeness`: English only,
Ryan, the same short text/sampling/float32/SDPA/two-thread options and 256-token
ceiling. French is reported as `not_requested`. This is a budget/completeness
measurement, not an optimization or installed qualification.

The harness selects verification/load/generation/encoding caps of
30/120/120/15 seconds under a 270-second total child deadline. It passes a typed
120-second execution and an explicit private diagnostic selector through
`native_child.infer` to `NativeQwenBackend`. The backend validates that selected
limit; the owner enforces it. Reports retain effective phase/total/execution
budgets and the phase or total deadline exceeded. The normal installed socket
and registration still reject execution timeouts over 30 seconds; their
120-second startup/150-second original-job ceiling is unchanged. The selector
is absent from the closed installed socket configuration.

The completeness mode retains 6 GiB sampled RSS, 0.5-second observation staleness,
default Metal watermark, fallback disabled, bounded logs and kill/reap cleanup.
It downloads only the existing pinned permitted public checkpoint. No cache or
retry is introduced. A valid WAV is distinct from verified ending/quality; the
report keeps `ending_verified: false` until separately checked. Running this
opt-in mode requires a separate deliberate runtime authorization.


## FP16 candidate: numerical gate, not full-model qualification

`--diagnostic-precision float16` requires the explicit EN completeness mode.
Only the private diagnostic child/backend receive this selector; installed
registration/socket admission remains MPS/float32/SDPA. The committed checkpoint
manifest is unchanged; the owned diagnostic tree gets a copied execution tuple
for FP16. Weight paths, sizes, revisions and hashes are unchanged. There is no
automatic dtype fallback, BF16 admission or watermark override. The actual
selected model dtype is checked and recorded before generation.

Before any model download, cheap Apple CI runs the pinned 12Hz talker and code
predictor on MPS in FP32 and FP16 using tiny, random parameters (one layer,
64-wide hidden state, original codec vocabulary and 16 code groups). It checks
finite raw activations/logits, actual dtype/device, processed probabilities and
sampled-token validity, with a 90-second owned deadline and the existing sampled
RSS/log/kill/reap bounds. No pretrained weights or tokenizer decoder are loaded.
Passing does **not** prove trained-model numerical safety, audio quality,
performance, expected ending, or installed FP16 support.

[PyTorch 2.10 supports Half on MPS](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/mps/OperationUtils.h#L564).
[Qwen PR355](https://github.com/QwenLM/Qwen3-TTS/pull/355) is an unmerged proposal
reporting FP16 overflow on CUDA, not an endorsed MPS fix.
[Our pinned Transformers 4.57.3 sampler](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/utils.py#L2579)
already promotes returned logits to FP32 before sampling. This inherited cast is
reported; no new upstream/model cast is added. It cannot repair earlier
projection overflow. The gate separately records a representational overflow
sentinel and verifies rejection of injected nonfinite data before sampling.
No new Qwen wheel, lock upgrade or logits patch is introduced.
