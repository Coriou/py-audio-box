# Apple Silicon CI: first dependency slice

Implemented against `25d92e393b0f5d6e9038c51a9c89e38d96fd0678`.
Local validation is source-only. No GitHub workflow was triggered, no host ML
was imported, and no model weights were downloaded. Apple Silicon / Metal
synthesis is **UNQUALIFIED**. The existing unit workflow and Dockerfile are
outside this slice.

The automatic push/PR job installs the native wheel closure with pip
`--require-hashes --only-binary=:all:`, runs `pip check`, then imports real
Torch, torchaudio, Transformers, tokenizers and Qwen. It checks Darwin/arm64,
CPython 3.11, exact direct versions, the Qwen loader symbol, a CPU tensor
operation, and real MPS built/available flags. It never constructs a model.
Only `tests.test_ci_apple_silicon` and `tests.worker.test_native_protocol` run;
the latter uses inert TLS fixtures. No Linux-wide unittest discovery occurs.
Imports and CPU checks do not establish synthesis compatibility.

[GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
lists standard `macos-15` as M1 arm64, 3 CPUs, 7 GB RAM, 14 GB SSD, free for
public repositories. Private repositories consume their minutes allowance.
No large/paid runner is selected. Each job has a 20 minute deadline.

## Lock provenance

Generated with existing `uv 0.11.15`, targeting `aarch64-apple-darwin`, Python
3.11, binary-only, hashes enabled. No environment installation or builds.
The 91-package closure fixes qwen-tts 0.1.1, torch 2.10.0 (without `+cpu`),
torchaudio 2.10.0, transformers 4.57.3 and tokenizers 0.22.2. Transitive
versions are fixed in the lock; resolution does not prove imports will pass.

An ephemeral local metadata gateway forwarded public PyPI Simple/PEP 658
metadata, preserving advertised SHA256 values, exposing only universal,
macOS ARM64/universal2 wheel links, and refusing wheel payload requests.
Each resolution attempt had a cumulative 10 MiB network-body cap. An initial
uncompressed attempt hit the cap and produced no lock; the successful
compressed attempt transferred 8,512,514 bytes (91 packages, 14.30 seconds).
A preceding URL-encoding attempt failed resolution after 2,572,197 bytes.
These are per-attempt bounds, not a claim of less than 10 MiB total traffic.
No large wheel payload was acquired. The lock header records the equivalent
upstream PyPI compile command; the temporary gateway is not an install index.

Independent bounded PyPI version-JSON checks confirmed these lock hashes:

| Native artifact | SHA256 |
| --- | --- |
| torch-2.10.0-2-cp311-none-macosx_11_0_arm64.whl | `418997cb02d0a0f1497cf6a09f63166f9f5df9f3e16c8a716ab76a72127c714f` |
| torch-2.10.0-cp311-none-macosx_11_0_arm64.whl | `b7bd80f3477b830dd166c707c5b0b82a898e7b16f59a7d9d42778dd058272e8b` |
| torchaudio-2.10.0-cp311-cp311-macosx_11_0_arm64.whl | `bcab0e39eb18da84cba1a0c87f600abb6ce97c882200cb46e841caea106f037f` |
| qwen_tts-0.1.1-py3-none-any.whl | `11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d` |

Sources: [Torch](https://pypi.org/pypi/torch/2.10.0/json),
[torchaudio](https://pypi.org/pypi/torchaudio/2.10.0/json),
[Qwen](https://pypi.org/pypi/qwen-tts/0.1.1/json).
Hash lists can contain multiple compatible wheel builds/ABI versions;
pip selects a wheel compatible with the actual CPython 3.11 runner.

Official action tag refs were checked with `git ls-remote` against the action
repositories: checkout v4.3.1 `34e114876b0b11c390a56381ad16ebd13914f8d5`,
setup-python v5.6.0 `a26af69be951a213d495a4c3e4e4022e16d87065`, and
upload-artifact v4.6.2 `ea165f8d65b6e75b540449e92b4886f43607fa02`.
[Checkout release](https://github.com/actions/checkout/releases/tag/v4.3.1),
[Python release](https://github.com/actions/setup-python/releases/tag/v5.6.0),
[artifact release](https://github.com/actions/upload-artifact/releases/tag/v4.6.2).

## Manual tensor probe and remaining gate

After the parent publishes this workflow to the default branch, GitHub makes
`workflow_dispatch` available. No dispatch has occurred here. Select
`mps_probe=true` to run the separate job after dependency checks succeed.
It starts a fresh child with fallback disabled before Torch import, retains
PyTorch's conservative default memory watermark, checks MPS availability,
allocates two values explicitly on MPS, performs arithmetic, synchronizes,
checks the result's MPS device and copied values, and fails without fallback.
The parent enforces 30 seconds, kills the process group on expiry, and waits
for the exact child. The workflow deadline additionally bounds the job.

Only `apple-ci/mps-probe.json` is uploaded, with a 10 MiB serialization bound
and 7 day retention. There is no workspace/directory artifact upload, no
r409 private clones, audio, weights, `.env`, or Hugging Face cache publication.
No secrets, `pull_request_target`, persistent checkout credentials, caches,
model downloads or synthetic success substitutions are configured.

A successful tensor probe establishes only that small MPS tensor operation.
Full Qwen synthesis remains deferred until a public reference fixture and a
measured fit within 7 GB are known. The next finite gate must use the product
native child with a 30 second synthesis limit, exact checkpoint/revision and
public input provenance, actual MPS device evidence, bounded output and
resource measurements, and verified termination/reap. It must not use
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` or CPU fallback to obtain a pass.

Local checks: `python3 -m unittest tests.test_ci_apple_silicon
 tests.worker.test_native_protocol -v`, `actionlint
.github/workflows/apple-silicon.yml`, and `git diff --check`. No actual native
dependency installation/import, hosted CI run, synthesis, or Apple hardware
qualification is claimed by these source checks.
