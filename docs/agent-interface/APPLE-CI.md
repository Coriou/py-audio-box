# Apple Silicon CI: first dependency slice

This workflow is authored source, not evidence that this revision has executed.
Native dependency feasibility, client protocol closure, tensor probes and installed
synthesis are separate evidence tiers. Apple CPU/Metal synthesis remains
**UNQUALIFIED**; ARM64 runtime is **not_run**.

The automatic `dependencies` job provisions SoX (`brew install sox`,
`sox --version`), installs the existing native binary hash lock and runs pip
check. Its report records native Python/architecture, exact dependency versions,
SoX executable/version, CPU tensor result and MPS built/available flags, without
constructing a model. The report is printed and uploaded even after failure;
missing reports fail visibly. No import or tensor pass qualifies synthesis.

The distinct `clients` job uses native arm64 CPython 3.11 on `macos-15`.
It creates fresh `.venv-client` and `.venv-service` environments, installs
`requirements/client.lock` and `service/requirements.lock` respectively with
`--only-binary=:all: --require-hashes`, and runs pip check in each. It executes
the real CLI help parser, then `tests.clients.test_voice_clients` using the
actual temporary Core/Dispatcher ASGI fixture. Explicitly selected
`tests.test_ci_apple_silicon.MacClientSmoke` starts a loopback fixture HTTP
service and actual MCP stdio child, initializes ClientSession, lists tools,
calls speech_discover and compares MCP/CLI discovery with HTTP. It loads no
engine. The fixture has an OS-assigned port, private HOME/environment, bounded
readiness/RPC waits and owned termination/kill/wait. SDK stdio context owns its
child cleanup. The inert host suite requires no FastAPI/MCP; dependency absence
in the selected real smoke is a failure, never a successful skip.

`apple-ci/clients.json` identifies commit, runner image/OS/architecture, Python,
installed SDK versions and separate passed/failed/not_run stage fields.
Skipped downstream steps do not claim installation or runtime qualification.
An incompatible Mac hash-locked wheel fails the job: retain its exact pip
package/pin diagnostic as the first blocker. A dedicated Mac client closure is
a narrow follow-up, not authorization to relock the dependency graph.

Jobs have 20 minute deadlines; child installs have 300 second deadlines,
fixture readiness 15 seconds and MCP protocol 30 seconds. Only explicit
Darwin-safe tests run, never Linux-wide discovery. Reports are exact JSON paths,
with 7-day retention and missing-file errors; no workspace, credentials, weights
or model cache is uploaded. No Docker, model construction/acquisition, secrets,
paid runner or weight cache is configured. Standard public runner compute is
$0; artifact storage remains subject to repository quotas and retention.
See [GitHub runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

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

## Next finite ARM64 preview slice — not implemented / not_run

Use existing Kokoro CPU through linux/arm64 Docker on free `ubuntu-24.04-arm`,
paired with the current amd64 baseline. Native macOS CPU requires a new adapter
and is outside this package. Full ARM64 synthesis is not required to describe a
preview on already qualified platforms.

1. Create separate `requirements/kokoro-linux-arm64.lock` and
   `service/requirements-linux-arm64.lock` at existing compatible pins using
   uv 0.11.15: `uv pip compile INPUT --constraint EXISTING_LOCK --python-version
   3.11 --python-platform aarch64-manylinux_2_28 --generate-hashes --no-build
   -o NEW_LOCK`. Metadata resolution is not wheel-install or synthesis evidence.
2. Add explicit `--platform` to `scripts/cpu-kokoro-export.py`, defaulting to
   linux/amd64; thread it through probe create, build config, image architecture,
   installation policy and retained receipts. Inspect downstream guards; never
   bypass amd64 guards to accept ARM64 evidence.
3. Add an architecture-specific descriptor. The existing Kokoro build descriptor
   pins an amd64 FFmpeg SHA256 (`0dafc136...`); never reuse it for ARM64. Verify
   exact ARM64 Debian package versions, executable hash and base-image manifest
   membership. ARM64 availability of the base digest remains unverified. Preserve
   amd64 defaults and all resource bounds.
4. Parameterize only control/worker-export release-image lock selection. Preserve
   Pocket/Qwen policy and qualification. Add one bounded pinned-action workflow
   pairing amd64 and ubuntu-24.04-arm with exact model provenance.
5. Future separately authorized runs must exercise installed HTTP/CLI/MCP
   generate/export/download/restart and retain actual WAVs, receipts,
   image/source/model hashes and qualification.json. No secret-dependent Pocket
   or paid Metal lane. Pinned bounded CPU-STT fidelity acquisition/run requires
   separate authorization; absent fidelity cannot pass the synthesis gate.

Stop at the first exact pin/platform blocker and preserve amd64 behavior. No
model or dependency upgrades are implied. Import-only or lock-only green means
dependency feasibility, never Apple CPU runtime qualification.
