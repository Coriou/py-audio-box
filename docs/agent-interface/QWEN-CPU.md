# Qwen CPU LOT1 — source only

This is an unbuilt, unqualified Linux amd64 / CPython 3.11 runtime recipe.
It does not make the full headless CPU or Apple release journey available.
Human listening confirmed incorrect French numbers and missing liaisons. There
is no quality approval; earlier release planning does not supersede that feedback.
Apple Metal remains separately unqualified.

`requirements/qwen-cpu.in` declares Qwen TTS 0.1.1, CPU Torch and Torchaudio
2.10.0+cpu, Transformers 4.57.3 and Tokenizers 0.22.2. The CPU lock retains the
86-distribution r409 closure, including the Torch artifact previously reused by
that experiment. Every artifact hash and version also matches `poetry.lock`.
SoX 1.5.0 is the sole source artifact. Its exact source hash stays in the
installed lock; the recipe separates that entry from the wheel-only remainder.
`qwen-sox-build.lock` pins r409's Setuptools 82.0.0 and NumPy 1.26.4 wheel
prerequisites. SoX builds without dependency resolution or build isolation;
Setuptools supplies bdist_wheel. No other sdist is permitted.
The installed lock uses the Pocket composition rule: runtime pins win overlaps,
and service-only pinned, hashed packages are added. This composition has not
been installed or checked for runtime compatibility. The control image keeps
its existing independent service lock.

The `worker-qwen` stage lives in the existing release Dockerfile. It installs
its own runtime into the image, with no Pocket venv or external temporary mount.
It embeds the versions, public Debian artifact URLs, byte counts and SHA-256
hashes of r409's 16-package SoX closure. The pinned Bookworm base supplies the
remaining base dependencies. Downloads are size bounded and hash checked before
`dpkg -i`; no apt resolver or unversioned package acquisition is used. Artifact
unavailability or a mismatch fails the build. The recipe is amd64 only. The
retained private system-stage footprint was about 52 MiB; this is not a build
space estimate. No binaries or private experiment paths are distributed.

Inference is offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). Model
provisioning must remain a separate, explicit operator action. The public,
read-only Base 0.6B descriptor pins revision
`5d83992436eae1d760afd27aff78a71d676296fc`, eleven files and 2,516,100,892 bytes.
Its hashes come from r409 acquisition metadata, including the nested speech
tokenizer weights. The acquired small config declares `model_type=qwen3_tts`,
`tts_model_type=base`, `tts_model_size=0b6`,
`tokenizer_type=qwen3_tts_tokenizer_12hz` and architecture
`Qwen3TTSForConditionalGeneration`. Runtime version declarations match the CPU
lock. CPU float32/SDPA and English/French are operator declarations, not hardware
or pronunciation qualification. Base has no preset speaker list; reference
voice preparation and retained provenance are still required.

The release Docker context now explicitly includes `qwen-installed.lock`,
`qwen-sox-build.lock` and this exact expected descriptor; it retains the
deny-by-default context rules.
The descriptor validator explicitly accepts vocab/merges for this Base checkpoint
and retains the existing tokenizer.json layout for other managed snapshots.

The worker now shares the tokenizer layout validator with provisioning and checks
Base config identity before accepting the managed snapshot. These source checks
do not establish installed execution or inference qualification.
The headless distribution manifest remains unchanged for the parent's LOT3.

There is no Qwen production install policy or enable command in this lot.
Direct image launch refuses. LOT2 must wire the existing certified daemon-role
worker protocol with the UID 10001 child, ownership, capabilities, configuration
and image seals intact. Do not bypass those gates or add a parallel launcher.
Then the parent can complete the archive/module closure, authorize a build with
fresh disk admission, qualify the installed offline lifecycle and inference,
and obtain human voice-quality approval. CPU and Apple require their own actual
qualification. The previously reported 7.3 GiB free and 6 GiB floor do not
establish sufficient future build capacity; remeasure and budget then.

Validation in this lot uses source parsing and finite mocked/stdlib tests only.
The bounded source-test container rerun uses an existing image, read-only source
and root, no network, and model-import blocking. It passed 53 tests, but the final
wheel-install `--no-deps` adjustment occurred during that run and received
separate source checks; this was not an exact-final-source rerun.
No image build, model download,
host ML import, paid API, key read, staging, commit or push is part of this change. r409's original
runtime RESULT established offline imports/version checks, not inference or
release authority.

## Lot A — managed CPU x-vector source journey

The Linux source lane now reconstructs preparation and saved-voice executions
from one trusted CPU clone template. Public `reference_text` must be empty;
the private typed descriptor and loader use `None`. Native transcript/ICL
preparation keeps its existing contract. Preparation accepts only the complete
canonical mono PCM16 WAV, at most 1 MiB and 60 seconds, and synthesizes the
separately authored audition. Existing candidate publication and CAS save retain
the reference independently of the audition. Saved generation uses the frozen
spoken plan and the Core-owned immutable version, never a public filesystem path.

For an explicitly configured source harness, `QwenContexts` supplies owner
resolution. The supervisor must separately register the **whole original
context** under `(principal, 'qwen-cpu', 'managed-clone')` in `qwen_contexts`.
Exact-version registrations retain their existing meaning. The supervisor
reconstructs and corroborates the complete derived context, verifies the current
hardware/profile, and obtains the exact private reference through the existing
reaped UID10001 descriptor helper. This does not add an installed family or
configuration policy. The mounted reference root remains `/references/assets`.

Managed Qwen uses the control-owned `0770` conditioning namespace. Validated
reference bytes become a float32 waveform/rate tuple before backend spawn;
there is no named reference snapshot in the scanned staging namespace. The same
post-reap empty-directory helper and terminal receipt/ACK ordering are retained.
Only the exact managed Qwen CPU conditioning disappearance joins the existing
Pocket resource-monitor exception; foreign entries and substitutions still fail.

`QWEN_CPU_SECONDS = 1800` is the CPU dispatch ceiling, persisted in existing
dispatch settings and bounded by the selected profile. The supervisor anchors
its monotonic cutoff once from the original job deadline before expensive input
verification. Start and the child retain that anchor; backend startup and
inference receive only remaining time. Observe/result handling cannot replenish
it. Lifecycle RPC timeouts remain separate. Native and other-engine budgets
retain their existing ceilings. A new retry job follows existing reap/ACK gates
and retains frozen inputs; restarting an old attempt cannot renew its cutoff.

Added offline injected fixtures cover preparation/reference/audition provenance,
CAS and exact replay, immutable reopen/generation, HTTP/CLI/MCP upload through
publication, cancellation and retry after reap/ACK and reopen, descriptor and
namespace refusals, loader tuple transfer, and execution beyond 30 seconds under
the original cutoff. The final bounded offline run passed **115 tests**, with no
failures, errors or skips. Astra independently verified the source and receipt
hashes. This documentation update follows that verified snapshot; executable
source is unchanged. Engines were injected and real model imports were blocked.

Recovery now accepts exact typed Qwen preparation pairs alongside Pocket pairs.
Fresh installed Qwen enrollment, current installed CPU
availability after the existing 300-second observation window, archive closure,
and integrated daemon retention remain Lot B work. Integrated A+B review and
operational qualification are still required. There is no installed CPU release
or real inference claim.
