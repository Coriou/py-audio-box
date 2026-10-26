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
hashes of the 24-package SoX closure completed from r409's retained official
Bookworm Packages metadata. The original 16 pins remain unchanged. The pinned
Bookworm base supplies libc6 >= 2.34, gcc-12-base = 12.2.0-14+deb12u1,
libbz2-1.0, liblzma5 >= 5.1.1alpha+20120614 and zlib1g >= 1:1.2.11.dfsg;
the recipe checks this boundary before installation. Downloads are size bounded and hash checked before
`dpkg -i`; no apt resolver or unversioned package acquisition is used. Artifact
unavailability or a mismatch fails the build. The recipe is amd64 only. The
retained private system-stage footprint was about 52 MiB; this is not a build
space estimate. No binaries or private experiment paths are distributed.


The actual worker build at source `6349441fad721fd9ab688b79766df86d72564c1e`
failed at `dpkg`, before any pip step: libsox-fmt-base and libvorbisfile3 lacked
FLAC/Ogg/sndfile/Vorbis dependencies. r409's extraction had resolved against a
different base (`793f998d…`), so its downloaded subset was insufficient for this
slim image. Recursive Depends/Pre-Depends review adds the five missing packages
and libsndfile1's three MP3/Opus dependencies. Exact added pins are below; URLs
are `https://deb.debian.org/debian/` plus the Filename fields retained in the
source test fixture and embedded in the Dockerfile.

| Package | Version | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| libflac12 | 1.4.2+ds-2 | 198124 | `4d8431c274eef13ec9ee2c0bf988c34b4342367dec053b7f69a432525be5b1fe` |
| libogg0 | 1.3.5-3 | 23412 | `f67acb0477aed0354f411172e37af338f3ac6bd4f3766134a2cf5539884b5a28` |
| libsndfile1 | 1.2.0-1+deb12u1 | 196540 | `02dc3f08fe8476e82e005f412e3b7289ff641180f33a9be8073f4bde61c85e8e` |
| libvorbis0a | 1.3.7-1 | 93028 | `01c14f9d1109650077a4c5c337c285476a6932d4d448f5a190c2b845724dbf21` |
| libvorbisenc2 | 1.3.7-1 | 80568 | `75dd4d6f904c7db82f5112e60d8efea9e81dfedf94e970c5da7af2b3d81643c0` |
| libmp3lame0 | 3.100-6 | 364932 | `b1292c749172e7623547ecf684da57d3ec6ca22109f08b11bbb9e6be6e6beb95` |
| libmpg123-0 | 1.31.2-1+deb12u1 | 146980 | `a10ad0d59995b859797a5f20d2833672051d0b2c7dab4c71f2e79258b2e3f631` |
| libopus0 | 1.3.1-3 | 195192 | `c172e212f9039e741916aa8e12f3670d1e049dc0c16685325641288c2d83faa7` |

Provenance: r409's retained main Packages.lz4 SHA-256 is
`38163a87d0269f1af22bdddc092bf8e6c605f981642a09561e60565358f15e07`;
security Packages.lz4 is
`2fee886f46c37e24a2741b8da6ac8c3b1fde2d76f479771caa0b77def424b795`.
The retained main InRelease date is 2026-07-11 10:16:37 UTC; security is
2026-10-26 04:42:52 UTC. These are the same retained index bytes used by r409,
not a newly resolved apt closure. The base sources' 20261005T000000Z snapshot
comment and the base digest are unchanged; the retained acquisition actually
used deb.debian.org. This repair neither advances those pins nor introduces a
snapshot-date claim for the mirror metadata.

The static test embeds the relevant original Packages fields for all 24
artifacts, checks URL/hash/size/version pins, recursively traverses dependency
alternatives, and checks the explicit base boundary. No extra manifest or build
context inclusion is needed. This is a source repair only: parent review,
commit, fresh build admission and an actual worker build remain required.

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
That earlier source-validation lot performed no image build, model download,
host ML import, paid API, key read, staging, commit or push. r409's original
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

At the Lot A checkpoint, recovery accepted exact typed Qwen preparation pairs alongside Pocket pairs.
Fresh installed Qwen enrollment, current installed CPU
availability after the existing 300-second observation window, archive closure,
and integrated daemon retention were still Lot B work, completed in source below.
Operational qualification remains required. There is no installed CPU release
or real inference claim.

## LotB installed daemon source family

The explicit `speech-box docker-qwen` command selects a fresh Qwen CPU daemon
installation. Its public finite commands are `install`, `up`, `status`, and
`stop`; `install` runs import and initialization as separate helper phases; restart uses `stop` then `up` with the same retained handle. Acquisition
and image building are separate operator actions. Startup never downloads,
repairs, pulls, or chooses a model from filesystem metadata.

Only Qwen/Qwen3-TTS-12Hz-0.6B-Base revision
`5d83992436eae1d760afd27aff78a71d676296fc`, the canonical eleven-file descriptor,
CPU float32 SDPA, and transcript-free x-vector cloning are allowed. The source
recipe remains the existing `worker-qwen` target, including its pinned SoX source
build and reviewed wheels. These source changes do not qualify that image.

The helper retains registration and locks. Core runs as UID 10001; the supervisor
runs as root with the existing dropped-capability isolated execution children.
Workers receive only the read-only `core/assets` volume subpath. The existing
checked reference descriptor bridge, inherited descriptor, 0770 namespace,
original 1800-second deadline, ACK/reap, cancellation, publication, and ownership
fences apply. Public API, CLI, and MCP use the existing dispatcher journey:
upload → prepare → inspect results → CAS save → frozen spoken generation.
Duplicate replay remains ahead of live checks; historical identity is immutable.

Installed CPU availability is checked separately from the frozen Linux hardware
observation. No recorded timestamp is refreshed. Non-installed Linux contexts
retain their existing five-minute observation expiry. Apple and CUDA are deferred.

Lot B passed independent Astra source review and 30 exact-final-source offline
tests, with no failures, errors or skips. This prose update follows that verified
snapshot. Actual target-image build and installed CPU qualification remain.
The earlier 24 GiB capacity admission observation is not future capacity proof.
No dirty-source archive is a release. Source tests establish neither installed
runtime nor hardware or audio quality acceptance.

The installed CPU public identity is exactly `Qwen3-TTS-12Hz-0.6B-Base`
(the existing public model-name convention without an upstream namespace).
Plans, saved voices, replay and history retain that identity. Execution
descriptors and bundle metadata retain checkpoint `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
and revision `5d83992436eae1d760afd27aff78a71d676296fc`: one exact binding,
without alias normalization or fallback. The eleven-file manifest pin and
CPU float32/SDPA requirements are unchanged.

### Bounded live observation history

Only the explicitly checked installed Qwen CPU family uses internal authenticated
state polls through the dispatcher's existing single executor. ALIVE polls add no
command rows. A STOPPED poll schedules a journaled `observe`; only its matching
STOPPED reply permits collection. A contradictory confirmation quarantines.
UNKNOWN is confirmed through the durable observation path; failed or unbound polls
retain an uncertain `observe` and quarantine. These reads change neither the
original deadline nor the independent watchdog's TERM/KILL and reap obligations.

Cancellation and resource stops share one stop request per runtime, serviced before
the cadence guard, including after pending start or observation I/O. Receipt
verification, publication, reservation release and ACK ordering remain unchanged.
Five clean attempts retain 25 lifecycle commands (prepare, start, terminal observe,
collect, ACK per attempt), plus at most five stops. ACK retries and diagnostic
failures retain additional history. This is not unlimited retention: the aggregate
2048-row ceiling counts every SQL table, including inventory and sqlite_sequence;
the 16 MiB and other inventory/worker limits still apply. Existing history remains
charged. Simulated duration tests are source evidence, not installed CPU or audio
qualification. The bounded r420 run passed 180 tests and 103 subtests in 103.73
seconds; Astra verified the exact source and receipts. This prose correction
follows that snapshot. Actual qualification must still measure final aggregate
installation capacity.

The r420 correction preserves exact job, event, dispatch, command and saved-voice
rows across reopen, while charging new inventory scans, objects and history to
the same aggregate ceiling. The full-table injected journey runs five simulated
1800-second attempts with pair publication, save, failed generation, retry and
reopen replay. UNKNOWN-after-cancel uses an ambiguous stop reply configured before
the RPC can apply; the generic fixture's STOPPED mutation must not overwrite it.
STOPPED confirmation errors and UNKNOWN replies retain durable observation
evidence and quarantine without collection or ACK. These remain offline source
checks, with no installed runtime, release or audio acceptance claim.

### Config metadata validation (r427 source fix)

For the reviewed Base BPE checkpoint, only `config.json`'s root,
`talker_config`, `speaker_encoder_config`, and
`talker_config.code_predictor_config` may contain `_name_or_path` metadata,
and only as the exact empty string. These are upstream `PretrainedConfig`
objects; the field records provenance rather than selecting an asset. The
pinned config contains one such field, in the code predictor. Other locations
retain asset-reference validation. Real `*_path`/`*_file` declarations remain
nonempty strings naming listed local assets; remote code declarations remain
refused. Base `vocab.json` is instead a nonempty token-to-nonnegative-integer
mapping (booleans and containers refused), so token spellings are not loader
keys. No JSON bytes, model pins, or manifest identity are rewritten.

This source fix requires review, commit and rebuild before a separately
authorized actual measurement; it supplies no inference or audio qualification.

### Numeric UID startup environment (r429 source fix)

Before importing `lib.tts` (and therefore Torch/Qwen), the spawned Qwen owner
sets HOME, user names, temporary paths, TorchInductor/Triton/Torch and Hugging
Face cache paths to controlled private state under `/tmp/qwen-child-*`.
Inherited values are overwritten. The parent creates the mode-0700 directory
and owns its plain path without a GC/atexit deletion finalizer. Cleanup requires
a confirmed integer child exit status after join, or a process-start failure
with no child obligation. Unknown status or failed reap retains storage even
after owner disposal; a detected unreaped child disables reuse. Start attempts
are recorded before `Process.start()`: exceptions can follow OS child creation
even with no PID and false `is_alive()`. Ambiguity retains process ownership and
private storage and disables reuse until successful join and integer exit status
confirm reap. Failures before start invocation have no child obligation. Later
verified reap permits idempotent cleanup, but does not re-enable the backend.
There is no passwd entry requirement or device/model fallback. Model paths,
offline policy, UID 20000 and container isolation remain unchanged. Writable
space remains subject to the runtime's bounded temporary filesystem; no image
environment change is required.

Retained Torch source corroborates `cache_dir()` -> `default_cache_dir()` ->
`getpass.getuser()` -> passwd lookup when user environment is absent.
`torch/_inductor/codecache.py:2669` also calls `default_cache_dir()` directly
at module import, bypassing TORCHINDUCTOR_CACHE_DIR. Controlled USER/LOGNAME
and temporary paths therefore accompany the explicit cache override.
The prior actual trace loses the dependency traceback at the child error
transport, so the exact historical intervening import frames are unavailable.
Offline import-seam checks supply source evidence only. Review, commit and
rebuild precede any separately authorized inference/runtime qualification.

### Bounded public model import (r431 source fix)

The Qwen `docker-qwen install` import has a fixed 600-second budget, anchored
before helper creation. The sender transmits its original UTC expiry once; the
helper validates the finite remaining budget (at most 600 seconds), converts it
to its local monotonic clock, and uses it for the wire, chunk RPCs, and final
sealed-tree descriptor verification. No file or chunk renews this deadline.
Host and Docker UTC clocks must agree; clock skew can cause a conservative
refusal. There is no CLI timeout override. Qwen initialization and later client
phases retain 120 seconds; helper initialization retains 100 seconds. Pocket and
Kokoro budgets are unchanged.

This bound follows r430's actual failure: the canonical eleven-file bundle is
2,516,100,892 bytes, and at least 973,078,528 bytes had reached the destination
before the 100-second helper deadline. That observation suggests roughly
9.3 MiB/s and about 259 seconds for transfer alone. 600 seconds allows transfer
and final rehash. At the 4 GiB total cap, transfer alone requires about
6.83 MiB/s; rehash must also finish within the same budget.
It is a source budget decision, not proof that a fresh installation will finish.
The existing 2 GiB per-file, 4 GiB total, no-follow paths, incremental SHA256,
UID/mode, lock, configuration, manifest and receipt checks remain required.
A stopped helper alone never proves import completion. Failed/expired imports
retain intent and partial volumes; initialization requires verified import-done.

The prior actual engine inference passed; public installed import and all later
installed HTTP/CLI/MCP, cancellation/reap, restart/history and publication stages
remain unqualified. Review, commit and new images precede any actual retry.
