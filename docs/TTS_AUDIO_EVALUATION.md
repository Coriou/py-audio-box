# Standalone audio evaluation (r410)

`benchmarks/tts/audio_judge.py` is a stdlib-only, serial synchronous judge.
Dry run is the default. Original audio is untouched; no resampling or host ML.
Judgments do not establish human acceptance, MOS, speaker identity, or release
qualification. No paid calls or credential reads occurred in this preparation.
The user authorized a small pilot with a key; this document does not execute it.

## Input and transmission

The input matrix uses `schema_version: 1`, with `samples` containing `status`
and `audio: {path, sha256}`. Only successful rows are selected. `--select 0,3`
optionally selects zero-based matrix indices. Declared original hashes must match.
Paths must lie under the explicit root; traversal and symlink components are
refused. Use canonical paths rather than `/tmp` or `/var` aliases.

Both matrix (4 MiB maximum) and WAV (20 MiB maximum) use a directory-descriptor
walk from `/`, opening directories with `O_DIRECTORY|O_NOFOLLOW`, then files with
`O_NOFOLLOW|O_NONBLOCK`. The opened descriptor must be regular. Reads are bounded
to limit+1 bytes; before/after identity, size, timestamps and final directory-entry
identity must match. Growth, replacement and symlink swaps fail closed.

PCM WAV only: 1–2 channels, 8–96 kHz, 1–4 bytes/sample, at most 120 seconds.
Truncated PCM is rejected. A fresh metadata-free PCM WAV is constructed from the
validated exact frames with unchanged channels, sample width and rate. Ancillary
chunks and trailing bytes are excluded. `original_sha256` (also the compatibility
`sha256` field) and `transmitted_sha256` are recorded explicitly and both enter
cache identity. Only the reconstructed WAV bytes are transmitted.

The API receives the fixed rubric and audio, never filenames, engine labels,
expected text, blind-map data or Ben's findings. Ordinary identical requests dedup.
`--replicate-id blind-repeat-2` explicitly changes cache identity for a planned
independent blind repeat; it is local metadata, never sent in the prompt.

## Model and reservation

Each plan fetches unauthenticated public endpoint and model metadata. It verifies
canonical slug `google/gemini-3.8-flash-20260902`, audio input, strict output and
reasoning parameters, supported `low` effort, standard endpoint availability,
prices and the 65,536-token completion ceiling. The request pins the canonical
slug and `google-ai-studio`, with `only`, no fallback and required parameters.
The full endpoint snapshot and reasoning metadata are retained privately.

The selected endpoint advertises `supports_multiple_audio_references: false`:
`--reference` is refused unless the selected endpoint explicitly advertises true.
The first pilot uses individual clips with no identity scoring. Gemini thinking
is mandatory; requests use `reasoning.effort: low`, never disable thinking.
The requested `max_tokens: 2048` is not treated as a billing guarantee.

Verified standard prices on 2026-10-26 are $0.75/M input/audio tokens and
$3.75/M output/internal-reasoning tokens; discounts are ignored. Reservations use
32 audio tokens/sec rounded up per clip, UTF-8 prompt/schema bytes plus 1024
framing tokens, and the **full 65,536 completion tokens including thinking** at
the higher completion/internal rate. Each pilot request must calculate at most
$0.25 and reserves a fixed $0.25. Execution admits at most four new unique
requests and cumulative retained reservations no greater than the supplied cap,
which must be positive and at most $1.

This is a hard **local reservation/admission rule**, not a server billing
limit or a guarantee about token accounting, provider behavior or future prices.
`estimated_usd_at_requested_cap` is explicitly separate from the full-ceiling
calculation and reservation. A provider-side key budget is an independent control.

## Prepared private calibration pilot

The original r408 `listening/blind-map.json` and repository
`docs/agent-interface/evaluation/pocket-comparison-r408/listening-review.json`
were checked against both source WAV and neutral listening WAV hashes. These
are existing r408 notes, not new labels for r409 output. Selection order:

- sample013: incorrect amount.
- sample018: correct amount.
- sample004: omitted tail.
- sample017: complete, robotic delivery.

Private files under `~/.local/state/py-audio-box/audio-evaluation-r410/`:
`calibration/matrix.json` contains only input audio paths/hashes and status;
`calibration/provenance.json` retains the exact blind-map records, its hash and
known notes locally; `manifest.json` contains the four-request dry-run plan.
Existing source audio is reused without copying. Provenance and findings are
never included in the request body.

Actual dry run: **4 samples, 4 unique/new requests, $1.00 reserved**.
Full-ceiling calculated total: **$0.990666**. Estimate at the requested 2048-token
cap: **$0.038346**, not admission authority. Durations in selection order:
2.88, 4.736, 9.28 and 12.309333333333333 seconds.
The state had no ledger file before or after preparation, hence no retained
submissions in this tool's ledger; this is not an account-wide billing audit.

The earlier whole-r409 plan had 32 samples / 24 unique requests and reported
$0.228852 using the requested output cap. That is superseded as admission
reasoning: full-ceiling planning calculates $5.942916 and reserves $6, refusing execution
under $1. Only the four calibration requests are prepared for the first pilot.

Dry run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/tts/audio_judge.py \
  --root /Users/ben/.local/state/py-audio-box \
  --matrix audio-evaluation-r410/calibration/matrix.json
```

Exact execution command, prepared but **not run** (uses existing `.env` only
when executing, or an already-set `OPENROUTER_API_KEY`):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/tts/audio_judge.py \
  --root /Users/ben/.local/state/py-audio-box \
  --matrix audio-evaluation-r410/calibration/matrix.json \
  --mode sync-audio --execute --max-cost-usd 1.00 --env-file .env
```

## Persistence and judgment limits

Private state directory must be owned and 0700; atomic JSON files are 0600,
with file/directory fsync. Exclusive locking covers reservation and execution.
All new reservations persist before submission. Cache hits cost no new local
reservation; completed reservations remain counted. Timeouts, process death,
invalid judgments or missing billing metadata retain reservations and block
further execution. There is no automatic retry, release or settlement redesign.
Never delete an uncertain ledger to regain budget; reconcile with provider
billing manually. Local state cannot recover an unreturned request ID.

Keys are read only after execution admission. Literal env-file assignments only;
no sourcing, interpolation, global export or logging credentials. No redirects,
provider fallback or automatic retry. Timeout: CLI 1–60 seconds, default 60. The `run()` API accepts only finite
integer/float deadlines with `0 < timeout <= 60` (excluding booleans), validated
before state locking, file writes, credential reads or reservation.
The ledger retains request ID, resolved provider/model, usage/cost and validated
judgment or bounded uncertainty state; raw provider error bodies are not stored.

A single call returns a heard transcript plus acoustic judgments. Those outputs
are correlated; this is not independent ASR and acoustic evidence. A future
two-pass approach and expected-text comparison are deferred. No acceptance
threshold is established by this four-clip calibration. Batch audio is unsupported;
`sync-audio` is the only mode and has no claimed batch discount.

## Verification and public sources

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_audio_judge.py -v
```

16 stdlib tests pass with synthetic PCM and mocked submissions: bounded race
reads, symlink swaps, matrix rejection, exact-frame metadata-free transmission,
cache/replicate selection, reservations, uncertain-state retention, schema,
credential isolation and locks. No live submission compatibility, billing or
judgment quality is claimed.

- [Endpoint metadata](https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints)
- [Model catalogue and reasoning metadata](https://openrouter.ai/api/v1/models)
- [Reasoning effort and Gemini thinking](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Audio input](https://openrouter.ai/docs/guides/overview/multimodal/audio)
- [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
