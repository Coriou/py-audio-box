# Finite STT feedback (r412 local comparison)

`benchmarks/tts/stt_check.py` provides serial speech-to-text feedback for the
user-authorized CPU/Apple listening loop. It runs on existing PCM WAV files;
it does not run synthesis, host ML, model downloads, or builds. STT disagreement
is a **REVIEW**, never a synthesis FAIL. A listener must confirm a synthesis
problem separately. CLEAR means only “no anomaly detected,” not acceptance.
Existing r408 human labels are not transferred to the new r409 48-item set.
Proper-name spelling differences are not proof of a pronunciation problem.
No acoustic quality, MOS, or reference-audio evaluation is performed.

## Verified transport and pricing

Public official sources checked on 2026-10-26:

- [OpenRouter transcription contract](https://openrouter.ai/blog/tutorials/transcription-on-openrouter/)
  (updated 2026-09-24): `POST https://openrouter.ai/api/v1/audio/transcriptions`,
  JSON base64 audio, optional ISO-639-1 language, JSON transcript and usage.
  Its pricing section defines Whisper catalog rates as dollars per audio second.
- [Live transcription model metadata](https://openrouter.ai/api/v1/models?output_modalities=transcription):
  exact ID and canonical slug `openai/whisper-large-v3`, audio-to-transcription,
  `pricing.prompt = "0.0000075"`, `pricing.completion = "0"`.
- [Model API page](https://openrouter.ai/openai/whisper-large-v3/api):
  confirms endpoint and model; its displayed $0.000008/second is rounded.
  The exact plan uses the catalog's $0.0000075/second, not token conversion.

The transcription-specific routing documentation explicitly says `only`,
`order`, `allow_fallbacks`, and `sort` do not pin this endpoint. That takes
precedence over generic routing text on the model page. This tool sends no
provider block and makes no provider guarantee. Provider and resolved model
are `unknown` unless returned by the service. A returned different model
stops execution as uncertain; the tool never selects another model. The
catalog exposes no immutable weights revision: the cache records the canonical
slug as the advertised model version and the snapshot states that limitation.
An upstream weights change without a slug change cannot be detected here.

Only these request fields are sent:

```json
{
  "model": "openai/whisper-large-v3",
  "language": "fr",
  "response_format": "json",
  "input_audio": {"data": "<metadata-free WAV base64>", "format": "wav"}
}
```

No expected text, names, engine, human feedback, prompt, or vocabulary hints
are sent. There are no chat requests, batch API jobs, audio URLs, reference
clips, automatic retries, or different-model fallbacks. Batch submission is
unsupported: each unique clip is one serial synchronous HTTP request. The
documented upstream processing timeout is about 60 seconds, not an audio
length promise; this client caps its own timeout at 60 seconds and WAVs at
120 seconds/20 MiB using the existing input helper. Actual execution evidence is retained privately, separately from fixture evidence.

## Plan and execute

Default command, from the repository:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/tts/stt_check.py
```

This performs only a public metadata GET and local WAV reads, then writes a
private planned manifest. It never loads a credential or submits audio.
Defaults are:

- Matrix/root: `~/.local/state/py-audio-box/voice-comparison-r409/matrix-r7.json`
  and its parent directory.
- Output: `~/.local/state/py-audio-box/stt-evaluation-r411` (owned 0700).
  Atomic manifest/ledger replacements use 0600 files under a held state lock.
- All 48 matrix slots must have successful audio; the tool is bounded to 1–48
  items and verifies every declared original SHA before planning.

The checked matrix has SHA256
`b84b2077a24e094e417f7852f29fd7c3c6c376294dca18ba7eb271fcfe97384d`.
The dry run produced **48 items / 40 unique requests**, with exact original
and transmitted SHA256 for each input in `manifest.json`. Unique duration
cost is **$0.00173116** at **$0.0000075/second**. The independent local
reservation is **$0.40** (minimum $0.01 per unique request). All 48 items now have nonempty validated annotations: **138 checks, zero missing**.
The private `critical-r411.json` preserves the original 120 checks and adds exact
`200` and `180` number checks on six decrease clauses, plus `sans dépasser`
negation checks on six rows. `200 euros` is not an exact source substring.
The original sidecar and SHA256 provenance are retained beside it. The manifest
records annotation coverage and a canonical annotation hash; its missing template is empty.
These planning figures are distinct from paid execution and listening qualification.

An explicitly authorized invocation can submit with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/tts/stt_check.py \
  --annotations "$HOME/.local/state/py-audio-box/voice-comparison-r409/critical-r411.json" \
  --env-file /absolute/path/to/userkeylocal.env \
  --execute --max-cost-usd 0.40
```

The shared literal key loader reads `OPENROUTER_API_KEY` from the process
environment first, otherwise from `--env-file`, only during execute after budget
checks. The file is never shell-sourced, interpolated, or exported; dry-run never
reads it. Supply the actual local file path above; never put the key in arguments.
Every invocation refreshes the public metadata; changed pricing changes the
plan and can cause budget rejection. `0 < --max-cost-usd <= 1` is mandatory
for execute. The cap applies to all retained reservations plus new ones, not
merely the current invocation. Reservations are never automatically refunded.
They are local stop controls, **not a server-side billing ceiling or guarantee**.
Duration pricing is an estimate; provider billing granularity is not assumed.

Before the first submission, the entire pending set is durably reserved.
Each request is marked in flight before HTTP. Timeout, malformed response,
unknown/invalid/over-reservation cost, unexpected model, or interruption leaves
an unresolved reservation and stops. There is no retry or automatic clearing;
resolve billing and retained state manually before further execute calls.
The full billing `usage` object (bounded to a 16 KiB audit object), request ID,
provider, resolved model, seconds, and actual cost are retained privately.
Seconds must be finite and positive; invalid or impossible usage stops serial
execution as uncertain. Local duration tolerance is max(0.25 seconds, 5% of
actual clip duration). Cost tolerance against usage seconds × catalog rate is
max($0.000001, 10% of that estimate). These are explicit anomaly thresholds,
not provider rounding rules. Every reconciliation remains `billing_check.status:
review` with unknown rounding, even within tolerance; exact billing is not asserted.
Usage seconds above 600 (for clips capped at 120 seconds), nonpositive/nonfinite usage or cost, missing cost, or cost above the reservation stops execution. A bounded duration discrepancy is `usage_review` and serial execution continues; the independent `price_review` flag remains visible even when both differ. A valid reported cost within its reservation but outside the catalog tolerance is `price_review`; serial execution continues. This does not establish invoice accuracy or a provider minimum price.
Errors retain only a fixed message. Raw server errors/payloads and credential
strings are not logged. A transcript containing the active credential is
rejected. The transcript is durable before billing validation and before
expected-text comparison; an uncertain billing result can therefore retain
a transcript for local comparison without resolving billing.

## Cache, imported transcripts, and local comparison

The module imports only helper functions from `benchmarks/tts/audio_judge.py`
using importlib: held-descriptor bounded reads, metadata-free PCM reconstruction,
HTTP without redirects, canonical hashes, private atomic writes/state lock,
and secret loading. It does not use acoustic verdicts. Matrix reads are bounded
at 4 MiB and reject symlinks and changing inputs; WAVs are verified again
against the planned metadata immediately before transmission.

STT keys include original and transmitted hashes, model ID, advertised model
version, language, and format/response parameters. Expected text, critical
annotations, engine, names, and human labels are absent from the STT key.
Changing the script alone reuses the persisted transcript without another
charge. Local comparison keys separately include transcript SHA, expected
text SHA, comparer version, language, annotations, and uncertainty. Results
live in `ledger.json` under `comparisons`; per-item keys are in the manifest.
Transcript text is private retained feedback, not a public report.

For an already available transcript, create a JSON object under the explicit
input root, keyed by the planned STT cache key:

```json
{"<cache_key>": {"text": "What was actually transcribed", "uncertainty": false}}
```

Then use `--import-transcripts transcripts.json` with the appropriate
`--input-root`. Import is a separate operation from execute. It accepts only
planned, unused keys, persists transcripts before comparison, and marks their
source `local_import_unverified`; it does not assert an OpenRouter call or
charge. The CLI still refreshes public pricing metadata but sends no audio.
Programmatic `compare(...)` is entirely local. Imported records are not
provider or synthesis qualification evidence.

## Critical annotations and conservative interpretation

Supply `--annotations critical.json` under the input root. Its object maps
zero-based matrix indices to lists; a row's `critical_literals` can also hold
the same list. The sidecar overrides a row only for specified indices. A
missing annotation list always requires REVIEW; an explicit empty list means
no critical checks were supplied; coverage counts it as missing. Do not treat
the planned empty template as an assessed annotation set.

```json
{
  "0": [
    {"kind": "amount", "expected": "1 250,50 euros", "heard_forms": ["1250,50 euros"]},
    {"kind": "identifier", "expected": "A-2047"},
    {"kind": "version", "expected": "B"},
    {"kind": "negation", "expected": "pas"},
    {"kind": "tail", "expected": "avant vendredi"},
    {"kind": "proper_name", "expected": "Coriou"}
  ]
}
```

Every `expected` must be an exact substring of that item's expected text.
Identifiers and versions retain exact surface checks with word boundaries;
`à 2047` never establishes identity with `A-2047`. Proper-name differences,
including case/diacritics, require `ASR_name_review`; SAS/SASS also receives
that annotation, never pronunciation proof. Negation uses NFC/casefold word
tokens. Tail checks use the existing normalized token stream and the last
nonempty clause of the exact annotated substring, split at sentence marks,
semicolons, or commas outside numeric literals. Quotes, punctuation, case, and
curly apostrophes do not create tail failures. Missing words still require
review; agreement on a suffix does not clear another content mismatch. Unsupported annotation kinds require REVIEW.

`amount` parses a whole literal to currency and integer minor units, with
EN comma grouping/dot decimals or FR space grouping/comma decimals. It does
not conflate `1250.50€` (125050 minor units, EN) with `125050€` (12505000).
`number` uses the same finite two-decimal locale parser without currency.
A small explicitly listed spoken vocabulary supports zero through ten plus
20 and 50 (EN/FR), units and optional cents; it is not a general number-word
parser. Explicit `heard_forms` allow local checking of alternative surfaces;
they never become request hints. Unsupported/ambiguous number forms, unknown
currencies (including ambiguous `$`/“dollars”; use explicit `USD`), and unsupported locales never establish equivalence. Even a
parsed equivalent critical form does not clear a textual ASR mismatch.

WER/CER reuse `lib/fidelity.py`'s deterministic edit-distance routine,
reporting edit counts and rates for NFC raw text and NFC/casefold/word-token
normalized text. No intelligibility score is derived. Any normalized mismatch,
empty/uncertain transcript, critical check requiring review, detected negation
loss, or changed/omitted final three words requires REVIEW regardless of low
WER. Generic negation/tail checks supplement explicit annotations; they do
not establish full semantic coverage. Timestamps are null: the default JSON
request does not ask for timings and this slice does not import or infer them.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_stt_check.py
```

54 STT fixture tests pass in the focused stdlib suite (45 existing, 9 added). They cover deduplication, dry-run credential isolation, exact
request fields, durable transcript ordering, script-only cache reuse,
identity invalidation, mock HTTP timeouts/unknown charge, retained reservation,
credential redaction/rejection, import, budget limits, changed audio, symlinks,
pricing rejection, EN/FR money/minor units and numbers, limited spoken forms,
identifiers/version separation, negation, tail omission, proper names,
empty/uncertain text, and NFC raw/normalized metrics. No test calls a service.

The shared HTTP transport enforces a whole-call deadline with a scoped POSIX
`setitimer` on the main thread, including connect, response reading, and JSON
parsing. It restores its prior default/ignored handler and inactive timer in
`finally`; it refuses a pre-existing timer/custom handler, non-main threads,
and platforms without this API before network I/O. Linux/Darwin CLI is supported;
Windows requires a future owned-process transport, rather than a socket-only
claim. No background transport thread is created. A local slow-drip HTTP server
regression verifies an actual finite deadline and handler restoration.

The authorized recovery uses `--reconcile-ledger-sha256 HASH --reconcile-cache-key KEY`
as a separate, one-time local operation. The operator must explicitly attest that
legacy reserved entries in that exact ledger were never sent after the sequential
loop stopped. This is not a generic inference from `reserved`. The operation
checks the ledger hash, transcript hash, and reported usage, writes read-only
0400 copies of the original ledger and manifest plus SHA256 hashes in a 0500
backup directory, and records the exact unsent keys in a reconciliation audit.
It retains the prior error, usage, provider unknowns, costs, and all reservations.
The known response becomes `usage_review` for a duration discrepancy, otherwise `price_review`, with `cost_unverified_invoice`. Each hash-pinned recovery preserves earlier reconciliation audits.
No root cause or invoice settlement is inferred.

Only explicitly audited never-sent reservations may resume without reserving
again. New entries are durably reserved; each is durably marked `sending` and
`submitted` before HTTP. A crash at that boundary requires manual investigation;
complete, price-review, usage-review, sending, and uncertain entries are never resubmitted.
Unknown transport, invalid response/schema, impossible duration, or charge above
reservation stops the one serial loop without automatic retry. All retained valid
transcripts receive local comparisons even when billing validation stops execution;
comparison availability does not resolve transport or invoice uncertainty.
The original cumulative $0.40 reservation remains retained for all 40 keys.


## Bounded recovery execution, 2026-10-26

The authorized recovery submitted 11 of the 22 audited unsent keys and stopped
on the first unresolved duration response: usage reports 10 seconds for a 6.08
second clip (local tolerance 0.304 seconds). Its transcript and usage are retained;
no root cause is established and no automatic retry occurred. The response-reported
cost is $0.000308333333333, below its $0.01 reservation. This is a duration
validation stop, not evidence of a transport failure or a provider minimum price.

Final retained statuses are 24 complete, 4 price_review, 1 uncertain, and 11
reserved never sent: 40 unique keys, unchanged cumulative reservation $0.40.
Total response-reported cost is $0.001816075833333, including the unresolved
response once; this is not invoice verification. The private
`resume-partial-report-r411.json` contains all 48 matrix rows, 138 planned critical
checks, 37 available row comparisons, per-engine/language issue counts, and raw
and normalized CER/WER. Missing transcripts remain unassessed. Proper-name and
other ASR warnings require listening and are not automated quality acceptance.

The final parent-authorized continuation treats a complete valid returned response
with finite positive usage at most 600 seconds and finite positive cost within its
retained reservation as known transmission. Duration and catalog-price mismatches
remain separate permanent review flags, with the same anomaly tolerances above.
No rounding, minimum charge, provider identity, or invoice accuracy is inferred.
Transport-unknown entries remain blocked and are never resent. The second explicit
key/hash reconciliation preserves the first audit and a new hashed read-only backup;
no ledger deletion, reservation release, or reset occurs.


## r412 bounded local recomparison

Comparer `r412-finite-v3` changes only local comparison keys, leaving STT keys,
transcripts, reservations, and billing states intact. Successful comparisons
checkpoint individually in the existing private ledger during normal runs, so
later comparison failure retains earlier work without another transcription.
The explicit EN orthographic map contains only `neighbourhood` → `neighborhood`.
`orthography_only` is informational and can remove mandatory text/tail review
only with nonempty annotations whose critical checks all detect no anomaly.
Raw and normalized edit metrics remain unchanged; this is no general stemming.
Finite numeric checks recognize listed spoken integers such as `three`/`trois`
and `3` without explicit heard forms. This does not rewrite global text, delete
numbers, or clear a remaining textual mismatch. Unsupported forms stay review.

The local import API recomputed all 48 retained comparisons into private
`~/.local/state/py-audio-box/stt-evaluation-r411/contentreport-r412.json`, with
comparer source SHA256 and original report/ledger hashes. Original r411 JSON
and ledger bytes were verified unchanged. No HTTP, credential reads, host ML,
or retranscription occurred. Exact plaintext expected/heard strings remain.

Counts below are anomaly triage, not quality passes or engine qualification:

| Group | Before CLEAR / REVIEW | After CLEAR / REVIEW |
| --- | --- | --- |
| kokoro/fr | 2 / 6 | 2 / 6 |
| kokoro/en | 0 / 8 | 4 / 4 |
| pocket-tts/fr | 2 / 6 | 2 / 6 |
| pocket-tts/en | 0 / 8 | 3 / 5 |
| qwen/fr | 2 / 6 | 2 / 6 |
| qwen/en | 2 / 6 | 6 / 2 |
| Total | 8 / 40 | 19 / 29 |

FR rows 34/35 retain critical amount review; EN row 46's added 1,000 retains
text and tail review. The 40 retained unique billing entries remain 34 complete,
5 price_review, 1 usage_review; reported cost stays $0.002404075833333, without
invoice verification. Qwen remains the user-preferred continuation, not an
STT-established quality winner. Production TTS normalization remains deferred
pending human confirmation.

The finite sign/boundary correction uses numeric-only surface matching: signs
(`-`, `−`, `+`), adjacent currency, decimal and grouping continuations cannot
supply an unsigned numeric substring. Unsupported signed annotations remain
REVIEW. A `signed_numeric_difference` alert also preserves review when generic
word normalization would erase a changed sign, including without annotations.
Identifier matching (for example `A-2047`) retains its separate surface rules.
Only local comparer keys change; STT request keys do not change.

A fresh local API recomparison is retained as `contentreport-r412-fixed.json`
in the same private r411 directory. All 48 rows were recomputed: 19 CLEAR and
29 REVIEW; rows 34, 35, 43 and 46 remain REVIEW. Its provenance records source
and original input hashes. Both older reports, the ledger and manifest were
verified byte-for-byte unchanged. This correction made no STT request.
