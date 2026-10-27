# Opt-in speech rendering: numeric-v1

This opt-in, stdlib-only preview renders explicitly annotated EUR amounts and
identifiers for `fr-FR` and `en-GB`. The existing named `speech_generate` operation accepts this opt-in through HTTP,
CLI and MCP. This source integration is not installed runtime qualification. The
three actual r414 rendered FR amount, EN amount and FR identifier pairs were
accepted by human listening; French liaison remains unresolved. Standard French
orthography is retained; this module never writes phonetic liaison hints.

Run from the repository root:

```sh
python3 scripts/preview-spoken-text.py request.json
# or JSON on stdin:
python3 scripts/preview-spoken-text.py < request.json
python3 -m unittest discover -s tests -p test_spoken_text.py -v
```

Example request (offsets are Python Unicode code points, end exclusive):

```json
{"original_text":"Le budget est de 1 250,50 euros.","locale":"fr-FR","profile":"numeric-v1","annotations":[{"type":"amount","start":17,"end":31,"literal":"1 250,50 euros","amount":"1250.50","currency":"EUR"}]}
```

The spoken result is `Le budget est de mille deux cent cinquante euros et
cinquante centimes.` Amount annotations include the currency word to prevent
accidentally duplicating euros. Supply exactly one of `amount` (unsigned ASCII
exact decimal string, dot decimal, up to two fractional digits) or `minor_units`
(integer, excluding bool). Range: 0 through 999999.99 inclusive. Negative and
explicitly signed amounts are unsupported in this closed profile. Zero is
`zéro euro` / `zero euros`; zero cents are omitted. One euro/centime/cent is
singular. Locale literals use optional correctly grouped ASCII spaces for French,
commas for English, and the locale decimal mark. Currency spelling is exactly
` euros`, even for input value one; output inflects it. Symbols, currency codes,
nonbreaking spaces, exponent notation and ambiguous/foreign grouping are rejected.
Literal and numeric value must agree. No float conversion occurs.

An identifier uses `type: "identifier"` and `policy: "spell-letters-digits"`
instead of amount/currency fields. ASCII letters/digits with optional internal
hyphens are supported; every letter and digit is spoken individually. A hyphen
becomes a comma and space, retaining group separation. `A-2047` becomes
`a, deux zéro quatre sept` / `A, two zero four seven`. Annotate version `B`
separately to obtain `bé` / `bee`. Leading zeros are retained. Letter names are
explicit tables; there is no implicit acronym or numeric policy.

All request/annotation fields are closed. At most 64 annotations are accepted. Annotations must be ordered and
nonoverlapping, with exact original literals and nonempty in-bounds spans.
Repeated identical literals require separate explicit spans. Unannotated digits,
dates, times and punctuation remain untouched. Input is printable, nonempty,
and at most 4096 characters; output is at most 4096 characters. This is a character
bound, not a tokenizer or runtime admission guarantee. The JSON preview transport
has a 65536-byte bound and rejects duplicate object keys. Output intentionally
contains original user text; use only printable, nonsecret preview inputs.

The result contains `original_text`, `spoken_text`, `locale`, `profile`, integer
`version: 1`, copied annotations, and `render_sha256`. The digest covers every
other result field, including all accepted request fields, using UTF-8 JSON with
sorted keys, compact separators, unescaped Unicode, and no NaN. Identical input
produces byte-identical canonical output. Changed annotations, text, profile or
version change the digest. Re-preview the original request; output is not accepted
as a request and spoken offsets must never be reused against the original text.
This digest records provenance; it is not an authentication signature.

For `speech_generate`, keep `text` exactly original and supply the closed optional
`text_rendering` object, for example:

```json
{"profile":"numeric-v1","version":1,"locale":"fr-FR","annotations":[{"type":"amount","start":17,"end":31,"literal":"1 250,50 euros","amount":"1250.50","currency":"EUR"}]}
```

Only integer version 1 is accepted (neither bool nor float). Caller-supplied
spoken text and hashes are rejected. The trusted resolver freezes `plan.speech_text`
after replacing the request and before its first execution validation. The record
contains the exact preview result. The shared pure accessor corroborates all
fields against the original request and pinned renderer, then returns the retained
`spoken_text` for inference. Missing, unsupported or changed records refuse;
history readers never repair or rerender retained history in place.

Locale must match the resolved settings: French/English for Qwen, fr/en for
Kokoro/Pocket where supported. Both original and expanded text must fit the profile
character limit; expanded UTF-8 must fit the execution descriptor byte limit.
The renderer 4096-character, HTTP 16 KiB request, private context 48000-byte and
existing output/resource limits remain. Preparation reference/audition roles are
unchanged. Without opt-in, requests and plans retain their exact raw bytes and no
record is added. Existing plan hashing carries provenance; there is no database,
publication or artifact migration and no backfill. Future renderer versions need
an explicit supported implementation, never an automatic substitution.

The existing duplicate-before-resolution, request fingerprint, historical retry,
asset allocation, cancellation, ACK and reap paths remain in charge. CLI JSON
rejects duplicate keys at every depth and nonfinite constants.

Both headless and native package manifests include `lib/spoken_text.py` and this
document. The focused offline suite passed 28 tests, including archive contents.
Installed Qwen lifecycle, actual installed CPU execution, Apple qualification and
the final release regression remain separate. These integration tests used injected
engines; they did not run model inference, installation or a Docker build.

Eight private jobs are prepared under
`~/.local/state/py-audio-box/qwen-fidelity-r414/`: four matched raw/spoken pairs
for FR amount, EN amount, FR identifier and an unchanged FR liaison control.
`jobs.json` preserves the r7 runner's job fields; `comparison.json` pins source
file hashes and case metadata. Each pair shares voice, language and seed. The initial preparation performed no ICL or audio generation. Actual subsequent
r414 listening acceptance is recorded privately in `human-listening-result.json`. The liaison pair deliberately has identical
text: numeric-v1 makes no liaison claim. `requests/` can be fed directly to this
preview script. Existing r7 sources and audio are read-only.
