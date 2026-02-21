# tests/local — Local test suite

Individual focused test scripts that exercise every feature of the project,
plus a main orchestrator that runs them all in sequence.

All scripts run from the **repo root** against the Docker container via `./run`.

---

## Quick start

```bash
# Full suite (all 19 scripts, ~4-6 hours on CPU)
make test-local

# Skip slow variant/clone tests (~1-2 hours, good for routine dev checks)
make test-local-fast          # SKIP_SLOW=1 SKIP_DESIGN=1

# CLI utilities only, no synthesis (~5 min)
make test-local-cli

# Synthesis features only (suites 03–09)
make test-local-synth

# CustomVoice features only (suites 10–15)
make test-local-cv

# Run specific suites
ONLY="01 07 09" make test-local

# Skip specific suites
SKIP="15 16 17" make test-local

# Run a single suite directly (no orchestrator)
bash tests/local/03-clone-profiles.sh
```

### With a local audio file (voice-split / voice-register)

Suites 18 and 19 are skipped automatically unless you supply a WAV/MP3 that
is accessible inside the container (i.e. lives under `./work/` on the host):

```bash
# Copy your file into ./work/ first
cp ~/my-recording.wav ./work/

# Then pass the container path
TEST_AUDIO=/work/my-recording.wav make test-local

# Also enable --start/--end trim tests
TEST_AUDIO=/work/my-recording.wav TEST_AUDIO_LONG=1 make test-local
```

---

## Environment variables

| Variable          | Default             | Description                                                 |
| ----------------- | ------------------- | ----------------------------------------------------------- |
| `SKIP_SLOW`       | `0`                 | `1` skips multi-take variant tests in 05 and all of 15, 17  |
| `SKIP_DESIGN`     | `1`                 | `0` enables suite 16 (loads the 1.7B VoiceDesign model)     |
| `ONLY`            | _(all)_             | Space-separated suite numbers to run, e.g. `"01 07"`        |
| `SKIP`            | _(none)_            | Space-separated suite numbers to exclude, e.g. `"18 19"`    |
| `TEST_AUDIO`      | _(unset)_           | Container path to a WAV/MP3; enables suites 18 and 19       |
| `TEST_AUDIO_LONG` | `0`                 | `1` also runs `--start`/`--end` trim sub-tests in 18 and 19 |
| `THREADS`         | `nproc`             | CPU thread count passed to every synthesis call             |
| `DTYPE`           | `float32`           | Model dtype — `float32`, `bfloat16`, or `float16`           |
| `CACHE`           | `/cache`            | Cache directory inside the container                        |
| `OUT`             | `/work/local-tests` | Output directory for all generated audio                    |

---

## Suite index

| #   | Script                   | What it covers                                                  | Slow gate     |
| --- | ------------------------ | --------------------------------------------------------------- | ------------- |
| 01  | `01-cli-utils.sh`        | `list-voices`, `list-speakers`, `capabilities` — no synthesis   | —             |
| 02  | `02-voice-clone.sh`      | `voice-clone` app: self-test, synth, `prepare-ref`              | —             |
| 03  | `03-clone-profiles.sh`   | All bundled clone voices × all three profiles (EN + FR)         | —             |
| 04  | `04-designed-clone.sh`   | Pre-baked designed-clone voice across profiles + QA             | —             |
| 05  | `05-variants.sh`         | `--variants`, `--select-best`, `--qa` scoreboard                | partial       |
| 06  | `06-chunking.sh`         | `--chunk` auto-splits long text into segments                   | —             |
| 07  | `07-text-file.sh`        | `--text-file` reads input from a file                           | —             |
| 08  | `08-save-profile.sh`     | `--save-profile-default` persists per-voice generation defaults | —             |
| 09  | `09-export-import.sh`    | `export-voice` / `import-voice` zip round-trip                  | —             |
| 10  | `10-custom-voice.sh`     | CustomVoice `--speaker` with no style instruction               | —             |
| 11  | `11-instruct.sh`         | CustomVoice `--instruct` free-form style strings                | —             |
| 12  | `12-instruct-style.sh`   | CustomVoice `--instruct-style` named templates                  | —             |
| 13  | `13-register-builtin.sh` | `register-builtin` saves a CustomVoice speaker as a named voice | —             |
| 14  | `14-tones.sh`            | `register-builtin --tone` + `speak --tone` variant selection    | —             |
| 15  | `15-variants-cv.sh`      | CustomVoice `--variants` + `--select-best` + `--qa`             | `SKIP_SLOW`   |
| 16  | `16-design-voice.sh`     | `design-voice` from text description + synthesis                | `SKIP_DESIGN` |
| 17  | `17-slow-clones.sh`      | Slow FR / child clone voices + multi-take                       | `SKIP_SLOW`   |
| 18  | `18-voice-split.sh`      | `voice-split` diarisation + voice registration from local audio | `TEST_AUDIO`  |
| 19  | `19-voice-register.sh`   | `voice-register` one-shot registration from local audio         | `TEST_AUDIO`  |

---

## Suite details

### 01 — CLI utilities

**`01-cli-utils.sh`** | ~5–10 min (first run downloads voice listing) | no synthesis

Tests the four informational sub-commands. Nothing is synthesised — if these
fail, nothing else will work.

| Test                                       | What it checks                                 |
| ------------------------------------------ | ---------------------------------------------- |
| `list-voices` (human)                      | Readable voice list is non-empty               |
| `list-voices --json`                       | JSON output is valid and contains known voices |
| `capabilities --json --skip-speaker-probe` | Reports model names and supported flags        |
| `list-speakers --json`                     | CustomVoice speaker list returned as JSON      |

**Update when:** a new voice is added, a model is renamed, or a top-level CLI
flag is added or removed.

---

### 02 — voice-clone pipeline

**`02-voice-clone.sh`** | ~15–20 min | no synthesis for self-test; 2 synth calls

Exercises the standalone `voice-clone` app: its self-test, direct synthesis
for two voices (one EN, one FR), and `prepare-ref` which pre-processes a
reference audio into the cache.

| Test                                 | What it checks                                |
| ------------------------------------ | --------------------------------------------- |
| `voice-clone self-test`              | App imports and prints help cleanly           |
| `synth: chalamet-en / balanced / EN` | Clone-only synthesis path works end-to-end    |
| `synth: rascar-capac / stable / FR`  | FR voice over the clone engine                |
| `prepare-ref: chalamet-en`           | Reference audio preprocessing writes to cache |

> **Note:** `prepare-ref` takes `--voice`, not `--out`. It has no output path
> flag — results go to `--cache`.

**Update when:** the `voice-clone` app signature changes, a new `prepare-ref`
flag is added, or the set of bundled reference voices changes.

---

### 03 — clone profiles

**`03-clone-profiles.sh`** | ~30–60 min | 7 synthesis calls

Smoke-tests all three generation profiles (stable / balanced / expressive)
across the main bundled clone voices, covering both EN and FR. No QA — just
verifies each combination produces audio without error.

| Voice                | Profile                    | Language |
| -------------------- | -------------------------- | -------- |
| `david-attenborough` | stable                     | EN       |
| `chalamet-en`        | expressive                 | EN       |
| `chalamet-en-2`      | balanced                   | EN       |
| `hikaru-nakamura`    | balanced + `--temperature` | EN       |
| `rascar-capac`       | stable                     | FR       |
| `blitzstream`        | balanced                   | FR       |
| `blitzstream-2`      | expressive                 | FR       |

**Update when:** a profile is added/removed, a bundled voice is added or
retired, or `--temperature` flag semantics change.

---

### 04 — designed clone

**`04-designed-clone.sh`** | ~15–25 min | 3 synthesis calls

Tests the `smoke-designed` voice (a pre-baked designed clone checked into the
registry) across all three profiles, including one run with `--qa` to verify
the Whisper-based intelligibility scoring pipeline works end-to-end.

| Test                                  | What it checks                                     |
| ------------------------------------- | -------------------------------------------------- |
| `smoke-designed / stable / EN`        | Designed-clone engine works                        |
| `smoke-designed / expressive / EN`    | Expressive profile for designed clones             |
| `smoke-designed / balanced / qa / EN` | `--qa` runs Whisper, reports intelligibility ≥ 80% |

**Update when:** the designed-clone engine changes, `--qa` thresholds change,
or the Whisper model used for QA is swapped.

---

### 05 — variants + QA

**`05-variants.sh`** | ~20–40 min | 3 always-on + 1 behind `SKIP_SLOW`

Tests multi-take generation (`--variants N`) and the automatic best-take
selection / QA scoreboard features.

| Test                                                     | Gate          | What it checks                         |
| -------------------------------------------------------- | ------------- | -------------------------------------- |
| `chalamet-en / expressive / 3-variants select-best`      | always        | Generates 3 takes, picks best by score |
| `david-attenborough / stable / 2-variants qa scoreboard` | always        | 2 takes + Whisper QA in scoreboard     |
| `chalamet-en-2 / balanced / qa single-take`              | always        | `--qa` works even with a single take   |
| `rascar-capac / 2-variants select-best / FR`             | `SKIP_SLOW=0` | Multi-take round-trip for FR voice     |

**Update when:** `--variants`, `--select-best`, or `--qa` flag semantics
change, or scoring thresholds are adjusted.

---

### 06 — chunking

**`06-chunking.sh`** | ~35–50 min | 3 synthesis calls (multi-segment each)

Tests `--chunk`, which splits long input text at sentence boundaries and
concatenates the resulting audio segments.

| Test                                             | What it checks                                    |
| ------------------------------------------------ | ------------------------------------------------- |
| `david-attenborough / stable / --chunk / EN`     | 6-segment EN text stitched correctly              |
| `rascar-capac / stable / --chunk / FR`           | 5-segment FR text                                 |
| `chalamet-en / expressive / --chunk + --qa / EN` | Chunked output still scores ≥ 80% intelligibility |

**Update when:** the chunking algorithm or sentence-split logic changes, or
the maximum tokens-per-chunk is adjusted.

---

### 07 — text-file input

**`07-text-file.sh`** | ~15–25 min | 3 synthesis calls

Verifies that `--text-file <path>` reads synthesis input from a file instead
of an inline `--text` argument.

| Test                                                       | What it checks                              |
| ---------------------------------------------------------- | ------------------------------------------- |
| `chalamet-en-2 / balanced / --text-file / EN`              | File content is read and synthesised        |
| `rascar-capac / stable / --text-file / FR`                 | Works with FR content                       |
| `david-attenborough / stable / --text-file + --chunk / EN` | `--text-file` and `--chunk` can be combined |

> **Container path note:** temp files are written to `./work/` on the host
> (maps to `/work/` in the container). Host `/tmp` is not mounted. CLI args
> use `/work/` paths, not `/tmp/`.

**Update when:** `--text-file` path resolution or character encoding handling
changes.

---

### 08 — save-profile-default

**`08-save-profile.sh`** | ~15–20 min | 4 synthesis calls (2 save + 2 verify)

Tests `--save-profile-default`, which persists a generation profile as the
voice-specific stored default so subsequent calls don't need `--profile`.

| Test                                                               | What it checks                                |
| ------------------------------------------------------------------ | --------------------------------------------- |
| `blitzstream-2 / expressive / --save-profile-default / FR (save)`  | Profile written to voice config               |
| `blitzstream-2 / no --profile / FR`                                | Omitting `--profile` uses the saved one       |
| `david-attenborough / stable / --save-profile-default / EN (save)` | Same for an EN voice                          |
| `david-attenborough / no --profile / EN`                           | Saved default respected without explicit flag |

**Update when:** voice config persistence format changes or
`--save-profile-default` is renamed.

---

### 09 — export / import round-trip

**`09-export-import.sh`** | ~10–15 min | 2 synth + 2 export + 2 import calls

Exports a voice to a `.zip`, re-imports it with `--force`, then synthesises
with it to confirm the round-trip is lossless. Covers two voices (one EN,
one FR).

| Test                                           | What it checks                              |
| ---------------------------------------------- | ------------------------------------------- |
| `export-voice chalamet-en --dest /work/...zip` | Zip archive written to `/work/`             |
| `import-voice --zip ... --force`               | Zip re-imported, existing voice overwritten |
| `chalamet-en post round-trip / stable / EN`    | Synthesis still works after import          |
| Same sequence for `rascar-capac` (FR)          | Round-trip works for FR voice too           |

> **Key CLI detail:** `export-voice` takes the voice slug as a **positional**
> argument, not `--voice`:
>
> ```bash
> ./run voice-synth export-voice chalamet-en --dest /work/out.zip
> ```

**Update when:** the export zip format or metadata schema changes, or
`--force` is renamed.

---

### 10 — CustomVoice (no instruct)

**`10-custom-voice.sh`** | ~10–15 min | 3 synthesis calls

Tests the CustomVoice model (`--speaker`) without any style instruction. Verifies
that `list-speakers` works and all three standard profiles produce output.

| Test                   | What it checks                        |
| ---------------------- | ------------------------------------- |
| `list-speakers --json` | Returns built-in speaker list as JSON |
| `Ryan / stable`        | Baseline CustomVoice synthesis        |
| `Ryan / balanced`      | Balanced profile                      |
| `Ryan / expressive`    | Expressive profile                    |

**Update when:** new speakers are added to the CustomVoice model, or the
`--speaker` flag is renamed.

---

### 11 — CustomVoice `--instruct`

**`11-instruct.sh`** | ~15–25 min | 5 synthesis calls

Tests free-form style instructions passed at synthesis time via `--instruct`.
Covers a range of instruction registers to confirm the model accepts them
without error across different profiles.

| Instruction style                        | Profile    |
| ---------------------------------------- | ---------- |
| Podcast host (warm, energetic)           | expressive |
| Technical presenter (precise, no filler) | stable     |
| Audiobook narrator (slow, dramatic)      | stable     |
| Sports commentator (high energy)         | expressive |
| Meditation guide (calm, soothing)        | balanced   |

**Update when:** `--instruct` argument processing changes, or instruction
text goes through a template/formatting step.

---

### 12 — CustomVoice `--instruct-style`

**`12-instruct-style.sh`** | ~15–25 min | 5 synthesis calls (4 pass + 1 intentional fail)

Tests the named style template shorthand `--instruct-style <name>`, which
expands to a preset instruction string baked into the app.

| Template      | Expected result                                            |
| ------------- | ---------------------------------------------------------- |
| `serious_doc` | PASS — expands to documentary narration instruction        |
| `warm`        | PASS — expands to warm/friendly instruction                |
| `excited`     | PASS — expands to high-energy instruction                  |
| `audiobook`   | PASS — expands to audiobook narration instruction          |
| `formal`      | PASS — expands to formal/professional delivery instruction |

Full list of available templates: `audiobook`, `calm`, `energetic`, `excited`,
`formal`, `melancholic`, `nature_doc`, `sad`, `serious_doc`, `warm`, `whisper`.

**Update when:** a new style template is added — add it here. If a template
is removed, remove or update its test case.

---

### 13 — register-builtin

**`13-register-builtin.sh`** | ~20–30 min | 7 calls (3 register + 4 synth)

Tests `register-builtin`, which saves a CustomVoice speaker under a persistent
voice name (usable with `--voice` like a clone). Covers `--instruct-default`
(baked instruction string), `--instruct-default-style` (baked template), and
per-call override priority.

| Test                                                           | What it checks                                        |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| Register `test-ryan` (no default)                              | Registered without any instruction                    |
| Speak `test-ryan`, no instruct                                 | No instruction applied                                |
| Register `test-ryan` with `--instruct-default "..."`           | Custom string persisted to voice config               |
| Speak without `--instruct`                                     | Saved default applied automatically                   |
| Speak with `--instruct-style excited`                          | Per-call override takes precedence over saved default |
| Register `test-ryan-warm` with `--instruct-default-style warm` | Template-based default                                |
| Speak `test-ryan-warm` without override                        | Template default applied                              |

**Update when:** `register-builtin` flag names change, or the override
priority between per-call and saved defaults is altered.

---

### 14 — tones

**`14-tones.sh`** | ~20–30 min | 4 register + 4 synth

Tests `register-builtin --tone <name>` to create tone variants of the same
base voice, and `speak --tone <name>` to select among them at generation time.

| Test                                  | What it checks           |
| ------------------------------------- | ------------------------ |
| Register base `test-ryan`             | Base voice exists        |
| Register `test-ryan --tone calm`      | Tone variant registered  |
| Register `test-ryan --tone energetic` | Second tone variant      |
| Register `test-ryan --tone serious`   | Third tone variant       |
| Speak `--tone calm / stable`          | Correct variant selected |
| Speak `--tone energetic / expressive` | Correct variant selected |
| Speak `--tone serious / stable`       | Correct variant selected |
| Speak no `--tone / balanced`          | Falls back to base voice |

**Update when:** `--tone` registration or selection logic changes, or tones
are stored differently in the voice registry format.

---

### 15 — CustomVoice variants + select-best (slow)

**`15-variants-cv.sh`** | ~30–60 min | all tests gated behind `SKIP_SLOW=0`

Combines multi-take generation with CustomVoice speakers, covering `--tone`
variants, `--instruct`, and `--qa` scoring. Gated because each test generates
2–3 takes.

| Test                                                        | What it checks                             |
| ----------------------------------------------------------- | ------------------------------------------ |
| `test-ryan --tone calm / 2-variants select-best / balanced` | Tone-aware multi-take selection            |
| `Ryan / 3-variants select-best + --instruct / EN`           | Instruct + multi-take combined             |
| `Ryan / 2-variants qa scoreboard / EN`                      | QA scoreboard in the CustomVoice code path |

**Update when:** `--variants` or `--select-best` behaviour changes in the
CustomVoice code path.

---

### 16 — design-voice (slow)

**`16-design-voice.sh`** | ~20–40 min | all tests gated behind `SKIP_DESIGN=0`

Tests `design-voice`, which uses the 1.7B VoiceDesign model to generate a
brand-new synthetic voice from a text description, then speaks with it.

| Test                                                | What it checks                                       |
| --------------------------------------------------- | ---------------------------------------------------- |
| `design-voice: test-designed-broadcaster`           | Voice generated from description, stored in registry |
| Speak `test-designed-broadcaster / stable / EN`     | Designed voice works for synthesis                   |
| Speak `test-designed-broadcaster / expressive / EN` | Works across profiles                                |
| `design-voice: test-designed-narrator`              | A second design verifies multi-voice support         |
| Speak `test-designed-narrator / balanced / EN`      | Second designed voice works                          |

**Update when:** `design-voice` CLI flags change, the VoiceDesign model is
updated, or the voice file format it produces changes.

---

### 17 — slow clone voices (FR)

**`17-slow-clones.sh`** | ~30–60 min | all tests gated behind `SKIP_SLOW=0`

Exercises FR voices with longer reference audio that takes substantially more
time than standard EN clones. Includes a multi-take test for one FR voice.

| Test                                           | What it checks                     |
| ---------------------------------------------- | ---------------------------------- |
| `tiktok-child-fr-1 / stable / FR`              | Child FR voice synthesises cleanly |
| `jonathan-cohen / expressive / FR`             | Actor FR voice, expressive profile |
| `chalamet-fr / stable / FR`                    | FR version of the chalamet clone   |
| `jonathan-cohen / 2-variants select-best / FR` | Multi-take works for FR voices     |

**Update when:** slow FR voices are added or removed, or reference audio
processing improves enough to move these into the always-on suite.

---

### 18 — voice-split

**`18-voice-split.sh`** | ~10–15 min | requires `TEST_AUDIO=/work/file.wav`

Tests the `voice-split` app: diarises a local audio file, segments it, and
optionally registers a segment as a named clone voice. All tests skip
automatically when `TEST_AUDIO` is unset.

| Test                                                 | Gate                | What it checks                                 |
| ---------------------------------------------------- | ------------------- | ---------------------------------------------- |
| `voice-split 2 clips 15s (no voice-name)`            | `TEST_AUDIO` set    | Segments audio, returns clips, no registration |
| `voice-split 1 clip + --voice-name test-split-voice` | `TEST_AUDIO` set    | Single segment registered as a named voice     |
| Speak `test-split-voice / stable / EN`               | `TEST_AUDIO` set    | Registered voice synthesises cleanly           |
| `voice-split trimmed segment 0:30-2:00`              | `TEST_AUDIO_LONG=1` | `--start`/`--end` trim flags work              |

**Update when:** `voice-split` segmentation parameters change, `--voice-name`
flag is added/removed, or the registration format changes.

---

### 19 — voice-register

**`19-voice-register.sh`** | ~10–15 min | requires `TEST_AUDIO=/work/file.wav`

Tests the `voice-register` app: registers an existing local audio file directly
as a clone voice (no segmentation). All tests skip when `TEST_AUDIO` is unset.

| Test                                            | Gate                | What it checks                 |
| ----------------------------------------------- | ------------------- | ------------------------------ |
| `voice-register test-registered-voice`          | `TEST_AUDIO` set    | File registered as named voice |
| Speak `test-registered-voice / stable / EN`     | `TEST_AUDIO` set    | Registered voice works for EN  |
| Speak `test-registered-voice / expressive / EN` | `TEST_AUDIO` set    | Expressive profile also works  |
| `voice-register with --start/--end trim`        | `TEST_AUDIO_LONG=1` | Trim args accepted and applied |

**Update when:** `voice-register` CLI changes, the voice metadata format
changes, or trim flag names are updated.

---

## Shared library (`lib/common.sh`)

Sourced by every suite. Provides:

- **`speak LABEL [flags...]`** — wraps `./run voice-synth speak` with
  `--cache`, `--threads`, `--dtype`, `--out` injected automatically from env
- **`run_cmd LABEL CMD [args...]`** — runs any arbitrary command, tracks timing
  and pass/fail
- **`skip_test LABEL`** — records a skip without running anything
- **`print_summary [SUITE_LABEL]`** — prints the results table and exits 0/1

All three counters (`pass`, `fail`, `skip`) are local to each script instance.

## Orchestrator (`run-all.sh`)

Runs each suite as a sub-process and continues regardless of individual suite
failures. Prints a per-suite PASS/FAIL banner and a final summary. Exits 0
only if every executed suite passed.

---

## Adding a new test suite

1. Create `tests/local/NN-my-feature.sh`, using an existing script as a template.
2. Source the shared lib: `source tests/local/lib/common.sh`
3. Use `speak` for `voice-synth speak` calls; `run_cmd` for everything else.
4. End with `print_summary "NN-my-feature"`.
5. Add the filename to the `SUITES` array in `run-all.sh`.
6. Add a `make test-local-NN` Makefile target if the suite is a useful
   standalone group.
7. Add a section to this README.

## Updating tests after a feature change

- **CLI flag renamed:** `grep -r '<old-flag>' tests/local/` to find all
  occurrences, update them, then `bash -n <file>` to confirm no syntax errors
  before re-running.
- **New bundled clone voice:** add a `speak` call to `03-clone-profiles.sh`.
- **New CustomVoice speaker:** add a `speak` call to `10-custom-voice.sh`.
- **Voice removed from registry:** remove or comment out its `speak` call.
  The suite will fail with "voice not found" if a missing voice is referenced.
- **New `--instruct-style` template:** add a test case to `12-instruct-style.sh`
  and update the table in this README.
- **Profile added or removed:** update `03-clone-profiles.sh` and
  `04-designed-clone.sh`.
- **Export/import format changed:** update `09-export-import.sh`; remember
  `export-voice` takes the voice slug as a positional arg, not `--voice`.

---

## Platform notes

**macOS `mktemp`:** the template X's must appear at the very end of the string.
An extension suffix like `.txt` after the X's causes `mkstemp failed ... File exists`.
Suite 07 avoids this by writing to `./work/` with PID-based names rather than
using `mktemp`.

**Container volume mounts:** only `./work → /work` and `./cache → /cache` are
mounted. Host `/tmp` is not visible inside the container. Any test that writes
temp files must use `./work/` on the host and pass `/work/` paths to CLI args.

**CPU runtime:** synthesis runs at ~15–20× real-time on CPU (a 5 s clip takes
~80–100 s). Full suite runtime is 4–6 hours on a modern Mac. Use
`make test-local-fast` (`SKIP_SLOW=1 SKIP_DESIGN=1`) for routine development
checks.
