# Optional CPU synthesis

Qwen keeps its existing commands and default. This separate CLI loads only the
chosen optional runtime; Kokoro uses presets, Pocket uses an authorized WAV clone
reference. French and English are selected explicitly. Unsupported controls fail.

Docker-only Kokoro example:

```sh
./run-cpu kokoro build
./run-cpu kokoro provision --language fr --dest /cache/tts/kokoro
./run-cpu kokoro speak --assets /cache/tts/kokoro --language fr \
  --preset ff_siwis --text "Bienvenue dans cette présentation." \
  --output /work/narration.wav --threads 2 --json
./run-cpu kokoro export --input /work/narration.wav \
  --output /work/narration-48k.wav --rate 48000 --channels 1
```

`speak`, `benchmark` and `export` run with `--network none`; only `provision` enables
networking. Synthesis assets are mounted read-only. Existing takes are preserved.
Provisioning requires a fresh directory and verifies the pinned Kokoro file hashes.
Each WAV has a JSON sidecar with exact bundle hashes, parameters and timing.

For Pocket, first obtain legitimate access to `kyutai/pocket-tts` as the account
owner. The wrapper forwards an existing `HF_TOKEN` only for provisioning, without
embedding it in an image. It never accepts access conditions on your behalf.

```sh
./run-cpu pocket-tts build
./run-cpu pocket-tts provision --language fr --dest /cache/tts/pocket-fr
./run-cpu pocket-tts speak --assets /cache/tts/pocket-fr --language fr \
  --reference /work/authorized-reference.wav --text "Bonjour." \
  --output /work/pocket-fr.wav --seed 1729 --threads 2 --json
```

Provision each Pocket language separately. The pinned package supplies immutable
Hub revisions; generated configs use portable bundle paths. External pickle caches
are never used. The default inference deadline is 600 seconds including model load;
`--timeout` changes it explicitly. A timeout terminates and reaps the worker. For Pocket, the parent owns a private
0700 temporary namespace for conditioning snapshots and model-load config. It
removes that namespace after confirmed worker reap on success, error or timeout;
it preserves it if worker reap fails. This applies to both `speak` and `batch`.

The metadata's `offline_requested` records library settings, while network isolation
is established by the Docker launch. The CLI alone does not certify its own Docker
network configuration. Naturalness and speaker identity require listening; successful
WAV generation or ASR scores do not establish acoustic acceptance.

See [the benchmark protocol](../../docs/TTS_BENCHMARK.md). Dependency locks are
separate under `requirements/`; regenerate intentionally with `uv pip compile
--python-version 3.11 --python-platform x86_64-unknown-linux-gnu --generate-hashes`
and the matching `.in` file. Pocket needs the CPU PyTorch index. Rebuild and recheck
offline synthesis whenever a lock, phonemizer, voice asset or checkpoint changes.

## Reuse one model for a batch

`batch` runs sequential jobs for one explicitly selected backend, language and
preset (or approved Pocket reference). It keeps that model in a supervised spawned
worker; this removes repeated model initialization between successful takes.
It does not create a general service or change the `speak` command.

Create `/work/jobs-fr.json` with a JSON list:

```json
[
  {"id": "intro", "text": "Bienvenue dans cette présentation.", "output": "/work/intro.wav"},
  {"id": "budget", "text": "Le budget est de 1 250,50 euros.", "output": "/work/budget.wav", "timeout": 120}
]
```

```sh
./run-cpu kokoro batch --assets /cache/tts/kokoro --language fr \
  --preset ff_siwis --threads 2 --jobs /work/jobs-fr.json \
  --report /work/batch-fr.json --timeout 600 --json
```

All jobs, IDs, output paths and report collisions are validated before any model
loads. Every job requires `id`, exact `text` and `.wav` `output`; its optional
positive finite `timeout` overrides the top-level deadline. Relative output paths
are relative to the jobs JSON directory. IDs and artifact paths must be unique;
existing WAVs, sidecars and reports are never overwritten. At most 1,000 jobs and 16 MiB of jobs JSON are
accepted. Batch supports no silent fallback or per-job language/voice changes.
Use separate batches for French and English or different voices.

The timeout applies separately to each job. The first deadline includes process
startup and backend initialization; subsequent deadlines apply to the reused model.
A timeout, inference error or per-job output preparation error discards and reaps
the worker before another job can
start. Default behavior stops and records remaining jobs as skipped.
`--continue-on-error` records failures and starts a fresh worker for the next job,
which is consequently cold. The parent alone publishes final audio/metadata after
a timely completed result; timed-out staging files are cleaned after worker reap.
A previous successful take remains preserved even if a later job fails.

Each sidecar and report retain exact text, job ID, preset/reference identity, asset
manifest and worker generation. For Pocket, the authorized reference hash is checked
before and after model initialization, and the returned conditioning hash must
match before publication. The adapter conditions on a private byte snapshot; a
changed or divergent reference is rejected. Snapshot/conditioning costs remain
in the backend metadata. First successful jobs report
`cold_model`; subsequent jobs on the same worker report `warm_model`.
`backend_initialization_seconds` / per-job `load_seconds` include backend imports,
asset checks and initialization on cold jobs and are zero on warm jobs.
Backend `metadata.model_load_seconds`, when supplied, describes the one-time model
session load and is distinct from those broader initialization costs.
`generation_seconds`, audio `export_seconds` and worker `total_seconds` describe
those measured intervals; report-only `job_wall_seconds` additionally includes parent
startup, IPC and publication. Sidecar/report `parent_seconds_before_publication`
share the measured interval through result receipt; final publication time appears
only in the report wall interval. Report `batch_wall_seconds` includes plan validation,
all job processing, restarts and final worker cleanup before report publication. Actual installed runtime versions are collected
once per worker and retained separately from the provisioning manifest versions. Lower `rtf` is faster (seconds / audio seconds).
This is actual model reuse, not a renamed cold-process repeat. Naturalness, text
fidelity and voice identity still require listening.

Kokoro disables ONNX CPU arena reserves; dedicated CPU images use a fixed glibc
mmap threshold to return large freed buffers. This saves RAM with some allocation
overhead. Weights remain FP32. Completed audio is released between
jobs while the model stays loaded. See [memory measurements and their scope](../../docs/TTS_MEMORY.md).
