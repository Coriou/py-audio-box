# job-runner

`job-runner` is the producer/validation CLI for the Redis stream-backed queue described in `docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md`.

Core commands:

```bash
# Validate YAML + voice availability + preview text load
./run job-runner plan /app/jobs.example.yaml

# Idempotent enqueue by request_id
./run job-runner enqueue /app/jobs.example.yaml

# Producer bridge from beatsheet JSON
./run job-runner enqueue-beatsheet /work/beatsheet.json --voice newsroom

# Compile deterministic manifest and staged text files
./run job-runner compile /app/jobs.example.yaml --out /work/job-batch

# Local validation run (direct python, not nested ./run)
./run job-runner run /app/jobs.example.yaml --out /work/job-batch

# Queue and result inspection
./run job-runner status --json
./run job-runner history                         # last 20 done/failed (most recent first)
./run job-runner history --status failed --limit 50
./run job-runner result topic:beat-001 --json
./run job-runner report /app/jobs.example.yaml --json

# Operational controls
./run job-runner retry topic:beat-001
./run job-runner flush
```

## YAML schema

`jobs.example.yaml` shows the canonical format:

- root can be either:
  - `jobs: [...]` with optional `defaults: {...}`
  - a top-level list of job objects
- each job is normalized through `lib/jobqueue.normalize_job_spec`
- if `request_id` is missing, it is derived from `output_name` (`topic/beat-001` -> `topic:beat-001`)
- `output_name` must be relative (no leading `/`, no `..`, no trailing `/`)

Example:

```yaml
defaults:
  language: English
  profile: balanced
  variants: 1
  select_best: false
  chunk: false

jobs:
  - output_name: topic/beat-001
    voice: newsroom
    text: "First line"

  - output_name: topic/beat-002
    speaker: Ryan
    voice: null
    text: "Second line"
```

## Compile manifest contract

`compile` writes:

- staged text files at `<out>/<output_name>/text.txt`
- deterministic manifest at `<out>/manifest.json`
- each job argv includes:
  - `voice-synth speak`
  - `--text-file <.../text.txt>`
  - `--out-exact <.../<output_name>>`
  - `--json-result <.../<output_name>/result.json>`

The manifest can be executed with:

```bash
./run job-runner execute-manifest /work/job-batch/manifest.json
```
