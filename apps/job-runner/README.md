# job-runner

`job-runner` is the producer/validation CLI for the Redis stream-backed TTS queue.
It is the primary interface for planning, enqueueing, compiling, executing, and
inspecting synthesis jobs. Full architecture details live in
[docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md](../../docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md).

## Architecture

```
                     ┌──────────────┐
  YAML / beatsheet ──►  job-runner  ├──── plan / compile / run  (local, no Redis)
                     │   (CLI)      ├──── enqueue / report      (writes to Redis stream)
                     └──────┬───────┘
                            │
                  ┌─────────▼──────────┐
                  │   Redis Stream      │  pab:jobs:stream
                  │   (consumer group)  │  pab:jobs:result / failed
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │   job-watcher       │  leases → provisions Vast → synth → upload → ack
                  │   (VPS daemon)      │
                  └────────────────────┘
```

## Quick Start

```bash
# Validate YAML + voice availability + preview text load
./run job-runner plan /app/jobs.example.yaml

# Idempotent enqueue by request_id
./run job-runner enqueue /app/jobs.example.yaml

# Queue and result inspection
./run job-runner status --json
./run job-runner history
./run job-runner result topic:beat-001 --json

# Quick dashboard (status + recent history)
make jobs-dashboard
```

## Command Reference

### Offline Commands (no Redis required)

| Command | Description |
|---------|-------------|
| `plan YAML` | Validate schema, check voice availability, preview jobs table |
| `compile YAML --out DIR` | Materialize text files + deterministic `manifest.json` |
| `run YAML [--dry-run]` | Compile + execute locally (validation path) |
| `execute-manifest MANIFEST` | Execute a pre-compiled `manifest.json` |

### Redis Commands

| Command | Description |
|---------|-------------|
| `enqueue YAML` | Idempotent enqueue from YAML to Redis stream |
| `enqueue-beatsheet JSON --voice V` | Enqueue from beatsheet JSON (producer bridge) |
| `status [--json]` | Queue depth, pending, stale PEL, lock state, watcher health |
| `history [--limit N] [--status done\|failed\|all]` | Recent job history (most recent first) |
| `result REQUEST_ID` | Fetch done/failed record for a specific request |
| `report YAML` | Cross-reference all jobs in YAML against Redis state |
| `retry REQUEST_ID [--yes]` | Requeue a terminal-failed job for another attempt |
| `flush [--hard] [--yes]` | Delete transient queue keys (--hard also clears results) |

Every Redis command accepts `--redis-url` and `--json` flags.

## Makefile Shortcuts

```bash
make jobs-plan FILE=/app/jobs.example.yaml   # plan
make jobs-enqueue FILE=/app/jobs.example.yaml # enqueue
make jobs-status                              # queue status
make jobs-status ARGS='--json'                # JSON queue status
make jobs-history                             # recent history
make jobs-history ARGS='--status failed'      # failed only
make jobs-result ID=topic:beat-001            # single result
make jobs-report FILE=/app/jobs.example.yaml  # full report
make jobs-retry ID=topic:beat-001 ARGS='--yes'# retry failed job
make jobs-flush                               # soft flush
make jobs-flush ARGS='--hard --yes'           # hard flush
make jobs-dashboard                           # status + history
make jobs-logs                                # latest execution.json
```

## YAML Schema

`jobs.example.yaml` shows the canonical format:

- Root can be either:
  - `jobs: [...]` with optional `defaults: {...}`
  - a top-level list of job objects
  - a single job object
- Each job is normalized through `lib/jobqueue.normalize_job_spec`
- If `request_id` is missing, it is derived from `output_name` (`topic/beat-001` → `topic:beat-001`)
- `output_name` must be relative (no leading `/`, no `..`, no trailing `/`)

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

Default-level `callback_data` is deep-merged with per-job `callback_data`.

## Compile Manifest Contract

`compile` writes:

- Staged text files at `<out>/<output_name>/text.txt`
- Deterministic manifest at `<out>/manifest.json`
- Each job argv includes:
  - `voice-synth speak`
  - `--text-file <.../text.txt>`
  - `--out-exact <.../<output_name>>`
  - `--json-result <.../<output_name>/result.json>`

The manifest can be executed with:

```bash
./run job-runner execute-manifest /work/job-batch/manifest.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection URL |
| `WATCHER_LEASE_IDLE_MS` | `180000` | PEL idle threshold for stale detection (ms) |

All watcher-specific env vars are documented in `.env.example`.

## Operational Playbook

### Check if the system is healthy

```bash
make jobs-dashboard
# or
./run job-runner status
```

Key indicators:
- `outstanding > 0` with `watcher_last_seen` recent → working normally
- `stale_pel > 0` → watcher may have crashed; entries will be auto-reclaimed
- `lock_owner: -` with `outstanding > 0` → watcher not running

### Investigate a failed job

```bash
./run job-runner result topic:beat-001 --json
```

### Retry a specific failed job

```bash
./run job-runner retry topic:beat-001 --yes
```

### Reset the entire queue (development only)

```bash
./run job-runner flush --hard --yes
```

### Run a batch locally for validation

```bash
./run job-runner run /app/jobs.example.yaml --dry-run --json
```
