# job-watcher

`job-watcher` is the daemon that consumes Redis stream leases, provisions Vast, executes manifests remotely, uploads artifacts, and writes final Redis status records.

## Required environment

- `REDIS_URL`
- `VAST_API_KEY`
- `VAST_SSH_KEY`
- `S3_BUCKET`
- `RCLONE_REMOTE`

Optional defaults follow `docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md` (`WATCHER_*`, `S3_PREFIX`, `VAST_*`, `GHCR_*`).

## Run

```bash
# One scheduling cycle
python apps/job-watcher/job-watcher.py --once

# Daemon mode
python apps/job-watcher/job-watcher.py
```

Using Docker Compose:

```bash
docker compose -f docker-compose.watcher.yml up -d watcher
docker compose -f docker-compose.watcher.yml logs -f watcher
```

## Behavior summary

- reclaims stale leases with `XAUTOCLAIM`
- moves due retry payloads from `pab:jobs:retry:zset` back to stream
- acquires renewable distributed lock (`pab:watcher:lock`)
- provisions one Vast instance per run and drains multiple batches
- compiles deterministic manifests (`voice-synth speak --out-exact --json-result`)
- uploads to `s3://<bucket>/<prefix>/<output_name>/` via `rclone`
- writes done/failed records and ACKs stream entries
