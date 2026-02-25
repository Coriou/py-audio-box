# Production Job Runner Implementation Plan
> Reviewed and rewritten on February 22, 2026 for reliability, automation, and Vast.ai cost efficiency.
> Implement in order. Do not skip validation gates.

---

## 0) Review Outcome (what changed and why)

The previous draft had several production risks that would materially hurt reliability or cost:

1. Destructive queue pop (`RPOP`) could lose jobs on watcher crash before requeue.
2. Fixed lock TTL without renewal could expire during long runs and allow double provisioning.
3. `job_id` mixed dedupe and request identity, which can drop callback context for repeated text.
4. Current `voice-synth speak` output layout is timestamped, not deterministic per `output_name`.
5. One-instance-per-batch flow wastes money when jobs keep arriving during an active run.
6. No first-class retry strategy (transient vs permanent failure) or dead-letter policy.
7. No explicit OOM policy for parallel synthesis experiments.

This plan addresses all of the above.

---

## 1) Success Criteria

### Reliability SLOs

- No silent job loss under process crash, container restart, or VPS reboot.
- At-least-once processing with idempotent completion semantics.
- 99.9% of accepted jobs end in `done` or `failed` (never stuck) within retry budget.

### Cost/Performance SLOs

- Keep GPU instance idle time under `WATCHER_IDLE_GRACE` (default 75s).
- Minimize provision churn by draining additional queued jobs before teardown.
- Default synthesis concurrency is safe (`1`) and never crashes GPU with OOM storms.

### Automation SLOs

- Producer only needs Redis access plus schema contract.
- Full queue lifecycle is machine-readable (`status`, `result`, metrics).
- No manual steps required during normal operation.

---

## 2) Final Architecture (v1)

```
beatsheet project
  -> enqueue (Redis Stream)
job-watcher daemon (VPS, Docker)
  -> leases jobs from stream (consumer group)
  -> compiles deterministic task batch
  -> provisions Vast instance
  -> runs synthesis sequentially (safe mode)
  -> uploads outputs to Spaces
  -> writes result/failure records
  -> ACKs stream entries
```

Key choice: use Redis Streams + consumer group, not plain list pop.  
Reason: stream pending entries are recoverable after crashes via `XAUTOCLAIM`.

---

## 3) Data Model and Queue Semantics

## 3.1 Identity model

Use two IDs:

- `request_id`: unique job identity from producer (example: `why-pineapple...:beat-001`)
- `content_hash`: deterministic hash of synthesis-defining inputs (audio-equivalence key)

Why split IDs:

- `request_id` tracks business request lifecycle and callback metadata.
- `content_hash` enables optional cache/reuse logic later without losing request-level state.

## 3.2 Job schema (`lib/jobqueue.py`)

```python
@dataclass
class JobSpec:
    request_id: str
    queued_at: str                 # ISO-8601 UTC

    # synthesis source (exactly one required)
    voice: str | None
    speaker: str | None

    text: str
    language: str | None
    tone: str | None
    instruct: str | None
    instruct_style: str | None
    profile: str | None
    variants: int                  # >= 1
    select_best: bool
    chunk: bool

    temperature: float | None
    top_p: float | None
    repetition_penalty: float | None
    max_new_tokens: int | None

    output_name: str               # stable path key, e.g. topic/beat-001
    callback_data: dict

    # system
    content_hash: str              # "sha256:..."
    attempt: int                   # starts at 0
```

## 3.3 Hashing (`make_content_hash`)

Hash only audio-defining fields:

`voice, speaker, text, language, tone, instruct, instruct_style, profile, variants, select_best, chunk, temperature, top_p, repetition_penalty, max_new_tokens`

Do not hash: `request_id`, `queued_at`, `output_name`, `callback_data`, `attempt`.

## 3.4 Redis keys

```
pab:jobs:stream                    # XADD queue
pab:jobs:group                     # consumer group: "watchers"
pab:jobs:result                    # HSET request_id -> result JSON
pab:jobs:failed                    # HSET request_id -> failure JSON
pab:jobs:req:index                 # HSET request_id -> state marker (idempotent enqueue)
pab:jobs:first_queued              # oldest queued ts hint
pab:jobs:retry:zset                # retry schedule (score=unix_ts)
pab:watcher:lock                   # distributed lock value=run_id (renewed)
pab:watcher:heartbeat              # watcher heartbeat
pab:watcher:instance               # current Vast instance id
pab:metrics:*                      # counters/timers
```

## 3.5 Enqueue contract

`enqueue(job)` behavior:

1. Validate schema and normalize defaults.
2. Compute `content_hash`.
3. Atomically reject duplicate `request_id` enqueue if already indexed.
4. `XADD` to stream with serialized job payload.
5. `SETNX pab:jobs:first_queued`.

Idempotency rule: duplicate `request_id` does not create a second stream record.

## 3.6 Processing semantics

- Worker reads with `XREADGROUP`.
- Job is considered leased while in pending entries list (PEL).
- Job is ACKed only after `done` or terminal `failed` record is written.
- Crash recovery: new watcher calls `XAUTOCLAIM` for stale PEL entries.

This removes the job-loss window present in list pop designs.

## 3.7 Retry policy

Classify errors:

- `transient`: infrastructure/network/SSH/rclone/remote host failure
- `permanent`: invalid voice, bad schema, unsupported speaker, empty text

Rules:

- `transient`: exponential backoff (`30s`, `2m`, `10m`) up to `WATCHER_MAX_ATTEMPTS` (default 3)
- `permanent`: immediate terminal failure
- retries are rescheduled via `pab:jobs:retry:zset` and re-enqueued to stream with `attempt+1`

---

## 4) Repository Layout After Implementation

```
lib/
  jobqueue.py                      NEW  stream-backed queue client + schema
apps/
  job-runner/
    job-runner.py                  NEW  enqueue/plan/compile/run/status/result/report/retry
  job-watcher/
    job-watcher.py                 NEW  daemon: lease -> batch -> vast -> upload -> ack
Dockerfile.watcher                 NEW
docker-compose.watcher.yml         NEW
jobs.example.yaml                  NEW
scripts/
  vast-deploy.sh                   MOD  hard cap + machine-readable summary
  vast-deploy-json.sh              NEW  optional wrapper if needed
Makefile                           MOD  jobs-* targets
pyproject.toml                     MOD  redis + pyyaml
apps/voice-synth/voice-synth.py    MOD  deterministic output mode for automation
```

---

## 5) `apps/voice-synth` Required Automation Fix

Current behavior writes to timestamped subdirectories. That is not deterministic enough for queue automation.

Add to `voice-synth speak`:

- `--out-exact DIR`  
  Writes `take_XX.wav`, `best.wav` (when enabled), `takes.meta.json`, `text.txt` directly under `DIR` with no extra nested timestamp folder.

- `--json-result FILE`  
  Writes a compact machine-readable summary (selected take, timings, files, model metadata).

Job runner and watcher should use `--out-exact /work/<output_name>`.

---

## 6) `apps/job-runner/job-runner.py`

Runs in toolbox container for local validation and enqueue tooling.

## 6.1 Commands

### `plan YAML_FILE`

- Validate schema and resolve defaults.
- Verify local voice slugs exist for all `voice` jobs.
- Print preview table and estimated character totals by voice.
- No Redis side effects.

### `enqueue YAML_FILE`

- Render jobs.
- Generate `request_id` (from `output_name` unless explicitly provided).
- Push to Redis stream using idempotent enqueue.
- Return status per row: `queued | duplicate_request | already_done`.

### `enqueue-beatsheet BEATSHEET_JSON ...`

Primary producer bridge:

- One job per `beats[i].narration`
- `output_name = "{topicId}/beat-{id:03d}"`
- `request_id = "{topicId}:beat-{id:03d}"`

### `compile YAML_FILE --out DIR`

- Materialize staged text files.
- Build deterministic batch manifest (`manifest.json`) containing exact per-job command args.
- Group by synthesis source to minimize model reload churn.
- No shell-escaped freeform command file as source of truth.

### `run YAML_FILE [--dry-run]`

- Local execution path for validation only.
- Uses direct Python invocation, not nested `./run` calls from inside container.

### `status [--json]`

- Stream length, pending count, stale PEL count, retry queue size, lock state, active instance.

### `result REQUEST_ID [--json]`

- Fetch final state from Redis (`done` or `failed`).

### `report YAML_FILE [--json]`

- Status for all jobs in YAML by `request_id`.

### `retry REQUEST_ID [--yes]`

- Requeue terminal failed job with `attempt=0` after explicit confirmation.

### `flush [--yes]`

- Deletes only queue-related transient keys, never historical result hashes unless `--hard`.

---

## 7) `apps/job-watcher/job-watcher.py`

Runs as VPS daemon and is the only component that provisions Vast.

## 7.1 Environment variables

Required:

```
REDIS_URL
VAST_API_KEY
VAST_SSH_KEY
S3_BUCKET
RCLONE_REMOTE
```

Optional core:

```
S3_PREFIX=voice-results
WATCHER_POLL_INTERVAL=10
WATCHER_BATCH_MIN=1
WATCHER_BATCH_MAX_WAIT=300
WATCHER_BATCH_MAX_JOBS=128
WATCHER_IDLE_GRACE=75
WATCHER_MAX_RUNTIME=240
WATCHER_MAX_ATTEMPTS=3
WATCHER_LOCK_TTL=120
WATCHER_LOCK_RENEW_EVERY=40
WATCHER_LEASE_IDLE_MS=180000
WATCHER_VOICES_DIR=/cache/voices
WATCHER_WORK_DIR=/work_remote
WATCHER_SYNTH_CONCURRENCY=1
```

## 7.2 Main loop behavior

1. Recover stale stream leases (`XAUTOCLAIM`).
2. Move due retry jobs from ZSET back to stream.
3. Evaluate trigger:
   - start when `queued >= WATCHER_BATCH_MIN`, or
   - oldest queued age >= `WATCHER_BATCH_MAX_WAIT`.
4. Acquire distributed lock with token (`run_id`), start renewal thread.
5. Provision Vast once.
6. Drain loop while instance alive:
   - lease up to `WATCHER_BATCH_MAX_JOBS`
   - compile and execute
   - upload + ack
   - if queue empty, wait up to `WATCHER_IDLE_GRACE` for new work before destroy
7. Destroy instance and release lock.

This avoids repeated spin-up costs during burst traffic.

## 7.3 Lock correctness

- Lock value is run token (`run_id`).
- Renewal and release use compare-and-delete semantics (Lua) so one watcher cannot release another watcher's lock.
- If lock renewal fails, watcher aborts new leasing and exits safely.

## 7.4 Voice availability check

- Validate required voice dirs before provisioning.
- Permanent-fail invalid voice jobs immediately (no retry).
- Built-in `speaker` jobs bypass voice-dir check.

## 7.5 Batch compile format

Manifest example:

```json
{
  "run_id": "watcher-20260222-211500",
  "jobs": [
    {
      "request_id": "topic:beat-001",
      "output_name": "topic/beat-001",
      "out_exact": "/work/topic/beat-001",
      "argv": [
        "voice-synth", "speak",
        "--voice", "rascar-capac",
        "--language", "French",
        "--text-file", "/work/topic/beat-001/text.txt",
        "--out-exact", "/work/topic/beat-001"
      ]
    }
  ]
}
```

## 7.6 Remote execution strategy

Use a single remote executor command per batch:

- upload staged `/work` files and selected `/cache/voices` subset
- run `python /app/apps/job-runner/job-runner.py execute-manifest /work/<run_id>/manifest.json`
- executor writes `/work/<run_id>/execution.json` with per-job success/failure

Reason: one controlled execution flow is easier to audit/retry than freeform shell lines.

## 7.7 Upload and ack

For each successful job:

1. Validate required output files exist and are non-empty.
2. Upload to `s3://{bucket}/{prefix}/{output_name}/`.
3. Write done result hash by `request_id`.
4. `XACK` stream entry.

If upload fails, classify as transient and schedule retry (no ACK until retry message persisted and original handled).

## 7.8 SIGTERM behavior

- Stop leasing new work.
- Finish current critical section (or checkpoint and abort cleanly).
- Ensure lock renewal thread stops.
- Destroy instance if owned.
- Leave unacked leased jobs recoverable via stream reclaim.

---

## 8) Parallel Processing and OOM Policy

Default for production launch:

- `WATCHER_SYNTH_CONCURRENCY=1` (strict sequential synthesis on GPU)

Rationale:

- Most stable and predictable memory behavior.
- Avoids hidden quality regressions and CUDA OOM thrash.

Experimental path (opt-in only):

1. Add controlled concurrency flag (`2` max initially).
2. Run qualification matrix by GPU SKU with worst-case text length and variants.
3. Require:
   - zero OOM over 100 batch runs,
   - P95 latency improvement > 20%,
   - no quality metric regression.
4. If any OOM occurs in production, auto-fallback to concurrency `1` and mark host profile unsafe.

Do not enable multi-process synthesis globally before passing qualification.

---

## 9) Vast Deploy Changes

`scripts/vast-deploy.sh` must support robust automation:

1. Add hard runtime cap:

```bash
if [[ -n "${VAST_MAX_DURATION:-}" ]]; then
  CREATE_ARGS+=(--max-dph-duration "$VAST_MAX_DURATION")
fi
```

2. Add machine-readable completion output (`--summary-json PATH`) containing:
   - `instance_id`
   - `job_name`
   - `work_remote_dir`
   - task-level exit summary
3. Add strict task mode (`--fail-fast`) for watcher use.
4. Preserve existing trap cleanup behavior as final guard.

---

## 10) Storage and Result Contract

Result keying: by `request_id`.

```json
{
  "request_id": "topic:beat-001",
  "content_hash": "sha256:abcd...",
  "status": "done",
  "attempt": 1,
  "run_id": "watcher-20260222-211500",
  "completed_at": "2026-02-22T21:23:14Z",
  "output_name": "topic/beat-001",
  "outputs": {
    "best": "s3://bucket/voice-results/topic/beat-001/best.wav",
    "takes": [
      "s3://bucket/voice-results/topic/beat-001/take_01.wav"
    ],
    "meta": "s3://bucket/voice-results/topic/beat-001/takes.meta.json"
  },
  "callback_data": {
    "topic_id": "topic",
    "beat_id": 1
  }
}
```

Failure record:

```json
{
  "request_id": "topic:beat-001",
  "status": "failed",
  "attempt": 3,
  "error_type": "transient_exhausted",
  "error": "...",
  "failed_at": "2026-02-22T21:25:10Z",
  "job": { "...original spec..." }
}
```

---

## 11) Docker and Compose (`watcher`)

`Dockerfile.watcher` requirements:

- base: `python:3.11-slim-bookworm`
- install: `bash`, `jq`, `curl`, `openssh-client`, `rsync`, `rclone`
- pip: `vastai`, `redis`, `pyyaml`
- default command: `python3 apps/job-watcher/job-watcher.py`

`docker-compose.watcher.yml`:

- mount repo, cache, work_remote, SSH key, rclone config
- restart policy `unless-stopped`
- add healthcheck (`job-runner status --json` or watcher ping script)
- one active replica in production (plus optional passive standby host)

---

## 12) Observability and Operations

Mandatory:

- Structured JSON logs with `run_id`, `request_id`, `stream_id`, `attempt`, `instance_id`
- Counters:
  - jobs queued/done/failed/retried
  - provisioning count/failures
  - upload failures
- Timers:
  - queue wait
  - synth duration
  - upload duration
  - total job latency

Recommended:

- Emit periodic utilization line:
  - queued jobs
  - instance state
  - current batch age
  - estimated cost/hr from chosen offer

---

## 13) Producer Integration Contract (other project)

Producer only needs:

1. Build `JobSpec` payload with unique `request_id`.
2. Push via Redis stream (or call `job-runner enqueue-beatsheet`).
3. Poll `pab:jobs:result` and `pab:jobs:failed` by `request_id`.

Beatsheet mapping:

- input: `beatsheet.topicId`, `beats[].id`, `beats[].narration`
- `request_id = "{topicId}:beat-{id:03d}"`
- `output_name = "{topicId}/beat-{id:03d}"`

---

## 14) Testing Plan (must pass before production)

## 14.1 Unit tests

- hash determinism (`content_hash`)
- schema validation and defaults
- lock compare-and-delete/renew logic
- retry classifier

## 14.2 Integration tests (local Docker)

- Redis stream enqueue -> lease -> ack happy path
- reclaim stale PEL entries after simulated crash
- retry scheduling and exhaustion
- deterministic output directory (`--out-exact`)

## 14.3 Failure/chaos tests

- kill watcher after lease before ack
- kill watcher during Vast runtime
- break rclone upload
- simulate SSH timeout
- ensure no silent loss and expected final status

## 14.4 Cost tests

- burst traffic (100+ jobs)
- confirm single provision drains multiple micro-batches
- measure idle teardown under `WATCHER_IDLE_GRACE`

---

## 15) Implementation Order

1. `lib/jobqueue.py` (stream client, schema, ids, lock scripts)
2. `pyproject.toml` deps (`redis`, `pyyaml`)
3. `apps/voice-synth/voice-synth.py` (`--out-exact`, `--json-result`)
4. `apps/job-runner/job-runner.py` (`plan`, `enqueue`, `compile`, `status`, `result`, `report`, `retry`)
5. `jobs.example.yaml`
6. `scripts/vast-deploy.sh` (`--max-dph-duration`, `--summary-json`, `--fail-fast`)
7. `Dockerfile.watcher`
8. `docker-compose.watcher.yml`
9. `apps/job-watcher/job-watcher.py`
10. `Makefile` jobs targets
11. `.env.example` updates
12. integration/chaos/cost test suite

---

## 16) Deferred (explicitly out of v1)

- Multi-GPU or horizontal worker scaling.
- Global audio dedupe reuse by `content_hash` (cache pointer mode).
- Automatic signed URL issuance inside watcher (can be separate API service).

---

## 17) Practical Defaults for Day 1

Use these until real traffic metrics justify tuning:

```
WATCHER_BATCH_MIN=1
WATCHER_BATCH_MAX_WAIT=300
WATCHER_BATCH_MAX_JOBS=128
WATCHER_IDLE_GRACE=75
WATCHER_POLL_INTERVAL=10
WATCHER_MAX_RUNTIME=240
WATCHER_SYNTH_CONCURRENCY=1
WATCHER_MAX_ATTEMPTS=3
```

These settings prioritize low idle waste and high safety.
