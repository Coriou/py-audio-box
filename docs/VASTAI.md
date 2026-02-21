# vast.ai Integration

> Cloud GPU automation for py-audio-box — provision, run, pull results, destroy.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Scripts Reference](#scripts-reference)
   - [scripts/vast](#scriptsvast)
   - [scripts/vast-deploy.sh](#scriptsvast-deploysh)
   - [scripts/vastai-setup.sh](#scriptsvastai-setupsh)
6. [Makefile Targets](#makefile-targets)
7. [Common Workflows](#common-workflows)
8. [Task File Format](#task-file-format)
9. [Environment Variables](#environment-variables)
10. [Instance Lifecycle](#instance-lifecycle)
11. [Outputs & Results](#outputs--results)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The integration lets you:

- **Spin up a cheap cloud GPU** with a pre-built CUDA image (no setup, deps are baked in)
- **Sync your local code** to `/app` on the instance via `rsync`
- **Run one or many tasks** (`run-direct` invocations) streamed live to your terminal
- **Pull results** from `/work` on the instance into `work_remote/<job>/` locally
- **Auto-destroy** the instance when done (cost control via `trap`)

**No host-side Python/pip installs required** — the `vastai` CLI runs inside the CPU Docker container via `scripts/vast`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Local machine (macOS / Linux)                          │
│                                                         │
│  scripts/vast          ← thin Docker wrapper for vastai │
│  scripts/vast-deploy.sh ← full lifecycle orchestrator   │
│  Makefile               ← ergonomic targets             │
│                                                         │
│  docker run :latest (CPU image)                         │
│    └── vastai CLI  ─────────────────────────────────────┼──► vast.ai API
└─────────────────────────────────────────────────────────┘
         │  provision + SSH + rsync
         ▼
┌─────────────────────────────────────────────────────────┐
│  vast.ai GPU instance                                   │
│                                                         │
│  Image:  ghcr.io/coriou/voice-tools:cuda               │
│  /app    ← synced from local repo                       │
│  /work   ← task outputs (pulled back after run)         │
│  /cache  ← model weights (HF_HOME / TORCH_HOME)         │
│                                                         │
│  ./run-direct <app> [args]  ← task runner               │
└─────────────────────────────────────────────────────────┘
```

### Docker image variants

| Tag        | Base                      | Use                                      |
| ---------- | ------------------------- | ---------------------------------------- |
| `:latest`  | python:3.11-slim-bookworm | CPU-only, CLI tools (including `vastai`) |
| `:cuda`    | CUDA 12.4 + cu124 wheels  | Production GPU workloads (Volta SM 7.0+) |
| `:cuda128` | CUDA 12.8 + cu128 wheels  | Blackwell SM 10.0+ GPUs                  |

The `:cuda` image is deployed to vast.ai by default (`VAST_IMAGE`).

---

## Prerequisites

### Host machine

| Requirement    | Why                                                       |
| -------------- | --------------------------------------------------------- |
| Docker running | `scripts/vast` executes `vastai` inside the CPU container |
| `ssh`          | Instance connectivity                                     |
| `rsync`        | Code sync + result pull                                   |
| `jq`           | Parses `--raw` JSON from the vast.ai API                  |

Install `jq` if missing:

```bash
brew install jq       # macOS
apt install jq        # Debian/Ubuntu
```

### Authentication

Get your API key from [cloud.vast.ai/console/cli](https://cloud.vast.ai/console/cli) and export it once per session (or add to your shell profile):

```bash
export VAST_API_KEY=your_key_here
```

The key is also read from `~/.config/vastai/vast_api_key` (plain text, no quotes).

### SSH key

`vast-deploy.sh` auto-detects your default SSH key in preference order:

```
~/.ssh/id_ed25519
~/.ssh/id_rsa
~/.ssh/id_ecdsa
~/.ssh/id_dsa
```

Override with `--ssh-key PATH` if you use a non-default key.

---

## Quick Start

```bash
# 1. Set your API key
export VAST_API_KEY=xxxxxxxxxxxx

# 2. Browse available GPU offers
make vast-search

# 3a. Run a single task (provision → run → pull → destroy)
make vast-run ARGS='-- voice-synth speak --voice myvoice --text "Hello world"'

# 3b. Run a batch of tasks from a file
make vast-run TASKS=my-jobs.txt

# 3c. Open an interactive shell on a fresh instance
make vast-shell
```

Results land in `work_remote/<job-name>/` on your local machine.

---

## Scripts Reference

### `scripts/vast`

A **zero-install wrapper** around the `vastai` CLI. It runs `vastai` inside the `:latest` (CPU) Docker container so no Python package installs are needed on the host.

```
Usage:  ./scripts/vast <vastai-subcommand> [args...]
```

**How it works:**

1. Resolves `VAST_API_KEY` from env or `~/.config/vastai/vast_api_key`
2. Runs `docker run --rm --network host ghcr.io/coriou/voice-tools:latest`
3. Inside the container: writes the key to `~/.config/vastai/vast_api_key`, then `exec vastai "$@"`
4. TTY passthrough: `-it` when stdout is a terminal (coloured tables), stdin-only when piping `--raw` JSON

**Common uses:**

```bash
# Browse offers
./scripts/vast search offers 'gpu_ram >= 20 reliability > 0.98' --order 'dph+'

# List your running instances
./scripts/vast show instances

# Inspect a specific instance (JSON)
./scripts/vast show instance 12345 --raw | jq .

# Destroy an instance manually
./scripts/vast destroy instance 12345

# Get SSH connection URL
./scripts/vast ssh-url 12345
```

---

### `scripts/vast-deploy.sh`

The **full lifecycle orchestrator**. Eight sequential stages:

```
1. search offers  →  find cheapest qualifying GPU
2. create instance  →  provision with --ssh --direct
3. wait for running  →  poll status every 5 s
4. wait for SSH  →  probe port until accepting (up to 6 min)
5. rsync code  →  local repo → /app on instance
6. run tasks  →  ./run-direct <task> for each task; streams output live
7. pull results  →  /work on instance → work_remote/<job>/ locally
8. destroy  →  via trap cleanup EXIT (always runs unless --keep)
```

**Usage:**

```bash
./scripts/vast-deploy.sh [OPTIONS] [-- TASK [TASK...]]
```

**Options:**

| Flag                | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| `-- TASK [TASK...]` | Inline task(s) — everything after `--` is passed as individual tasks |
| `--tasks FILE`      | Read tasks from a file (one per line, `#` comments ignored)          |
| `--shell`           | Interactive SSH session; disables auto-destroy                       |
| `--job NAME`        | Job label (default: `py-audio-box-YYYYMMDD_HHMMSS`)                  |
| `--ssh-key PATH`    | Override SSH private key path                                        |
| `--no-sync`         | Skip rsync of local code to `/app`                                   |
| `--no-pull`         | Skip pulling `/work` results after tasks                             |
| `--keep`            | Do NOT destroy the instance when done                                |
| `--dry-run`         | Print all commands, touch nothing                                    |
| `-h`, `--help`      | Show full help                                                       |

---

### `scripts/vastai-setup.sh`

A **manual first-run helper** for interactive sessions. Only needed if you SSH into a raw instance rather than using `vast-deploy.sh`. It:

1. Pulls latest code (`git pull --ff-only` if `.git` present, else shallow clone + rsync)
2. Verifies CUDA + torch (`gpu_name`, SM version, VRAM, torch version)

Logs everything to `/var/log/vastai-setup.log`.

> **Note:** `vast-deploy.sh` handles everything `vastai-setup.sh` does automatically. You only run `vastai-setup.sh` if you SSH into an instance that was provisioned manually through the vast.ai web UI.

---

## Makefile Targets

```
make vast-search                     # List cheapest qualifying offers
make vast-status                     # Show your running instances
make vast-run TASKS=my-jobs.txt      # Full lifecycle from a tasks file
make vast-run ARGS='-- voice-synth speak --voice x --text "hi"'
make vast-shell                      # Provision + interactive SSH
make vast-destroy ID=12345           # Destroy a specific instance
make vast-pull ID=12345 JOB=myjob    # rsync /work from a running instance
```

Pass extra `vast-deploy.sh` flags through `ARGS`:

```bash
make vast-run ARGS='--no-sync --keep --dry-run' TASKS=my-jobs.txt
```

---

## Common Workflows

### Batch job from a file

Create a tasks file:

```bash
# my-jobs.txt
# Lines starting with # are ignored

voice-register --url "https://youtu.be/XXXX" --voice-name myvoice
voice-synth speak --voice myvoice --text "First sentence."
voice-synth speak --voice myvoice --text "Second sentence."
```

Run it:

```bash
make vast-run TASKS=my-jobs.txt
```

Results appear in `work_remote/py-audio-box-YYYYMMDD_HHMMSS/`.

---

### Single one-off command

```bash
make vast-run ARGS='-- voice-synth speak --voice myvoice --text "Hello"'
```

---

### Interactive debugging session

```bash
make vast-shell
# SSH drops you into /app on the instance
# Instance is NOT auto-destroyed when you exit — you keep it
# When done:
make vast-destroy ID=<instance_id>
```

---

### Keep instance alive after tasks (debugging)

```bash
make vast-run ARGS='--keep' TASKS=my-jobs.txt
# Check results, SSH if needed, then:
make vast-destroy ID=<instance_id>
```

---

### Dry run (validate without spending money)

```bash
make vast-run ARGS='--dry-run' TASKS=my-jobs.txt
# Prints every command it would run, touches nothing
```

---

### Pull results from a manually-managed instance

If you provisioned an instance with `--keep` or `--shell` and want to pull `/work` from it:

```bash
make vast-pull ID=12345 JOB=my-session-name
# Results → work_remote/my-session-name/
```

---

### Custom GPU query

The default search query is:

```
reliability > 0.98 gpu_ram >= 20 compute_cap >= 700 inet_down >= 200 disk_space >= 50 rented=False
```

Override it for a specific run:

```bash
VAST_QUERY='gpu_ram >= 40 compute_cap >= 800 inet_down >= 500 rented=False' \
  make vast-run TASKS=my-jobs.txt
```

See all available filter fields: `./scripts/vast search offers --help`

---

### Use a different image (e.g. cuda128 for Blackwell GPUs)

```bash
VAST_IMAGE=ghcr.io/coriou/voice-tools:cuda128 \
VAST_QUERY='compute_cap >= 1000 gpu_ram >= 20' \
  make vast-run TASKS=my-jobs.txt
```

---

## Task File Format

Each non-blank, non-comment line becomes one `./run-direct <line>` invocation on the instance.

```bash
# my-jobs.txt

# Register a voice from a YouTube video
voice-register --url "https://youtu.be/XXXX" --voice-name myvoice --text "Sample text."

# Synthesise speech
voice-synth speak --voice myvoice --text "Hello from vast.ai"

# Split vocals from a video
voice-split --url "https://youtu.be/YYYY" --start 60 --end 120
```

- Lines are executed sequentially, one at a time
- A non-zero exit from a task emits a warning but does NOT stop the remaining tasks
- Each line is passed verbatim to `./run-direct`; shell quoting applies to the whole line

---

## Environment Variables

| Variable       | Default                           | Description                                                                |
| -------------- | --------------------------------- | -------------------------------------------------------------------------- |
| `VAST_API_KEY` | _(required)_                      | vast.ai API key from [cloud.vast.ai](https://cloud.vast.ai/console/cli)    |
| `VAST_IMAGE`   | `ghcr.io/coriou/voice-tools:cuda` | Docker image deployed to the instance                                      |
| `VAST_DISK`    | `60`                              | Disk size in GB for the instance                                           |
| `VAST_QUERY`   | see above                         | vastai search offers query string                                          |
| `GHCR_TOKEN`   | _(optional)_                      | GitHub PAT with `read:packages` — only needed if the GHCR image is private |

`VAST_IMAGE`, `VAST_DISK`, and `VAST_QUERY` can also be passed as `make` variables:

```bash
VAST_DISK=100 make vast-run TASKS=my-jobs.txt
```

---

## Instance Lifecycle

```
search offers
      │
      ▼
create instance  ──────────────────────────────────┐
      │                                            │ trap cleanup EXIT
      ▼                                            │ (always fires)
poll status: scheduling → loading → running        │
      │                                            │
      ▼                                            │
poll SSH port (up to 6 min, 5 s intervals)         │
      │                                            │
      ▼                                            │
rsync local /app → /app on instance                │
      │                                            │
      ▼                                            │
run tasks (./run-direct per task, live output)     │
      │                                            │
      ▼                                            │
rsync /work → work_remote/<job>/                   │
      │                                            │
      └─────────────────────► destroy instance ◄──┘
                               (unless --keep / --shell)
```

**On Ctrl-C / error:** the `trap cleanup EXIT` fires and destroys the instance automatically, preventing orphaned billed instances.

**`--shell` mode:** sets `KEEP=1` implicitly (you're interacting with it, don't kill it under your feet). You are responsible for `make vast-destroy ID=...` when done.

---

## Outputs & Results

- Task output (stdout/stderr) streams **live** to your terminal during execution
- Files written to `/work` on the instance are pulled to `work_remote/<job-name>/` locally after all tasks complete
- The job name defaults to `py-audio-box-YYYYMMDD_HHMMSS`; set a custom one with `--job`:

```bash
./scripts/vast-deploy.sh --job blitzstream-ep42 --tasks my-jobs.txt
# Results → work_remote/blitzstream-ep42/
```

---

## Troubleshooting

### `jq not found`

```bash
brew install jq
```

### `VAST_API_KEY not set`

```bash
export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key)
# or
export VAST_API_KEY=your_key_here
```

### `No offers match the query`

Relax the search constraints:

```bash
# Lower VRAM requirement or remove compute_cap floor
VAST_QUERY='reliability > 0.95 gpu_ram >= 16 inet_down >= 100 rented=False' \
  make vast-run TASKS=my-jobs.txt
```

### SSH did not become available within 6 minutes

Usually means the instance is still pulling the Docker image (can take 2-4 min on cold pull). Check the instance log on the vast.ai console. Rarely, the image pull fails — try again or use `--dry-run` first to verify the config is correct.

### Task failed but instance was destroyed

All task outputs are streamed live to your terminal. If you need to inspect state after a failure, add `--keep`:

```bash
make vast-run ARGS='--keep' TASKS=my-jobs.txt
# SSH in manually:
ssh -p <PORT> root@<HOST>
# When done:
make vast-destroy ID=<ID>
```

### Private GHCR image pull fails on instance

Pass your GitHub PAT:

```bash
GHCR_TOKEN=ghp_xxxx make vast-run TASKS=my-jobs.txt
```

### Orphaned instances (forgot --keep / script was killed)

```bash
make vast-status       # list running instances
make vast-destroy ID=12345
```

Or destroy directly:

```bash
./scripts/vast destroy instance 12345
```
