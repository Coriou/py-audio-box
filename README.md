# toolbox

A portable, cacheable Python runtime for ML/audio scripts.
Everything runs in Docker — zero host installs, fast re-runs, shared cache across all apps.

---

## Prerequisites

- **Docker** with Compose v2 (`docker compose` — not the old `docker-compose`)
- No Python, Poetry, ffmpeg, or GPU drivers needed on the host

First-time setup:

```bash
make build        # build the shared image (~5 min, cached on rebuild)
```

---

## Quick start

```bash
# Extract clean voice clips from a YouTube video
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 5 --length 30

# ── One-shot voice registration (recommended) ─────────────────────────────────
# Download → Demucs split → clone prompt → test synthesis in a single command
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough \
    --text "Nature is the greatest artist."

# ── Named voice workflow (step by step) ──────────────────────────────────────
# 1. Extract + register a named voice
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough --clips 5

# 2. Process ref audio, build clone prompt (only once)
./run voice-clone synth \
    --voice david-attenborough \
    --text "Nature is the greatest artist."

# 3. Fast iteration — no ref processing or prompt building on re-runs
./run voice-synth list-voices
./run voice-synth speak --voice david-attenborough \
    --text "Welcome to the natural history of the Earth."
./run voice-synth speak --voice david-attenborough \
    --text "..." --variants 4 --qa
# If you built a tone-labelled prompt (--tone sad during voice-clone synth):
./run voice-synth speak --voice david-attenborough --tone sad \
    --text "..."

# ── Or with an existing WAV file ─────────────────────────────────────────────
./run voice-clone synth --ref-audio /work/myclip.wav --text "Hello, world"

# Get a shell inside the container (explore, debug, prototype)
make shell

# See all available apps
./run
```

Outputs land in `./work/` on your host.
Everything expensive (model weights, downloads, Demucs runs) is cached in `./cache/` —
re-runs with different flags are nearly instant.

---

## Repository structure

```
apps/
  <name>/
    <name>.py       ← entry point (required, must match dir name)
    README.md       ← app-level docs: purpose, args, output format
cache/              ← shared persistent cache — gitignored, never delete unless resetting
  voices/
    <slug>/         ← named voice registry; shared across all apps
work/               ← default output directory — gitignored
lib/
  voices.py         ← shared voice-registry library (zero extra deps)
Dockerfile          ← single shared image: Python 3.11 + Poetry + ffmpeg + torch stack
docker-compose.yml  ← mounts: ./  → /app, ./work → /work, ./cache → /cache
pyproject.toml      ← Python dependencies for the shared image
poetry.lock         ← committed — pins exact versions for reproducible builds
run                 ← launcher: ./run <app> [args…]
Makefile            ← shortcuts for build / shell / clean
```

---

## Voice registry

All apps share a named voice registry at `/cache/voices/<slug>/`.
A **slug** is lowercase ASCII with hyphens (e.g. `david-attenborough`).

### Directory layout

```
/cache/voices/
  david-attenborough/
    voice.json          ← identity, source info, ref metadata, prompt index
    source_clip.wav     ← raw clip from voice-split (44.1 kHz)
    ref.wav             ← processed 24 kHz segment from voice-clone
    prompts/
      <hash>_<model>_full_v1.pkl        ← clone prompt (used by voice-synth)
      <hash>_<model>_full_v1.meta.json
```

Each stage is optional — a voice progresses from `source_clip.wav` → `ref.wav`
→ `prompts/*.pkl` as you run the pipeline steps.

### Progressive pipeline

```bash
# Step 1 — register the voice with a source clip
./run voice-split --url "..." --voice-name david-attenborough

# Step 2 — process ref + build prompt  (first time: ~45s; cached thereafter)
./run voice-clone synth --voice david-attenborough --text "..."

# Step 3 — fast synthesis (model load + generate only, no VAD or Whisper)
./run voice-synth speak --voice david-attenborough --text "..."

# ── Or do all three steps with one command ────────────────────────────────────
./run voice-register \
    --url "..." \
    --voice-name david-attenborough \
    --text "Nature is the greatest artist."

# Inspect all registered voices + their pipeline status
./run voice-synth list-voices

# Rename / delete / share voices
./run voice-synth rename-voice old-slug new-slug
./run voice-synth delete-voice my-voice --yes
./run voice-synth export-voice my-voice               # → /work/my-voice.zip
./run voice-synth import-voice --zip /work/my-voice.zip
```

### Shared library

`lib/voices.py` contains `VoiceRegistry` and `validate_slug`, imported by all
apps via:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lib"))
from voices import VoiceRegistry, validate_slug
```

Zero external dependencies — stdlib only.

---

## Mount contract (inside the container)

| Host path | Container path | What goes there                                                   |
| --------- | -------------- | ----------------------------------------------------------------- |
| `./`      | `/app`         | All source code (live bind-mount — edits take effect immediately) |
| `./work`  | `/work`        | Default output directory for all apps                             |
| `./cache` | `/cache`       | Shared persistent cache: models, downloads, segments              |

Scripts should default `--cache /cache` and `--out /work`.
`XDG_CACHE_HOME` is also set to `/cache` inside the container, so tools like HuggingFace
Hub and Torch Hub write there automatically.

---

## Creating a new app

### 1. Scaffold

```
apps/
  my-tool/
    my-tool.py    ← entry point
    README.md     ← see conventions below
```

### 2. Entry point conventions (`my-tool.py`)

```python
#!/usr/bin/env python3
"""One-line description."""

import argparse
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",   default="/work",   help="Output directory")
    ap.add_argument("--cache", default="/cache",  help="Persistent cache directory")
    # ... your args
    args = ap.parse_args()
    # ... your logic

if __name__ == "__main__":
    main()
```

Key rules:

- `--cache` defaults to `/cache` (the shared mount)
- `--out` defaults to `/work`
- Use `argparse.ArgumentDefaultsHelpFormatter` so `--help` is always informative
- Cache expensive work to disk under `--cache`; check before recomputing
- Print `[cache hit]` lines so it's obvious what was skipped

### 3. Run it

```bash
./run my-tool --help
./run my-tool --out /work
```

No config changes needed. `./run` discovers apps by scanning `apps/*/`.

### 4. (Optional) Add a Makefile shortcut

```makefile
.PHONY: my-tool
my-tool:  ## Run my-tool. Usage: make my-tool ARGS='--flag value'
	./run my-tool $(ARGS)
```

### 5. App README conventions

Each `apps/<name>/README.md` should cover:

- What the app does (1–3 sentences)
- Usage examples (copy-paste ready `./run` commands)
- All CLI flags with types and defaults
- Output format / file naming
- What gets cached and where

---

## Dependencies

All apps share one image. To add a Python package:

```bash
# 1. Add it to pyproject.toml
# 2. Regenerate the lock file (no local Poetry needed)
docker compose run --rm toolbox bash -c "poetry lock"

# 3. Rebuild
make build
```

To add a system package (ffmpeg, sox, …): edit the `apt-get install` block in `Dockerfile`, then `make build`.

`poetry.lock` is committed — it guarantees reproducible builds and makes `make build` faster
(only the install layer re-runs when deps change).

---

## Makefile reference

```
make build              Build (or rebuild) the toolbox image
make build-no-cache     Force a clean image rebuild (ignores all layer cache)
make shell              Interactive bash shell inside the container
make clean-work         Delete output files in ./work/
make clean-cache        Wipe ./cache/ — models and downloads will re-run
```

App shortcuts (pass extra flags via `ARGS=`):

```
make voice-split     ARGS='--url "https://..." --clips 5 --length 30'
make voice-clone     ARGS='synth --ref-audio /work/myclip.wav --text "Hello, world"'
make voice-synth     ARGS='speak --voice <id> --text "Hello"'
make voice-register  ARGS='--url "https://..." --voice-name my-voice --text "Hello"'
```

---

## Agent / automation guidelines

When an AI agent or script works with this repo:

- **Don't install packages on the host.** All Python work happens inside the container.
- **The image is already built.** Use `./run <app>` or `docker compose run --rm toolbox …` directly; only run `make build` if `pyproject.toml` or `Dockerfile` changed.
- **Cache is safe to read, never safe to delete mid-run.** The `./cache` directory is the source of truth for all expensive computation. Treat it as append-only during a run.
- **Adding an app = adding a file.** Drop `apps/<name>/<name>.py` and it's runnable with `./run <name>`. No registration, no config changes.
- **Check `--help` first.** Every script uses `ArgumentDefaultsHelpFormatter`; `./run <app> --help` is always the authoritative reference for flags and defaults.
- **Output is always in `./work/`** (mounted at `/work`) unless `--out` is overridden. Look there for results.
- **Prefer `--voice <slug>` over `--ref-audio <path>`.** Named voice slugs are the canonical way to reference voices across all apps. Run `./run voice-synth list-voices` to enumerate available voices. Register a new voice with `--voice-name <slug>` on `voice-split`, `voice-clone synth`, or `voice-synth design-voice`. When `--voice <slug>` is used with `voice-clone synth`, the built prompt is automatically registered back to that voice — no separate `--voice-name` needed.
- **Use `voice-register` for one-shot pipeline runs.** `./run voice-register --url "..." --voice-name <slug> --text "..."` chains voice-split → voice-clone synth and leaves a fully synthesis-ready voice in a single command. Re-runs are safe and cached.
