# toolbox

A portable, cacheable Python runtime for ML/audio scripts.
Everything runs in Docker — zero host installs, fast re-runs, shared cache across all apps.

---

## Prerequisites

- **Docker** with Compose v2 (`docker compose` — not the old `docker-compose`)
- No Python, Poetry, ffmpeg, or GPU drivers needed on the host for CPU mode

First-time setup:

```bash
make build        # build the shared CPU image (~5 min, cached on rebuild)
```

---

## Platforms

### CPU (default — works everywhere)

No extra requirements. The default image runs purely on CPU and works on
macOS, Linux, and Windows (Docker Desktop or WSL 2).

```bash
make build
./run voice-synth speak --voice my-voice --text "Hello"
```

### GPU (NVIDIA — faster inference)

Requires an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# 1. Install NVIDIA Container Toolkit on the host (once)
#    Linux / WSL 2:
#      sudo apt-get install -y nvidia-container-toolkit
#      sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
#    Windows (Docker Desktop):
#      Settings → Docker Engine → add the nvidia runtime (see docker-compose.gpu.yml)

# 2. Build the CUDA image (once; layers shared with the CPU image)
make build-gpu

# 3. Run any app with GPU acceleration
TOOLBOX_VARIANT=gpu ./run voice-synth speak --voice my-voice --text "Hello"
TOOLBOX_VARIANT=gpu ./run voice-register --url "..." --voice-name my-voice --text "Hello"
```

Acceleration details:

- `TOOLBOX_VENDOR=gpu` overlays `docker-compose.gpu.yml` — no other flags needed
- Device is detected at runtime via `torch.cuda.is_available()`
- `--dtype` defaults to `auto`: `float16` on pre-Ampere GPUs (Maxwell / Pascal /
  Volta / Turing), `bfloat16` on Ampere+
- Demucs (voice-split) and Silero VAD run on GPU automatically
- Whisper (voice-clone / voice-synth QA) switches to `float16` on GPU
- The Qwen3-TTS model loads onto the GPU; expect ~3–10× speedup over CPU

### Windows

The `./run` bash script works in Git Bash and WSL 2.
For native PowerShell there is a `run.ps1` equivalent:

```powershell
# CPU
.\run.ps1 voice-synth speak --voice my-voice --text "Hello"

# GPU
$env:TOOLBOX_VARIANT = "gpu"
.\run.ps1 voice-synth speak --voice my-voice --text "Hello"
```

Everything else (`make`, `docker compose`) works identically on Windows
(use Git Bash or enable make via `winget install GnuWin32.Make`).

## Cheat sheet

### Register a voice (one command, fully cached)

```bash
# From a YouTube URL
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough \
    --text "Nature is the greatest artist."

# From a local audio file  (put the file in ./work/ first)
./run voice-register \
    --audio /work/my-recording.wav \
    --voice-name my-voice \
    --text "Hello, this is a test."

# With a timestamp range (skip the rest of the audio)
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --start 1:23 --end 5:00 \
    --voice-name speaker \
    --text "Hello world."

# Same, from a local file
./run voice-register \
    --audio /work/interview.mp3 \
    --start 0:45 --end 3:30 \
    --voice-name interviewee \
    --text "Hello world."
```

Timestamp formats: `90` (seconds), `1:30` (MM:SS), `1:30.5`, `1:02:30` (HH:MM:SS).

### Synthesise with a registered voice

```bash
./run voice-synth speak --voice david-attenborough --text "Welcome."
./run voice-synth speak --voice david-attenborough --text "..." --variants 4 --qa
./run voice-synth speak --voice david-attenborough --tone excited --text "..."
./run voice-synth list-voices
```

### Extract voice clips only (no synthesis)

```bash
# YouTube
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 5 --length 30
./run voice-split --url "..." --start 10:00 --end 15:00 --clips 5 --voice-name my-voice

# Local file
./run voice-split --audio /work/recording.wav --clips 5 --length 30
./run voice-split --audio /work/recording.wav --start 0:30 --end 4:00 --voice-name my-voice
```

### Build clone prompt from an existing clip

```bash
./run voice-clone synth --voice david-attenborough --text "Nature is the greatest artist."
./run voice-clone synth --ref-audio /work/myclip.wav --text "Hello, world"
```

### Voice management

```bash
./run voice-synth rename-voice old-slug new-slug
./run voice-synth delete-voice my-voice --yes
./run voice-synth export-voice my-voice          # → /work/my-voice.zip
./run voice-synth import-voice --zip /work/my-voice.zip
```

### Utilities

```bash
make shell            # interactive bash shell inside the container
make shell-gpu        # same, GPU image
make clean-work       # delete ./work/ outputs
make clean-cache      # wipe ./cache/ (models + downloads re-run)
./run                 # list all available apps
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
./run voice-split --url "..." --voice-name david-attenborough           # YouTube
./run voice-split --audio /work/clip.wav --voice-name my-voice          # local file
./run voice-split --url "..." --start 1:30 --end 4:00 --voice-name speaker  # trimmed

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
make build                  Build (or rebuild) the CPU toolbox image
make build-gpu              Build the CUDA 12.4 GPU image  →  voice-tools:cuda
make build-no-cache         Force a clean CPU rebuild (no layer cache)
make build-gpu-no-cache     Force a clean GPU rebuild (no layer cache)
make shell                  Interactive bash shell inside the CPU container
make shell-gpu              Interactive bash shell inside the GPU container
make clean-work             Delete output files in ./work/
make clean-cache            Wipe ./cache/ — models and downloads will re-run
make publish                Build + push all images to GHCR (cpu → latest, cuda → cuda)
make publish-cpu            Build + push CPU image only
make publish-cuda           Build + push CUDA image only
```

App shortcuts (pass extra flags via `ARGS=`):

```
make voice-split     ARGS='--url "https://..." --clips 5 --length 30'
make voice-split     ARGS='--audio /work/file.wav --clips 5'
make voice-clone     ARGS='synth --ref-audio /work/myclip.wav --text "Hello, world"'
make voice-synth     ARGS='speak --voice <slug> --text "Hello"'
make voice-register  ARGS='--url "https://..." --voice-name my-voice --text "Hello"'
make voice-register  ARGS='--audio /work/file.wav --voice-name my-voice --text "Hello"'
```

---

## Agent / automation guidelines

When an AI agent or script works with this repo:

- **Don't install packages on the host.** All Python work happens inside the container.
- **The image is already built.** Use `./run <app>` or `docker compose run --rm toolbox …` directly; only run `make build` if `pyproject.toml` or `Dockerfile` changed. For GPU hosts, use `TOOLBOX_VARIANT=gpu ./run …` and run `make build-gpu` after Dockerfile changes.
- **Cache is safe to read, never safe to delete mid-run.** The `./cache` directory is the source of truth for all expensive computation. Treat it as append-only during a run.
- **Adding an app = adding a file.** Drop `apps/<name>/<name>.py` and it's runnable with `./run <name>`. No registration, no config changes.
- **Check `--help` first.** Every script uses `ArgumentDefaultsHelpFormatter`; `./run <app> --help` is always the authoritative reference for flags and defaults.
- **Output is always in `./work/`** (mounted at `/work`) unless `--out` is overridden. Look there for results.
- **Prefer `--voice <slug>` over `--ref-audio <path>`.** Named voice slugs are the canonical way to reference voices across all apps. Run `./run voice-synth list-voices` to enumerate available voices. Register a new voice with `--voice-name <slug>` on `voice-split`, `voice-clone synth`, or `voice-synth design-voice`. When `--voice <slug>` is used with `voice-clone synth`, the built prompt is automatically registered back to that voice — no separate `--voice-name` needed.
- **Use `voice-register` for one-shot pipeline runs.** `./run voice-register --url "..." --voice-name <slug> --text "..."` (YouTube) or `./run voice-register --audio /work/file.wav --voice-name <slug> --text "..."` (local file) chain voice-split → voice-clone synth and leave a fully synthesis-ready voice in a single command. Pass `--start`/`--end` (e.g. `--start 1:30 --end 5:00`) to trim the source before processing. Re-runs are safe and cached.
- **GPU mode is opt-in.** Prefix any `./run` command with `TOOLBOX_VARIANT=gpu` to use the GPU image (requires `make build-gpu` once). On Windows PowerShell use `$env:TOOLBOX_VARIANT = "gpu"` then the normal `.\run.ps1` command. No `--dtype` flag needed; dtype is chosen automatically.
