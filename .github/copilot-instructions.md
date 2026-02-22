# Copilot Instructions — py-audio-box

## Architecture overview

Docker-first Python ML toolbox. Everything runs inside a shared container; **zero host installs** required (no Python, Poetry, ffmpeg, etc. on the host).

### Components

| Layer           | Path                          | Role                                                           |
| --------------- | ----------------------------- | -------------------------------------------------------------- |
| **Entry point** | `./run` / `./run-direct`      | Launches any app via `docker compose run`                      |
| **Apps**        | `apps/<name>/<name>.py`       | Self-contained CLI scripts (one per capability)                |
| **Shared lib**  | `lib/`                        | Helpers imported by all apps via explicit `sys.path` injection |
| **Cache**       | `./cache/` → `/cache` (mount) | Models (HF, torch) + voice registry persisted across runs      |
| **Work output** | `./work/` → `/work` (mount)   | Generated audio files                                          |

### Apps

- **`voice-split`** — downloads YouTube audio, runs Demucs + Silero VAD, exports clean voice clips
- **`voice-register`** — one-shot pipeline: split → clone → register a named voice
- **`voice-clone`** — builds a `clone_prompt` pickle from a reference `.wav` using Qwen3-TTS
- **`voice-synth`** — synthesises speech; sub-commands: `speak`, `list-voices`, `list-speakers`, `capabilities`, `register-builtin`, `design-voice`

### Shared lib (`lib/`)

- **`tts.py`** — Qwen3-TTS model loading (`load_tts_model`), `synthesise`, `synthesise_custom_voice`, device/dtype helpers, language resolution, generation profiles
- **`voices.py`** — `VoiceRegistry` class; voice registry lives at `/cache/voices/<slug>/` with `voice.json`, `ref.wav`, and `prompts/` directory
- **`audio.py`** — ffmpeg wrappers (`normalize_audio`, `trim_audio_encode`), `analyse_acoustics`, `score_take_selection`/`rank_take_selection`
- **`vad.py`** — Silero VAD wrapper
- **`styles.yaml`** — named instruction-template presets for `--instruct`

Apps access the library by injecting the path before their imports:

```python
_LIB = str(Path(__file__).resolve().parent.parent.parent / "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
```

## Running apps

```bash
# CPU (default)
./run voice-synth speak --voice my-voice --text "Hello"

# GPU (requires NVIDIA Container Toolkit + CUDA image built)
TOOLBOX_VARIANT=gpu ./run voice-synth speak --voice my-voice --text "Hello"

# Inside container (no Docker prefix needed — used by remote deployments)
./run-direct voice-synth speak --voice my-voice --text "Hello"
```

## Build & publish

```bash
make build             # CPU image (local dev)
make build-gpu         # GPU image (local dev)

# GHCR publish — two-step CUDA workflow:
make publish-cuda-base # Rebuild heavy layer: torch + flash-attn (run when deps change)
make publish-cuda      # Rebuild thin app layer ~50 MB (run on every code change)
make publish-cpu       # CPU image only
```

`scripts/publish.sh` aborts on an unclean working tree by default (ensures reproducible SHA tags). Use `--allow-dirty` to override during development.

## Image tags

| Tag         | PyTorch                | GPU support             | Rebuild trigger                      |
| ----------- | ---------------------- | ----------------------- | ------------------------------------ |
| `latest`    | 2.10+cpu               | None                    | Code or dep change                   |
| `cuda-base` | 2.6+cu124 + flash-attn | SM 7.0–9.0              | Dep / torch / flash-attn change only |
| `cuda`      | (FROM cuda-base)       | SM 7.0–9.0              | Every code change                    |
| `cuda128`   | 2.10+cu128             | SM 8.0+ (no flash-attn) | Dep change                           |

> **flash-attn is critical**: without it, Qwen3-TTS custom attention layers fall back to a Python CPU loop → 0% GPU utilisation. Requires SM 8.0+ (Ampere). On SM 7.x the `:cuda` image works but uses `sdpa`.

## Testing

```bash
make test               # pytest inside container (unit/integration)
make test-local         # modular shell test suite (tests/local/run-all.sh)
make test-local-fast    # fast subset: SKIP_SLOW=1 SKIP_DESIGN=1
make synth-test         # comprehensive CPU synthesis test matrix
make vast-remote-test   # full suite on a fresh vast.ai GPU instance

# Run only specific suite numbers:
ONLY="03 04" ./tests/local/run-all.sh
SKIP="17 18" ./tests/local/run-all.sh
```

Test suites in `tests/local/` are numbered `01`–`19` (shell scripts); pytest files are in `tests/*.py`.

## Voice registry schema

Voices are stored at `/cache/voices/<slug>/`:

- `voice.json` — metadata (slug, engine, ref info, prompts index, generation defaults)
- `ref.wav` — 24 kHz processed reference segment
- `prompts/<model_tag>_<mode>_v<N>.pkl` — serialised Qwen3-TTS prompt token tensors

**Engine types**: `clone_prompt`, `custom_voice`, `designed_clone`

`PROMPT_SCHEMA_VERSION` (defined in `lib/tts.py`) gates pickle compatibility — bump it when the prompt format changes.

## Remote deployment (vast.ai)

```bash
# Env: VAST_API_KEY (required), GHCR_TOKEN (recommended — avoids anonymous rate-limiting)
source .env

make vast-shell         # provision GPU + open interactive SSH shell
make vast-run ARGS='-- voice-synth speak --voice myvoice --text "Hello"'
make vast-run TASKS=my-jobs.txt   # tasks file: one app invocation per line
make vast-status        # list running instances
make vast-destroy ID=12345
```

`scripts/vast-deploy.sh` handles the full lifecycle: search offer → create instance → wait for running → authenticate GHCR → clone repo → run tasks → rsync `/work` back → destroy. Lock file at `/tmp/vast-deploy.lock` prevents concurrent runs.

Default GPU query: `reliability > 0.98 gpu_ram >= 20 compute_cap >= 800 compute_cap < 1200`. Exclude `compute_cap >= 1200` (Blackwell SM 12.0 / RTX 5xxx) — those require the `:cuda128` image.

### Cost discipline (critical)

- **Auto-destroy is the default** — `vast-deploy.sh` destroys the instance after tasks complete. Never use `--keep` unless actively debugging; always destroy manually when done.
- **No double-spending** — check `make vast-status` before provisioning. Never create a new instance if one is already running. The lock file `/tmp/vast-deploy.lock` prevents concurrent `vast-deploy.sh` runs but does not prevent manual double-provisioning.
- **Select best value** — sort by `dlperf_per_dphtotal` (DLPerf per dollar), not raw speed. Use `make vast-search` to compare offers before provisioning:
  ```bash
  make vast-search   # sorted by DLPerf/$ descending
  ```
- **Batch work** — combine multiple synthesis jobs into a single `--tasks` file rather than spinning up separate instances per job.
- **Prefer `--push-cache ./cache/voices`** — uploading the voice registry avoids re-cloning voices remotely (saves time and therefore money).
- **`VAST_MAX_MONTHLY_PRICE`** — deploy script prompts for confirmation when the selected offer exceeds this ceiling (default: $40/month ≈ $0.054/hr). Set lower to add a guardrail.

## Key environment variables (`.env`)

| Variable          | Purpose                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `VAST_API_KEY`    | vast.ai API key                                                          |
| `GHCR_TOKEN`      | GitHub PAT (`read:packages`) — authenticated GHCR pulls                  |
| `TOOLBOX_VARIANT` | `gpu` to overlay docker-compose.gpu.yml at runtime                       |
| `HF_HOME`         | Set to `/cache/huggingface` in image (models survive container restarts) |
| `TORCH_HOME`      | Set to `/cache/torch` in image                                           |
| `TORCH_DEVICE`    | Set to `cuda` in the CUDA image                                          |

## Docker / container conventions

- **No virtualenv**: `POETRY_VIRTUALENVS_CREATE=false` — all packages install into system Python as root
- **BuildKit cache mounts** on `/root/.cache/pypoetry` and `/root/.cache/pip` speed up rebuilds
- `ARG COMPUTE` (declared after expensive cached layers) controls which PyTorch wheels are installed — changing it only invalidates the thin CUDA-swap layer
- `/work` and `/cache` directories are baked into the image so they always exist even without a volume mount
