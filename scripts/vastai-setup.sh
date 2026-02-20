#!/usr/bin/env bash
# scripts/vastai-setup.sh — one-shot setup on a vast.ai instance
#
# Run this on the instance as root after SSHing in:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Coriou/py-audio-box/main/scripts/vastai-setup.sh)
#
# Designed for the "PyTorch (Vast)" template (vastai/pytorch) which ships
# Python 3.11, pip, and CUDA. No Docker needed — scripts run directly.
#
# What it does:
#   1. Clones the repo to ~/py-audio-box
#   2. Installs Python deps via pip (from poetry.lock)
#   3. Installs system deps (ffmpeg, sox) if missing
#   4. Creates /work and /cache symlinks so default paths just work
#   5. Verifies GPU + prints a ready-to-run example

set -euo pipefail

REPO_URL="https://github.com/Coriou/py-audio-box.git"
REPO_DIR="${HOME}/py-audio-box"

# ── colour helpers ─────────────────────────────────────────────────────────────
BOLD='\033[1m' RESET='\033[0m'
GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m' RED='\033[0;31m' DIM='\033[2m'
hr()   { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────"; }
log()  { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}  ✓${RESET}  %s\n" "$*"; }
info() { printf "${DIM}     %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  !${RESET}  %s\n" "$*"; }
die()  { printf "${RED}  ✗  %s${RESET}\n" "$*" >&2; exit 1; }
hms()  { local s=$(( SECONDS - $1 )); printf "%dm%02ds" $((s/60)) $((s%60)); }

hr
printf "${BOLD}  py-audio-box · vast.ai setup${RESET}\n"
hr

# ── 1. clone repo ──────────────────────────────────────────────────────────────
log "Cloning repo"
if [[ -d "$REPO_DIR/.git" ]]; then
  warn "Repo already exists — pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
ok "Repo at $REPO_DIR"

cd "$REPO_DIR"

# ── 2. python version check ────────────────────────────────────────────────────
log "Checking Python"
PYTHON=$(command -v python3.11 || command -v python3 || command -v python)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VER at $PYTHON"
[[ "${PY_VER}" < "3.10" ]] && die "Python 3.10+ required (got $PY_VER)"

# ── 3. system deps ─────────────────────────────────────────────────────────────
log "System deps (ffmpeg, sox)"
MISSING=()
command -v ffmpeg &>/dev/null || MISSING+=(ffmpeg)
command -v sox    &>/dev/null || MISSING+=(sox)
if [[ ${#MISSING[@]} -gt 0 ]]; then
  info "Installing: ${MISSING[*]}"
  apt-get update -qq && apt-get install -y -qq "${MISSING[@]}"
fi
ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
ok "sox    $(sox --version 2>&1 | awk '{print $NF}')"

# ── 4. python deps ─────────────────────────────────────────────────────────────
log "Python deps"
T0=$SECONDS

# Install poetry if not present.
if ! command -v poetry &>/dev/null; then
  info "Installing poetry …"
  pip install -q poetry
fi

# Install all deps into the system Python (no virtualenv).
info "Running poetry install --no-root …"
poetry config virtualenvs.create false
poetry install --no-root --no-interaction -q

# Detect CUDA version to pick the right torch wheel index.
CUDA_VER=$(nvidia-smi 2>/dev/null \
  | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
if [[ -z "$CUDA_VER" ]]; then
  warn "nvidia-smi not found or no GPU — skipping CUDA torch reinstall"
  TORCH_INDEX=""
elif "$PYTHON" -c "v='${CUDA_VER}'.split('.'); exit(0 if (int(v[0]),int(v[1])) >= (12,8) else 1)" 2>/dev/null; then
  info "CUDA $CUDA_VER → reinstalling torch with cu128 wheels"
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif "$PYTHON" -c "v='${CUDA_VER}'.split('.'); exit(0 if (int(v[0]),int(v[1])) >= (12,4) else 1)" 2>/dev/null; then
  info "CUDA $CUDA_VER → reinstalling torch with cu124 wheels"
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else
  info "CUDA $CUDA_VER → reinstalling torch with cu121 wheels"
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

# Reinstall torch + torchaudio from the CUDA index (poetry installed CPU ones).
if [[ -n "$TORCH_INDEX" ]]; then
  pip install -q --no-cache-dir torch torchaudio --index-url "$TORCH_INDEX"
fi

ok "Python deps installed  ($(hms $T0))"

# ── 5. work / cache dirs + symlinks ───────────────────────────────────────────
log "Dirs & symlinks"
mkdir -p "$REPO_DIR/work" "$REPO_DIR/cache"

# /work and /cache symlinks let the apps use their default paths without flags.
ln -sfn "$REPO_DIR/work"  /work  2>/dev/null || true
ln -sfn "$REPO_DIR/cache" /cache 2>/dev/null || true
chmod +x "$REPO_DIR/run-direct"
ok "/work  → $REPO_DIR/work"
ok "/cache → $REPO_DIR/cache"

# ── 6. GPU sanity check ────────────────────────────────────────────────────────
log "GPU check"
"$PYTHON" - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("  WARN: CUDA not available — CPU mode only")
else:
    name = torch.cuda.get_device_name(0)
    sm   = torch.cuda.get_device_capability(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    ver  = torch.version.cuda
    print(f"  {name}  |  SM {sm[0]}.{sm[1]}  |  {mem} GB VRAM  |  CUDA {ver}")
PYEOF

# ── 7. persist TORCH_DEVICE=cuda ──────────────────────────────────────────────
if ! grep -q 'TORCH_DEVICE' ~/.bashrc 2>/dev/null; then
  echo 'export TORCH_DEVICE=cuda  # set by py-audio-box vastai-setup' >> ~/.bashrc
fi
export TORCH_DEVICE=cuda

# ── done ──────────────────────────────────────────────────────────────────────
hr
printf "${GREEN}${BOLD}  ✓  Ready in $(hms 0)!${RESET}\n\n"
printf "  ${CYAN}cd ${REPO_DIR}${RESET}\n\n"
printf "  ${BOLD}# Register a voice from YouTube:${RESET}\n"
printf "  ${BOLD}./run-direct voice-register \\\\\n"
printf "    --url \"https://www.youtube.com/watch?v=XXXX\" \\\\\n"
printf "    --voice-name my-voice \\\\\n"
printf "    --text \"Hello world.\"\n${RESET}\n"
printf "  ${BOLD}# Synthesise:${RESET}\n"
printf "  ${BOLD}./run-direct voice-synth speak --voice my-voice --text \"Hello.\"\n${RESET}\n"
hr
