#!/usr/bin/env bash
# scripts/vastai-setup.sh — one-shot setup on a fresh vast.ai instance
#
# Run this on the vast.ai instance as root after SSHing in:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Coriou/py-audio-box/main/scripts/vastai-setup.sh)
#
# Or copy & paste the whole file.
#
# What it does:
#   1. Clones the repo to ~/py-audio-box
#   2. Pulls ghcr.io/coriou/voice-tools:cuda (skips build entirely)
#   3. Verifies GPU passthrough + CUDA version
#   4. Prints a ready-to-run example command

set -euo pipefail

REPO_URL="https://github.com/Coriou/py-audio-box.git"
REPO_DIR="${HOME}/py-audio-box"
REGISTRY="ghcr.io/coriou/voice-tools"

# Auto-select image tag based on host GPU SM version.
# Run a quick nvidia-smi check (available on vast.ai hosts without Docker).
detect_image_tag() {
  local sm
  sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  if [[ -n "$sm" ]] && [[ "$sm" -ge 100 ]]; then
    echo "cuda128"   # Blackwell SM 10.0+ → cu128 wheels
  else
    echo "cuda"      # everything else (Volta SM 7.0+) → cu124 wheels
  fi
}

IMAGE_TAG=$(detect_image_tag)
IMAGE="${REGISTRY}:${IMAGE_TAG}"

# ── colour helpers ─────────────────────────────────────────────────────────────
BOLD='\033[1m' RESET='\033[0m'
GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m' RED='\033[0;31m' DIM='\033[2m'
hr()   { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────"; }
log()  { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}  ✓${RESET}  %s\n" "$*"; }
info() { printf "${DIM}     %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  !${RESET}  %s\n" "$*"; }
die()  { printf "${RED}  ✗  %s${RESET}\n" "$*" >&2; exit 1; }

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

# ── 2. ensure docker compose v2 is available ──────────────────────────────────
log "Checking Docker"
docker info &>/dev/null || die "Docker daemon not running"
# vast.ai machines have docker-compose-plugin; fall back to standalone
if ! docker compose version &>/dev/null; then
  warn "docker compose plugin missing — installing"
  apt-get update -qq && apt-get install -y -qq docker-compose-plugin
fi
ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"
ok "Compose $(docker compose version --short)"

# ── 3. pull CUDA image ─────────────────────────────────────────────────────────
log "Pulling $IMAGE"
if [[ "$IMAGE_TAG" == "cuda128" ]]; then
  info "Blackwell detected — using cu128 wheels (:cuda128)"
  # Fallback: if cuda128 hasn't been published yet, try cuda with a warning
  if ! docker pull "$IMAGE" 2>/dev/null; then
    warn ":cuda128 not yet published — falling back to :cuda (PTX JIT on first run)"
    IMAGE_TAG="cuda"
    IMAGE="${REGISTRY}:cuda"
    docker pull "$IMAGE"
  fi
else
  info "Using cu124 wheels (:cuda) — Volta SM 7.0+ optimised"
  docker pull "$IMAGE"
fi
docker pull "$IMAGE"
ok "Image pulled"

# ── 4. tag as local name so docker-compose.gpu.yml resolves it ────────────────
docker tag "$IMAGE" voice-tools:cuda 2>/dev/null || true
info "Tagged as voice-tools:cuda"

# ── 5. GPU sanity check ────────────────────────────────────────────────────────
log "Verifying GPU passthrough"
GPU_INFO=$(docker run --rm --gpus all "$IMAGE" python -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available'); sys.exit(1)
name = torch.cuda.get_device_name(0)
sm   = torch.cuda.get_device_capability(0)
mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
ver  = torch.version.cuda
print(f'{name}  |  SM {sm[0]}.{sm[1]}  |  {mem} GB  |  CUDA {ver}')
" 2>/dev/null)
ok "GPU: $GPU_INFO"

# ── 6. create work / cache dirs ───────────────────────────────────────────────
mkdir -p "$REPO_DIR/work" "$REPO_DIR/cache"

# ── done ──────────────────────────────────────────────────────────────────────
hr
printf "${GREEN}${BOLD}  ✓  Ready!${RESET}\n\n"
printf "  ${DIM}cd${RESET} ${CYAN}${REPO_DIR}${RESET}\n"
printf "  ${DIM}then run e.g.:${RESET}\n\n"
printf "  ${BOLD}TOOLBOX_VARIANT=gpu ./run voice-register \\\\\n"
printf "    --url \"https://www.youtube.com/watch?v=XXXX\" \\\\\n"
printf "    --voice-name my-voice \\\\\n"
printf "    --text \"Hello world.\"\n${RESET}\n"
hr
