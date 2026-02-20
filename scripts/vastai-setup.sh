#!/usr/bin/env bash
# scripts/vastai-setup.sh — first-SSH init when running ghcr.io/coriou/voice-tools:cuda
#                           as the vast.ai Docker image.
#
# HOW TO USE:
#   1. On vast.ai "Edit instance" / template, set:
#        Docker Image:  ghcr.io/coriou/voice-tools:cuda
#        On-start cmd:  cd /app && git pull --ff-only && mkdir -p /work /cache
#   2. SSH in — deps are already installed (baked into the image)
#   3. Run:  TORCH_DEVICE=cuda ./run-direct voice-register ...
#
# This script is only needed if you want a one-shot interactive check / first-run
# setup after SSH-ing in.  For automated starts, use the "On-start cmd" field above.

set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
[[ -d "$APP_DIR/.git" ]] || APP_DIR="${HOME}/py-audio-box"

BOLD='\033[1m' RESET='\033[0m'
GREEN='\033[0;32m' CYAN='\033[0;36m' DIM='\033[2m' YELLOW='\033[1;33m'
hr()   { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────"; }
log()  { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}  ✓${RESET}  %s\n" "$*"; }
info() { printf "${DIM}     %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  !${RESET}  %s\n" "$*"; }

hr
printf "${BOLD}  py-audio-box · vast.ai instance init${RESET}\n"
hr

# ── 1. pull latest code ────────────────────────────────────────────────────────
log "Code"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
  ok "Pulled latest → $APP_DIR"
else
  warn "No git repo at $APP_DIR — image code is at the build-time version"
fi
cd "$APP_DIR"

# ── 2. verify torch + CUDA ─────────────────────────────────────────────────────
log "GPU check"
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('  WARN: CUDA not available (CPU mode)')
else:
    name = torch.cuda.get_device_name(0)
    sm   = torch.cuda.get_device_capability(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f'  {name}  |  SM {sm[0]}.{sm[1]}  |  {mem} GB VRAM  |  CUDA {torch.version.cuda}')
    print(f'  torch {torch.__version__}')
"

# ── 3. ensure dirs exist ───────────────────────────────────────────────────────
mkdir -p /work /cache
chmod +x "$APP_DIR/run-direct" 2>/dev/null || true
export TORCH_DEVICE=cuda
if ! grep -q 'TORCH_DEVICE' ~/.bashrc 2>/dev/null; then
  echo 'export TORCH_DEVICE=cuda' >> ~/.bashrc
fi

# ── done ──────────────────────────────────────────────────────────────────────
hr
printf "${GREEN}${BOLD}  ✓  Ready!${RESET}\n\n"
printf "  ${CYAN}cd ${APP_DIR}${RESET}\n\n"
printf "  ${BOLD}TORCH_DEVICE=cuda ./run-direct voice-register \\\\\n"
printf "    --url \"https://www.youtube.com/watch?v=XXXX\" \\\\\n"
printf "    --voice-name my-voice \\\\\n"
printf "    --text \"Hello world.\"\n${RESET}\n"
hr
printf "    --url \"https://www.youtube.com/watch?v=XXXX\" \\\\\n"
printf "    --voice-name my-voice \\\\\n"
printf "    --text \"Hello world.\"\n${RESET}\n"
printf "  ${BOLD}# Synthesise:${RESET}\n"
printf "  ${BOLD}./run-direct voice-synth speak --voice my-voice --text \"Hello.\"\n${RESET}\n"
hr
