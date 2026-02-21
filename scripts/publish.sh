#!/usr/bin/env bash
# scripts/publish.sh — build and push voice-tools images to GHCR
#
# Usage:
#   ./scripts/publish.sh                   build + push: latest (cpu) and cuda
#   ./scripts/publish.sh cpu               CPU image only
#   ./scripts/publish.sh cuda              CUDA/GPU image only  (cu124, SM 7.0+)
#   ./scripts/publish.sh cuda128           CUDA/GPU image for Blackwell (cu128, SM 10.0+)
#   ./scripts/publish.sh --no-cache        force full rebuild (skips layer cache)
#   ./scripts/publish.sh --tag v1.2.3      also push a version tag
#   ./scripts/publish.sh --dry-run         print commands without running them
#   ./scripts/publish.sh cuda --no-cache --tag v2.0.0
#
# Images pushed:
#   ghcr.io/coriou/voice-tools:latest     CPU  (runs on any linux/amd64 host)
#   ghcr.io/coriou/voice-tools:cuda       GPU  (CUDA 12.4 / cu124, Volta SM 7.0+)
#   ghcr.io/coriou/voice-tools:cuda128    GPU  (CUDA 12.8 / cu128, Blackwell SM 10.0+)
#
# CPU is always built first — its layers are cached on disk, so the CUDA build
# only needs to push the single CUDA-swap layer delta (~2 GB vs ~6 GB total).

set -euo pipefail

REGISTRY="ghcr.io/coriou/voice-tools"
PLATFORM="linux/amd64"

# ── colour helpers ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m' RESET='\033[0m'
  GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m' RED='\033[0;31m' DIM='\033[2m'
else
  BOLD='' RESET='' GREEN='' CYAN='' YELLOW='' RED='' DIM=''
fi

hr()    { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────"; }
log()   { printf "${BOLD}==> %s${RESET}\n" "$*"; }
ok()    { printf "${GREEN}  ✓${RESET}  %s\n" "$*"; }
info()  { printf "${DIM}     %s${RESET}\n" "$*"; }
warn()  { printf "${YELLOW}  !${RESET}  %s\n" "$*"; }
die()   { printf "${RED}  ✗${RESET}  %s\n" "$*" >&2; exit 1; }
hms()   { local s=$(( SECONDS - $1 )); printf "%dm%02ds" $((s/60)) $((s%60)); }

# ── arg parsing ────────────────────────────────────────────────────────────────
VARIANTS=()
NO_CACHE=""
EXTRA_TAG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    cpu|cuda|cuda128|all) VARIANTS+=("$1"); shift ;;
    --no-cache)   NO_CACHE="--no-cache"; shift ;;
    --tag)        EXTRA_TAG="$2"; shift 2 ;;
    --tag=*)      EXTRA_TAG="${1#--tag=}"; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) die "Unknown argument: $1. Run with --help for usage." ;;
  esac
done

# Default: build both
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(cpu cuda)
# Expand "all"
for i in "${!VARIANTS[@]}"; do
  [[ "${VARIANTS[$i]}" == "all" ]] && { VARIANTS=(cpu cuda cuda128); break; }
done

# Deduplicate while preserving order; cpu always before cuda variants (cache efficiency)
# (bash 3 compatible — no associative arrays)
ordered=()
_contains() { local needle="$1"; shift; for el in "$@"; do [[ "$el" == "$needle" ]] && return 0; done; return 1; }
for v in cpu cuda cuda128; do
  for want in "${VARIANTS[@]}"; do
    if [[ "$want" == "$v" ]] && ! _contains "$v" "${ordered[@]+"${ordered[@]}"}"; then
      ordered+=("$v")
    fi
  done
done
VARIANTS=("${ordered[@]}")

# ── sanity checks ──────────────────────────────────────────────────────────────
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=""
[[ -n "$(git status --porcelain 2>/dev/null)" ]] && GIT_DIRTY="-dirty"

hr
printf "${BOLD}  voice-tools publish${RESET}  ${DIM}%s%s${RESET}\n" "$GIT_SHA" "$GIT_DIRTY"
info "registry : $REGISTRY"
info "platform : $PLATFORM"
info "variants : ${VARIANTS[*]}"
[[ -n "$EXTRA_TAG" ]] && info "extra tag: $EXTRA_TAG"
[[ -n "$NO_CACHE"  ]] && warn "layer cache disabled (--no-cache)"
[[ $DRY_RUN -eq 1  ]] && warn "DRY RUN — no images will be built or pushed"
hr

# Check GHCR login (skip in dry-run)
if [[ $DRY_RUN -eq 0 ]]; then
  if ! docker buildx imagetools inspect "${REGISTRY}:latest" &>/dev/null && \
     ! grep -q "ghcr.io" ~/.docker/config.json 2>/dev/null; then
    die "Not logged in to ghcr.io. Run: echo TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
  fi
fi

# ── build function ─────────────────────────────────────────────────────────────
build_variant() {
  local variant="$1"       # cpu | cuda
  local primary_tag
  local -a extra_args=()

  case "$variant" in
    cpu)
      primary_tag="${REGISTRY}:latest"
      # no COMPUTE arg → Dockerfile defaults to cpu (no-op swap layer)
      ;;
    cuda)
      primary_tag="${REGISTRY}:cuda"
      extra_args+=(--build-arg COMPUTE=cu124)
      ;;
    cuda128)
      primary_tag="${REGISTRY}:cuda128"
      extra_args+=(--build-arg COMPUTE=cu128)
      ;;
  esac

  local -a cmd=(
    docker buildx build
    --platform "$PLATFORM"
    -t "$primary_tag"
    -t "${REGISTRY}:${variant}-${GIT_SHA}${GIT_DIRTY}"
  )
  [[ -n "$EXTRA_TAG" ]] && cmd+=(-t "${REGISTRY}:${EXTRA_TAG}$([[ $variant == cuda128 ]] && echo -cuda128 || [[ $variant == cuda ]] && echo -cuda || echo "")")
  cmd+=("${extra_args[@]}")
  [[ -n "$NO_CACHE" ]] && cmd+=("$NO_CACHE")
  cmd+=(--push .)

  local t0=$SECONDS
  hr
  log "Building ${BOLD}${variant}${RESET}  →  ${CYAN}${primary_tag}${RESET}"
  info "also tagging: ${variant}-${GIT_SHA}${GIT_DIRTY}"
  [[ -n "$EXTRA_TAG" ]] && info "also tagging: ${EXTRA_TAG}$([[ $variant == cuda ]] && echo -cuda || echo "")"
  echo

  if [[ $DRY_RUN -eq 1 ]]; then
    printf "${DIM}  $ %s${RESET}\n" "${cmd[*]}"
  else
    "${cmd[@]}"
    echo
    ok "${variant} pushed  $(hms $t0)"
  fi
}

# ── run ────────────────────────────────────────────────────────────────────────
T_START=$SECONDS

for variant in "${VARIANTS[@]}"; do
  build_variant "$variant"
done

hr
if [[ $DRY_RUN -eq 1 ]]; then
  printf "${YELLOW}  dry run complete — no images were pushed${RESET}\n"
else
  printf "${GREEN}${BOLD}  ✓  Done${RESET}  ${DIM}$(hms $T_START) total  ·  %d variant(s) pushed${RESET}\n" "${#VARIANTS[@]}"
  echo
  printf "${DIM}  Pull with:${RESET}\n"
  for v in "${VARIANTS[@]}"; do
    [[ "$v" == cpu  ]] && printf "    docker pull ${CYAN}${REGISTRY}:latest${RESET}\n"
    [[ "$v" == cuda ]] && printf "    docker pull ${CYAN}${REGISTRY}:cuda${RESET}\n"
  done
fi
hr
