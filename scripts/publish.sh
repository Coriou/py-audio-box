#!/usr/bin/env bash
# scripts/publish.sh — build and push voice-tools images to GHCR
#
# Usage:
#   ./scripts/publish.sh                   build + push: latest (cpu) and cuda
#   ./scripts/publish.sh cpu               CPU image only
#   ./scripts/publish.sh cuda              CUDA/GPU image only  (cu124, Volta SM 7.0+)
#   ./scripts/publish.sh cuda128           CUDA/GPU image for Blackwell (cu128, SM 10.0+)
#   ./scripts/publish.sh all               All three variants
#   ./scripts/publish.sh --no-cache        force full rebuild (skips layer cache)
#   ./scripts/publish.sh --tag v1.2.3      also push a named version tag
#   ./scripts/publish.sh --dry-run         print commands without running them
#   ./scripts/publish.sh cuda --no-cache --tag v2.0.0
#
# Dirty-tree policy:
#   By default the script aborts if the working tree has uncommitted changes.
#   This ensures every pushed image is reproducible from a known commit.
#   Use --allow-dirty to override (no immutable SHA tag will be pushed).
#
# Images pushed:
#   ghcr.io/coriou/voice-tools:latest      CPU  (any linux/amd64 host)
#   ghcr.io/coriou/voice-tools:cuda-base   GPU base — torch+flash-attn, no app src (~4 GB, rarely rebuilt)
#   ghcr.io/coriou/voice-tools:cuda        GPU app  — FROM cuda-base + COPY app (~50 MB, rebuilt on code change)
#   ghcr.io/coriou/voice-tools:cuda128     GPU  (CUDA 12.8, Blackwell SM 10.0+)
#
# Two-step CUDA workflow:
#   make publish-cuda-base   ← run once (or when deps/torch/flash-attn change)
#   make publish-cuda        ← run on every code change (fast: only 50 MB layer)
#
# Each clean build also pushes an immutable SHA tag for pinning:
#   ghcr.io/coriou/voice-tools:latest-<sha>
#   ghcr.io/coriou/voice-tools:cuda-base-<sha>
#   ghcr.io/coriou/voice-tools:cuda-<sha>
#   ghcr.io/coriou/voice-tools:cuda128-<sha>

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
die()   { printf "${RED}  ✗  ERROR: %s${RESET}\n" "$*" >&2; exit 1; }
hms()   { local s=$(( SECONDS - $1 )); printf "%dm%02ds" $((s/60)) $((s%60)); }

# ── arg parsing ────────────────────────────────────────────────────────────────
VARIANTS=()
NO_CACHE=""
EXTRA_TAG=""
DRY_RUN=0
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    cpu|cuda-base|cuda|cuda128|all) VARIANTS+=("$1"); shift ;;
    --no-cache)    NO_CACHE="--no-cache"; shift ;;
    --tag)         EXTRA_TAG="$2"; shift 2 ;;
    --tag=*)       EXTRA_TAG="${1#--tag=}"; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) die "Unknown argument: $1. Run with --help for usage." ;;
  esac
done

# Default: build cpu + cuda
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(cpu cuda)

# Expand "all"
for i in "${!VARIANTS[@]}"; do
  [[ "${VARIANTS[$i]}" == "all" ]] && { VARIANTS=(cpu cuda-base cuda cuda128); break; }
done

# Deduplicate while preserving order: cpu always before cuda variants (cache efficiency)
# (bash 3 compatible — no associative arrays)
_contains() { local needle="$1"; shift; for el in "$@"; do [[ "$el" == "$needle" ]] && return 0; done; return 1; }
ordered=()
for v in cpu cuda-base cuda cuda128; do
  for want in "${VARIANTS[@]}"; do
    if [[ "$want" == "$v" ]] && ! _contains "$v" "${ordered[@]+"${ordered[@]}"}"; then
      ordered+=("$v")
    fi
  done
done
VARIANTS=("${ordered[@]}")

# ── git state ──────────────────────────────────────────────────────────────────
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=""
[[ -n "$(git status --porcelain 2>/dev/null)" ]] && GIT_DIRTY="true"

hr
printf "${BOLD}  voice-tools publish${RESET}  ${DIM}%s${RESET}\n" "$GIT_SHA"
info "registry : $REGISTRY"
info "platform : $PLATFORM"
info "variants : ${VARIANTS[*]}"
[[ -n "$EXTRA_TAG"  ]] && info "extra tag: $EXTRA_TAG"
[[ -n "$NO_CACHE"   ]] && warn "layer cache disabled (--no-cache)"
[[ $DRY_RUN -eq 1   ]] && warn "DRY RUN — no images will be built or pushed"
hr

# ── dirty-tree guard ───────────────────────────────────────────────────────────
if [[ -n "$GIT_DIRTY" ]]; then
  if [[ $ALLOW_DIRTY -eq 1 ]]; then
    warn "Working tree is dirty — only stable tags will be pushed (no immutable SHA tag)."
    warn "Commit your changes to get a pinnable :cuda-<sha> tag."
    echo
  else
    die "Working tree has uncommitted changes.
  Every published image must be reproducible from a known commit.
  Fix:
    git add -A && git commit -m '...'   # commit your changes, then re-run
    OR
    $(basename "$0") --allow-dirty      # override (skips the SHA tag)"
  fi
fi

# ── login check ───────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
  if ! grep -q "ghcr.io" ~/.docker/config.json 2>/dev/null; then
    die "Not logged in to ghcr.io. Run: echo TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
  fi
fi

# ── build function ─────────────────────────────────────────────────────────────
build_variant() {
  local variant="$1"
  local stable_tag
  local -a extra_args=()

  local dockerfile="Dockerfile"
  case "$variant" in
    cpu)
      stable_tag="${REGISTRY}:latest"
      ;;
    cuda-base)
      stable_tag="${REGISTRY}:cuda-base"
      dockerfile="Dockerfile.cuda-base"
      ;;
    cuda)
      stable_tag="${REGISTRY}:cuda"
      # Thin app layer — FROM cuda-base + COPY . /app (~50 MB)
      dockerfile="Dockerfile.cuda"
      ;;
    cuda128)
      stable_tag="${REGISTRY}:cuda128"
      extra_args+=(--build-arg COMPUTE=cu128)
      ;;
  esac

  # Immutable SHA tag — only on clean builds so GHCR is never polluted with -dirty refs
  local sha_tag="${REGISTRY}:${variant}-${GIT_SHA}"
  [[ "$variant" == "cpu" ]] && sha_tag="${REGISTRY}:latest-${GIT_SHA}"

  local -a cmd=(
    docker buildx build
    --platform "$PLATFORM"
    -f "$dockerfile"
    -t "$stable_tag"
  )

  # SHA tag only when tree is clean
  [[ -z "$GIT_DIRTY" ]] && cmd+=(-t "$sha_tag")

  # Optional named version tag (e.g. --tag v1.2.3)
  if [[ -n "$EXTRA_TAG" ]]; then
    local version_tag="${REGISTRY}:${EXTRA_TAG}"
    [[ "$variant" == "cuda"    ]] && version_tag="${REGISTRY}:${EXTRA_TAG}-cuda"
    [[ "$variant" == "cuda128" ]] && version_tag="${REGISTRY}:${EXTRA_TAG}-cuda128"
    cmd+=(-t "$version_tag")
  fi

  [[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")
  [[ -n "$NO_CACHE" ]] && cmd+=("$NO_CACHE")
  cmd+=(--push .)

  local t0=$SECONDS
  hr
  log "Building ${BOLD}${variant}${RESET}  →  ${CYAN}${stable_tag}${RESET}"
  if [[ -z "$GIT_DIRTY" ]]; then
    info "immutable : $sha_tag"
  fi
  [[ -n "$EXTRA_TAG" ]] && info "version   : ${REGISTRY}:${EXTRA_TAG}$([[ $variant == cuda ]] && echo -cuda || [[ $variant == cuda128 ]] && echo -cuda128 || echo "")"
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
  printf "${GREEN}${BOLD}  ✓  All done${RESET}  ${DIM}$(hms $T_START) total · ${#VARIANTS[@]} variant(s)${RESET}\n"
  echo
  printf "${DIM}  Pull the latest:${RESET}\n"
  for v in "${VARIANTS[@]}"; do
    case "$v" in
      cpu)       printf "    docker pull ${CYAN}${REGISTRY}:latest${RESET}\n" ;;
      cuda-base) printf "    docker pull ${CYAN}${REGISTRY}:cuda-base${RESET}  ${DIM}(heavy base, ~4 GB)${RESET}\n" ;;
      cuda)      printf "    docker pull ${CYAN}${REGISTRY}:cuda${RESET}  ${DIM}(thin app layer on cuda-base, ~50 MB)${RESET}\n" ;;
      cuda128)   printf "    docker pull ${CYAN}${REGISTRY}:cuda128${RESET}\n" ;;
    esac
  done
  if [[ -z "$GIT_DIRTY" ]]; then
    echo
    printf "${DIM}  Pin to this exact build (${GIT_SHA}):${RESET}\n"
    for v in "${VARIANTS[@]}"; do
      local_prefix="latest"
      [[ "$v" != "cpu" ]] && local_prefix="$v"
      printf "    docker pull ${DIM}${REGISTRY}:${local_prefix}-${GIT_SHA}${RESET}\n"
    done
    echo
    printf "${DIM}  Tip: run 'make publish-cuda-base' only when deps change,${RESET}\n"
    printf "${DIM}       then 'make publish-cuda' for code-only updates (~50 MB push).${RESET}\n"
  fi
fi
hr
