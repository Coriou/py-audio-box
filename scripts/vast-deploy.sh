#!/usr/bin/env bash
# scripts/vast-deploy.sh — full vast.ai lifecycle: provision → sync → run tasks → pull → destroy
#
# Usage:
#   ./scripts/vast-deploy.sh [OPTIONS] [-- TASK [TASK...]]
#
# Basic examples:
#   # Provision + run one task + pull results + destroy
#   ./scripts/vast-deploy.sh -- voice-synth speak --voice myvoice --text "Hello world"
#
#   # Run several tasks from a file (one ./run-direct invocation per non-comment line)
#   ./scripts/vast-deploy.sh --tasks my-jobs.txt
#
#   # Open an interactive SSH shell (no tasks, instance not auto-destroyed)
#   ./scripts/vast-deploy.sh --shell
#
#   # Dry-run: print every command, touch nothing
#   ./scripts/vast-deploy.sh --dry-run -- voice-synth speak --voice x --text "hi"
#
# Environment:
#   VAST_API_KEY   required — your vast.ai API key (console.vast.ai → CLI)
#   GHCR_TOKEN     recommended — GitHub PAT with read:packages scope so the remote host
#                  pulls ghcr.io as YOU rather than anonymously. Anonymous GHCR pulls are
#                  rate-limited and slow (~3 MB/s); authenticated pulls are full-speed.
#                  To create: github.com → Settings → Developer settings → Personal access
#                  tokens → Tokens (classic) → read:packages. Add to .env:
#                    GHCR_TOKEN=ghp_xxxxxxxxxxxx
#   GHCR_USER      GHCR username to authenticate as (default: Coriou)
#
# Search customisation:
#   VAST_QUERY              override the GPU search query string (see: vastai search offers --help)
#   VAST_IMAGE              override the Docker image to deploy  (default: ghcr.io/coriou/voice-tools:cuda)
#   VAST_DISK               override disk size in GB             (default: 60)
#   VAST_REPO               Git repo to clone into /app          (default: https://github.com/Coriou/py-audio-box)
#   VAST_MAX_MONTHLY_PRICE  price ceiling in $/month (default: 40).  Prompts for confirmation
#                           when the selected offer exceeds this.  Set to 0 to disable.
#
# Options:
#   --tasks FILE     tasks file (one "./run-direct app [args]" per non-comment line)
#   --shell          provision and drop into an interactive SSH session
#   --job NAME       label / job name (default: py-audio-box-YYYYMMDD_HHMMSS)
#   --ssh-key PATH   path to SSH private key (default: auto-detect ~/.ssh/id_{ed25519,rsa,ecdsa})
#   --no-clone       skip git clone/pull of repo into /app on the instance
#   --no-pull        skip rsyncing /work back to work_remote/ after tasks
#   --keep           don't destroy the instance when done (useful for debugging)
#   --dry-run        print commands, touch nothing
#   --yes            skip all interactive prompts (useful in CI / scripted pipelines)
#   -h, --help       show this help
#
# Task file format (--tasks FILE):
#   # comment lines are ignored
#   voice-register --url "https://youtu.be/XXXX" --voice-name myvoice --text "Hello."
#   voice-synth speak --voice myvoice --text "Generated on vast.ai"
#   !bash /app/tests/remote/run-all.sh       # raw shell command (! prefix bypasses run-direct)
#
# Each line becomes:  ./run-direct <line>  on the remote instance,
# UNLESS the line starts with '!' — then it is run as a raw shell command.

set -euo pipefail

# Load .env from repo root so VAST_API_KEY (and other vars) can live there
_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_DEPLOY_DIR}/../.env" ]] && { set -o allexport; source "${_DEPLOY_DIR}/../.env"; set +o allexport; }
unset _DEPLOY_DIR

# ── colour helpers (same palette as publish.sh) ────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m' RESET='\033[0m'
  GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m' RED='\033[0;31m' DIM='\033[2m'
else
  BOLD='' RESET='' GREEN='' CYAN='' YELLOW='' RED='' DIM=''
fi

hr()   { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────"; }
log()  { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}  ✓${RESET}  %s\n" "$*"; }
info() { printf "${DIM}     %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  !${RESET}  %s\n" "$*"; }
die()  { printf "${RED}  ✗  ERROR: %s${RESET}\n" "$*" >&2; exit 1; }
hms()  { local s=$(( SECONDS - ${1:-0} )); printf "%dm%02ds" $((s/60)) $((s%60)); }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── config / defaults ──────────────────────────────────────────────────────────
VAST_CLI="${REPO_ROOT}/scripts/vast"
DEPLOY_IMAGE="${VAST_IMAGE:-ghcr.io/coriou/voice-tools:cuda}"
DISK_GB="${VAST_DISK:-60}"
REPO_URL="${VAST_REPO:-https://github.com/Coriou/py-audio-box}"
JOB_NAME=""
TASKS_FILE=""
TASKS=()
SSH_KEY=""
SHELL_MODE=0
NO_CLONE=0
NO_PULL=0
PUSH_CACHE_SRC=""
PUSH_WORK_SRC=""
KEEP=0
DRY_RUN=0
ASKED_CONFIRM=0  # 1 once the user has already said yes to a price prompt

# Maximum price guard: prompt (or abort) when the selected offer costs more than
# this many dollars per month.  Set to 0 to disable the guard entirely.
# Override via VAST_MAX_MONTHLY_PRICE env var (e.g. in .env).
MAX_MONTHLY_PRICE="${VAST_MAX_MONTHLY_PRICE:-100}"

# GHCR credentials: when set, passed as --login so the remote host authenticates
# against ghcr.io instead of pulling anonymously (removes rate-limiting / slow CDN).
GHCR_USER="${GHCR_USER:-Coriou}"
GHCR_TOKEN="${GHCR_TOKEN:-}"

# Default GPU search query: reliable Ampere/Ada GPU with enough VRAM for Qwen-TTS,
# fast inet, and enough disk; ordered cheapest-first.
# compute_cap >= 800 (Ampere) and < 1200 (excludes Blackwell sm_120 which needs cu128+).
# Override via VAST_QUERY env var.
# inet_down >= 500 ensures at least 500 Mb/s downlink — pulls our ~4 GB CUDA
# image in ~60s instead of 5+ minutes on slower hosts.
DEFAULT_QUERY='reliability > 0.98 gpu_ram >= 20 compute_cap >= 800 compute_cap < 1200 inet_down >= 500 disk_space >= 50 rented=False'

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)          TASKS_FILE="$2"; shift 2 ;;
    --shell)          SHELL_MODE=1; shift ;;
    --job)            JOB_NAME="$2"; shift 2 ;;
    --ssh-key)        SSH_KEY="$2"; shift 2 ;;
    --no-clone)       NO_CLONE=1; shift ;;
    --no-sync)        NO_CLONE=1; shift ;;  # backwards compat alias
    --no-pull)        NO_PULL=1; shift ;;
    --push-cache)     PUSH_CACHE_SRC="${2:-./cache/voices}"; shift 2 ;;
    --push-work)      PUSH_WORK_SRC="$2"; shift 2 ;;
    --keep)           KEEP=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --yes|-y)         ASKED_CONFIRM=1; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \?//'
      exit 0
      ;;
    --)               shift; TASKS+=("$@"); break ;;
    -*) die "Unknown option: $1.  Use --help for usage." ;;
    *)  TASKS+=("$1"); shift ;;
  esac
done

# ── prerequisites ──────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "Docker not running or not in PATH."
command -v ssh    >/dev/null 2>&1 || die "ssh not found."
command -v rsync  >/dev/null 2>&1 || die "rsync not found (used to pull /work results).  Install with: brew install rsync"
command -v jq     >/dev/null 2>&1 || die "jq not found.  Install with: brew install jq"

[[ -x "$VAST_CLI" ]] || die "scripts/vast not found or not executable. Run: chmod +x scripts/vast"

if [[ -z "${VAST_API_KEY:-}" ]]; then
  KEY_FILE="${HOME}/.config/vastai/vast_api_key"
  [[ -f "$KEY_FILE" ]] || die "VAST_API_KEY env var not set and $KEY_FILE not found.\n  Get your key: https://cloud.vast.ai/console/cli/"
fi

# ── resolve tasks ──────────────────────────────────────────────────────────────
if [[ -n "$TASKS_FILE" ]]; then
  [[ -f "$TASKS_FILE" ]] || die "Tasks file not found: $TASKS_FILE"
  while IFS= read -r line; do
    # skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    TASKS+=("$line")
  done < "$TASKS_FILE"
fi

if [[ ${#TASKS[@]} -eq 0 && $SHELL_MODE -eq 0 ]]; then
  die "No tasks provided.  Use -- TASK [TASK...], --tasks FILE, or --shell.\n  See --help for usage."
fi

# ── resolve SSH key ────────────────────────────────────────────────────────────
# VAST_SSH_KEY env var (set in .env) takes precedence over --ssh-key and auto-detect.
# It must match the public key registered in your vast.ai account settings.
if [[ -z "$SSH_KEY" && -n "${VAST_SSH_KEY:-}" ]]; then
  SSH_KEY="${VAST_SSH_KEY}"
  # expand tilde manually (env vars loaded via `source` don't get tilde expansion)
  SSH_KEY="${SSH_KEY/#\~/$HOME}"
fi
if [[ -z "$SSH_KEY" ]]; then
  for candidate in ~/.ssh/ssh_key ~/.ssh/id_rsa ~/.ssh/id_ed25519 ~/.ssh/id_ecdsa ~/.ssh/id_dsa; do
    if [[ -f "$candidate" ]]; then
      SSH_KEY="$candidate"
      break
    fi
  done
fi
[[ -n "$SSH_KEY" && -f "$SSH_KEY" ]] || die "No SSH private key found.  Set VAST_SSH_KEY in .env or use --ssh-key PATH."

# ── job name ───────────────────────────────────────────────────────────────────
[[ -z "$JOB_NAME" ]] && JOB_NAME="py-audio-box-$(date +%Y%m%d_%H%M%S)"

# ── dry-run wrapper ────────────────────────────────────────────────────────────
# vast_cmd / ssh_cmd print the command instead of running it in dry-run.
# rsync_cmd is used only for pulling /work results back to the host.
vast_cmd()  {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "${DIM}  [dry] vast %s${RESET}\n" "$*"
    echo '{}'
  else
    "$VAST_CLI" "$@"
  fi
}
ssh_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "${DIM}  [dry] ssh %s${RESET}\n" "$*"
  else
    ssh "$@"
  fi
}
rsync_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "${DIM}  [dry] rsync %s${RESET}\n" "$*"
  else
    rsync "$@"
  fi
}

# ── instance state ─────────────────────────────────────────────────────────────
INSTANCE_ID=""
INSTANCE_HOST=""
INSTANCE_PORT=""
SSH_KNOWN_HOSTS_FILE="/tmp/vast_known_hosts_${JOB_NAME}"

# ── cleanup / trap ─────────────────────────────────────────────────────────────
cleanup() {
  local code=$?
  rm -f "$SSH_KNOWN_HOSTS_FILE"
  if [[ -n "$INSTANCE_ID" && $KEEP -eq 0 && $SHELL_MODE -eq 0 ]]; then
    echo
    warn "Cleaning up: destroying instance ${INSTANCE_ID} …"
    "$VAST_CLI" destroy instance "$INSTANCE_ID" 2>/dev/null || true
    ok "Instance ${INSTANCE_ID} destroyed."
  elif [[ -n "$INSTANCE_ID" && $KEEP -eq 1 ]]; then
    warn "Instance ${INSTANCE_ID} kept alive (--keep).  Destroy with:"
    info "  ./scripts/vast destroy instance ${INSTANCE_ID}"
    info "  make vast-destroy ID=${INSTANCE_ID}"
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ── helpers ────────────────────────────────────────────────────────────────────
instance_status() {
  # `vastai show instance <ID> --raw` returns a single JSON object (not an array).
  # The `status` field is ALWAYS null — the real field is `actual_status`.
  local raw
  raw=$("$VAST_CLI" show instance "$1" --raw 2>/dev/null) || { echo "unknown"; return; }
  echo "$raw" | jq -r '.actual_status // "unknown"'
}

instance_json() {
  # `vastai show instance <ID> --raw` returns a plain JSON object directly.
  "$VAST_CLI" show instance "$1" --raw 2>/dev/null
}

wait_for_status() {
  local id="$1" target="$2"
  local attempt=0 max=300  # 25 min max (large CUDA image pull can take 15-20 min)
  printf "     waiting for status='${CYAN}%s${RESET}'" "$target"
  while (( attempt < max )); do
    local status
    status=$(instance_status "$id")
    if [[ "$status" == "$target" ]]; then
      echo
      return 0
    fi
    # Hard failures — don't keep polling
    if [[ "$status" == "error" || "$status" == "exited" || "$status" == "deleted" ]]; then
      echo
      die "Instance $id entered unexpected status: $status"
    fi
    # 'unknown' often means the API returned an empty array right after creation — keep polling
    ((attempt++)) || true
    if (( attempt % 6 == 0 )); then
      printf " ${DIM}%s${RESET}" "$status"
    fi
    printf "."
    sleep 5
  done
  echo
  die "Instance $id did not reach '$target' within $(( max * 5 / 60 )) minutes."
}

wait_for_ssh() {
  local host="$1" port="$2"
  local attempt=0 max=72   # 6 minutes max (containers can take a while to pull the image)
  log "Waiting for SSH"
  printf "     polling port %s:%s" "$host" "$port"
  while (( attempt < max )); do
    if ssh \
      -o ConnectTimeout=5 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE" \
      -o BatchMode=yes \
      -i "$SSH_KEY" \
      -p "$port" \
      "root@$host" \
      'echo __vast_ok__' 2>/dev/null | grep -q '__vast_ok__'; then
      echo
      return 0
    fi
    ((attempt++))
    printf "."
    sleep 5
  done
  echo
  die "SSH did not become available within $(( max * 5 / 60 )) minutes."
}

run_ssh() {
  # run_ssh CMD — execute CMD on the provisioned instance
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE" \
    -o BatchMode=yes \
    -i "$SSH_KEY" \
    -p "$INSTANCE_PORT" \
    "root@$INSTANCE_HOST" \
    "$@"
}

# ── banner ─────────────────────────────────────────────────────────────────────
hr
printf "${BOLD}  py-audio-box · vast.ai deploy${RESET}\n"
printf "     job       : ${CYAN}%s${RESET}\n" "$JOB_NAME"
printf "     image     : %s\n" "$DEPLOY_IMAGE"
printf "     disk      : %s GB\n" "$DISK_GB"
if [[ ${#TASKS[@]} -gt 0 ]]; then
  printf "     tasks     : %d\n" "${#TASKS[@]}"
fi
[[ $DRY_RUN -eq 1 ]]   && printf "     ${YELLOW}DRY RUN — no changes will be made${RESET}\n"
[[ $KEEP -eq 1 ]]      && printf "     ${YELLOW}--keep — instance will NOT be destroyed${RESET}\n"
hr

# ── pre-flight: check for existing instances ───────────────────────────────────
# Guard against accidental double-provisioning.
# Primary gate: API instance check (runs even if lock was manually deleted).
# Secondary gate: flock-based lockfile (prevents concurrent runs on same machine).
#
# The API check fires first with a double-tap (2 s apart) to handle the brief
# window where vast.ai hasn't propagated a freshly-created instance yet.
VAST_LOCK="/tmp/vast-deploy.lock"
if [[ $DRY_RUN -eq 0 ]]; then
  # ── Primary gate: API check (first pass) ──────────────────────────────────
  EXISTING=$(${VAST_CLI} show instances --raw 2>/dev/null | jq 'length' 2>/dev/null || echo 0)
  if [[ "$EXISTING" -gt 0 ]]; then
    warn "You already have ${EXISTING} instance(s) running on vast.ai."
    ${VAST_CLI} show instances 2>/dev/null || true
    printf "\n"
    die "Refusing to provision another instance.  Destroy existing ones first:\n  make vast-destroy ID=<id>\n  make vast-status"
  fi

  # ── Secondary gate: lockfile (prevents concurrent local runs) ──────────────
  # flock(1) is Linux-only; macOS ships without it.  Use it when available,
  # fall back to bash noclobber (still race-resistant for interactive use).
  if command -v flock >/dev/null 2>&1; then
    exec 200>"$VAST_LOCK"
    if ! flock -n 200 2>/dev/null; then
      die "Another vast-deploy is already running (lock: $VAST_LOCK).\n  If this is stale: rm $VAST_LOCK"
    fi
    echo "$$" >&200
    trap 'flock -u 200 2>/dev/null; rm -f "$VAST_LOCK"' EXIT
  else
    # noclobber: fails if lock file already exists
    if ! ( set -o noclobber; echo "$$" > "$VAST_LOCK" ) 2>/dev/null; then
      LOCK_PID=$(cat "$VAST_LOCK" 2>/dev/null || echo '?')
      die "Another vast-deploy is already running (PID ${LOCK_PID}, lock: $VAST_LOCK).\n  If this is stale: rm $VAST_LOCK"
    fi
    trap 'rm -f "$VAST_LOCK"' EXIT
  fi

  # ── Primary gate: API check (second pass, after lock acquired) ─────────────
  # Brief pause then re-check: handles the race where two processes both passed
  # the first check before either provisioned (API takes ~1 s to propagate).
  sleep 2
  EXISTING=$(${VAST_CLI} show instances --raw 2>/dev/null | jq 'length' 2>/dev/null || echo 0)
  if [[ "$EXISTING" -gt 0 ]]; then
    warn "You already have ${EXISTING} instance(s) running on vast.ai."
    ${VAST_CLI} show instances 2>/dev/null || true
    printf "\n"
    die "Refusing to provision another instance.  Destroy existing ones first:\n  make vast-destroy ID=<id>\n  make vast-status"
  fi
fi

# ── 1. search offers ───────────────────────────────────────────────────────────
log "Searching for GPU offers"
QUERY="${VAST_QUERY:-$DEFAULT_QUERY}"
info "query: $QUERY"
info "order: best value first (dlperf_per_dphtotal desc)"

# Order by best value: highest DLPerf per dollar (dlperf_per_dphtotal descending).
# This picks the GPU with the most compute per $ spent, not just the cheapest.
OFFER_JSON=$(
  "$VAST_CLI" search offers "$QUERY" \
    --order 'dlperf_per_dphtotal-' \
    --raw 2>/dev/null
)

if [[ $DRY_RUN -eq 1 ]]; then
  OFFER_ID="DRY_RUN_OFFER"
  OFFER_GPU="(dry-run)"
  OFFER_DPH="0.00"
  OFFER_RAM="0"
else
  # `vastai search offers --raw` returns a plain JSON array (not {"offers":[...]})
  OFFER_COUNT=$(echo "$OFFER_JSON" | jq 'length')
  [[ "$OFFER_COUNT" -gt 0 ]] || die "No offers match the query.  Try relaxing VAST_QUERY."

  OFFER_ID=$(    echo "$OFFER_JSON" | jq -r '.[0].id')
  OFFER_GPU=$(   echo "$OFFER_JSON" | jq -r '.[0].gpu_name // "unknown"')
  # dph_total_adj includes estimated bandwidth costs — matches what the UI shows
  OFFER_DPH=$(   echo "$OFFER_JSON" | jq -r '.[0].dph_total_adj // .[0].dph_total // 0' | xargs printf "%.4f")
  # gpu_ram is in MB — convert to GB for display
  OFFER_RAM=$(   echo "$OFFER_JSON" | jq -r '(.[0].gpu_ram // 0) / 1024 | round')
  OFFER_VCPU=$(  echo "$OFFER_JSON" | jq -r '.[0].cpu_cores_effective // 0' | xargs printf "%.0f")
  OFFER_VALUE=$( echo "$OFFER_JSON" | jq -r '.[0].dlperf_per_dphtotal // 0' | xargs printf "%.1f")
fi

ok "Found offer ${OFFER_ID}"
printf "     GPU       : ${CYAN}%s${RESET}  (%s GB VRAM)   value: %s DLPerf/\$\n" "$OFFER_GPU" "$OFFER_RAM" "$OFFER_VALUE"

# ── price breakdown table (always shown) ─────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
  _p12h=$( awk -v h="$OFFER_DPH" 'BEGIN { printf "%.2f", h * 12  }')
  _pday=$( awk -v h="$OFFER_DPH" 'BEGIN { printf "%.2f", h * 24  }')
  _pwk=$(  awk -v h="$OFFER_DPH" 'BEGIN { printf "%.2f", h * 168 }')
  _pmo=$(  awk -v h="$OFFER_DPH" 'BEGIN { printf "%.2f", h * 730 }')
  printf "     price     :  ${BOLD}\$%s${RESET}/hr" "$OFFER_DPH"
  printf "  ·  \$%s / 12 h" "$_p12h"
  printf "  ·  \$%s / day"  "$_pday"
  printf "  ·  \$%s / week" "$_pwk"
  printf "  ·  \$%s / mo\n" "$_pmo"
else
  printf "     price     :  \$%s/hr\n" "$OFFER_DPH"
fi

# ── price guard ────────────────────────────────────────────────────────────────
# Compare the offer's hourly rate against the monthly ceiling.
# $X/mo ÷ 730 h/mo = hourly equivalent threshold.
if [[ $DRY_RUN -eq 0 && "$MAX_MONTHLY_PRICE" != "0" ]]; then
  MAX_DPH=$(awk -v mo="$MAX_MONTHLY_PRICE" 'BEGIN { printf "%.6f", mo / 730 }')
  OFFER_MO=$(awk -v h="$OFFER_DPH" 'BEGIN { printf "%.2f", h * 730 }')
  EXCEEDS=$(awk -v a="$OFFER_DPH" -v b="$MAX_DPH" 'BEGIN { print (a > b) ? 1 : 0 }')
  if [[ "$EXCEEDS" -eq 1 ]]; then
    printf "\n"
    printf "  ${YELLOW}${BOLD}⚠  Price warning${RESET}  —  \$%s/mo exceeds your ceiling of \$%s/mo  [VAST_MAX_MONTHLY_PRICE=%s]\n" \
      "$OFFER_MO" "$MAX_MONTHLY_PRICE" "$MAX_MONTHLY_PRICE"
    printf "\n"
    if [[ $ASKED_CONFIRM -eq 0 ]]; then
      printf "  Proceed anyway? [y/N] "
      read -r CONFIRM </dev/tty
      case "$CONFIRM" in
        [yY]|[yY][eE][sS]) ok "Confirmed — continuing."; printf "\n" ;;
        *) die "Aborted by user (price \$${OFFER_MO}/mo exceeds ceiling \$${MAX_MONTHLY_PRICE}/mo).\n  Raise the ceiling: VAST_MAX_MONTHLY_PRICE=${OFFER_MO} or relax VAST_QUERY."
        ;;
      esac
    else
      warn "Price exceeds ceiling (\$${OFFER_MO}/mo > \$${MAX_MONTHLY_PRICE}/mo) but --yes was set — continuing."
      printf "\n"
    fi
  fi
fi

# ── 2. create instance ────────────────────────────────────────────────────────
log "Provisioning instance"

# On-start: clone the public repo into /app (or pull if already there from a previous run)
# The Docker image bakes /app from COPY . /app, including .git with a local SSH remote.
# On the instance that SSH alias doesn't resolve, so plain 'git pull' fails silently.
# Fix: force the remote URL to HTTPS, then fetch+reset so we always have the latest commit.
_GIT_SYNC="git -C /app remote set-url origin ${REPO_URL} 2>/dev/null; git -C /app fetch --depth=1 origin main 2>/dev/null && git -C /app reset --hard FETCH_HEAD 2>/dev/null || (rm -rf /app 2>/dev/null; git clone --depth=1 ${REPO_URL} /app 2>/dev/null) || true"
ONSTART_CMD="chmod 700 /root/.ssh 2>/dev/null; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; ${_GIT_SYNC}; mkdir -p /work /cache; chmod +x /app/run-direct 2>/dev/null || true"

# Build create-instance args.
# --cancel-unavail: if the offer was rented between search and create, return an
# error immediately instead of silently creating a stopped/broken instance.
CREATE_ARGS=(
  create instance "$OFFER_ID"
  --image "$DEPLOY_IMAGE"
  --disk  "$DISK_GB"
  --ssh
  --direct
  --cancel-unavail
  --label "$JOB_NAME"
  --onstart-cmd "$ONSTART_CMD"
  --raw
)

# Pass GHCR credentials so the host pulls as an authenticated user, not anonymous.
# Anonymous GHCR pulls hit rate limits and are served by a slower CDN tier.
if [[ -n "$GHCR_TOKEN" && "$DEPLOY_IMAGE" == ghcr.io/* ]]; then
  CREATE_ARGS+=(--login "-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io")
  info "Using authenticated GHCR pull (user: ${GHCR_USER})"
else
  warn "GHCR_TOKEN not set — image will be pulled anonymously (may be slow). Add GHCR_TOKEN to .env"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  printf "${DIM}  [dry] vast %s${RESET}\n" "${CREATE_ARGS[*]}"
  INSTANCE_ID="DRY_RUN_ID"
else
  CREATE_JSON=$("$VAST_CLI" "${CREATE_ARGS[@]}")
  INSTANCE_ID=$(echo "$CREATE_JSON" | jq -r '.new_contract // .id // empty')
  [[ -n "$INSTANCE_ID" ]] || die "Failed to parse instance ID from: $CREATE_JSON"
fi

ok "Instance ${INSTANCE_ID} created (label: ${JOB_NAME})"

# ── 3. wait for running ────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
  log "Waiting for instance to become ready"
  # Instances go through: scheduling → loading → running (some skip 'loading').
  # wait_for_status tolerates 'unknown' (API returns empty array right after creation).
  wait_for_status "$INSTANCE_ID" "running"
  ok "Instance is running"

  # Extract SSH connection details via dedicated ssh-url command.
  # Returns:  ssh://root@<host>:<port>
  SSH_URL=$("$VAST_CLI" ssh-url "$INSTANCE_ID" 2>/dev/null)
  INSTANCE_HOST=$(echo "$SSH_URL" | sed 's|ssh://[^@]*@||;s|:.*||')
  INSTANCE_PORT=$(echo "$SSH_URL" | sed 's|.*:||')
  if [[ -z "$INSTANCE_HOST" || -z "$INSTANCE_PORT" ]]; then
    warn "ssh-url returned: '${SSH_URL}'"
    warn "Falling back to show instance JSON…"
    INST=$(instance_json "$INSTANCE_ID")
    INSTANCE_HOST=$(echo "$INST" | jq -r '.ssh_host // .public_ipaddr // empty')
    INSTANCE_PORT=$(echo "$INST" | jq -r '.ssh_port // 22')
    [[ -n "$INSTANCE_HOST" ]] || die "Could not determine SSH host. Instance JSON:\n$(echo "$INST" | jq '{id,status,ssh_host,public_ipaddr,ssh_port}')"
  fi
fi

info "SSH: root@${INSTANCE_HOST:-?}:${INSTANCE_PORT:-?}"

# ── 4. wait for SSH ────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
  wait_for_ssh "$INSTANCE_HOST" "$INSTANCE_PORT"
  ok "SSH is ready"
fi

# ── 5. clone / pull code ──────────────────────────────────────────────────────
if [[ $NO_CLONE -eq 0 ]]; then
  log "Cloning ${REPO_URL} → /app"
  if [[ $DRY_RUN -eq 0 ]]; then
    # Force HTTPS remote URL (the baked-in .git may have a local SSH alias) then fetch+reset.
    run_ssh "git -C /app remote set-url origin ${REPO_URL} 2>/dev/null || true; git -C /app fetch --depth=1 origin main && git -C /app reset --hard FETCH_HEAD || (rm -rf /app && git clone --depth=1 ${REPO_URL} /app); mkdir -p /work /cache; chmod +x /app/run-direct 2>/dev/null || true"
  else
    printf "${DIM}  [dry] ssh git clone/pull %s → /app${RESET}\n" "$REPO_URL"
  fi
  ok "Code ready"
fi

# ── 5.5 upload local cache (voices, etc.) to /cache on the instance ─────────
if [[ -n "$PUSH_CACHE_SRC" ]]; then
  PUSH_CACHE_SRC="${PUSH_CACHE_SRC/#\~/$HOME}"
  PUSH_CACHE_SRC="$(cd "$PUSH_CACHE_SRC" 2>/dev/null && pwd || echo "$PUSH_CACHE_SRC")"
  [[ -d "$PUSH_CACHE_SRC" ]] || die "--push-cache: directory not found: $PUSH_CACHE_SRC"
  # Destination is always under /cache/ on the instance; interior structure is preserved.
  # e.g. ./cache/voices → /cache/voices/   OR   ./cache/voices/rascar-capac → /cache/voices/rascar-capac/
  REMOTE_CACHE_PARENT="/cache/$(basename "$PUSH_CACHE_SRC")"
  log "Uploading cache → ${INSTANCE_HOST}:${REMOTE_CACHE_PARENT}/"
  info "src: $PUSH_CACHE_SRC"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "${DIM}  [dry] rsync %s → root@instance:%s/${RESET}\n" "$PUSH_CACHE_SRC" "$REMOTE_CACHE_PARENT"
  else
    run_ssh "mkdir -p ${REMOTE_CACHE_PARENT}"
    rsync_cmd -az --info=progress2 \
      -e "ssh -p ${INSTANCE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
      "${PUSH_CACHE_SRC}/" \
      "root@${INSTANCE_HOST}:${REMOTE_CACHE_PARENT}/"
    ok "Cache uploaded → ${REMOTE_CACHE_PARENT}/"
  fi
fi

# ── 5.6 upload local work files to /work on the instance ─────────────────────
if [[ -n "$PUSH_WORK_SRC" ]]; then
  PUSH_WORK_SRC="${PUSH_WORK_SRC/#\~/$HOME}"
  if [[ -d "$PUSH_WORK_SRC" ]]; then
    log "Uploading work dir → ${INSTANCE_HOST}:/work/"
    info "src: $PUSH_WORK_SRC"
    [[ $DRY_RUN -eq 1 ]] || {
      rsync_cmd -az --info=progress2 \
        -e "ssh -p ${INSTANCE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
        "${PUSH_WORK_SRC}/" "root@${INSTANCE_HOST}:/work/"
      ok "Work dir uploaded → /work/"
    }
  elif [[ -f "$PUSH_WORK_SRC" ]]; then
    REMOTE_FILE="/work/$(basename "$PUSH_WORK_SRC")"
    log "Uploading $(basename "$PUSH_WORK_SRC") → ${INSTANCE_HOST}:${REMOTE_FILE}"
    [[ $DRY_RUN -eq 1 ]] || {
      rsync_cmd -az --info=progress2 \
        -e "ssh -p ${INSTANCE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
        "${PUSH_WORK_SRC}" "root@${INSTANCE_HOST}:${REMOTE_FILE}"
      ok "File uploaded → ${REMOTE_FILE}"
    }
  else
    die "--push-work: path not found: $PUSH_WORK_SRC"
  fi
fi

# ── 6. run tasks ───────────────────────────────────────────────────────────────
if [[ $SHELL_MODE -eq 1 ]]; then
  # Interactive shell — disable auto-destroy on exit
  KEEP=1
  log "Opening interactive SSH shell"
  info "Instance will NOT be automatically destroyed."
  info "When done, run:  make vast-destroy ID=${INSTANCE_ID}"
  if [[ $DRY_RUN -eq 0 ]]; then
    ssh \
      -t \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE" \
      -i "$SSH_KEY" \
      -p "$INSTANCE_PORT" \
      "root@$INSTANCE_HOST"
  fi
else
  log "Running ${#TASKS[@]} task(s)"
  T0=$SECONDS
  declare -i TASK_NUM=0
  for task in "${TASKS[@]}"; do
    ((TASK_NUM++)) || true
    printf "\n${BOLD}  [%d/%d]${RESET}  %s\n" "$TASK_NUM" "${#TASKS[@]}" "$task"
    hr
    if [[ $DRY_RUN -eq 1 ]]; then
      if [[ "$task" == '!'* ]]; then
        printf "${DIM}  [dry] ssh … %s${RESET}\n" "${task#!}"
      else
        printf "${DIM}  [dry] ssh … ./run-direct %s${RESET}\n" "$task"
      fi
    else
      # Tasks prefixed with '!' bypass ./run-direct and run as raw shell commands.
      # Example task entry:  '!bash /app/tests/remote/run-all.sh'
      if [[ "$task" == '!'* ]]; then
        run_ssh "cd /app && ${task#!}" || {
          warn "Task ${TASK_NUM} exited non-zero — continuing with remaining tasks."
        }
      else
        # run_ssh executes the task; output streams directly to the caller's terminal
        run_ssh "cd /app && ./run-direct ${task}" || {
          warn "Task ${TASK_NUM} exited non-zero — continuing with remaining tasks."
        }
      fi
    fi
  done
  hr
  ok "All tasks done  $(hms $T0)"
fi

# ── 7. pull results ────────────────────────────────────────────────────────────
if [[ $NO_PULL -eq 0 && $SHELL_MODE -eq 0 && $DRY_RUN -eq 0 ]]; then
  log "Pulling results → work_remote/${JOB_NAME}/"
  PULL_DST="${REPO_ROOT}/work_remote/${JOB_NAME}"
  mkdir -p "$PULL_DST"
  rsync_cmd -az --progress \
    -e "ssh -p ${INSTANCE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
    "root@${INSTANCE_HOST}:/work/" \
    "${PULL_DST}/"
  ok "Results pulled → work_remote/${JOB_NAME}/"
fi

# ── 8. done ────────────────────────────────────────────────────────────────────
hr
if [[ $KEEP -eq 0 && $SHELL_MODE -eq 0 ]]; then
  printf "${GREEN}${BOLD}  ✓  Done${RESET}  (instance ${INSTANCE_ID} will be destroyed on exit)\n"
else
  printf "${GREEN}${BOLD}  ✓  Done${RESET}\n"
fi
hr
echo
