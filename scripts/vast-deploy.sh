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
#
# Search customisation:
#   VAST_QUERY     override the GPU search query string (see: vastai search offers --help)
#   VAST_IMAGE     override the Docker image to deploy  (default: ghcr.io/coriou/voice-tools:cuda)
#   VAST_DISK      override disk size in GB             (default: 60)
#
# Options:
#   --tasks FILE     tasks file (one "./run-direct app [args]" per non-comment line)
#   --shell          provision and drop into an interactive SSH session
#   --job NAME       label / job name (default: py-audio-box-YYYYMMDD_HHMMSS)
#   --ssh-key PATH   path to SSH private key (default: auto-detect ~/.ssh/id_{ed25519,rsa,ecdsa})
#   --no-sync        skip rsync of local code to /app on the instance
#   --no-pull        skip rsyncing /work back to work_remote/ after tasks
#   --keep           don't destroy the instance when done (useful for debugging)
#   --dry-run        print commands, touch nothing
#   -h, --help       show this help
#
# Task file format (--tasks FILE):
#   # comment lines are ignored
#   voice-register --url "https://youtu.be/XXXX" --voice-name myvoice --text "Hello."
#   voice-synth speak --voice myvoice --text "Generated on vast.ai"
#
# Each line becomes:  ./run-direct <line>  executed on the remote instance.

set -euo pipefail

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
JOB_NAME=""
TASKS_FILE=""
TASKS=()
SSH_KEY=""
SHELL_MODE=0
NO_SYNC=0
NO_PULL=0
KEEP=0
DRY_RUN=0

# Default GPU search query: reliable Volta+ GPU with enough VRAM for Qwen-TTS,
# fast inet, and enough disk; ordered cheapest-first.
# Override via VAST_QUERY env var.
DEFAULT_QUERY='reliability > 0.98 gpu_ram >= 20 compute_cap >= 700 inet_down >= 200 disk_space >= 50 rented=False'

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)          TASKS_FILE="$2"; shift 2 ;;
    --shell)          SHELL_MODE=1; shift ;;
    --job)            JOB_NAME="$2"; shift 2 ;;
    --ssh-key)        SSH_KEY="$2"; shift 2 ;;
    --no-sync)        NO_SYNC=1; shift ;;
    --no-pull)        NO_PULL=1; shift ;;
    --keep)           KEEP=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
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
command -v rsync  >/dev/null 2>&1 || die "rsync not found."
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
if [[ -z "$SSH_KEY" ]]; then
  for candidate in ~/.ssh/id_ed25519 ~/.ssh/id_rsa ~/.ssh/id_ecdsa ~/.ssh/id_dsa; do
    if [[ -f "$candidate" ]]; then
      SSH_KEY="$candidate"
      break
    fi
  done
fi
[[ -n "$SSH_KEY" && -f "$SSH_KEY" ]] || die "No SSH private key found.  Use --ssh-key PATH."

# ── job name ───────────────────────────────────────────────────────────────────
[[ -z "$JOB_NAME" ]] && JOB_NAME="py-audio-box-$(date +%Y%m%d_%H%M%S)"

# ── dry-run wrapper ────────────────────────────────────────────────────────────
# vast_cmd / ssh_cmd / rsync_cmd print the command instead of running it in dry-run.
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
  local attempt=0 max=120  # 10 min max (image pull can be slow)
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
# Guard against accidental double-provisioning (e.g. re-running while a previous
# instance is still up).  `show instances --raw` returns a plain JSON array.
if [[ $DRY_RUN -eq 0 ]]; then
  EXISTING=$(${VAST_CLI} show instances --raw 2>/dev/null | jq 'length')
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
info "order: cheapest first (dph+)"

OFFER_JSON=$(
  "$VAST_CLI" search offers "$QUERY" \
    --order 'dph_total_adj+' \
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

  OFFER_ID=$(  echo "$OFFER_JSON" | jq -r '.[0].id')
  OFFER_GPU=$( echo "$OFFER_JSON" | jq -r '.[0].gpu_name // "unknown"')
  # dph_total_adj includes estimated bandwidth costs — matches what the UI shows
  OFFER_DPH=$( echo "$OFFER_JSON" | jq -r '.[0].dph_total_adj // .[0].dph_total // 0' | xargs printf "%.4f")
  # gpu_ram is in MB — convert to GB for display
  OFFER_RAM=$( echo "$OFFER_JSON" | jq -r '(.[0].gpu_ram // 0) / 1024 | round')
  OFFER_VCPU=$(echo "$OFFER_JSON" | jq -r '.[0].cpu_cores_effective // 0' | xargs printf "%.0f")
fi

ok "Found offer ${OFFER_ID}"
printf "     GPU       : ${CYAN}%s${RESET}  (%s GB VRAM)\n" "$OFFER_GPU" "$OFFER_RAM"
printf "     price     : \$%s/hr\n" "$OFFER_DPH"

# ── 2. create instance ────────────────────────────────────────────────────────
log "Provisioning instance"

# Build onstart command: update code from git if .git exists, else skip
ONSTART_CMD="mkdir -p /work /cache; cd /app 2>/dev/null; git pull --ff-only 2>/dev/null || true; chmod +x run-direct 2>/dev/null || true"

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

# ── 5. sync code ───────────────────────────────────────────────────────────────
if [[ $NO_SYNC -eq 0 ]]; then
  log "Syncing code → /app"
  rsync_cmd -az --progress \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='.DS_Store' \
    --exclude='.vscode/' \
    --exclude='.ruff_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='work/' \
    --exclude='work_remote/' \
    --exclude='cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    -e "ssh -p ${INSTANCE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
    "${REPO_ROOT}/" \
    "root@${INSTANCE_HOST}:/app/"
  ok "Code synced"
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
      printf "${DIM}  [dry] ssh … ./run-direct %s${RESET}\n" "$task"
    else
      # run_ssh executes the task; output streams directly to the caller's terminal
      run_ssh "cd /app && ./run-direct ${task}" || {
        warn "Task ${TASK_NUM} exited non-zero — continuing with remaining tasks."
      }
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
