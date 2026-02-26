#!/usr/bin/env bash
# jobs-watch.sh — live progress display for a running local watcher batch
#
# Usage:
#   ./scripts/jobs-watch.sh [CONTAINER] [INTERVAL]
#
# Defaults:
#   CONTAINER  pab-synth-local  (or set WATCHER_CONTAINER env var)
#   INTERVAL   10 seconds

# No set -e: subcommand failures are expected and handled individually
set -uo pipefail

CONTAINER="${1:-${WATCHER_CONTAINER:-pab-synth-local}}"
INTERVAL="${2:-${WATCH_INTERVAL:-10}}"

# ── helpers ──────────────────────────────────────────────────────────────────

_bold()  { printf '\033[1m%s\033[0m' "$*"; }
_green() { printf '\033[32m%s\033[0m' "$*"; }
_yellow(){ printf '\033[33m%s\033[0m' "$*"; }
_red()   { printf '\033[31m%s\033[0m' "$*"; }
_grey()  { printf '\033[90m%s\033[0m' "$*"; }
_cyan()  { printf '\033[36m%s\033[0m' "$*"; }

_bar() {
    local done=$1 total=$2 width=${3:-30}
    local filled=$(( done * width / (total > 0 ? total : 1) ))
    local empty=$(( width - filled ))
    printf '['
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null || true
    printf '%0.s░' $(seq 1 $empty 2>/dev/null) 2>/dev/null || true
    printf ']'
}

_duration() {
    local secs=$1
    if   (( secs < 60 ));   then printf '%ds' "$secs"
    elif (( secs < 3600 )); then printf '%dm%02ds' "$(( secs/60 ))" "$(( secs%60 ))"
    else                         printf '%dh%02dm' "$(( secs/3600 ))" "$(( (secs%3600)/60 ))"
    fi
}

# Find the active batch dir for a given container name pattern
_find_batch_dir() {
    # Look for the newest watcher batch dir that has a manifest.json
    local latest
    latest=$(find work_remote -maxdepth 2 -name "manifest.json" -newer work_remote/.DS_Store 2>/dev/null \
             | head -1 | xargs dirname 2>/dev/null || true)

    if [[ -z "$latest" ]]; then
        # fallback: most recently modified batch dir
        latest=$(ls -td work_remote/watcher-*-b*/  2>/dev/null | head -1 | sed 's:/$::' || true)
    fi
    echo "$latest"
}

# Pull total beats from manifest
_total_from_manifest() {
    local batch_dir=$1
    python3 -c "
import json, sys
d = json.load(open('$batch_dir/manifest.json'))
print(len(d['jobs']))
" 2>/dev/null || echo "?"
}

# Pull topic_id from manifest
_topic_from_manifest() {
    local batch_dir=$1
    python3 -c "
import json, sys, pathlib
d = json.load(open('$batch_dir/manifest.json'))
# topic_id is the first path component of any job's output_name
print(pathlib.Path(d['jobs'][0]['output_name']).parts[0])
" 2>/dev/null || basename "$batch_dir"
}

# Count beats with result.json (= completed by voice-synth)
_count_done() {
    local batch_dir=$1
    find "$batch_dir" -name "result.json" -not -path "*/logs/*" 2>/dev/null | wc -l | tr -d ' '
}

# Count beats where result.json contains "failed" status
_count_failed() {
    local batch_dir=$1
    grep -rl '"status": "failed"' "$batch_dir" --include="result.json" 2>/dev/null | wc -l | tr -d ' '
}

# Current beat being synthesised (from container /proc)
_current_beat() {
    docker exec "$CONTAINER" sh -c \
        'cat /proc/*/cmdline 2>/dev/null | tr "\0" "\n" | grep -o "beat-[0-9]*" | head -1' \
        2>/dev/null || echo ""
}

# Parse the manifest entry for the given beat and return: voice / language /
# profile / variants / temperature and the first 120 chars of its text.
# Output is newline-separated key=value pairs for easy consumption in bash.
_current_job_details() {
    local batch_dir=$1 beat=$2
    [[ -z "$batch_dir" || -z "$beat" ]] && return
    python3 - "$batch_dir" "$beat" 2>/dev/null <<'PYEOF'
import json, sys, pathlib, textwrap
batch_dir, beat = pathlib.Path(sys.argv[1]), sys.argv[2]

# Load manifest
try:
    manifest = json.loads((batch_dir / 'manifest.json').read_text())
except Exception:
    sys.exit(0)

# Find matching job
job = next((j for j in manifest.get('jobs', []) if beat in j.get('output_name', '')), None)
if not job:
    sys.exit(0)

# Parse argv key→value pairs
argv = job.get('argv', [])
params: dict[str, str] = {}
i = 0
while i < len(argv):
    if argv[i].startswith('--') and i + 1 < len(argv) and not argv[i+1].startswith('--'):
        params[argv[i][2:]] = argv[i+1]
        i += 2
    else:
        i += 1

print(f"voice={params.get('voice', '?')}")
print(f"language={params.get('language', '?')}")
print(f"profile={params.get('profile', '?')}")
print(f"variants={params.get('variants', '?')}")
print(f"temperature={params.get('temperature', '?')}")

# Resolve text: try local text.txt (container /work → local work_remote)
text_file_container = params.get('text-file', '')
if text_file_container:
    # strip leading /work/ and prepend batch_dir
    rel = text_file_container.lstrip('/').split('/', 1)[-1]  # remove 'work/BATCH_NAME/'
    # just search under batch_dir for the beat text.txt
    candidates = list(batch_dir.rglob(f'*/{beat}/text.txt'))
    if not candidates:
        candidates = list(batch_dir.rglob('text.txt'))
    if candidates:
        text = candidates[0].read_text(encoding='utf-8').strip()
        snippet = textwrap.shorten(text, width=120, placeholder='…')
        print(f"text={snippet}")
PYEOF
}

# Redis lock TTL via job-runner status (run inside toolbox)
_lock_ttl() {
    ./run job-runner status --json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('lock',{}).get('ttl_ms','?'))" \
        2>/dev/null || echo "?"
}

# Wall-clock start time: prefer the batch manifest mtime (= when THIS batch
# was enqueued), fall back to the container start time only when no manifest
# is available yet.
_start_epoch() {
    local batch_dir=${1:-}
    if [[ -n "$batch_dir" && -f "$batch_dir/manifest.json" ]]; then
        python3 -c "import os; print(int(os.path.getmtime('$batch_dir/manifest.json')))" 2>/dev/null || echo "0"
    else
        docker inspect --format '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null \
            | python3 -c "
import sys, datetime
ts = sys.stdin.readline().strip()
try:
    dt = datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
    print(int(dt.timestamp()))
except:
    print(0)
" 2>/dev/null || echo "0"
    fi
}

# ── main loop ────────────────────────────────────────────────────────────────

echo ""
_bold "  jobs-watch"; echo "  —  refreshing every ${INTERVAL}s   (Ctrl-C to exit)"
echo ""

BATCH_DIR=""
TOTAL=""
TOPIC=""
START_EPOCH=0

while true; do
    # Resolve batch dir once (reuse until container stops)
    if [[ -z "$BATCH_DIR" ]]; then
        BATCH_DIR=$(_find_batch_dir)
        if [[ -n "$BATCH_DIR" ]]; then
            TOTAL=$(_total_from_manifest "$BATCH_DIR")
            TOPIC=$(_topic_from_manifest "$BATCH_DIR")
        fi
    fi

    # Container alive?
    CONTAINER_STATUS=$(docker inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")

    # Timestamps
    NOW=$(date +%s)
    if [[ "$START_EPOCH" -eq 0 && -n "$BATCH_DIR" ]]; then
        START_EPOCH=$(_start_epoch "$BATCH_DIR")
    fi
    ELAPSED=$(( NOW - START_EPOCH ))

    # Counts
    DONE=0; FAILED=0; CURRENT=""
    if [[ -n "$BATCH_DIR" ]]; then
        DONE=$(_count_done "$BATCH_DIR")
        FAILED=$(_count_failed "$BATCH_DIR")
    fi

    if [[ "$CONTAINER_STATUS" == "running" ]]; then
        CURRENT=$(_current_beat)
    fi

    # ETR
    ETR_STR="—"
    if [[ "$TOTAL" =~ ^[0-9]+$ && "$DONE" -gt 0 && "$TOTAL" -gt "$DONE" ]]; then
        REMAINING=$(( TOTAL - DONE ))
        PER_BEAT=$(( ELAPSED / DONE ))
        ETR=$(( REMAINING * PER_BEAT ))
        ETR_STR=$(_duration "$ETR")
    fi

    # ── render ───────────────────────────────────────────────────────────────
    clear
    echo ""
    _bold "  ── ${TOPIC:-pab-synth}  •  Local CPU synthesis ──────────────────────────"; echo ""
    echo ""

    # Progress bar
    if [[ "$TOTAL" =~ ^[0-9]+$ ]]; then
        printf "  Progress  "
        _bar "$DONE" "$TOTAL" 32
        printf "  "
        _bold "$DONE"
        printf " / %s beats" "$TOTAL"
        if [[ "$FAILED" -gt 0 ]]; then
            printf "  ("; _red "${FAILED} failed"; printf ")"
        fi
        echo ""
    fi

    echo ""

    # Current beat + job details
    if [[ -n "$CURRENT" ]]; then
        printf "  Synth now  "; _cyan "$CURRENT"; echo ""
        if [[ -n "$BATCH_DIR" ]]; then
            JOB_DETAILS=$(_current_job_details "$BATCH_DIR" "$CURRENT")
            if [[ -n "$JOB_DETAILS" ]]; then
                VOICE=$(echo "$JOB_DETAILS"    | grep '^voice='       | cut -d= -f2-)
                LANG=$(echo "$JOB_DETAILS"     | grep '^language='    | cut -d= -f2-)
                PROFILE=$(echo "$JOB_DETAILS"  | grep '^profile='     | cut -d= -f2-)
                VARIANTS=$(echo "$JOB_DETAILS" | grep '^variants='    | cut -d= -f2-)
                TEMP=$(echo "$JOB_DETAILS"     | grep '^temperature=' | cut -d= -f2-)
                JOB_TEXT=$(echo "$JOB_DETAILS" | grep '^text='        | cut -d= -f2-)
                # Params line
                printf "  Voice      "; _bold "${VOICE:-?}"
                [[ -n "$LANG"     ]] && printf "  •  %s" "$LANG"
                [[ -n "$PROFILE" ]] && printf "  •  profile:%s" "$PROFILE"
                [[ -n "$VARIANTS" && "$VARIANTS" != "1" ]] && printf "  •  %s variants" "$VARIANTS"
                [[ -n "$TEMP"    ]] && printf "  •  temp:%s" "$TEMP"
                echo ""
                # Text snippet
                [[ -n "$JOB_TEXT" ]] && { printf "  Text       "; _grey "\"${JOB_TEXT}\""; echo ""; }
            fi
        fi
    elif [[ "$CONTAINER_STATUS" == "running" ]]; then
        printf "  Synth now  "; _yellow "model loading…"; echo ""
    else
        printf "  Synth now  "; _grey "—"; echo ""
    fi

    # Timing
    printf "  Elapsed    %s\n" "$(_duration "$ELAPSED")"
    printf "  ETR        %s\n" "$ETR_STR"
    echo ""

    # Container + lock health
    if [[ "$CONTAINER_STATUS" == "running" ]]; then
        printf "  Container  "; _green "running"; echo ""
    else
        printf "  Container  "; _red "$CONTAINER_STATUS"; echo ""
    fi

    # Lock TTL (only query Redis if container is still up, to save time)
    if [[ "$CONTAINER_STATUS" == "running" ]]; then
        LOCK_TTL=$(_lock_ttl)
        if [[ "$LOCK_TTL" =~ ^[0-9]+$ ]]; then
            LOCK_SEC=$(( LOCK_TTL / 1000 ))
            if   (( LOCK_SEC > 60 )); then printf "  Lock TTL   "; _green "${LOCK_SEC}s"; echo ""
            elif (( LOCK_SEC > 0  )); then printf "  Lock TTL   "; _yellow "${LOCK_SEC}s ⚠"; echo ""
            else                           printf "  Lock TTL   "; _red "expired!"; echo ""
            fi
        else
            printf "  Lock TTL   "; _grey "—"; echo ""
        fi
    fi

    echo ""
    _grey "  Batch dir  ${BATCH_DIR:-discovering…}"; echo ""
    echo ""
    printf "  "; _grey "Updated $(date '+%H:%M:%S')  —  next in ${INTERVAL}s"; echo ""
    echo ""

    # Exit when done
    if [[ "$TOTAL" =~ ^[0-9]+$ && "$DONE" -ge "$TOTAL" && "$DONE" -gt 0 ]]; then
        echo ""
        _bold "  ✓ All $TOTAL beats complete!"; echo ""
        if [[ -n "$TOPIC" ]]; then
            OUT_DIR="work_remote/jobs_out/${TOPIC}"
            echo ""
            printf "  Export dir  "; _cyan "$OUT_DIR"; echo ""
            echo ""
            printf "  Beats:\n"
            ls "$OUT_DIR/beats/" 2>/dev/null | while read -r b; do
                printf "    "; _green "✓"; printf " %s\n" "$b"
            done
        fi
        echo ""
        break
    fi

    if [[ "$CONTAINER_STATUS" != "running" && "$DONE" -gt 0 ]]; then
        echo ""
        _yellow "  Container stopped. $DONE/$TOTAL beats completed."; echo ""
        echo ""
        break
    fi

    sleep "$INTERVAL"
done
