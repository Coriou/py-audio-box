#!/usr/bin/env bash
# Fetch MediaPipe Pose Landmarker models into the shared cache mount.
#
# Usage:
#   scripts/fetch_models.sh [--models-dir DIR] [--heavy] [--force]
#
# Defaults to the Lite + Full variants required by the auto-sfx POC backends
# ("mediapipe-lite", "mediapipe-full"). Heavy is opt-in. Files already present
# are skipped unless --force is given.
set -euo pipefail

MODELS_DIR="${AUTO_SFX_MODELS_DIR:-/cache/auto-sfx/models}"
BASE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
WANT_HEAVY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --heavy|--all) WANT_HEAVY=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "fetch_models.sh: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

VARIANTS=(lite full)
[[ $WANT_HEAVY -eq 1 ]] && VARIANTS+=(heavy)

mkdir -p "$MODELS_DIR"

fetch_one() {
  local variant="$1"
  local url="$BASE_URL/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task"
  local out="$MODELS_DIR/pose_landmarker_${variant}.task"
  local tmp="$out.part-$$"

  if [[ -s "$out" && $FORCE -eq 0 ]]; then
    echo "up-to-date: $out ($(wc -c <"$out" | tr -d ' ') bytes)"
    return 0
  fi

  echo "downloading $variant -> $out"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --tries=3 -O "$tmp" "$url"
  else
    echo "fetch_models.sh: need curl or wget" >&2
    rm -f "$tmp"
    return 1
  fi

  if [[ ! -s "$tmp" ]]; then
    echo "fetch_models.sh: empty download for $variant" >&2
    rm -f "$tmp"
    return 1
  fi

  mv "$tmp" "$out"
  echo "done: $out ($(wc -c <"$out" | tr -d ' ') bytes)"
}

status=0
for v in "${VARIANTS[@]}"; do
  fetch_one "$v" || status=1
done

if [[ $status -ne 0 ]]; then
  echo "fetch_models.sh: one or more downloads failed" >&2
fi
exit "$status"
