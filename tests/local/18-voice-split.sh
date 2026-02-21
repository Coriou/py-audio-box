#!/usr/bin/env bash
# tests/local/18-voice-split.sh
# voice-split: extract clean clips from a local audio file.
#
# Requires a real speech recording to produce meaningful results.
# Set TEST_AUDIO to a path accessible inside the container (e.g. /work/my.wav).
# The test is automatically SKIPPED when TEST_AUDIO is unset or the file
# does not exist inside the running container.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 18: voice-split (local audio file) ==="

_AUDIO="${TEST_AUDIO:-}"

# Check whether the file exists inside the container before attempting any run
if [[ -z "$_AUDIO" ]]; then
  skip_test "voice-split local file — set TEST_AUDIO=/work/file.wav to enable"
  print_summary "18-voice-split"
fi

# Basic: extract 2 clips of 15 s
run_cmd "voice-split: 2 clips 15s (no voice-name)" \
  ./run voice-split \
    --audio "$_AUDIO" \
    --clips 2 \
    --length 15 \
    --cache "$CACHE" \
    --out "$OUT/voice-split" \
    --max-scan-seconds 120

# Extract 1 clip and register as a named voice
run_cmd "voice-split: 1 clip + --voice-name test-split-voice" \
  ./run voice-split \
    --audio "$_AUDIO" \
    --clips 1 \
    --length 20 \
    --voice-name test-split-voice \
    --cache "$CACHE" \
    --out "$OUT/voice-split" \
    --max-scan-seconds 120

# Verify the registered voice is usable for synthesis
speak "test-split-voice / stable / EN (post voice-split registration)" \
  --voice test-split-voice \
  --text "Voice registered from a local audio file, tested directly after splitting." \
  --language English \
  --profile stable \
  --seed 5

# Timestamp trim (take a slice from the middle of the file)
if [[ "${TEST_AUDIO_LONG:-0}" == "1" ]]; then
  run_cmd "voice-split: trimmed segment 0:30-2:00" \
    ./run voice-split \
      --audio "$_AUDIO" \
      --start 0:30 \
      --end 2:00 \
      --clips 2 \
      --length 15 \
      --cache "$CACHE" \
      --out "$OUT/voice-split-trimmed"
else
  skip_test "voice-split trimmed segment (set TEST_AUDIO_LONG=1 to enable)"
fi

print_summary "18-voice-split"
