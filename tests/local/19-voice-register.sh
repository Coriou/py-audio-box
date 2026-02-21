#!/usr/bin/env bash
# tests/local/19-voice-register.sh
# voice-register: one-shot pipeline (voice-split + voice-clone) from a local file.
#
# Set TEST_AUDIO to a container-accessible path (e.g. /work/recording.wav).
# The test is automatically SKIPPED when TEST_AUDIO is unset.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 19: voice-register pipeline ==="

_AUDIO="${TEST_AUDIO:-}"

if [[ -z "$_AUDIO" ]]; then
  skip_test "voice-register from local file — set TEST_AUDIO=/work/file.wav to enable"
  print_summary "19-voice-register"
fi

# Full pipeline: split + clone + test synth
run_cmd "voice-register: test-registered-voice (from local file)" \
  ./run voice-register \
    --audio "$_AUDIO" \
    --voice-name test-registered-voice \
    --text "This voice was registered in a single step using the voice-register pipeline." \
    --cache "$CACHE" \
    --out "$OUT/voice-register" \
    --threads "$THREADS" \
    --dtype "$DTYPE" \
    --max-scan-seconds 120

# Use the freshly registered voice with voice-synth
speak "test-registered-voice / stable / EN" \
  --voice test-registered-voice \
  --text "Voice registered end-to-end. Synthesis directly after registration confirmed." \
  --language English \
  --profile stable \
  --seed 1

speak "test-registered-voice / expressive / EN" \
  --voice test-registered-voice \
  --text "Wow, everything worked perfectly in one shot! That is the beauty of automation." \
  --language English \
  --profile expressive \
  --seed 2

# Register with a timestamp trim (if long file)
if [[ "${TEST_AUDIO_LONG:-0}" == "1" ]]; then
  run_cmd "voice-register: test-registered-trim (with --start/--end)" \
    ./run voice-register \
      --audio "$_AUDIO" \
      --start 1:00 \
      --end 3:30 \
      --voice-name test-registered-trim \
      --text "Registered from a trimmed segment of the source audio file." \
      --cache "$CACHE" \
      --out "$OUT/voice-register-trim" \
      --threads "$THREADS" \
      --dtype "$DTYPE"
else
  skip_test "voice-register with --start/--end trim (set TEST_AUDIO_LONG=1)"
fi

print_summary "19-voice-register"
