#!/usr/bin/env bash
# tests/local/09-export-import.sh
# export-voice / import-voice round-trip, then verify synth still works.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 09: export-voice / import-voice round-trip ==="

_EXPORT_ZIP="/work/test-export-chalamet-en.zip"

run_cmd "export-voice: chalamet-en" \
  ./run voice-synth export-voice \
    chalamet-en \
    --dest "$_EXPORT_ZIP" \
    --cache "$CACHE"

run_cmd "import-voice: chalamet-en --force" \
  ./run voice-synth import-voice \
    --zip "$_EXPORT_ZIP" \
    --force \
    --cache "$CACHE"

speak "chalamet-en post round-trip / stable / EN" \
  --voice chalamet-en \
  --text "Import and export round-trip verified successfully. Voice is intact." \
  --language English \
  --profile stable \
  --seed 50

# Second voice round-trip
_EXPORT_ZIP2="/work/test-export-rascar-capac.zip"

run_cmd "export-voice: rascar-capac" \
  ./run voice-synth export-voice \
    rascar-capac \
    --dest "$_EXPORT_ZIP2" \
    --cache "$CACHE"

run_cmd "import-voice: rascar-capac --force" \
  ./run voice-synth import-voice \
    --zip "$_EXPORT_ZIP2" \
    --force \
    --cache "$CACHE"

speak "rascar-capac post round-trip / stable / FR" \
  --voice rascar-capac \
  --text "La voix a ete exportee, reimportee, et fonctionne parfaitement." \
  --language French \
  --profile stable \
  --seed 51

print_summary "09-export-import"
