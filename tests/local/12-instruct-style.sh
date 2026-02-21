#!/usr/bin/env bash
# tests/local/12-instruct-style.sh
# CustomVoice: --instruct-style template shortcuts from lib/styles.yaml.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 12: CustomVoice — --instruct-style templates ==="

speak "Ryan / stable / --instruct-style serious_doc / EN" \
  --speaker Ryan \
  --text "The following section outlines the key principles of acoustic signal processing." \
  --instruct-style serious_doc \
  --language English \
  --profile stable \
  --seed 3

speak "Ryan / balanced / --instruct-style warm / EN" \
  --speaker Ryan \
  --text "It has been a wonderful journey together and we are grateful you joined us for every step." \
  --instruct-style warm \
  --language English \
  --profile balanced \
  --seed 4

speak "Ryan / expressive / --instruct-style excited / EN" \
  --speaker Ryan \
  --text "This is unbelievable! Everything we worked so hard for is finally happening right now!" \
  --instruct-style excited \
  --language English \
  --profile expressive \
  --seed 5

speak "Ryan / stable / --instruct-style audiobook / EN" \
  --speaker Ryan \
  --text "Chapter one. The night was dark and perfectly still, save for the soft sound of distant rainfall." \
  --instruct-style audiobook \
  --language English \
  --profile stable \
  --seed 6

speak "Ryan / balanced / --instruct-style formal / EN" \
  --speaker Ryan \
  --text "Good evening. Our top story tonight: a major breakthrough in renewable energy storage." \
  --instruct-style formal \
  --language English \
  --profile balanced \
  --seed 7

print_summary "12-instruct-style"
