#!/usr/bin/env bash
# tests/local/08-save-profile.sh
# --save-profile-default: save a generation profile per voice, verify it sticks.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 08: --save-profile-default ==="

# Save expressive as default for blitzstream-2
speak "blitzstream-2 / expressive / --save-profile-default / FR (save)" \
  --voice blitzstream-2 \
  --text "Une voix bien reglee, c'est la cle d'un bon contenu audio." \
  --language French \
  --profile expressive \
  --save-profile-default \
  --seed 30

# Run again without --profile — should pick up saved default
speak "blitzstream-2 / no --profile (uses saved default) / FR" \
  --voice blitzstream-2 \
  --text "Le profil sauvegarde devrait maintenant s'appliquer automatiquement." \
  --language French \
  --seed 31

# Save stable as default for david-attenborough
speak "david-attenborough / stable / --save-profile-default / EN (save)" \
  --voice david-attenborough \
  --text "Nature has always been the greatest source of inspiration for science." \
  --language English \
  --profile stable \
  --save-profile-default \
  --seed 40

# Run without --profile
speak "david-attenborough / no --profile (uses saved default) / EN" \
  --voice david-attenborough \
  --text "Saved profile should be applied without specifying it again." \
  --language English \
  --seed 41

print_summary "08-save-profile"
