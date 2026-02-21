#!/usr/bin/env bash
# tests/local/17-slow-clones.sh
# Slow clone voices (FR child, FR actors) — skipped when SKIP_SLOW=1.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 17: slow clone voices (FR / child) ==="

if [[ "$SKIP_SLOW" == "0" ]]; then

  speak "tiktok-child-fr-1 / stable / FR" \
    --voice tiktok-child-fr-1 \
    --text "Il etait une fois, dans une foret tres loin d'ici, un petit renard qui cherchait son chemin." \
    --language French \
    --profile stable \
    --seed 99

  speak "jonathan-cohen / expressive / FR" \
    --voice jonathan-cohen \
    --text "Ecoutez, moi je vais vous dire un truc — c'est complique, mais c'est exactement pour ca que c'est interessant." \
    --language French \
    --profile expressive \
    --seed 200

  speak "chalamet-fr / stable / FR" \
    --voice chalamet-fr \
    --text "Je ne sais pas encore, mais je sens que quelque chose va changer. On verra bien." \
    --language French \
    --profile stable \
    --seed 77

  speak "jonathan-cohen / 2-variants select-best / expressive / FR" \
    --voice jonathan-cohen \
    --text "Non mais vraiment, vous avez vu ca ? C'est exactement ce genre de moment qu'on attendait." \
    --language French \
    --profile expressive \
    --variants 2 \
    --select-best \
    --seed 201

else
  skip_test "tiktok-child-fr-1 / stable / FR (SKIP_SLOW=1)"
  skip_test "jonathan-cohen / expressive / FR (SKIP_SLOW=1)"
  skip_test "chalamet-fr / stable / FR (SKIP_SLOW=1)"
  skip_test "jonathan-cohen 2-variants select-best / FR (SKIP_SLOW=1)"
fi

print_summary "17-slow-clones"
