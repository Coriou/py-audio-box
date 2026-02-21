#!/usr/bin/env bash
# scripts/local-synth-test.sh
# Comprehensive local CPU test matrix — exercises every feature of the project.
#
# Run from repo root:
#   ./scripts/local-synth-test.sh
#   SKIP_SLOW=1    ./scripts/local-synth-test.sh   # skip Q + O variant tests
#   SKIP_DESIGN=1  ./scripts/local-synth-test.sh   # skip design-voice

set -euo pipefail

CACHE="/cache"
OUT="/work/synth-test"
THREADS="$(nproc 2>/dev/null || echo 8)"
DTYPE="float32"
SKIP_SLOW="${SKIP_SLOW:-0}"
SKIP_DESIGN="${SKIP_DESIGN:-0}"

pass=0; fail=0; skip=0
results=()

run_cmd() {
  local label="$1"; shift
  echo ""
  echo "================================================"
  echo "  TEST: $label"
  echo "  CMD:  $*"
  echo "================================================"
  local t0 t1
  t0=$(date +%s)
  if "$@" 2>&1; then
    t1=$(date +%s)
    echo "  PASS  ($(( t1 - t0 ))s)"
    results+=("PASS  $(( t1 - t0 ))s    $label")
    (( pass++ )) || true
  else
    t1=$(date +%s)
    echo "  FAIL  ($(( t1 - t0 ))s)"
    results+=("FAIL  $(( t1 - t0 ))s    $label")
    (( fail++ )) || true
  fi
}

speak() {
  local label="$1"; shift
  run_cmd "$label" \
    ./run voice-synth speak \
      --cache "$CACHE" \
      --threads "$THREADS" \
      --dtype  "$DTYPE" \
      --out    "$OUT" \
      "$@"
}

skip_test() {
  echo ""; echo "  SKIP  $1"
  results+=("SKIP         $1")
  (( skip++ )) || true
}

echo "=== Local CPU Synthesis Test Matrix ==="
echo "  cache=$CACHE  out=$OUT  threads=$THREADS  dtype=$DTYPE"
echo ""

# ── 1. English documentary, stable ───────────────────────────────────────────
run_test "david-attenborough / stable / EN" \
  --voice david-attenborough \
  --text "The open ocean, covering more than seventy percent of our planet, remains one of the least explored environments on Earth." \
  --language English \
  --profile stable \
  --seed 1

# ── 2. English actor, expressive ─────────────────────────────────────────────
run_test "chalamet-en / expressive / EN" \
  --voice chalamet-en \
  --text "I didn't sleep at all last night. I kept thinking — what if this is actually real?" \
  --language English \
  --profile expressive \
  --seed 42

# ── 3. English actor v2, balanced + QA ───────────────────────────────────────
run_test "chalamet-en-2 / balanced / qa / EN" \
  --voice chalamet-en-2 \
  --text "Sometimes the simplest questions are the hardest ones to answer honestly." \
  --language English \
  --profile balanced \
  --seed 7 \
  --qa

# ── 4. French narration, stable ───────────────────────────────────────────────
run_test "rascar-capac / stable / FR" \
  --voice rascar-capac \
  --text "Les secrets du passé ne disparaissent jamais vraiment. Ils attendent, tapis dans l'ombre, jusqu'au moment où quelqu'un les retrouve." \
  --language French \
  --profile stable \
  --seed 3

# ── 5. French streamer, balanced ─────────────────────────────────────────────
run_test "blitzstream / balanced / FR" \
  --voice blitzstream \
  --text "Voilà, vous avez bien compris la stratégie. Maintenant on passe au niveau suivant, et ça va être intense." \
  --language French \
  --profile balanced \
  --seed 5

# ── 6. French child TTS, stable ──────────────────────────────────────────────
run_test "tiktok-child-fr-1 / stable / FR" \
  --voice tiktok-child-fr-1 \
  --text "Il était une fois, dans une forêt très loin d'ici, un petit renard qui cherchait son chemin." \
  --language French \
  --profile stable \
  --seed 99

# ── 7. 3-variants + select-best ───────────────────────────────────────────────
run_test "chalamet-en / expressive / 3-variants select-best" \
  --voice chalamet-en \
  --text "This is the moment everything changes. Are you ready?" \
  --language English \
  --profile expressive \
  --variants 3 \
  --select-best \
  --seed 100

# ── 8. Designed clone engine ──────────────────────────────────────────────────
run_test "smoke-designed / stable / designed_clone" \
  --voice smoke-designed \
  --text "A gentle breeze carried the scent of pine and cedar through the valley below." \
  --language English \
  --profile stable \
  --seed 11

# ── 9. blitzstream-2, expressive ─────────────────────────────────────────────
run_test "blitzstream-2 / expressive / FR" \
  --voice blitzstream-2 \
  --text "Franchement c'est incroyable. J'aurais jamais pensé que ça allait se passer comme ça. Mais voilà, on est là, et c'est réel." \
  --language French \
  --profile expressive \
  --seed 20

# ── 10. hikaru-nakamura, balanced, temperature override ──────────────────────
run_test "hikaru-nakamura / balanced / temperature=0.8" \
  --voice hikaru-nakamura \
  --text "Every move has a purpose. You have to think three, four, five moves ahead if you want to stay competitive at this level." \
  --language English \
  --profile balanced \
  --temperature 0.8 \
  --seed 55

# ── 11. jonathan-cohen, French, 2-variants select-best ───────────────────────
if [[ "$SKIP_SLOW" != "--skip-slow" ]]; then
  run_test "jonathan-cohen / expressive / 2-variants select-best / FR" \
    --voice jonathan-cohen \
    --text "Écoutez, moi je vais vous dire un truc — c'est compliqué, mais c'est exactement pour ça que c'est intéressant." \
    --language French \
    --profile expressive \
    --variants 2 \
    --select-best \
    --seed 200
else
  echo "  SKIP  jonathan-cohen 2-variants (--skip-slow)"; (( skip++ )) || true
fi

# ── 12. chalamet-fr, stable, French ──────────────────────────────────────────
if [[ "$SKIP_SLOW" != "--skip-slow" ]]; then
  run_test "chalamet-fr / stable / FR" \
    --voice chalamet-fr \
    --text "Je ne sais pas encore, mais je sens que quelque chose va changer. On verra bien." \
    --language French \
    --profile stable \
    --seed 77
else
  echo "  SKIP  chalamet-fr stable (--skip-slow)"; (( skip++ )) || true
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  RESULTS  pass=%d  fail=%d  skip=%d\n" "$pass" "$fail" "$skip"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for r in "${results[@]}"; do echo "  $r"; done
echo ""
echo "  Output files: ./work/synth-test/"
