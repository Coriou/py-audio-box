#!/usr/bin/env bash
# tests/remote/03-chunking.sh
# Long-text chunking: --chunk flag + --text-file input with multi-chunk content.
# Mirrors tests/local/06-chunking.sh + tests/local/07-text-file.sh
set -euo pipefail
cd /app 2>/dev/null || true
source tests/remote/lib/common.sh

echo "=== 03: chunking + text-file (${TARGET}) ==="

# ── Write text files to $OUT ──────────────────────────────────────────────────
_tf_en="${OUT}/remote_chunk_en_$$.txt"
_tf_fr="${OUT}/remote_chunk_fr_$$.txt"
_tf_attn="${OUT}/remote_chunk_attn_$$.txt"

mkdir -p "${OUT}"

cat > "$_tf_en" << 'EOF'
Voice synthesis at scale requires careful attention to latency, throughput, and naturalness. When generating long-form content, a chunking strategy splits the input into manageable segments while preserving prosodic continuity. Each chunk is synthesised independently and the resulting audio is concatenated. This approach enables arbitrarily long outputs without exceeding the model context window. The result should sound seamless to a human listener.
EOF

cat > "$_tf_fr" << 'EOF'
La langue française possède une musicalité particulière qui tient à son rythme syllabique régulier et à ses liaisons. Contrairement à l'anglais, chaque syllabe est donnée un poids quasi égal, ce qui confère au discours une fluidité reconnaissable. Cette spécificité acoustique rend la synthèse vocale en français particulièrement délicate. Les modèles doivent apprendre non seulement les phonèmes mais aussi les schémas prosodiques propres à cette langue.
EOF

cat > "$_tf_attn" << 'EOF'
Chapter one. The expedition had been underway for three weeks when they first spotted the anomaly on the horizon. It was a structure unlike anything recorded in the geological surveys. Dr Chen lowered her binoculars and turned to the rest of the team. Something had been here before us, she said quietly. The silence that followed was heavier than the desert air. Nobody disagreed.
EOF

# Inline chunked synthesis
speak "david-attenborough / stable / --chunk / EN" \
  --voice david-attenborough \
  --text "The ocean is the lifeblood of our planet. It covers seventy percent of the surface. It drives weather and regulates temperature. Life began in its waters three billion years ago. Today it faces extraordinary pressure from warming seas, plastic pollution, and overfishing. Yet it still holds secrets we have barely begun to uncover." \
  --language English \
  --profile stable \
  --chunk \
  --seed 20

speak "rascar-capac / stable / --chunk / FR" \
  --voice rascar-capac \
  --text "Le langage est bien plus qu'un outil de communication. Il façonne notre pensée, structure notre mémoire, et donne forme à nos émotions les plus intimes. Chaque mot porte en lui une histoire, un contexte, une culture entière. C'est pour cela que certaines choses ne se traduisent pas. Elles restent, irréductibles, dans la langue qui les a fait naître." \
  --language French \
  --profile stable \
  --chunk \
  --seed 21

# Text-file input
speak "chalamet-en / balanced / --text-file / EN  (long prose)" \
  --voice chalamet-en \
  --text-file "$_tf_en" \
  --language English \
  --profile balanced \
  --chunk \
  --seed 22

speak "rascar-capac / stable / --text-file / FR  (long prose)" \
  --voice rascar-capac \
  --text-file "$_tf_fr" \
  --language French \
  --profile stable \
  --chunk \
  --seed 23

speak "david-attenborough / stable / --text-file + --qa / EN  (narrative)" \
  --voice david-attenborough \
  --text-file "$_tf_attn" \
  --language English \
  --profile stable \
  --chunk \
  --qa \
  --seed 24

rm -f "$_tf_en" "$_tf_fr" "$_tf_attn"

print_summary "03-chunking"
