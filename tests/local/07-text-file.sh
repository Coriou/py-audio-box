#!/usr/bin/env bash
# tests/local/07-text-file.sh
# --text-file input: reads synthesis text from a file instead of inline.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source tests/local/lib/common.sh

echo "=== 07: --text-file input ==="

# Files must be in ./work/ (mounted as /work/ inside the container).
# Use unique names via $$ (pid) to avoid collisions across parallel runs.
_hf1="work/synth_text_$$.1.txt"
_hf2="work/synth_text_$$.2.txt"
_hf_chunk="work/synth_text_$$.chunk.txt"

# Container paths (./work is mounted as /work)
_cf1="/work/synth_text_$$.1.txt"
_cf2="/work/synth_text_$$.2.txt"
_cf_chunk="/work/synth_text_$$.chunk.txt"

mkdir -p work
printf 'Voice synthesis from a text file works exactly the same as inline text. This test verifies the file path is resolved correctly and the content is read in full.' \
  > "$_hf1"

printf "La synthese vocale a partir d un fichier texte fonctionne de la meme facon que le texte en ligne. Ce test verifie que le chemin est correct et que le contenu est lu integralement." \
  > "$_hf2"

printf 'This is the first sentence in a chunked text file. It should be synthesised in sequence with the rest. Here comes the second sentence. And now a third, to ensure the chunking logic handles multiple segments gracefully from a file source.' \
  > "$_hf_chunk"

speak "chalamet-en-2 / balanced / --text-file / EN" \
  --voice chalamet-en-2 \
  --text-file "$_cf1" \
  --language English \
  --profile balanced \
  --seed 20

speak "rascar-capac / stable / --text-file / FR" \
  --voice rascar-capac \
  --text-file "$_cf2" \
  --language French \
  --profile stable \
  --seed 21

speak "david-attenborough / stable / --text-file + --chunk / EN" \
  --voice david-attenborough \
  --text-file "$_cf_chunk" \
  --language English \
  --profile stable \
  --chunk \
  --seed 22

rm -f "$_hf1" "$_hf2" "$_hf_chunk"

print_summary "07-text-file"
