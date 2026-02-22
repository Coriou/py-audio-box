#!/usr/bin/env bash
# tests/remote/06-long-form.sh
# Long-form synthesis: ~60-90 seconds of audio per test.
#
# This suite exercises the model under sustained generation load — the kind
# encountered in real audiobook, podcast, and narration production jobs.
# Each test targets ~60–90s of output audio, covering:
#   - Single-take synthesis without chunking  (within the model's token budget)
#   - Multi-chunk long prose via --chunk      (exercises chunk stitching)
#   - EN + FR languages
#   - Multiple voice identities
#
# RTF values here are the most meaningful indicator of usable throughput:
# a GPU that does 2× RTF takes twice as long as real-time to produce 60s clips.
# Compare the benchmark/*.json files across targets to see which hardware wins.
set -euo pipefail
cd /app 2>/dev/null || true
source tests/remote/lib/common.sh

echo "=== 06: long-form synthesis (~60-90s audio per test, TARGET=${TARGET}) ==="

# ── Test 1: David Attenborough / EN / stable / ~65s / no-chunk ────────────────
# ~700 chars → expected audio ~55-70s (single take, no chunking needed)
speak "david-attenborough / stable / EN / ~65s / no-chunk" \
  --voice david-attenborough \
  --text "The forest at dawn holds its breath. Light seeps through the canopy in thin golden shafts, illuminating worlds invisible to those who hurry past. In this ancient woodland, older than memory, the first creatures stir. A deer pauses at the edge of a clearing, head raised, listening. Somewhere above, a woodpecker begins its morning work, the rhythmic drumming echoing through stands of oak and beech. These trees have witnessed centuries of such mornings. They remember seasons of drought and flood, storms that felled their neighbours, springs that returned with improbable tenderness. To walk here is to move through time itself. The forest does not know urgency. It knows only the steady accumulation of years, the patient turning of seasons, the quiet persistence of life." \
  --language English \
  --profile stable \
  --seed 60

# ── Test 2: Rascar-Capac / FR / stable / ~60s / no-chunk ─────────────────────
# ~730 chars → expected audio ~55-65s French
speak "rascar-capac / stable / FR / ~60s / no-chunk" \
  --voice rascar-capac \
  --text "La montagne m'a toujours fasciné par son silence particulier. Ce n'est pas l'absence de bruit, mais quelque chose de plus profond — une présence qui écrase les préoccupations ordinaires sous le poids de sa permanence. On grimpe pendant des heures, les muscles qui brûlent, le souffle court, et puis soudain, un plateau s'ouvre. La vallée en dessous semble appartenir à un autre monde. Les villages, les routes, les champs soigneusement dessinés — tout cela paraît fragile depuis là-haut, presque provisoire. Et c'est exactement ce que la montagne enseigne: que nos constructions, nos histoires, nos petites querelles ne sont que des annotations en marge d'une page dont la montagne est le texte principal. Il faut venir ici de temps en temps pour remettre les choses à leur juste place." \
  --language French \
  --profile stable \
  --seed 61

# ── Test 3: Chalamet-EN / EN / balanced / ~90s / --chunk ─────────────────────
# ~1050 chars → 2 chunks → expected audio ~80-95s total
speak "chalamet-en / balanced / EN / ~90s / --chunk" \
  --voice chalamet-en \
  --text "The first time I genuinely understood what it meant to be afraid was not in a dark alley or during some dramatic confrontation. It was on a Tuesday afternoon in an unremarkable office building, sitting across from a woman who looked at my work with absolute stillness for what felt like a very long time. That silence contained the possibility of everything: acceptance and rejection, continuation and ending, the whole architecture of what had seemed certain dissolving in real time. I had spent years preparing for precisely that moment. I had practised what I would say, anticipated every objection, rehearsed confidence until it was almost indistinguishable from the real thing. None of it mattered in the end. What mattered was the silence, and what lived inside it, and whether I could sit with the not-knowing long enough to let the answer arrive on its own terms. That is a skill nobody teaches you. You learn it only by failing to have it at the worst possible moments." \
  --language English \
  --profile balanced \
  --chunk \
  --seed 62

# ── Test 4: Hikaru-Nakamura / EN / expressive / ~65s / no-chunk ──────────────
# ~710 chars → expected audio ~60-75s
speak "hikaru-nakamura / expressive / EN / ~65s / no-chunk" \
  --voice hikaru-nakamura \
  --text "Twenty moves in and the position is already completely beyond what any engine would call normal. Both kings have castled into absolute firestorms. White has sacrificed the exchange for what looks like completely unclear compensation, and the spectators have no idea what is happening. But here is what I know: the chess that creates unforgettable games is not the chess that computation finds optimal. It is the chess that emerges from two human minds pushing each other to the absolute edge of their understanding. Right now, on board three, that is exactly what we are watching. The knight on d5 is simultaneously protected and exposed. The clock is ticking. Both players are deep in thought. Whatever comes next will echo in tournament halls for a very long time." \
  --language English \
  --profile expressive \
  --seed 63

# ── Test 5: Rascar-Capac / FR / expressive / ~80s / --chunk ──────────────────
# ~960 chars → 2 chunks → expected audio ~75-90s French
speak "rascar-capac / expressive / FR / ~80s / --chunk" \
  --voice rascar-capac \
  --text "Permettez-moi de vous parler d'une découverte qui a changé ma façon de voir le monde, ou plutôt d'entendre le monde. J'avais toujours cru que la musique était quelque chose que l'on écoutait passivement, que les sons arrivaient à nous et que nous les recevions sans effort particulier. Puis un jour, un ami musicien m'a demandé d'écouter autrement. Non pas avec les oreilles, mais avec tout le corps. Il m'a appris à sentir les vibrations dans mes pieds, à percevoir les basses comme une chaleur dans la poitrine, à entendre les harmoniques comme des couleurs au-dessus du son principal. Cette expérience a duré peut-être dix minutes. Mais elle a tout changé. Depuis ce jour, je n'écoute plus la musique de la même façon. Je l'habite. Et ce que j'ai découvert, c'est que la frontière entre écouter et ressentir n'existe pas vraiment. Elle a toujours été une illusion confortable." \
  --language French \
  --profile expressive \
  --chunk \
  --seed 64

print_summary "06-long-form"
