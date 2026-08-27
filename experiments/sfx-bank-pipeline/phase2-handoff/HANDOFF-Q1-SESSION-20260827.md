# Handoff — Q1 blind-review + Q2 gold-annotation session (2026-08-27)

Read together with `phase2-handoff/HANDOFF.md` (the phase-2 master script).
This file records where THAT session ended and exactly what is left.

## TL;DR for the next orchestrator

1. **Q1 (blind listening)** — **COMPLETE (2026-08-27, this session).** Owner
   imported his ratings onto the new origin and submitted **160/160** (payload
   validated: 0 missing fields). Archived as
   `review/blind/blind-ratings-Ben-final-160of160.json`; `incoming/` is empty;
   the one-time `import-ratings.html` page was removed per plan. For history:
   the real pre-import state was **140/160**, not 150/160 — the template never
   rendered the whole-clip "Overall preference" row (fixed this session), so
   those 10 ratings were impossible and their missing-pills jumped to nothing
   (the reported "click to jump doesn't work").
2. **Q2 (human-gold timing)** — NOT started. `annotations/` is empty. Owner
   must annotate pass `A` on walk1 → man → ocean-short via the gold
   annotator (steps in HANDOFF.md §0; fps 30/29.97/30; save files as
   `<clip>.A.gt.json` in `work/sfx-gold/sessions/20260826-demo/annotations/`).
   Orchestrator audits each file (start ≤ contact ≤ end, window brackets
   contact, fps matches session, notes state the "visible-first-touch"
   convention). Walk1 pass `B` happens days later, unrehearsed.
3. **Q4 (bake-off)** — blocked until ≥3 impulse clips have contact_frame
   (walk1 + man qualify today; walk2 is the third — its current event plan is
   wrong, see Truth findings). Command in HANDOFF.md §4.
4. **Q3 (shot list)** — manifest repaired this session: ocean rows' dangling
   `surf_bed_01_6ace93b1` anchor blanked (no surf class exists in any run's
   bank). The five `needs_clip` rows await physical shooting by the owner.

## Q1 FINAL verdict (complete 160/160 archive; only Event keep/replace changed vs the 140/160 preview)
| | bank | mirelo16 |
|---|---|---|
| Whole-clip keep/replace | 2 / 3 | 0 / 5 |
| Event keep/replace | 2 / 18 | 4 / 16 |
| Dims mean (1–5) | sync 2.4 · sem 2.0 · bel 2.2 · aq 1.8 · pal 2.0 · art 2.0 | sync 2.2 · sem 2.0 · bel 2.0 · aq 1.8 · pal 2.6 · art 2.0 |
| Event pref by clip | man 1.2 · oceans 5.0/4.0 · walks 1.0 | man 2.6 · oceans 2.0/2.0 · walks 1.0 |

Label mapping: man A=bank/B=mirelo16 · oceans A=mirelo16/B=bank ·
walk1 A=bank/B=mirelo16 · walk2 A=mirelo16/B=bank.

**Truth findings from owner notes (feed `docs/auto-sfx-phase2-truth-repair.md`):**
- man: "footsteps on grass" label wrong — surface is road/gravel.
- ocean-long/ocean-short: "rain falling" label wrong — it is waves/surf.
- walk2: event plan itself is wrong genre (sword whoosh/metal clang/explosion/
  camera shutter over a grass-walking video) → both systems score 1.0; plan
  must be regenerated or the clip dropped before it counts as an impulse clip.
- bank man render: noise pollution drowns steps; step-to-step sound
  inconsistent (palette strike). mirelo16 man: steps accurate, noise bed
  annoying. Neither shippable on man.
- Designed bank bed clearly wins on oceans → supports §3 bank-overlap
  expansion (camera shutter, wipers, gear shift, whoosh, bass-dive).

## Infrastructure notes

- Serving (repo root required):
  - Gold annotator: `python3 -m http.server 8765` →
    `http://127.0.0.1:8765/experiments/auto-sfx/gold_annotator.html?session=work/sfx-gold/sessions/20260826-demo/session.json`
    (session paths resolve against origin root — fixed this session).
  - Blind review: `python3 experiments/sfx-bank-pipeline/build_blind_review.py --run-dir work/sfx-pipeline/20260826-poc --serve --port 8898` →
    `http://127.0.0.1:8898/`. **Port 8899 is permanently taken** by an unrelated
    node service (`pfi-sniper-residential-host`, returns 401) — do NOT use it.
    The `--serve` variant is required: only it implements `POST /save`; a plain
    `python3 -m http.server` will happily serve the page and silently drop
    submissions. A server is running right now via hub process `blind-serve`;
    if it is gone, restart with the command above (repo root as cwd).
    Ratings recovery: the owner's ratings live in browser localStorage under the
    OLD origin (`:8899`); `review/blind/import-ratings.html` (served at
    `http://127.0.0.1:8898/import-ratings.html`) one-click-imports the
    140/160 export into the current origin. Regenerate it from the export JSON
    if missing. Delete once Q1 is archived.
- Rebuild blind page after template edits:
  `python3 experiments/sfx-bank-pipeline/build_blind_review.py --run-dir work/sfx-pipeline/20260826-poc`
  (rewrites `review/blind/index.html`; keeps existing `assignment.json` so
  blinding/seed is stable. NEVER hand-edit the built index.html.)
- `hub` broker was broken this session (ENOENT on broker.sock; report filed).
  Background services were run via `bash` with `async: true, timeout: 0`.
  IMPORTANT: plain `async: true` jobs die after 300s — a `--serve` process
  MUST use `timeout: 0`.
- Ratings POST contract: `POST /save` (≤32MB JSON) →
  `review/blind/incoming/blind-ratings-<annotator>-<ts>-<hex>.json`, fsynced.
  Payload keys: `clip|scope|id|label|field` (fields: the six dims, `pref`,
  `keep`, `notes`, `text` for general notes). Field-level truth lives INSIDE
  each object, not in the key.

## UI state (design-agent redesign + follow-up fixes, owner-accepted)

- Merged A|B event tables; event click cues both videos 0.75s before the
  moment, plays System A then System B ~2.5s each (sequential by design).
- Missing-pill strip is in normal flow above the clip list (NOT in the sticky
  bar — a sticky-bar variant inflated to 175px+ at ≤1150px and was reverted;
  commits 563d091, e25a060).
- **Screenshot trap for the next orchestrator:** stitched full-page captures
  bake the sticky bar at every section boundary and headless Chrome renders
  unloaded videos as black rectangles. Neither is a real defect — verify
  against the live DOM (`querySelectorAll('.progress').length === 1`) before
  believing a screenshot.
- Rating scale semantics: 1–5, higher = better; per-event pref is per-system
  (not pairwise). keep/replace = "would you ship it without swapping?".

## Commits from this session

- 4e05d03 blind-review render+Range fix; gold session-path resolution; manifest anchor repair
- a470db9 input→persistence wiring (was entirely absent) + general media notes
- 64ff398 design-agent UI redesign landing + missing-ratings jump list
- 563d091 pill dedupe (double label loop) + bar layout hardening
- e25a060 pills out of sticky bar into page flow
