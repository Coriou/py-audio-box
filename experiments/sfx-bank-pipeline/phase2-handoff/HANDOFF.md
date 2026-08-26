# Owner handoff — phase-2 evaluation steps

Everything below is minutes-scale per item. Do items in order; each is
independent of the others.

## 1. Blind listening — rate the EXISTING POC renders (Q1)

```bash
python3 experiments/sfx-bank-pipeline/build_blind_review.py \
    --run-dir work/sfx-pipeline/20260826-poc --serve --port 8899
# open http://127.0.0.1:8899/
```

- 5 clips × 2 anonymous systems × 20 visible events. Everything is already
  rendered — zero generation cost.
- Required: per-system six 1–5 dimensions + keep/replace on the whole clip;
  per-event overall preference + keep/replace.
- The keep/replace question means: *would you ship it without swapping the sound?*
- On submit the system identity reveal appears; ratings autosave to
  localStorage AND land in `review/blind/incoming/*.json` when served.
  The Download button is the file:// fallback.

## 2. Human-gold timing annotations (Q2 prerequisite)

Two independent passes, annotator IDs `A` and `B` (owner decision 2026-08-26):

```bash
python3 -m http.server 8765   # repo root
# open http://127.0.0.1:8765/experiments/auto-sfx/gold_annotator.html?session=work/sfx-gold/sessions/20260826-demo/session.json
```

- Demo session covers walk1 / man / ocean-short (fps probed: 30 / 29.97 / 30).
- Pass A: annotate impulses honestly first instinct. Pass B: start fresh
  (different localStorage name "B"), same clips, ideally a day later. Save
  both JSONs into `annotations/<clip>.<A|B>.gt.json`. Disagreements are kept,
  never merged.
- Deliverable threshold for the bake-off to run: ≥3 impulse-bearing clips with
  contact_frame filled (walk1/man qualify).
- New owner-shot phone clips: record `source: shot-by-owner YYYY-MM-DD,
  device <model>` inside each event's notes; no third-party footage without a
  recorded license URL (manifest registry tracks it).

## 3. Shot list for bank-overlap clips (Q3 prerequisite)

`experiments/sfx-bank-pipeline/phase2-handoff/bank_overlap.manifest.json`
lists needs_clip rows (phone-camera OK, ≥30 fps, ≤15 s each): camera shutter
close-up exercising BOTH COMCam analog + focus-beep assets · wipers full cycle
incl. lever start/stop · manual gear shift clanks interior · whoosh pass-by ·
one scene where a designed bass-dive cue fits. After shooting: add rows
`{"clip": "...", "status": "seeded", "expected_bank_entry": "<indexed id>"}`
— retrieval-eval's forced top-5 page then tags best/acceptable/wrong per event.

## 4. Bake-off once annotations exist

```bash
/tmp/pab-sfx-bench/bin/python experiments/sfx-bank-pipeline/timing_bakeoff.py \
    --gold work/sfx-gold/sessions/*/annotations/*.gt.json \
    --detected experiments/auto-sfx/gt/walk1.agent-draft.gt.json \
    --report-dir work/sfx-gold/reports
```
(For real detector runs point --detected at auto-sfx runs/events.json.)
