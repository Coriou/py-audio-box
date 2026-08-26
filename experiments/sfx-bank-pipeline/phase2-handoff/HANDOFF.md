# Owner handoff — phase-2 evaluation steps

Everything below is minutes-scale per item; items are independent. **Section 0
is written FOR THE NEXT AGENT/ORCHESTRATOR** running this session with the
owner at the keyboard — follow it verbatim when guiding.

---

## 0. AGENT SCRIPT — guiding the owner's first annotations

You are walking the owner through their very first human-gold timing pass.
Goal of the guide: owner produces defensible frame-level events, not perfect
ones. Talk them through clip-by-clip; verify each step on screen before moving
on.

### Pre-flight (agent does, owner watches)

1. Serve from repo root and prove it before opening anything:
   ```bash
   cd <repo-root> && python3 -m http.server 8765 &
   curl -sf -o /dev/null http://127.0.0.1:8765/work/sfx-bench/20260825-165347/inputs/man.mp4 && echo SERVE-OK
   ```
   If `SERVE-OK` does not print, stop and fix serving first.
2. Open exactly:
   `http://127.0.0.1:8765/experiments/auto-sfx/gold_annotator.html?session=work/sfx-gold/sessions/20260826-demo/session.json`
3. **Known pitfall — if the browser console shows**
   `Unsafe attempt to load URL file:///... 'file:' URLs are treated as unique
   security origins`, the page was opened over `file://`. The page itself now
   detects this and shows a red explainer banner instead of hanging after the
   name prompt (owner reported this 2026-08-26). Do NOT debug it further:
   point the owner at the `http://localhost:8765/...` URL from step 2. Same
   remedy if the annotator-name prompt appears but no clip list follows
   (session fetch failed) — check server cwd is the REPO ROOT, since clip
   paths resolve as `/work/...`.
4. Annotator name: owner types `A` (pass B later uses `B`). Stored locally;
   per-pass files land as `<clip>.A.gt.json` / `<clip>.B.gt.json`.

### Guided rubric (say this while annotating walk1 together)

- **What counts as contact**: the frame where shoe/surface contact becomes
  visible — *not* the audio-salient instant and *not* full weight-bearing
  necessarily. CTO §3.1 lists acceptable ambiguities (heel strike vs toe,
  compression, gravel displacement): pick one convention (visible-first-touch)
  and state it once into the notes of every event to keep A/B comparable.
- **Motion blur / fps quantization**: choose the FIRST frame where contact is
  discernible; if two frames tie, take the earlier and set confidence medium.
- **Acceptable window semantics**: `[contact−1, contact+1]` default is correct;
  widen only when the frame is genuinely ambiguous (e.g., 60 fps source). The
  bake-off treats the window as soft ground truth and ±33/50/100 ms as hard
  gates — do not inflate windows to be safe; record real uncertainty via
  confidence instead.
- **Pass structure**: one event per visible impulse. Do NOT annotate "gait"
  continuously — each footfall is its own impulse event; the bake-off matches
  detector contacts 1-to-1 against these.
- **Repetitive type**: reserve for wipers/hammering-style sequences where
  individual events are indistinct; normal footsteps are impulses even though
  they repeat.
- **When unsure a footstep happened**: still annotate with confidence low +
  note; missed-event analysis needs false-anchor candidates too.
- **Per event capture order**: scrub near the moment → Set start → step to
  visible contact → Set contact → step past settle → Set end → verify the
  readout numbers → Save event.
- **Agent verifies between clips**: events table shows sensible s/c/e triples
  (start ≤ contact ≤ end), windows bracket contact, FPS shown matches session
  listing (walk1 30, man 29.97 — sub-second timestamps must use the right
  fps), file names end `.A.gt.json`.

### Pass-B disagreement protocol

Owner repeats walk1+man days later under name `B` without re-reading pass A.
Never reconcile by hand: keep both files; the bake-off reports inter-rater
spread alongside detector error.

### Quality bar before calling the session done

≥3 impulse-bearing clips annotated · every event has contact_frame AND notes
line stating the contact convention used · windows untouched unless justified
· both `A` and `B` files exist for walk1.

## 1. Blind listening — rate the EXISTING POC renders (Q1)

```bash
python3 experiments/sfx-bank-pipeline/build_blind_review.py \
    --run-dir work/sfx-pipeline/20260826-poc --serve --port 8899
# open http://127.0.0.1:8899/
```

- 5 clips × 2 anonymous systems × 20 visible events; zero generation cost.
- Required: per-system six 1–5 dimensions + keep/replace on the whole clip;
  per-event overall preference + keep/replace.
- Keep/replace means: *would you ship it without swapping the sound?*
- On submit the identity reveal appears; ratings autosave to localStorage AND
  post to `review/blind/incoming/*.json` when served; Download is the fallback.

## 2. Human-gold timing annotations (Q2 prerequisite)

Same serve command as §0 step 1; page URL in §0 step 2. Demo session covers
walk1 / man / ocean-short (fps probed: 30 / 29.97 / 30). Threshold for the
bake-off: ≥3 impulse clips with contact_frame filled (walk1/man qualify).
Newly shot phone clips: record `source: shot-by-owner YYYY-MM-DD, device
<model>` in each event's notes; no third-party footage without a recorded
license URL.

## 3. Shot list for bank-overlap clips (Q3 prerequisite)

`experiments/sfx-bank-pipeline/phase2-handoff/bank_overlap.manifest.json`
lists needs_clip rows (phone-camera OK, ≥30 fps, ≤15 s each): camera shutter
close-up exercising BOTH COMCam analog + focus-beep assets · wipers full cycle
incl. lever start/stop · manual gear shift clanks interior · whoosh pass-by ·
one scene where a designed bass-dive cue fits. After shooting add rows
`{"clip": "...", "status": "seeded", "expected_bank_entry": "<indexed id>"}` —
retrieval-eval's forced top-5 page then tags best/acceptable/wrong per event.

## 4. Bake-off once annotations exist

```bash
/tmp/pab-sfx-bench/bin/python experiments/sfx-bank-pipeline/timing_bakeoff.py \
    --gold work/sfx-gold/sessions/*/annotations/*.gt.json \
    --detected experiments/auto-sfx/gt/walk1.agent-draft.gt.json \
    --report-dir work/sfx-gold/reports
```
(For real detector runs point --detected at auto-sfx runs/events.json.)
