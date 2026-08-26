# CTO Review — Reliable Vision-Timed, SFX-Bank-Conditioned Foley Pipeline

**Date:** 2026-08-26  
**Repository:** `Coriou/py-audio-box`  
**Status:** continue POC / no production architecture selected yet  
**Primary objective:** find the simplest commercially viable architecture that can produce convincing Foley, tightly synchronized to video, while preferentially retaining the sonic identity of an owned/reference SFX library.

---

## 1. Executive decision

**Continue. The POC did not falsify the product thesis. It falsified one proposed implementation detail: treating generated V2A audio as the authoritative timing clock.**

The work already completed is useful and should be retained. In particular:

- the bank indexing/retrieval machinery;
- sample-exact placement;
- transient-preserving DSP;
- provider/cost ledger;
- per-event auditable decisions;
- A/B rendering infrastructure.

The test did provide good evidence that extracting onset timings from a generated Mirelo/Kling/ControlFoley-style soundtrack is not reliable enough to serve as the sole event clock. On the tested walking clips, the best result was only 2/4 contacts at a median 68.2 ms error; Mirelo missed all four contacts on one clip. Loosening the onset detector generated more candidates without resolving the timing problem.

However, **do not promote the opposite claim — “vision is a perfect clock” — to fact yet.** The current validation compares generated-audio onsets against another automated detector (`auto-sfx`) and then aligns output to those same detector events. Exact placement against that clock is therefore exact *by construction*. It does not establish that the vision detector itself agrees with the frame at which a human sound editor would place the Foley event. The report currently makes this jump too quickly.

There is also no perceptual result yet: nobody has listened to the A/B renders. The report correctly acknowledges this.

### Working architectural hypothesis

For the next phase, work on this decomposition:

```text
VIDEO
  │
  ├──► semantic event proposal
  │       "left foot hits gravel"
  │       "camera shutter"
  │       "door starts closing"
  │       "cloth rub"
  │
  ├──► native-frame timing refinement
  │       onset / contact / interval / intensity envelope
  │
  ▼
EVENT MODEL
  │
  ├──► SFX-bank retrieval ───────► direct placement
  │                                  │
  │                                  ├─ minor DSP
  │                                  │
  │                                  └─ reference-conditioned adaptation
  │
  └──► no acceptable bank match ──► generative fallback
                                      │
                                      ▼
                              final event stems
                                      │
                                      ▼
                             deterministic mix

```

In shorthand:

> **Vision = when and what**  
> **Bank = sonic identity**  
> **DSP = cheap adaptation**  
> **Reference-conditioned generation = repair/variation**  
> **Pure V2A = fallback / semantic sketch / baseline**

That is now our leading hypothesis, **not yet our conclusion**.

---

# 2. What the current POC actually proves

## 2.1 Good result: V2A audio is a poor authoritative onset clock

I accept this result for the current class of generated soundtracks.

The report tested multiple hosted V2A outputs rather than simply Mirelo, and the failure did not disappear under more permissive onset-detection settings. Generated tracks appear to contain plausible audiovisual rhythm without consistently exposing clean, frame-corresponding acoustic transients.

This is believable technically because the generated Foley is often mixed into ambience. The engineer measured footstep onset energy only around 1–6 dB above surrounding texture; pure spectral-flux detection consequently exploded to implausible event counts.

Therefore:

**Do not spend another cycle trying to perfect SuperFlux extraction from Mirelo output as our primary clock.**

We may still extract:

- semantic hints;
- approximate intensity envelopes;
- ambience characteristics;
- additional event proposals;
- rough continuous-motion dynamics.

But those timestamps should not determine final transient placement.

---

## 2.2 Good result: deterministic placement machinery is worth preserving

The current implementation already provides:

- normalized bank assets;
- onset maps;
- CLAP embeddings;
- tier selection;
- attack-preserving placement;
- post-transient compression;
- provider cost accounting;
- review artifacts.

That is a useful skeleton and does not need rewriting.

The principle that **an impulsive transient should not be time-stretched just to fit a visual interval** is correct. Keep it.

---

## 2.3 Good result: the retrieval system can reject obviously wrong sounds

The current bank was initially one gear-shift clank and appropriately rejected it for all twenty unrelated events. That shows the system is not indiscriminately forcing a bank sound onto every event.

That is useful but weak evidence.

It does **not** tell us:

- whether it picks the best of ten plausible footsteps;
- whether it understands gravel vs concrete vs wood;
- whether it preserves a coherent sound palette;
- whether CLAP ranking correlates with a Foley editor's choice;
- whether the direct/DSP/A2A tier decisions are sensible.

Those are now central questions.

---

# 3. What I do not accept yet

## 3.1 “Vision-derived contacts are a perfect clock”

This conclusion is premature.

The ground truth used by the report is itself `auto-sfx` output. Until we independently annotate visible events by hand, we're measuring:

```text
generated-audio detector
        versus
vision detector

```

not:

```text
system
        versus
human-perceived correct synchronization

```

The distinction matters.

A foot touching the ground visually is not necessarily the acoustically salient instant in every shot:

- heel strike vs full-foot contact;
- toe strike;
- shoe compression;
- loose gravel displacement;
- object deformation;
- motion blur;
- 24/25/30/60 fps temporal quantization;
- editorial anticipation or lag that sounds more natural than geometric contact.

The next benchmark needs **human frame-level reference annotations**.

---

## 3.2 Retrieval quality has effectively not been tested

The addendum increases the bank to eight files, but they are camera, designed and vehicle assets while the current benchmark remains heavily footstep-oriented. The report correctly says this still cannot answer retrieval quality.

Do not run another statistical retrieval analysis until the bank and videos overlap semantically.

Either:

1. add appropriate footsteps/impacts/water/etc. to the bank;

or preferably also:

2. create test clips specifically corresponding to the new camera/vehicle/designed assets.

A mismatch that produces rejection is a useful unit test, not a product benchmark.

---

## 3.3 Human quality has not been evaluated at all

This is the highest-priority gap.

The report explicitly states that no one listened to the A/B output.

Before optimizing model architecture, **listen to what we already built**.

Some objectively imperfect sync offsets may be perceptually acceptable. Conversely, sample-perfect placement can sound absurd if:

- the wrong physical sound was selected;
- its perspective is wrong;
- the transient is too hard;
- its acoustic space disagrees with the scene;
- every step repeats the same sample;
- gain is wrong;
- the tail is unnaturally truncated.

The ultimate metric is not timestamp MAE. It is whether a human believes the scene.

---

# 4. Do not treat all Foley events the same

This is the biggest conceptual improvement I want in the next architecture.

A generic event schema of:

```json
{
  "label": "footstep",
  "timestamp": 1.42
}

```

is insufficient.

There are fundamentally different temporal classes.

## A. Impulsive events

Examples:

- footsteps;
- shutter clicks;
- button presses;
- door latches;
- impacts;
- glass taps.

Representation:

```text
contact / peak timestamp
+ optional start/end
+ intensity

```

These are ideal for deterministic transient placement.

---

## B. Continuous gestures

Examples:

- cloth movement;
- scraping;
- dragging;
- pouring;
- wheel rolling;
- zipper;
- hand sliding over material.

Representation:

```text
start
end
time-varying intensity envelope

```

A single onset is not enough.

---

## C. Repetitive sequences

Examples:

- walking/running;
- windshield wipers;
- machinery;
- repetitive hammering.

Representation:

```text
sequence-level identity
+ ordered event timestamps
+ per-event intensity

```

The retrieval system should preferably choose a **family of related variants**, not independently retrieve the same sample N times.

---

## D. Beds / ambience

Examples:

- ocean;
- rain;
- traffic;
- room tone;
- wind.

Representation:

```text
scene interval
+ overall intensity envelope
+ environment/spatial metadata

```

Do not force these through the one-shot retrieval path.

---

## E. Hybrid events

Examples:

```text
door:
    hinge movement  ────────────────
                           latch ─╲_
                                  ▲ transient

```

A high-quality door sound may require:

- continuous creak/body;
- terminal impact;
- latch click;
- room tail.

Treat these as structured Foley objects rather than one WAV.

---

# 5. Upgrade the internal event representation

Create one canonical schema used by every detector, retrieval stage and renderer.

Suggested minimum:

```json
{
  "event_id": "evt_001",
  "class": "footstep",
  "subclass": "shoe_ground_contact",

  "temporal_type": "impulse",
  "start_s": 1.383,
  "contact_s": 1.417,
  "end_s": 1.620,

  "confidence": 0.91,

  "source": {
    "object": "left shoe",
    "material": "rubber/leather",
    "surface": "gravel"
  },

  "physical": {
    "intensity": 0.68,
    "velocity": 0.54,
    "size": 0.45
  },

  "visual": {
    "x": 0.47,
    "y": 0.81,
    "offscreen": false
  },

  "audio_strategy": null
}

```

Fields can be unknown.

The important change is that **our downstream system stops pretending every event is simply** `label + timestamp`**.**

---

# 6. Build a real gold benchmark before improving models

Create a small internal benchmark — call it `FoleyScope` or similar.

It does not need thousands of videos.

It needs **carefully selected videos with useful diversity and proper ground truth.**

## Suggested first set

Aim for roughly 30–50 short clips covering:


| Family      | Example scenes                                  |
| ----------- | ----------------------------------------------- |
| Footsteps   | hard floor, gravel, grass, stairs, running      |
| Impacts     | hand/table, dropped object, tool/metal          |
| Mechanisms  | door, drawer, camera shutter, switches          |
| Vehicles    | wipers, gear/action, engine/mechanical movement |
| Cloth/body  | jacket movement, bag, fabric handling           |
| Continuous  | scrape, drag, pour/flow                         |
| Natural     | water/surf, wind                                |
| Multi-event | 2–4 overlapping categories                      |


Current camera/vehicle assets are useful: find or shoot clips that actually exercise them rather than continuing to benchmark them against footsteps.

---

## Human timing annotations

For each visually identifiable event:

- annotation by **two people** where practical;
- annotate the video frame, not generated audio;
- store:
  - event label;
  - start;
  - contact/peak;
  - end;
  - ambiguity;
  - confidence;
- keep disagreements instead of hiding them.

Example:

```json
{
  "contact_frame": 143,
  "acceptable_frames": [142, 144]
}

```

This is much better ground truth than claiming an arbitrary millisecond is physically absolute.

---

# 7. Measure the correct things

## Timing metrics

For impulsive events report:

- recall at ±33 ms;
- recall at ±50 ms;
- recall at ±100 ms;
- false positives/minute;
- median onset error;
- p95 onset error.

Do **not** report median error alone: a detector that finds only two easy events can appear excellent.

---

## Retrieval metrics

For every event for which a bank match exists:

- top-1 acceptable?
- top-3 contains acceptable candidate?
- chosen sample preferred by human?
- wrong-category selection?
- palette/style consistency?

CLAP cosine is an internal feature, not a quality metric.

---

## Listening metrics

Blind the identity of systems.

Ask for 1–5 ratings for:

- synchronization;
- semantic correctness;
- physical believability;
- audio quality;
- consistency with the chosen SFX palette;
- artifacts;
- overall preference.

Also ask one binary question:

> **Would you keep this result without manually replacing the sound?**

That is a useful product metric.

---

# 8. Rework retrieval: CLAP alone is not enough

Current scoring is:

```text
0.6 × audio cosine
+
0.4 × text cosine

```

This is acceptable for first-pass retrieval but too generic for professional Foley.

Use **hierarchical retrieval**.

## Stage 1 — semantic candidate filtering

CLAP/text/tags answer:

> Is this broadly the correct kind of sound?

For example:

```text
footstep
  └── gravel
        └── shoe

```

---

## Stage 2 — acoustic/morphological reranking

Measure cheap deterministic descriptors:

- duration;
- transient strength;
- onset position;
- RMS envelope;
- decay time;
- spectral centroid;
- spectral bandwidth;
- spectral rolloff;
- zero-crossing/noisiness;
- harmonicity;
- dominant pitch where meaningful;
- stereo width;
- number/onset density of transients.

Then rank based on event type.

For an impact, transient shape may matter enormously.

For cloth, duration/envelope/noisiness may matter more.

For ambience, texture and spectral profile dominate.

There should **not** be one global `0.30` match threshold for every sound category.

---

# 9. Vision timing should be hierarchical too

Do not look for one giant model that directly gives perfect Foley timestamps for every physical phenomenon.

Use two stages.

## Stage 1 — semantic proposal

A multimodal VLM can inspect the video and propose:

```text
0.8–1.6 s: person takes step on gravel
2.4–3.1 s: person opens metal door
3.6–4.0 s: jacket moves

```

This stage answers **what and approximately where**.

Current Gemini video APIs, for example, allow custom FPS sampling, with higher rates specifically recommended for fast action; this makes VLMs useful as proposal mechanisms but not necessarily sufficient as the final sub-frame clock.

Qwen3-VL is another plausible proposal/temporal-grounding candidate because timestamp alignment is a first-class capability in its video stack.

---

## Stage 2 — native-frame specialist refinement

Once we know:

> “there is probably a left-foot strike between 1.30 and 1.55 s”

analyse only those frames.

Examples:

### Footsteps

Use:

- pose/keypoints;
- ankle/heel trajectory;
- vertical velocity;
- foot-ground proximity;
- change in direction;
- optical flow.

### Impacts

Use:

- object tracking;
- hand/object velocity;
- collision/proximity;
- abrupt trajectory change;
- deformation where visible.

### Doors/mechanisms

Use:

- tracked moving geometry;
- angular/linear velocity;
- motion stop;
- latch/contact point.

### Continuous events

Use:

- optical-flow magnitude;
- motion envelope;
- object-specific motion.

This should be substantially easier and cheaper than asking a multimodal transformer to infer millisecond timing directly.

---

# 10. Test an existing visual onset detector before building ours

**FoleyCrafter is worth dissecting specifically for its timing component.**

Its pipeline contains a separate `VideoOnsetNet` timestamp detector; the public inference code loads that detector independently and converts its predictions into temporal conditioning before audio generation.

This is exactly the sort of reusable architectural component we should benchmark against `auto-sfx`.

Important caveat: the overall FoleyCrafter generation pipeline relies on an Auffusion model whose released weights carry a non-commercial CC license, so **do not assume the complete system is production-safe**.

Task:

> Isolate and benchmark the onset detector independently. Before adopting it commercially, verify the detector checkpoint's own license and training-data provenance separately from the rest of FoleyCrafter.

Do not assume “repository Apache” automatically covers every downloaded weight.

---

# 11. Consider eventually owning the timing model

This may be strategically much easier than owning an entire generative Foley model.

The **Greatest Hits** audiovisual dataset is publicly distributed under Creative Commons Attribution 4.0 and contains impact/scratch videos with annotations; its original work also contains material/action semantics and impact locations.

That suggests an attractive future path:

```text
video
   ↓
small specialist timing/envelope model
   ↓
event timestamps/intensity

```

trained independently on permissively usable datasets plus our own annotated clips.

This model would:

- be small;
- be cheap to host;
- avoid generating copyrighted audio;
- provide a defensible technical differentiator;
- avoid dependency on one Chinese/US inference vendor;
- be easier to fine-tune per physical event family.

Do **not** train this yet. First establish that timing is truly the remaining bottleneck using human-gold evaluation.

---

# 12. Reference-audio-conditioned V2A remains strategically important

This research direction maps almost perfectly to our desired product.

## ControlFoley

ControlFoley explicitly supports:

```text
video
+
reference audio
→
video-synchronized audio retaining reference timbre

```

That is precisely the desired abstraction.

However, the released model weights are **CC BY-NC 4.0**, despite permissively licensed code. Do not make them a production dependency without an explicit commercial agreement.

Use it as a research benchmark only if our usage is legally cleared.

### What we want to learn from it

Not:

> “Can we ship ControlFoley?”

Instead:

> **“How much quality do we gain when video timing and reference timbre are solved jointly?”**

If ControlFoley destroys our hybrid pipeline in listening tests, that's valuable information. It tells us what missing capability we need to license or reproduce independently.

---

## AC-Foley

AC-Foley is another 2026 model designed specifically around **reference-audio-guided acoustic transfer**, including fine-grained timbre/material control.

Its released weights are likewise CC BY-NC, so treat it as research/non-production unless we obtain appropriate rights.

Benchmarking it against ControlFoley would be useful if available locally.

---

## MultiFoley

MultiFoley is perhaps the closest conceptual match to the eventual product: the research explicitly contemplates selecting reference sounds from an SFX library and generating Foley synchronized to video.

Unfortunately, the official repository still does not provide a usable public model/code release from which we can build a production integration.

Keep it on the watchlist rather than spending engineering time attempting to reconstruct it.

---

## FoleyGenEx

FoleyGenEx is another very recent reference-audio-controlled V2A system emphasizing frame-level synchronization and audio extension.

Again: interesting research target, not currently a dependable hosted production dependency.

---

# 13. Stable Audio 3 should remain in the stack — but as an adapter

The POC correctly validated Stable Audio 3 SFX audio-to-audio end-to-end at **$0.024/generation**, including deterministic seeds and WAV output when explicitly requested. It also caught the dangerous 30-second/default-MP3 behavior.

Fal's current Stable Audio 3 Small SFX A2A endpoint is expressly designed around **reference audio + prompt → new sound variation**.

That makes it interesting for:

```text
bank asset:
    "heavy leather footstep on gravel"

visual event:
    lighter/faster footstep

SA3:
    reference bank WAV
    +
    "short light leather shoe step on loose gravel"
          ↓
new variation

```

Then **we align its generated transient ourselves**.

This preserves the architecture:

```text
vision determines time
bank determines identity
SA3 modifies morphology

```

instead of asking SA3 to understand the video.

### Important

SA3 should not automatically run for every event.

Order of preference:

1. direct asset;
2. cheap deterministic DSP;
3. SA3 A2A;
4. full generative V2A fallback.

That preserves both margins and identity.

---

# 14. Mirelo remains valuable, just not as the clock

One correction to keep in the engineer notes: **Mirelo is currently available on Fal.**

Current Fal endpoints include:

```text
mirelo-ai/sfx1.6/video-to-video

```

at **$0.01 per generated second**, and the older:

```text
mirelo-ai/sfx-v1/video-to-audio

```

currently listed around **$0.007 per second/sample**.

SFX 1.6 does not expose reference-audio conditioning through its documented Fal V2V interface; it remains primarily video + generation controls/prompt.

Keep Mirelo for three uses:

### Baseline

Every experiment should compare against:

> “just give the video to Mirelo.”

If our complex stack doesn't clearly beat that, complexity is unjustified.

### Missing-event fallback

When no suitable bank sound exists:

```text
generate with Mirelo

```

and use the resulting audio rather than silently producing nothing.

### Semantic/intensity sketch

Investigate whether its output is useful for:

- broad scene dynamics;
- ambience;
- event intensity;
- sounds that vision failed to propose.

Just don't trust extracted transients as authoritative timestamps.

---

# 15. ThinkSound / PrismAudio: quality benchmark, license caution

ThinkSound remains worth a targeted quality run. Fal currently advertises a hosted ThinkSound endpoint at roughly **$0.001 per compute second** and marks it for commercial use.

However, the upstream project states that the released code/models/dataset package is intended for research/educational use and that commercial use of its released model assets is not permitted.

That creates the same kind of provider-vs-upstream-license ambiguity we encountered previously.

Therefore:

- **quality benchmark:** yes;
- **production dependency:** only after Fal confirms in writing that its commercial agreement covers our intended usage and territories.

Its successor **PrismAudio** is also worth tracking as a pure V2A quality baseline, not as the main bank-conditioned architecture.

---

# 16. Do not use Synchformer as the primary clock

Synchformer is useful for measuring/coarsely correcting audiovisual offset, and multiple V2A systems use Synchformer-derived features or metrics. But its released synchronization task is temporal-offset classification over comparatively coarse windows, not a purpose-built 20–50 ms Foley contact detector.

Use it, if useful, for:

- global A/V sanity checking;
- scoring generated alternatives;
- detecting gross desynchronization.

Do not use it to decide:

> “the shoe touches the ground at 1.417 s.”

There is also an unresolved public question around commercial rights for its pretrained checkpoints, so treat the weights separately from the MIT-licensed code.

---

# 17. Architecture bake-off: test three systems, not twenty models

The next experiment should explicitly compare three architectures.

## Architecture A — deterministic bank

```text
vision event detection
→ bank retrieval
→ DSP
→ deterministic placement

```

No generative audio except optional ambience.

This is our **control for complexity**.

---

## Architecture B — reference-conditioned research model

Where legally acceptable for internal evaluation:

```text
video
+
best bank reference
→ ControlFoley / AC-Foley

```

This tells us what an end-to-end AC-V2A system can currently accomplish.

Not a production candidate yet.

---

## Architecture C — hybrid production candidate

```text
vision proposals/refinement
        ↓
bank retrieval
        ↓
 ┌──────────────────────────────────┐
 │ strong exact match → direct      │
 │ close match       → DSP          │
 │ identity match    → SA3 A2A      │
 │ no match          → Mirelo       │
 └──────────────────────────────────┘
        ↓
deterministic event placement
        ↓
scene-aware mix

```

My current expectation is that **C will be the best product architecture**, but the experiment should be able to prove me wrong.

---

# 18. Next implementation work

## P0 — fix evaluation first

Before adding significant new modelling:

### 1. Listen to the existing A/B page

Record notes per event.

Do this before modifying the current output.

### 2. Create human-gold timing annotations

Minimum schema:

```text
event
class
frame_start
frame_contact
frame_end
acceptable_window
confidence

```

### 3. Build bank-aligned test clips

Use the camera/vehicle/designed assets already imported.

Also add sufficient target assets for existing walk videos.

### 4. Make evaluation blinded

The review page should not say:

```text
Mirelo
Our pipeline
ControlFoley

```

until after the rating.

Randomize A/B/C ordering.

---

# 19. P1 — improve retrieval without generation

Do this before training anything.

Expand the usable bank to at least several plausible alternatives per tested category.

For the POC, a useful minimum would be on the order of **10–20 variants for important one-shot classes**, not because that is a production requirement, but because retrieval cannot be meaningfully evaluated with one candidate.

Implement:

```text
semantic filter
    ↓
CLAP/text retrieval
    ↓
acoustic feature reranking
    ↓
event-specific scoring

```

Log the top five candidates, not only the selected one.

Then have humans indicate:

```text
best
acceptable
wrong

```

This lets us determine whether retrieval fails because:

- the bank has no appropriate sound;
- embedding search cannot find it;
- scoring chooses the wrong candidate;
- adaptation damages a good candidate.

Those are very different failures.

---

# 20. P2 — vision timing bake-off

Against human annotations compare:

### Current

`auto-sfx`

### Existing specialist

isolated FoleyCrafter `VideoOnsetNet`, if licensing/provenance allows evaluation

### Proposed two-stage

VLM event proposal + native-frame physical refinement

### Optional research baselines

other onset/contact networks if genuinely relevant

Do **not** compare their eventual mixed audio yet.

First compare their event detections in isolation.

That will tell us whether visual timing genuinely needs more work.

---

# 21. P3 — test sound adaptation separately from timing

For each human-annotated event, give the system the **correct event timestamp manually**.

Then compare only sound selection/adaptation:

```text
A: best direct bank sample
B: bank + DSP
C: bank reference + Stable Audio 3 A2A
D: ControlFoley/AC-Foley reference-conditioned output, where permitted
E: Mirelo pure V2A

```

This is critical.

Right now timing errors, retrieval errors and sound-generation errors are entangled.

We need to know:

> If timing were solved perfectly, can our bank/adaptation stack produce good enough Foley?

If the answer is no, there is no reason to spend weeks perfecting vision timing.

---

# 22. P4 — recombine only proven components

After the isolated timing and sound-quality benchmarks:

```text
best event detector
+
best retrieval/adaptation strategy
+
best fallback

```

Then run full end-to-end blind listening.

No new model should enter the production candidate unless it improves one measured bottleneck.

---

# 23. Mixing changes

The report already identified that global `-16 LUFS` normalization can hit true-peak limits and leave quiet material around `-24.8 LUFS`.

For the next version:

### Keep individual event stems

Do not immediately flatten the Foley track.

Persist:

```text
stems/
  evt_001.wav
  evt_002.wav
  evt_003.wav
  ambience.wav

```

This will dramatically improve debugging.

### Match gain locally

Use event/local-scene loudness rather than forcing the final composition to one global loudness target too early.

### Preserve pristine sources

Evaluation renders should use lossless 48 kHz WAV internally.

Only encode when making the browser review asset.

### Add simple perspective metadata

Eventually account for:

- distance from camera;
- screen location;
- environment;
- indoor/outdoor;
- source visibility.

A correct shoe sound at the wrong perceived distance sounds wrong even when perfectly synchronized.

---

# 24. Physical adaptation: implement conservatively

The report notes these heuristics are not yet implemented.

Add them, but keep the range small.

Examples:

```text
higher visual impact velocity
→ stronger transient / moderate gain

larger/heavier object
→ slight downward pitch / darker spectrum

small/light object
→ slight upward pitch / shorter body

farther source
→ lower level / less HF / more room contribution

```

Do not make cartoonishly deterministic mappings like:

```text
2× visual velocity = 2× gain

```

These should be subtle priors.

Always retain the unmodified bank source as an A/B control.

---

# 25. Provider and license registry

Commercial safety needs to become data rather than tribal knowledge.

Add a registry such as:

```yaml
mirelo_16:
  provider: fal
  commercial_status: provider_claimed
  territory: global
  reference_audio: false
  price_model: per_second
  production_allowed: pending_contract_review

controlfoley:
  provider: local
  weights_license: CC-BY-NC-4.0
  production_allowed: false

ac_foley:
  provider: local
  weights_license: CC-BY-NC-4.0
  production_allowed: false

stable_audio_3_sfx_a2a:
  provider: fal
  reference_audio: true
  production_allowed: pending_contract_review

```

Separate:

```text
QUALITY STATUS

```

from:

```text
COMMERCIAL STATUS

```

A model can win every benchmark and still be disqualified from production.

---

# 26. Research matrix


| Candidate                       | Video timing          | Reference audio             | Useful now?                      | Production status            |
| ------------------------------- | ---------------------: | ---------------------------: | -------------------------------- | ---------------------------- |
| **Our vision + bank**           | ✅ deterministic       | ✅ exact source              | **Core candidate**               | controllable                 |
| **Mirelo 1.6**                  | approximate/generated | ❌                           | strong baseline + fallback       | promising hosted option      |
| **Mirelo v1**                   | approximate/generated | ❌                           | cheap baseline                   | hosted                       |
| **Stable Audio 3 SFX A2A**      | ❌                     | ✅                           | **adaptation candidate**         | hosted/commercial-oriented   |
| **ControlFoley**                | ✅ model-controlled    | ✅                           | **important research benchmark** | ❌ public weights NC          |
| **AC-Foley**                    | ✅                     | ✅                           | important research benchmark     | ❌ public weights NC          |
| **MultiFoley**                  | ✅                     | ✅                           | architectural reference          | no usable public release     |
| **FoleyGenEx**                  | ✅                     | ✅                           | watch/research                   | not deployable today         |
| **FoleyCrafter onset detector** | ✅ visual onset        | N/A                         | **timing component to test**     | verify checkpoint separately |
| **ThinkSound**                  | generated             | no useful bank conditioning | quality/cost baseline            | upstream commercial conflict |
| **Synchformer**                 | coarse A/V alignment  | N/A                         | QC/reranking                     | not event clock              |


---

# 27. Provisional gates

These numbers are **engineering targets, not sacred product requirements**. Revise them after listening.

## Timing

For visible impulsive events:

```text
≥ 90% recall within ±50 ms
median error ≤ 33 ms
p95 error ≤ ~80 ms
low false-positive rate

```

If listeners cannot distinguish ±50 ms from ±80 ms in realistic material, loosen the metric rather than optimizing a meaningless number.

---

## Retrieval

For bank-covered categories:

```text
top-3 acceptable candidate ≥ 80%
top-1 acceptable ≥ ~65–70%
catastrophically wrong class < 3%

```

---

## Overall product

On bank-covered clips, the hybrid system should:

```text
beat or tie pure Mirelo in ≥ ~60% of blinded comparisons

```

while demonstrating clearly better **palette consistency**.

If it does not, we need to question whether the bank-conditioned product advantage is real.

---

## Economics

Track:

```text
$/generated video second
% events handled at $0
% events requiring A2A
% requiring full V2A fallback
latency p50/p95

```

A desirable architecture has the majority of ordinary events handled through:

```text
retrieval + DSP = effectively zero marginal inference

```

with expensive generation acting as a minority repair path.

---

# 28. What not to do next

Do **not**:

- fine-tune a palette model yet;
- spend days tuning SuperFlux against Mirelo;
- scale the library to thousands of sounds before proving retrieval works on 50–100;
- benchmark another dozen pure V2A models without a specific hypothesis;
- claim mechanical sample alignment equals perceptual sync;
- claim `auto-sfx` is ground truth;
- adopt a non-commercial research checkpoint into the shipping architecture;
- build editing-software integration;
- optimize infrastructure;
- push product UI.

The report's current engineering foundation is sufficient to answer the much more important unknowns first. The old Intel environment is also already hitting model/runtime constraints, so heavy ML experimentation should remain containerized and move to suitable GPU inference where necessary rather than contaminating the architecture with x86-specific compromises.

---

# 29. Concrete next-run brief

The next orchestrator should aim to answer **five questions**, in this order.

### Q1 — Are our current outputs actually perceptually bad or good?

Listen blindly to the existing renders.

Deliver scored results and concrete failure examples.

---

### Q2 — Can vision provide reliable event timing against human annotations?

Create human-gold labels and benchmark:

```text
auto-sfx
vs
isolated specialist onset detector
vs
VLM-proposal + native-frame refinement

```

Do not involve audio generation in this test.

---

### Q3 — Can an owned SFX bank reliably provide the right sonic identity?

Create an overlapping bank/video dataset.

Evaluate retrieval top-1/top-3 with humans.

Upgrade retrieval only if failures justify it.

---

### Q4 — Does reference-audio adaptation improve a good bank sound?

With timestamps held perfect, compare:

```text
direct
DSP
SA3 A2A
ControlFoley / AC-Foley research benchmark
Mirelo

```

Measure both physical believability and palette retention.

---

### Q5 — Does the hybrid whole beat the best omni-model?

Only after Q1–Q4:

```text
best vision clock
+
best bank/adaptation path
+
Mirelo fallback

```

versus:

```text
Mirelo alone

```

Blind listening decides.

---

# 30. Desired end state

We are not trying to prove that a complicated multi-model system is clever.

We are trying to converge on the **minimum stack that reliably sounds better**.

The ideal result may ultimately be surprisingly simple:

```text
video
  ↓
small event/timing model
  ↓
structured event list
  ↓
owned SFX retrieval
  ↓
mostly direct samples + modest DSP
  ↓
occasional reference-audio generation
  ↓
Mirelo only when the bank cannot solve it

```

That would be a very strong product architecture because:

- synchronization is controllable;
- customers can bring their own sonic identity;
- inference cost stays low;
- outputs are editable;
- the system is explainable;
- provider lock-in is limited;
- the expensive model is not responsible for every sound;
- commercial licensing becomes much easier to reason about.

But we should remain prepared for the experiment to show that reference-conditioned V2A is perceptually so superior that it deserves a larger role.

The next phase is therefore **not “build the stack.”**

It is:

> **separate timing, retrieval, adaptation and generation; benchmark each independently; then recombine only the components that actually win.**

That is the shortest path toward something we can trust.