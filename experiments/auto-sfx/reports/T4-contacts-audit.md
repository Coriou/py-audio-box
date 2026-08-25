# T4 — State-Machine Semantic Audit (`contacts.py`)

Phase 2 Checkpoint A input · audit only, no code changed · line refs to
`experiments/auto-sfx/src/auto_sfx/contacts.py` @ 2026-08-25.
Question per directive §4-T4: for every transition — what math fires it, what
biomechanical event it *claims*, what evidence ties the two together, and how
it fails. Vocabulary: §5.3 error classes.

---

## 1. Per-transition audit table

Thresholds: `thr = speed_drop_frac(0.35) × max(swing_peak, 0.4 bs/s)`
(L589-590, L107-109); `confirm = 2` frames; `min_stance_s = 0.45 × 0.25 s =
0.1125 s` (L114, L586); ascent veto `0.05 × scale` (L118); hybrid envelope
tolerance `0.15 × scale` (L122). Speed = pairwise displacement of the foot
*reference* (mean of usable heel/ankle/toe, L276-294) / dt / body scale,
stamped at the later frame (L574-582). **No temporal smoothing is applied to
landmarks inside this module** — the machine consumes raw per-frame values.

| Transition | Mathematical condition (lines) | Claimed biomechanical event | Correspondence evidence | Failure modes (§5.3) |
|---|---|---|---|---|
| UNKNOWN→SWING | first finite sample with `sp > thr` (L698-701) | foot airborne, mid-swing | plausible: swing speeds ≫ stance speeds in body-scale units; no event emitted | none direct (state only); a slow shuffle below `0.35×0.4=0.14 bs/s` never leaves UNKNOWN → all contacts missed (MISSED_CONTACT) |
| UNKNOWN→STANCE (seeded) | first finite sample with `sp ≤ thr` (L702-706) | "already planted before segment" | none needed — deliberately emits nothing; sets `stance_confirmed=False` so EXIT relaxation is disabled | safe by construction; cost is a MISSED_CONTACT for the one contact whose stance straddles a shot boundary/pose gap |
| SWING→CANDIDATE | first sample with `sp < thr` (L707-713) | **initial contact: foot JUST transitioned swing→support** | **none found.** No evidence in code or artifacts that first-sub-threshold frame coincides with touchdown | **EARLY_PHASE (primary)**: swing-foot speed decelerates smoothly *before* touchdown; the first frame below 35 % of peak precedes IC by the deceleration tail (frames at 30 fps). Also DUPLICATE_CONTACT / EARLY_PHASE via mid-swing apex dip: the apex of a sinusoidal swing is momentarily slow; the ascent veto checks ONE pre-run frame (`ry[p0-1]-ry[p0] > 0.05·scale`, L630-633) — at apex approach that frame can already be near-flat, so the veto passes; the duration guard (`dur ≥ 0.1125 s`) sits exactly inside the apex-dip duration band (~23 % of swing ≈ 0.09–0.12 s at normal cadence, L111-114 comment) → some apex dips survive as candidates |
| CANDIDATE→(close_run)→STANCE | run ends when `sp ≥ thr` (L714-717 → L598-688); confirm iff `n_frames ≥ 2` ∧ (`dur ≥ 0.1125 s` ∨ relaxed) ∧ ¬ascent ∧ (hybrid: min envelope height ≤ 0.15·scale) | sustained support confirmed | partial: persistence + envelope DO separate "planted" from "glance"; but see stamp policy below | **Event timestamp = FIRST low frame `seg[p0]`** (L616, L1029-1031), with `latest_s = +1 frame` (L1032). The claimed uncertainty (±1 frame) is far narrower than the true uncertainty (the whole pre-touchdown deceleration). This stamp policy, not the state machine per se, is what turns deceleration lead into signed EARLY_PHASE error. LATE_STANCE events arise when the reference speed stays high past touchdown (toe still whipping inflates the mean-landmark speed, L276-294) → trigger delayed past IC |
| CANDIDATE→UNKNOWN (reject) | any guard fails (L613-642) | no contact | conservative by design ("missing beats inventing", L23-24) | MISSED_CONTACT only — acceptable per priority order |
| STANCE→EXIT | `sp ≥ thr` (L718-720) | toe-off / foot leaving ground | plausible as state change; **no toe_off event is ever emitted** (kind hardcoded `"initial_contact"`, L1038) | none direct; TOE_OFF_CONFUSED_AS_CONTACT enters via the next row instead |
| EXIT→CANDIDATE (relaxed) | `sp < thr` again after EXIT (L721-728); opens run with `relaxed_run = stance_confirmed[0]` | "re-plant right after an airborne moment" | **weak.** Relaxation skips BOTH the duration guard and the ascent veto (L625: `if not was_relaxed`); the only guard left is `n_frames ≥ confirm(2)`. `stance_confirmed` attests the *previous* stance was real — it says nothing about evidence that the foot re-planted | **LATE_STANCE / TOE_OFF_CONFUSED_AS_CONTACT / DUPLICATE_CONTACT (primary)**: any brief speed dip right after toe-off — toe drag, pose jitter on the reference mean, the toe-whip lull between heel-lift and toe-lift — confirms on 2 frames of persistence and is stamped as a new initial contact |
| EXIT→SWING | `sp > 0.7 × max(peak, 0.4)` (L729-731) | foot accelerating into swing | plausible | none direct |

Post-processing: `_suppress_duplicates` collapses only **same-side** pairs
within `min_step` (L920-931) — two independent per-foot machines may confirm
the same instant, and nothing prevents it → **SIMULTANEOUS_LR survives by
construction**. `_cadence_and_alternation` (L934-956) only *diagnoses*
cadence violations and same-side consecutiveness (`LEFT_RIGHT_DISCONTINUITY`);
it never removes or re-ranks → **consecutive same-side survives by
construction** (both observed in A1). Side identity is inherited verbatim from
the backend's per-side tracks — no anatomical check exists in this module →
WRONG_SIDE / LANDMARK_SWAP are backend-inherited (A3).

## 2. Focused scrutiny

**Low foot velocity as contact cue — the planted-foot problem.** A planted
foot has near-zero velocity for the *entire* stance (~0.6–0.8 × step interval),
while the transition swing→support is instantaneous. "Low velocity" is
therefore a *state* feature, not a *transition* feature: every stance frame is
equally low, and the machine's pick — the FIRST low frame after a high-speed
run — equals the physical IC only if deceleration were instantaneous. Real
swing kinematics (and pose smoothing) make deceleration a multi-frame tail,
so `first-low-frame ≤ IC` systematically. **The machine detects "foot appears
stable" and labels it "foot just became stable".** The temporal ambiguity
window = duration of the sub-threshold deceleration tail ≈ (thr / deceleration
slope); with thr at 35 % of a ~1–3 bs/s peak and deceleration over ~30–40 % of
swing, the expected lead is O(2–6 frames at 30 fps). This single mechanism
predicts A1's early-phase population and its spread.

**Local extrema / support band.** The hybrid envelope (`_env_heights`,
L524-544) is a ±0.5 s centred running MAX of foot y (floor-most local level) —
zero-phase, so no group delay, and honestly "not a ground plane". Its failure
mode: on stairs/slopes or when the *other* foot is lower in the window, the
band drops and a genuinely planted foot reads `height > 0.15·scale` →
candidate rejected (MISSED_CONTACT); conversely near the band edge it adds no
timing information (it gates, it does not localise). The strike label
(heel/forefoot, L669-676) uses a single frame's heel−toe y difference with
eps = 0.015·scale — a classification, not a timing feature.

**Velocity sign changes.** Not used. The machine never looks at
`d(speed)/dt` or velocity-direction reversal; IC in sagittal view coincides
with the *vertical* velocity sign flip of the heel, which the horizontal+vertical
mixed `hypot` reference speed destroys (L581). Cheap, well-identified
improvement candidate for B-track (T5).

**Stance persistence.** `confirm=2` frames + `0.1125 s` duration. The apex-dip
analysis above shows the duration guard is mis-centred: it was chosen against
the *apex dip* (~11 % of step interval, L111-114) with a 0.45× margin, but
relaxed runs skip it entirely, and real deceleration tails shorten the
*effective* pre-IC low run so the guard also trims early candidates'
survival probabilistically rather than physically.

## 3. Smoothing / group-delay inventory (pose→decision path)

| Stage | Filter | Window | Zero-phase? | Group delay |
|---|---|---|---|---|
| `contacts.py` velocity/hybrid | none (raw pairwise displacement) | 1 frame diff | — | 0 frames |
| `contacts.py` envelope | centred running max | ±0.5 s | yes | 0 |
| `contacts.py` pelvis | centred MA, length = one sway period (ACF-estimated) or `min_step` fallback (L837-852) | ~0.25–2.0 s (k = 8–60 frames @30 fps) | yes (odd taps) | 0 — but see §4: *amplitude* distortion, not delay |
| `contacts.py` camera diag | centred MA high-pass | 0.5 s | yes | 0 (diagnostics only) |
| MediaPipe **video** mode | internal landmark tracker smoothing (backend-side, not this module) | unknown here | no (causal, recursive) | **unmeasured — T8's first job**; typical 2–4 frames |
| MediaPipe image mode / RTMPose | none known | — | — | 0 (to be confirmed in T8) |

Consequence: any *temporal* bias measured against GT must first be split into
backend-filter delay vs algorithm stamp policy; with `image`/`rtmpose` runs as
near-zero-filter controls, the same GT decomposes the two (T8 ablation matrix).

## 4. Dominant-failure hypothesis (ranked) — A1/A2

1. **Stamp policy on a state feature (explains most of A1).** The event time is
   the first frame below `0.35 × swing_peak` (L616, L1029-1031) — the start of
   a possibly-long low-speed region, not the swing→support transition.
   Swing deceleration precedes touchdown → systematic EARLY_PHASE with frame-
   scale spread; toe-whip inflating the reference mean after touchdown →
   LATE_STANCE population; both directions observed in A1. Code chain:
   L574-582 (mixed reference speed) → L707-713 (SWING→CAND on threshold
   crossing) → L616 (stamp at run start) → L1029-1031 (`earliest = time`,
   ±1 frame window).
2. **EXIT-relaxed confirmation fabricates post-toe-off contacts.** L721-728 +
   L625: after a confirmed stance, ANY 2-frame speed dip re-opens a candidate
   that skips duration and ascent guards. Toe drag / jitter / the heel-lift→
   toe-lift lull become `initialContact` events in terminal stance →
   LATE_STANCE, TOE_OFF_CONFUSED_AS_CONTACT, DUPLICATE_CONTACT. This is the
   cheapest mechanism to falsify: every relaxed run is recoverable from
   diagnostics by re-simulating the machine and logging `was_relaxed`.
3. **Pelvis (Zeni) method is phase-fragile by construction (A2).** Three
   stacked hazards: (a) the sway-null window length comes from the FIRST ACF
   local maximum in [0.225, 2.4] s (L757-783, L837-841) — hip-translation
   noise easily locks it onto a sub-band, putting the smoothing null off the
   sway band and letting step-rate sway peaks masquerade as stride-rate strike
   peaks; (b) even with a correct window, "max anterior ankle-relative-to-hip
   offset" in monocular 2D is not demonstrated to coincide with IC for these
   camera geometries (Zeni's validation is 3D lab kinematics); (c) peak
   prominence floor `0.005 × scale` (L144) is tiny relative to residual sway
   ripple. Any of the three yields events at a *different gait phase* than the
   velocity machine — exactly A2's 0–14 % agreement. T12 must label which
   physical phase pelvis actually fires on (GT), not retune it to agree.

Minor findings: `_PELVIS_PROM_FRAC` defined twice (L135 dead, L144 live);
`KIND_GAIN["toe_off"]` in `markers.py` is unreachable (no toe_off events are
ever produced); `latest_s = +1 dt` understates uncertainty for every event
(relevant to metrics-v2 phase classification, T14).

## 5. What this audit does NOT claim

- No GT has been consulted; magnitudes (2–6 frames) are mechanism-derived
  estimates, to be measured by T2/T3 on operator marks.
- Pose-backend internal filtering is unverified (T8 owns the ablation).
- The audit does not propose threshold patches — per directive, B-track
  changes wait for Checkpoint A.
