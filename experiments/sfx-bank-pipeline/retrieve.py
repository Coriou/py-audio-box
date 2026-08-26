#!/usr/bin/env python3
"""Stage 4 — retrieval + placement planning with the tiered strategy.

Event sources per clip (in priority order):

1. ``vision``    — auto-sfx foot-contact events (frame-exact by construction;
                   used wherever ground truth exists, since stage 2 proved no
                   hosted V2A meets a 50 ms frame-lock bar).
2. ``v2a``       — onsets detected in the best-scoring V2A output (clips with
                   no contact ground truth).
3. ``bed``       — continuous sources (surf) become one full-length bed event.

Each event is embedded (CLAP) as a snippet of the winning oracle track,
zero-shot labelled, scored against every bank entry
(``0.6·audio_cos + 0.4·text_cos``), and assigned a strategy tier:

- ``direct``         candidate fits timing/intensity almost as-is
- ``dsp``            fits after gain ±12 dB and/or tail trim / ≤1.35× tempo
- ``sa3``            right identity, needs variation (Stable Audio 3 a2a)
- ``mirelo-fallback`` nothing suitable → crop the oracle segment ($0)

Outputs ``plan/<clip>.placement.json``. No fal calls happen here; the render
stage executes any ``sa3`` intents through the ledgered caller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from clap_embedder import CLAP_SR, cosine, embed_audio, label_embeddings
from common import (
    CANONICAL_PROMPTS,
    atomic_json,
    utc_now,
)
from index_bank import DEFAULT_LABELS
from oracle_events import ANALYSIS_SR, measure_event, superflux_envelope
from events_v2 import infer_temporal_type, upgrade_event_v1

SNIPPET_PRE_S = 0.02
SNIPPET_POST_S = 0.90
AUDIO_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
SCORE_FLOOR = 0.30            # below: nothing suitable → mirelo-fallback
SA3_MIN_SCORE = 0.45          # identity good enough to spend a variation on
DIRECT_GAIN_DB = 3.0
DSP_GAIN_DB = 12.0
DSP_TEMPO_MAX = 1.35          # compress body/tail by up to this factor
EVENT_GUARD_S = 0.02          # silence kept before the next event
TAIL_SLACK_S = 0.70           # last event may decay past clip end
TOP_K = 5
MAX_SA3_INTENTS = 4           # hard cap ≈ $0.096 within the $5 budget

#: Multiplicative rank penalty per prior use of the same bank entry within one
#: repetitive sequence (event model v2 diversity pressure; CTO §4C).
SEQUENCE_REUSE_PENALTY = 0.85

#: Fallback preference for clips without contact ground truth.
V2A_PREFERENCE: tuple[str, ...] = ("controlfoley", "kling-v2a", "mirelo16", "sonilo11")

AMBIENCE_KEYWORDS = ("waves", "surf", "ocean")


def is_ambience_clip(clip: str) -> bool:
    prompt = CANONICAL_PROMPTS[clip].lower()
    return any(word in prompt for word in AMBIENCE_KEYWORDS)


def load_best_oracle(run_dir: Path, clip: str, validation: dict[str, Any]) -> tuple[str, Path]:
    """Winning V2A model + its analysis WAV for one clip."""
    entry = validation["clips"].get(clip, {})
    model = entry.get("best_model")
    if model is None:
        for candidate in V2A_PREFERENCE:
            if (run_dir / "oracle" / f"{clip}.{candidate}.wav").exists():
                model = candidate
                break
    wav_path = run_dir / "oracle" / f"{clip}.{model}.wav"
    return model, wav_path


def load_audio_48k(path: Path, start_s: float = 0.0,
                   end_s: float | None = None) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    assert sample_rate == ANALYSIS_SR, f"{path}: expected {ANALYSIS_SR} Hz"
    audio = audio.astype(np.float32)
    start = int(start_s * ANALYSIS_SR)
    stop = int(end_s * ANALYSIS_SR) if end_s is not None else audio.size
    segment = audio[max(0, start):min(audio.size, stop)]
    # 22050 → 48000 via polyphase (up 320 / down 147).
    return resample_poly(segment, 320, 147).astype(np.float32)


def build_events(run_dir: Path, clip: str, clip_duration: float,
                 validation: dict[str, Any],
                 contacts: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Assemble the placed-event list and its source tag for one clip."""
    model, wav_path = load_best_oracle(run_dir, clip, validation)
    sample_rate, oracle_audio = wavfile.read(wav_path)
    oracle_audio = oracle_audio.astype(np.float64)
    envelope, logmel = superflux_envelope(oracle_audio)

    if contacts:
        source = "vision"
        events = []
        for entry in contacts:
            frame = int(round(entry["time_s"] * ANALYSIS_SR / 128))
            measured = measure_event(oracle_audio, frame, logmel)
            measured["id"] = entry["id"]
            measured["contact_s"] = entry["time_s"]
            # Placement time is the CONTACT itself (vision is the clock);
            # measured onset drift is reported, not applied.
            measured["time_s"] = entry["time_s"]
            events.append(measured)
        return source, events

    if is_ambience_clip(clip):
        rms = float(np.sqrt(np.mean(np.square(oracle_audio))))
        return "bed", [{
            "id": "ev_bed",
            "time_s": 0.0,
            "duration_s": round(clip_duration, 3),
            "intensity_db": round(max(-120.0, 20.0 * np.log10(max(rms, 1e-9))), 2),
            "salience": 0.0,
            "centroid_hz": 0.0,
            "kind": "ambience-bed",
        }]

    payload_path = run_dir / "oracle" / f"{clip}.{model}.events.json"
    payload = json.loads(payload_path.read_text())
    return f"v2a:{model}", [{**event, "id": event["id"]} for event in payload["events"]]


def score_candidates(snippet: np.ndarray, snippet_label: str,
                     bank_entries: list[dict[str, Any]],
                     embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    snippet_embedding = embed_audio(snippet)
    text_query = label_embeddings((f"the sound of {snippet_label}",))[0]
    ranked = []
    for entry in bank_entries:
        file_embedding = embeddings.get(entry["id"])
        if file_embedding is None:
            continue
        audio_cos = cosine(snippet_embedding, file_embedding)
        text_cos = cosine(text_query, file_embedding)
        ranked.append({
            "entry_id": entry["id"],
            "rel_path": entry["rel_path"],
            "audio_cos": round(audio_cos, 4),
            "text_cos": round(text_cos, 4),
            "score": round(AUDIO_WEIGHT * audio_cos + TEXT_WEIGHT * text_cos, 4),
        })
    ranked.sort(key=lambda item: (-item["score"], item["rel_path"]))
    return ranked[:TOP_K]


def classify_snippet(snippet: np.ndarray, labels: tuple[str, ...]) -> tuple[str, float]:
    matrix = label_embeddings(labels)
    embedding = embed_audio(snippet)
    scores = matrix @ embedding
    top = int(np.argmax(scores))
    return labels[top], float(scores[top])


def decide_tier(event: dict[str, Any], candidates: list[dict[str, Any]],
                bank_by_id: dict[str, dict[str, Any]], clip_duration: float,
                event_times: list[float], allow_sa3: bool) -> dict[str, Any]:
    index = event_times.index(event["time_s"]) if event["time_s"] in event_times else None
    if event.get("kind") == "ambience-bed":
        gap = clip_duration
    elif index is not None and index + 1 < len(event_times):
        gap = event_times[index + 1] - event["time_s"] - EVENT_GUARD_S
    else:
        gap = clip_duration - event["time_s"] + TAIL_SLACK_S
    gap = max(gap, 0.15)

    decision: dict[str, Any] = {"available_gap_s": round(gap, 3)}
    if event.get("kind") == "ambience-bed":
        decision.update(tier="direct", ops=["gain"], reason="ambience bed loop-fill")
        if candidates:
            best = candidates[0]
            meta = bank_by_id[best["entry_id"]]
            decision.update({
                "candidate": best,
                "first_onset_s": round(meta["onsets_s"][0], 4) if meta["onsets_s"] else 0.0,
                "tail_s": round(max(0.05, meta["duration_s"]), 3),
                "gain_db": round(max(-DSP_GAIN_DB, min(DSP_GAIN_DB,
                                  event.get("intensity_db", -20.0) - meta["rms_db"])), 2),
                "raw_gain_db": round(event.get("intensity_db", -20.0) - meta["rms_db"], 2),
            })
        else:
            decision.update(tier="mirelo-fallback", reason="no bank candidates")
        return decision
    if not candidates:
        decision.update(tier="mirelo-fallback", reason="no bank candidates")
        return decision

    best = candidates[0]
    meta = bank_by_id[best["entry_id"]]
    first_onset = meta["onsets_s"][0] if meta["onsets_s"] else 0.0
    tail_s = max(0.05, meta["duration_s"] - first_onset)
    gain_db = event["intensity_db"] - meta["rms_db"]

    decision.update({
        "candidate": best,
        "first_onset_s": round(first_onset, 4),
        "tail_s": round(tail_s, 3),
        "gain_db": round(max(-DSP_GAIN_DB, min(DSP_GAIN_DB, gain_db)), 2),
        "raw_gain_db": round(gain_db, 2),
    })

    if best["score"] < SCORE_FLOOR:
        decision.update(tier="mirelo-fallback",
                        reason=f"best score {best['score']} < floor {SCORE_FLOOR}")
        return decision
    if abs(gain_db) <= DIRECT_GAIN_DB and tail_s <= gap + 0.05:
        decision.update(tier="direct", ops=["gain"], reason="fits timing and intensity")
        return decision
    if abs(gain_db) <= DSP_GAIN_DB:
        if tail_s <= gap * DSP_TEMPO_MAX:
            ops = ["gain"]
            if tail_s > gap:
                ops.append("tempo-compress")
            else:
                ops.append("tail-trim")
            decision.update(tier="dsp", ops=ops,
                            reason="fits after local DSP")
            return decision
        if allow_sa3 and best["score"] >= SA3_MIN_SCORE:
            decision.update(tier="sa3", ops=["generate-variation"],
                            reason="identity right, timing impossible locally")
            return decision
        decision.update(tier="mirelo-fallback",
                        reason="candidate too long and SA3 unavailable/below bar")
        return decision
    if allow_sa3 and best["score"] >= SA3_MIN_SCORE:
        decision.update(tier="sa3", ops=["generate-variation"],
                        reason="intensity mismatch beyond DSP")
        return decision
    decision.update(tier="dsp", ops=["gain", "tail-trim"],
                    reason="gain clamped inside DSP range; tail trimmed")
    return decision


def plan_clip(run_dir: Path, clip: str, clip_duration: float,
              validation: dict[str, Any], contacts_payload: list[dict[str, Any]],
              bank_entries: list[dict[str, Any]], embeddings: dict[str, np.ndarray],
              labels: tuple[str, ...], allow_sa3: bool) -> dict[str, Any]:
    model, wav_path = load_best_oracle(run_dir, clip, validation)
    source, events = build_events(run_dir, clip, clip_duration, validation,
                                  contacts_payload)
    event_times = [event["time_s"] for event in events]
    bank_by_id = {entry["id"]: entry for entry in bank_entries}
    chosen_by_label: dict[str, list[str]] = {}
    plans = []
    for event in events:
        snippet = load_audio_48k(
            wav_path,
            max(0.0, event["time_s"] - SNIPPET_PRE_S),
            event["time_s"] + SNIPPET_POST_S,
        )
        label, label_score = classify_snippet(snippet, labels)
        candidates = score_candidates(snippet, label, bank_entries, embeddings)

        # Repetitive-sequence diversity (event model v2): previously-used
        # candidates inside one sequence lose a little rank each reuse.
        sequence_id = f"{label}" if infer_temporal_type(label) == "repetitive" else None
        if sequence_id and chosen_by_label.get(label):
            usage: dict[str, int] = {}
            for entry_id in chosen_by_label[label]:
                usage[entry_id] = usage.get(entry_id, 0) + 1
            penalized = [
                {**c, "score": round(c["score"] * SEQUENCE_REUSE_PENALTY ** usage.get(c["entry_id"], 0), 4)}
                for c in candidates
            ]
            penalized.sort(key=lambda item: (-item["score"], item["rel_path"]))
        decision = decide_tier(event, candidates, bank_by_id, clip_duration,
                               event_times, allow_sa3)
        entry_id = (decision.get("candidate") or {}).get("entry_id")
        if sequence_id and decision.get("tier") not in (None, "mirelo-fallback") and entry_id:
            chosen_by_label.setdefault(label, []).append(entry_id)
        plans.append(upgrade_event_v1({
            **{key: event[key] for key in
               ("id", "time_s", "duration_s", "intensity_db") if key in event},
            **({"kind": event["kind"]} if "kind" in event else {}),
            "label": label,
            "label_score": round(label_score, 4),
            "candidates": candidates,
            **decision,
        }, label=label, clip=clip, sequence_id=sequence_id))

    return {
        "schema_version": 2,
        "generated_at": utc_now(),
        "clip": clip,
        "oracle_model": model,
        "event_source": source,
        "clip_duration_s": clip_duration,
        "events": plans,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-sa3", action="store_true",
                        help="disable Stable Audio 3 variation tier")
    parser.add_argument("--labels-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = DEFAULT_LABELS
    if args.labels_file:
        labels = tuple(line.strip() for line in args.labels_file.read_text().splitlines()
                       if line.strip())

    bank_index = json.loads((args.run_dir / "bank" / "index.json").read_text())
    bank_entries = bank_index["entries"]
    archive = np.load(args.run_dir / "bank" / "embeddings.npz")
    embeddings = {key: archive[key] for key in archive.files}
    validation = json.loads((args.run_dir / "validation" / "timing.json").read_text())

    from common import BENCH_RUN_DIR, load_contacts, REPO_ROOT
    inputs_index = json.loads(
        (BENCH_RUN_DIR / "inputs" / "inputs-index.json").read_text())
    durations = {
        entry["key"]: float(entry["duration_s"])
        for entry in inputs_index["clips"]
    }
    auto_sfx_runs = REPO_ROOT / "work" / "auto-sfx" / "runs"

    plan_dir = args.run_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    allow_sa3 = not args.no_sa3
    sa3_budget = MAX_SA3_INTENTS

    for clip in ("walk1", "walk2", "man", "ocean-long", "ocean-short"):
        contacts = load_contacts(clip, auto_sfx_runs)
        plan = plan_clip(args.run_dir, clip, durations[clip], validation,
                         contacts, bank_entries, embeddings, labels, allow_sa3)
        spent = sum(1 for event in plan["events"] if event["tier"] == "sa3")
        if spent > sa3_budget:
            # Re-plan this clip without SA3 once the intent budget is exhausted.
            plan = plan_clip(args.run_dir, clip, durations[clip], validation,
                             contacts, bank_entries, embeddings, labels,
                             allow_sa3=False)
            spent = sum(1 for event in plan["events"] if event["tier"] == "sa3")
        sa3_budget -= spent
        atomic_json(plan_dir / f"{clip}.placement.json", plan)
        tiers = {}
        for event in plan["events"]:
            tiers[event["tier"]] = tiers.get(event["tier"], 0) + 1
        print(f"{clip}: source={plan['event_source']} tiers={tiers}")
    print(f"SA3 intents used: {MAX_SA3_INTENTS - sa3_budget} "
          f"(≤ {MAX_SA3_INTENTS}, est ${MAX_SA3_INTENTS * 0.024:.3f} max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
