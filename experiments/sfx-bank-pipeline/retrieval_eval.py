#!/usr/bin/env python3
"""Retrieval evaluation harness + bank↔video overlap manifest (phase-2 Q3).

Implements docs/sfx-bank-foley-phase2-eval-brief.md §3.4 and CTO report §19
(top-5 logging, human best/acceptable/wrong tags, failure localization).

Subcommands
    manifest   Build/seed bank_overlap.manifest.json: for each test clip which
               indexed bank entry SHOULD win. Owner fills/expands.
    top5       Recompute forced top-5 candidates per plan event with the same
               scoring as retrieve.py (0.6·audio-cos + 0.4·text-cos of CLAP
               embeddings), regardless of the 0.30 tier floor, and emit a
               blinded-ish tagging page + questions JSON.
    tally      Aggregate downloaded tag JSONs into top-1/top-3 acceptable rates,
               chosen==human-best, wrong-class %, and failure-localization counts.

Scoring constants are intentionally duplicated from retrieve.py to keep this
tool import-light (torch loads lazily via clap_embedder only in `top5`).
KEEP IN SYNC: SNIPPET_PRE_S/SNIPPET_POST_S/AUDIO_WEIGHT/TEXT_WEIGHT/TOP_K.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from clap_embedder import embed_audio, cosine, label_embeddings  # noqa: E402

SNIPPET_PRE_S = 0.02     # sync: retrieve.py
SNIPPET_POST_S = 0.90    # sync: retrieve.py
AUDIO_WEIGHT = 0.6       # sync: retrieve.py
TEXT_WEIGHT = 0.4        # sync: retrieve.py
TOP_K = 5                # sync: retrieve.py

V2A_PREFERENCE = ("controlfoley", "kling-v2a", "mirelo16", "sonilo11")  # sync: retrieve.py

LOCALIZATION_TAGS = ("bank-lacks-sound", "search-miss", "scoring-wrong",
                     "adaptation-damage", "not-applicable")
CANDIDATE_TAGS = ("best", "acceptable", "wrong")


# ---------------------------------------------------------------- shared helpers

def load_bank(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    index = json.loads((run_dir / "bank" / "index.json").read_text())
    emb_path = run_dir / "bank" / "embeddings.npz"
    embeddings: dict[str, np.ndarray] = {}
    if emb_path.is_file():
        with np.load(emb_path) as z:
            for key in z.files:
                embeddings[key] = np.asarray(z[key], dtype=np.float32)
    return index["entries"], embeddings


def snippet_source_wav(run_dir: Path, clip: str) -> Path | None:
    """The oracle WAV whose content the original retrieval queried."""
    for model in V2A_PREFERENCE:
        p = run_dir / "oracle" / f"{clip}.{model}.wav"
        if p.is_file():
            return p
    globs = sorted((run_dir / "oracle").glob(f"{clip}.*.wav")) if (run_dir / "oracle").is_dir() else []
    return globs[0] if globs else None


def compute_snippet(run_dir: Path, clip: str, time_s: float) -> tuple[np.ndarray | None, str]:
    """Replicates retrieve.py's snippet extraction (analysis wav slice → 48 k)."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    wav = snippet_source_wav(run_dir, clip)
    if wav is None:
        return None, f"no oracle wav for {clip}"
    sr, audio = wavfile.read(wav)
    audio = audio.astype(np.float32)
    start = int(max(0.0, time_s - SNIPPET_PRE_S) * sr)
    stop = int((time_s + SNIPPET_POST_S) * sr)
    seg = audio[start:min(audio.size, stop)]
    audio_48k = resample_poly(seg, 320, 147).astype(np.float32)
    return audio_48k, ""


def rank_candidates(snippet_48k: np.ndarray, event_label: str,
                    entries: list[dict[str, Any]],
                    embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Same formula as retrieve.score_candidates but NOT truncated by any floor."""
    text_query = label_embeddings((f"the sound of {event_label}",))[0]
    snip_emb = embed_audio(snippet_48k)
    ranked = []
    for entry in entries:
        fe = embeddings.get(entry["id"])
        if fe is None:
            continue
        a = float(cosine(snip_emb, fe))
        t = float(cosine(text_query, fe))
        ranked.append({
            "entry_id": entry["id"],
            "rel_path": entry.get("rel_path", ""),
            "audio_cos": round(a, 4),
            "text_cos": round(t, 4),
            "score": round(AUDIO_WEIGHT * a + TEXT_WEIGHT * t, 4),
        })
    ranked.sort(key=lambda item: (-item["score"], item["rel_path"]))
    return ranked[:TOP_K]


# ---------------------------------------------------------------- top5 page

TOP5_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Retrieval top-5 tagging</title>
<style>
body{font-family:-apple-system,sans-serif;background:#111418;color:#d7dce2;margin:0 auto;max-width:920px;padding:22px;}
table{border-collapse:collapse;width:100%;font-size:.83rem;} td,th{padding:5px 8px;border-bottom:1px solid #232a31;text-align:left;}
.cand{background:#161c23;border-radius:6px;margin:6px 0;padding:6px 10px;}
.tag{margin-right:12px;} select,textarea{background:#171d24;color:#d7dce2;border:1px solid #2a3340;border-radius:4px;}
button{background:#2f6fed;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer;}
.ev{border:1px solid #232a31;border-radius:8px;padding:10px;margin:16px 0;}
.note{color:#8794a1;font-size:.76rem;}
</style></head><body>
<h1>Forced top-5 retrieval tagging</h1>
<p class="note">Tag each candidate best / acceptable / wrong (exactly one “best” ideally), pick where the
pipeline failed when applicable, then download the tags JSON next to this page.</p>
<div style="position:sticky;top:0;background:#111418;padding:8px 0;">
<button id="dl">Download tags JSON</button> <span id="done" class="note"></span></div>
<div id="root">__CONTENT__</div>
<script id="meta" type="application/json">__META__</script>
<script>
"use strict";
const META=JSON.parse(document.getElementById("meta").textContent);
const R=META.questions.map(()=>({}));
function collect(){
 const out={exported_utc:new Date().toISOString(),run_id:META.run_id,tool:"retrieval_eval",questions:[]};
 META.questions.forEach((q,i)=>out.questions.push({event_id:q.event_id,clip:q.clip,label:q.label,time_s:q.time_s,tags:R[i]}));
 return out;
}
document.getElementById("dl").onclick=()=>{
 const blob=new Blob([JSON.stringify(collect(),null,2)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download="retrieval-tags-"+META.run_id+".json";a.click();URL.revokeObjectURL(a.href);
};
document.getElementById("root").addEventListener("change",ev=>{
 const d=ev.target.dataset;if(!d.q)return;
 R[d.q][d.k]=ev.target.value;
 let n=0;R.forEach(r=>{if(Object.keys(r).length)n++;});
 document.getElementById("done").textContent=n+"/"+META.questions.length+" events touched";
});
</script></body></html>
"""


def cmd_top5(run_dir: Path, clip_filter: str | None, out_dirname: str = "retrieval-eval") -> int:
    plans_dir = run_dir / "plan"
    out_dir = run_dir / out_dirname
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, embeddings = load_bank(run_dir)
    copied = set()
    repo_root = _HERE.parents[1]

    def player_src(entry_id: str) -> str:
        e = next(e for e in entries if e["id"] == entry_id)
        src = repo_root / e["working_wav"]
        target_name = f"{entry_id}{src.suffix or '.wav'}"
        if entry_id not in copied:
            audio_dir.mkdir(parents=True, exist_ok=True)
            if not (audio_dir / target_name).exists():
                shutil.copyfile(src, audio_dir / target_name)
            copied.add(entry_id)
        return f"audio/{target_name}"

    html_rows = []
    meta_questions = []
    for plan_path in sorted(plans_dir.glob("*.placement.json")):
        plan = json.loads(plan_path.read_text())
        clip = plan["clip"]
        if clip_filter and clip != clip_filter:
            continue
        events = [e for e in plan.get("events", []) if e.get("time_s") is not None]
        # batch embedding is slow on CPU — one wav read per clip, model once globally
        for ev_i, ev in enumerate(events):
            q_idx = len(meta_questions)
            snippet, err = compute_snippet(run_dir, clip, ev["time_s"])
            rows_html = ""
            if err or snippet is None or not embeddings:
                rows_html = f"<p class='note'>cannot score: {err or 'no embeddings'}</p>"
            else:
                ranked = rank_candidates(snippet, ev["label"], entries, embeddings)
                if not ranked:
                    rows_html = "<p class='note'>no scorable bank entries</p>"
                for c_i, cand in enumerate(ranked):
                    src = player_src(cand["entry_id"])
                    e_meta = next(x for x in entries if x["id"] == cand["entry_id"])
                    dur = e_meta.get("duration_s")
                    rows_html += f"""
<div class="cand" data-q="{q_idx}">
  <strong>{c_i+1}. {cand['entry_id'][:46]}…</strong>
  <span class="note">score {cand['score']} (audio {cand['audio_cos']} · text {cand['text_cos']}) · {dur}s · {cand['rel_path']}</span>
  <div><audio controls preload="none" src="{src}" style="width:100%"></audio></div>
  <span class="tag">{"".join(f'<label class="tag"><input type="radio" data-q="{q_idx}" data-k="rank{c_i}" name="q{q_idx}-r{c_i}" value="{t}"> {t}</label>' for t in CANDIDATE_TAGS)}</span>
</div>"""
            html_rows += f"<div class='ev'><h3>{clip} · {ev['id']} — {ev['label']} @ {float(ev['time_s']):.2f}s"
            html_rows += f" <span class='note'>(tier {ev.get('tier','—')})</span></h3>"
            html_rows += """
  <div><label>failure mode:
  <select data-q="{q}" data-k="localization">
    <option value="">—</option>{opts}
  </select></label></div>
  <textarea data-q="{q}" data-k="notes" rows="1" placeholder="notes" style="width:100%"></textarea>
""".format(q=q_idx, opts="".join(f'<option value="{t}">{t}</option>' for t in LOCALIZATION_TAGS))
            html_rows += rows_html + "</div>"
            meta_questions.append({
                "event_id": ev["id"], "clip": clip, "label": ev["label"],
                "time_s": ev["time_s"], "plan_tier": ev.get("tier"),
                "snippet_model": snippet_source_wav(run_dir, clip).name if snippet_source_wav(run_dir, clip) else None,
            })

    meta_b64 = json.dumps({"run_id": run_dir.name, "questions": meta_questions})
    html = TOP5_TEMPLATE.replace("__CONTENT__", "".join(html_rows)).replace("__META__", meta_b64)
    (out_dir / "top5.html").write_text(html)
    (out_dir / "top5-questions.json").write_text(json.dumps(meta_questions, indent=2))
    print(f"wrote {out_dir}/top5.html ({len(meta_questions)} events) + {len(copied)} audio copies")
    print(f"serving tip: cd '{run_dir}' && python3 -m http.server 8770 → open /{out_dirname}/top5.html")
    return 0


# ---------------------------------------------------------------- manifest

def cmd_manifest(out_path: Path, seed_run: Path | None) -> int:
    rows: list[dict[str, Any]] = []
    if seed_run and seed_run.is_dir():
        plans = seed_run / "plan"
        for plan_path in sorted(plans.glob("*.placement.json")):
            clip = plan_path.name.replace(".placement.json", "")
            plan = json.loads(plan_path.read_text())
            first_label = (plan.get("events") or [{}])[0].get("label", "")
            if "ocean" in clip or "surf" in first_label.lower():
                expected = "surf_bed_01_6ace93b1"
            elif plan.get("event_source") == "contacts-vision":
                expected = "gravel_step_02_8a87f006 or grass_step_01_2c596168 (owner picks)"
            else:
                expected = ""
            rows.append({"clip": clip, "status": "seeded",
                         "expected_bank_entry": expected,
                         "events": [{"id": e["id"], "label": e.get("label")} for e in plan.get("events", [])[:4]]})
    for needed, hint in [
        ("camera-shutter-closeup", "shoot/close-up exercising COMCam analog shutter + focus beep"),
        ("wipers-daylight", "car windshield wipers start/run/stop"),
        ("gear-shift-interior", "manual gear shift with clanks (ICE cabin)"),
        ("whoosh-passby", "fast object pass-by for DSGNWhsh whoosh"),
        ("bass-dive-moment", "designed bass dive cue scene (owner judgment)"),
    ]:
        rows.append({"clip": needed, "status": "needs_clip", "expected_bank_entry": "", "hint": hint})
    doc = {"schema_version": 1, "kind": "bank-overlap-manifest",
           "note": "expected_bank_entry must be an indexed bank id (suffix-hashed). "
                   "Owner expands bank toward 10–20 variants per important class before statistics.",
           "rows": rows}
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"wrote {out_path} ({sum(1 for r in rows if r['status']=='needs_clip')} needs_clip placeholders)")
    return 0


# ---------------------------------------------------------------- tally

TALLY_FIXTURE_TAGS = {
    "questions": [
        {"event_id": "e1", "tags": {"rank0": "best"}},
        {"event_id": "e2", "tags": {"rank0": "wrong", "rank1": "acceptable", "localization": "scoring-wrong"}},
        {"event_id": "e3", "tags": {"rank2": "best", "rank3": "acceptable", "localization": "search-miss"}},
    ]
}


def tally_tags(tag_files: list[Path]) -> dict[str, Any]:
    n_q = 0
    top1_ok = top3_ok = wrong_chosen = 0
    loc: dict[str, int] = {}
    for f in tag_files:
        data = json.loads(f.read_text())
        for q in data.get("questions", []):
            n_q += 1
            tags = q.get("tags", {})
            vals = [v for k, v in tags.items() if k.startswith("rank")]
            has_acc_top1 = vals and vals[0] in ("best", "acceptable")
            top1_ok += bool(has_acc_top1)
            top3_ok += bool(any(v in ("best", "acceptable") for v in vals[:3]))
            wrong_chosen += bool(vals and vals[0] == "wrong")
            locval = tags.get("localization")
            if locval:
                loc[locval] = loc.get(locval, 0) + 1
    pct = lambda x: round(100.0 * x / n_q, 1) if n_q else None  # noqa: E731
    return {"questions_tagged": n_q,
            "top1_acceptable_pct": pct(top1_ok),
            "top3_acceptable_pct": pct(top3_ok),
            "top1_wrong_pct": pct(wrong_chosen),
            "failure_localization_counts": loc}

def cmd_selftest() -> int:
    fix_path = Path("/tmp/retrieval_eval_fixture_tags.json")
    fix_path.write_text(json.dumps(TALLY_FIXTURE_TAGS))
    res = tally_tags([fix_path])
    expect = {"questions_tagged": 3, "top1_acceptable_pct": 66.7,
              "top3_acceptable_pct": 100.0, "top1_wrong_pct": 33.3}
    ok = True
    for k, v in expect.items():
        got = res[k]
        good = abs(got - v) < 1e-9 if isinstance(v, float) else got == v
        print(("ok  " if good else "FAIL"), k, "=", got)
        ok &= good
    assert res["failure_localization_counts"] == {"scoring-wrong": 1, "search-miss": 1}, res
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, default=Path("work/sfx-gold/bank_overlap.manifest.json"))
    m.add_argument("--seed-from", type=Path, default=Path("work/sfx-pipeline/20260826-poc"))

    t = sub.add_parser("top5")
    t.add_argument("--run-dir", type=Path, default=Path("work/sfx-pipeline/20260826-poc"))
    t.add_argument("--clip")

    tl = sub.add_parser("tally")
    tl.add_argument("--tags", nargs="*", type=Path, default=[])
    tl.add_argument("--selftest", action="store_true")

    args = parser.parse_args()
    if args.cmd == "manifest":
        return cmd_manifest(args.out, args.seed_from)
    if args.cmd == "top5":
        return cmd_top5(args.run_dir.resolve(), args.clip)
    if args.cmd == "tally":
        if args.selftest:
            return cmd_selftest()
        print(json.dumps(tally_tags(list(args.tags)), indent=2))
        return 0
    parser.error("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
