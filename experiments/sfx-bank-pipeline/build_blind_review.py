#!/usr/bin/env python3
"""Stage 6b — blinded listening review over an existing sfx-bank-pipeline run.

Phase-2 Q1 harness (docs/sfx-bank-foley-phase2-eval-brief.md §3.1): re-present
the ALREADY-RENDERED A/B outputs of a run under randomized anonymous labels
(System A / B / …), collect 1-5 ratings (synchronization, semantic correctness,
physical believability, audio quality, palette consistency, artifacts), per-event
preference + the binary keep-replace question, and only reveal which label was
which system AFTER all required ratings are submitted. Zero new generation.

Outputs under <run>/review/blind/:
    assignment.json      label -> system map (+ seed, provenance)
    index.html           self-contained blinded rating page (file:// or http)
    RATING_GUIDE.md      owner instructions

Ratings persistence is dual-mode:
    - always: "Download ratings JSON" button in the page;
    - when served via `build_blind_review.py --serve`, submit also POSTs to
      /save which writes incoming/blind-ratings-<annotator>-<ts>.json (fsynced).

No system names appear in the served DOM before reveal: everything is encoded
in one base64 configuration blob decoded client-side.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import secrets
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CLIP_RATING_DIMENSIONS = [
    "sync",
    "semantic",
    "believability",
    "audio_quality",
    "palette",
    "artifacts",
]
DIMENSION_LABELS = {
    "sync": "Synchronization",
    "semantic": "Semantic correctness",
    "believability": "Physical believability",
    "audio_quality": "Audio quality",
    "palette": "Palette consistency",
    "artifacts": "Artifacts",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def discover_clips(run_dir: Path) -> dict[str, list[str]]:
    """Return {clip: [system, ...]} for clips having >=2 rendered systems and a plan."""
    review = run_dir / "review"
    plan_dir = run_dir / "plan"
    clips: dict[str, list[str]] = {}
    if not review.is_dir():
        return clips
    known: dict[str, set[str]] = {}
    for mp4 in sorted(review.glob("*.mp4")):
        stem = mp4.name[: -len(".mp4")]
        if "." not in stem:
            continue
        clip, system = stem.split(".", 1)
        if not (plan_dir / f"{clip}.placement.json").is_file():
            continue
        known.setdefault(clip, set()).add(system)
    for clip, systems in sorted(known.items()):
        if len(systems) >= 2:
            clips[clip] = sorted(systems)
    return clips


def load_events(plan_path: Path) -> list[dict[str, Any]]:
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            "id": ev.get("id", ""),
            "label": ev.get("label", ""),
            "time_s": round(float(ev.get("time_s", 0.0)), 3),
            "tier": ev.get("tier", ""),
        }
        for ev in plan.get("events", [])
        if ev.get("id")
    ]


def build_assignment(clips: dict[str, list[str]], events: dict[str, list[dict[str, Any]]],
                     run_dir: Path, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    per_clip: dict[str, Any] = {}
    for clip, systems in clips.items():
        order = list(systems)
        rng.shuffle(order)
        labels = [chr(ord("A") + i) for i in range(len(order))]
        per_clip[clip] = {
            "map": {label: system for label, system in zip(labels, order)},
            "events": events.get(clip, []),
        }
    return {
        "schema_version": 1,
        "kind": "blind-review-assignment",
        "run_dir": str(run_dir),
        "generated_utc": utc_now(),
        "seed": seed,
        "clips": per_clip,
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Blinded Foley Review</title>
<style>
:root { color-scheme: dark; }
body { font-family: -apple-system, sans-serif; background:#111418; color:#d7dce2; margin:0 auto; max-width:900px; padding:24px; }
h1 { font-size:1.25rem; } h2 { font-size:1.05rem; border-bottom:1px solid #2c333b; padding-bottom:4px; margin-top:36px; }
h3 { font-size:.92rem; color:#9fb0c0; margin-bottom:4px; }
video { width:100%; max-width:640px; background:#000; border-radius:6px; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
td, th { padding:4px 8px; text-align:left; vertical-align:top; border-bottom:1px solid #232a31; }
.dim { min-width:11rem; }
.opt { white-space:nowrap; }
label.opt { margin-right:10px; cursor:pointer; }
textarea { width:100%; background:#171c22; color:#d7dce2; border:1px solid #2c333b; border-radius:4px; }
button { background:#2f6fed; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:.9rem; cursor:pointer; }
button:disabled { background:#39414a; cursor:not-allowed; }
#reveal { margin-top:18px; padding:14px; background:#18202a; border:1px solid #2f6fed44; border-radius:8px; display:none; }
.progress { position:sticky; top:0; background:#111418ee; padding:10px 0; z-index:5; }
.eventblock { margin:14px 0 22px; padding:10px; background:#151a20; border-radius:8px; }
.note { color:#8794a1; font-size:.78rem; }
</style></head><body>
<h1>Blinded Foley Review</h1>
<p class="note">Rate each anonymous system honestly. System identities are revealed only after every
required rating is submitted. Per clip: rate all six dimensions and answer keep/replace per system.
Per visible event: give overall preference (1&ndash;5) and keep/replace per system.</p>
<div class="progress"><span id="prog">…</span>
<button id="submit" disabled>Submit &amp; reveal</button>
<button id="download" type="button">Download ratings JSON</button></div>
<div id="root"></div>
<div id="reveal"></div>
<script id="cfg" type="application/json">__CFG_B64__</script>
<script>
"use strict";
const CFG = JSON.parse(atob(document.getElementById("cfg").textContent));
const DIMS = __DIMS__;
const DIM_LABELS = __DIM_LABELS__;
const LS_KEY = "blind-ratings:" + CFG.run_id;
let annotator = localStorage.getItem("blind-annotator") || "";
while (!annotator) { annotator = prompt("Your name (stored locally, saved into the ratings file):") || ""; }
localStorage.setItem("blind-annotator", annotator);
const ratingsKey = (clip, scope, id, label) => clip+"\\u0000"+scope+"\\u0000"+id+"\\u0000"+label;
let R = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
function getRating(clip, scope, id, label){ return R[ratingsKey(clip,scope,id,label)] || {}; }
function setRating(clip, scope, id, label, field, value){
  const k = ratingsKey(clip,scope,id,label);
  const cur = R[k] || {}; cur[field] = value; R[k] = cur;
  localStorage.setItem(LS_KEY, JSON.stringify(R)); refreshProgress();
}
function completeness(){
  let total = 0, done = 0;
  for (const [clip, info] of Object.entries(CFG.clips)){
    const nSys = Object.keys(info.map).length;
    total += DIMS.length * nSys + 2 * nSys;
    for (let i=0;i<nSys;i++){ const lbl = CFG.labels[i];
      for (const dim of DIMS){ if (getRating(clip,"clip",dim,lbl)[dim] != null) done++; }
      if (getRating(clip,"clip","overall",lbl)["keep"]) done++;
      if (getRating(clip,"clip","overall",lbl)["pref"] != null) done++;
    }
    for (const ev of info.events){
      total += 2 * nSys;
      for (let i=0;i<nSys;i++){ const lbl = CFG.labels[i];
        if (getRating(clip,"event",ev.id,lbl)["pref"] != null) done++;
        if (getRating(clip,"event",ev.id,lbl)["keep"]) done++;
      }
    }
  }
  return {done, total};
}
function refreshProgress(){
  const {done,total} = completeness();
  document.getElementById("prog").textContent = `${done}/${total} required ratings recorded`;
  document.getElementById("submit").disabled = done !== total;
}
function collectRatings(){
  const out = { annotator, exported_utc: new Date().toISOString(), run_id: CFG.run_id,
    tool: "build_blind_review",
    cfg_b64: document.getElementById("cfg").textContent.trim(), ratings: {} };
  for (const [k,v] of Object.entries(R)){ out.ratings[k.replace(/\\u0000/g,"|")] = v; }
  return out;
}
function build(){
  const root = document.getElementById("root");
  let html = "";
  for (const [clip, info] of Object.entries(CFG.clips)){
    html += `<h2>${clip}</h2>`;
    for (const label of CFG.labels.slice(0, Object.keys(info.map).length)){
      html += `<h3>System ${label}</h3>
        <video controls preload="metadata" src="${info.videos[label]}"></video>`;
      if (info.events.length){
        html += `<div class="eventblock"><table>
          <tr><th class="dim">Event</th><th>System ${label}: overall preference</th><th>Keep without replacing?</th><th>Notes</th></tr>`;
        for (const ev of info.events){
          html += `<tr data-ev="${ev.id}"><td class="dim">${ev.label}<br><span class="note">${ev.id} @ ${ev.time_s.toFixed(2)}s &middot; tier ${ev.tier}</span></td>`
            + cellHtml(clip,"event",ev.id,label,"pref","scale")
            + cellHtml(clip,"event",ev.id,label,"keep","keep")
            + cellHtml(clip,"event",ev.id,label,"notes","notes") + `</tr>`;
        }
        html += `</table></div>`;
      }
      html += `<div class="eventblock"><table>
        <tr><th class="dim">Whole clip</th><th>System ${label}: six dimensions 1-5</th><th>Keep without replacing?</th><th>Notes</th></tr>`;
      for (const dim of DIMS){
        html += `<tr><td class="dim">${DIM_LABELS[dim]}</td>`
          + cellHtml(clip,"clip",dim,label,dim,"scale")
          + (dim===DIMS[0] ? cellHtml(clip,"clip","overall",label,"keep","keep") +
             cellHtml(clip,"clip","overall",label,"notes","notes") :
             `<td></td><td></td>`) + `</tr>`;
      }
      html += `</table></div>`;
    }
  }
  root.innerHTML = html;
}
function cellHtml(clip, scope, id, label, field, kind){
  const val = (getRating(clip, scope, id, label))[field];
  const ds = `data-clip="${clip}" data-scope="${scope}" data-id="${id}" data-label="${label}" data-field="${field}"`;
  if (kind === "scale"){
    let h = `<td>`;
    for (let v=1; v<=5; v++) h += `<label class="opt"><input type="radio" ${ds} value="${v}" ${val===v?"checked":""}></label>`;
    return h + `</td>`;
  }
  if (kind === "keep"){
    return `<td><label class="opt"><input type="radio" ${ds} data-value="keep" ${val==="keep"?"checked":""}>keep</label>` +
      `<label class="opt"><input type="radio" ${ds} data-value="replace" ${val==="replace"?"checked":""}>replace</label></td>`;
  }
  return `<td><textarea rows="1" ${ds}>${val==null?"":String(val).replace(/</g,"&lt;")}</textarea></td>`;
}
document.getElementById("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(collectRatings(), null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `blind-ratings-${annotator}.json`;
  a.click(); URL.revokeObjectURL(a.href);
});
document.getElementById("submit").addEventListener("click", async () => {
  const payload = JSON.stringify(collectRatings());
  if (location.protocol === "http:" || location.protocol === "https:"){
    try { await fetch("/save", {method:"POST", body: payload}); } catch(e){}
  }
  const rev = document.getElementById("reveal");
  let rows = "<table><tr><th>Label</th><th>Actual system</th></tr>";
  const tally = {};
  for (const [clip, info] of Object.entries(CFG.clips)){
    rows += `<tr><td colspan="2"><strong>${clip}</strong></td></tr>`;
    for (const [lbl, sys] of Object.entries(info.map)){
      rows += `<tr><td>System ${lbl}</td><td>${sys}</td></tr>`;
      const r = getRating(clip,"clip","overall",lbl);
      if (r.keep){
        tally[sys] = tally[sys] || {keep:0, replace:0};
        tally[sys][r.keep] += 1;
      }
    }
  }
  rows += "</table><p>Keep-replace tallies: <code>" + JSON.stringify(tally) + "</code></p>" +
          "<p>Ratings are already captured via localStorage — please still download/send the JSON.</p>";
  rev.innerHTML = rows; rev.style.display = "block";
  window.scrollTo({top: rev.offsetTop - 80, behavior: "smooth"});
});
build(); refreshProgress();
</script></body></html>
"""


def encode_config(assignment: dict[str, Any], run_id: str) -> tuple[str, int]:
    """Base64-encode the full client config incl. real system names + video srcs."""
    clips_out: dict[str, Any] = {}
    n_labels = 1
    for clip, info in assignment["clips"].items():
        labels = [chr(ord("A") + i) for i in range(len(info["map"]))]
        n_labels = max(n_labels, len(labels))
        clips_out[clip] = {
            "map": info["map"],
            "videos": {label: f"../{clip}.{system}.mp4" for label, system in info["map"].items()},
            "events": info["events"],
        }
    cfg = {
        "run_id": run_id,
        "labels": [chr(ord("A") + i) for i in range(n_labels)],
        "clips": clips_out,
    }
    blob = base64.b64encode(json.dumps(cfg, sort_keys=True).encode()).decode()
    return blob, n_labels


GUIDE = """# Blinded listening guide (Q1)

Open the page, rate, send back the JSON. Takes minutes per clip.

1. Serve (recommended, enables one-click save):
   ```bash
   python3 experiments/sfx-bank-pipeline/build_blind_review.py \\
       --run-dir {run_dir} --serve   # then open the printed URL
   ```
   (or open `review/blind/index.html` directly over file:// — playback works in
   Chrome; if blocked, use serve mode.)
2. Enter your name once (persisted locally).
3. Per clip, per anonymous system: six 1–5 dimensions + keep/replace + notes.
4. Per visible event: overall preference 1–5 + keep/replace. Disagreement with
   your own taste is data — record what you hear.
5. The keep/replace question means exactly: *would you ship this sound without
   manually swapping it out?*
6. Submit when every required control is filled; identity reveal appears.
7. Ratings also autosave to localStorage; press "Download ratings JSON" and put
   the file next to this README (or rely on serve mode's automatic save).

Definitions: sync = would a viewer notice timing errors; semantic = right kind
of sound for the action; believability = physically plausible for the scene;
audio quality = fidelity/artifact-free in isolation; palette = consistent sonic
identity across events; artifacts = glitches, truncation, pumping.
"""


class SaveHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _serve_static(self, path: str) -> None:
        target = (self.run_dir / path.lstrip("/")).resolve()
        if not str(target).startswith(str(self.run_dir.resolve())) or not target.is_file():
            self.send_error(404)
            return
        ctype = "text/html" if target.suffix == ".html" else (
            "application/javascript" if target.suffix == ".js" else "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_static("review/blind/index.html")
        elif path.endswith(".mp4"):
            # Media lives under review/; cfg uses "../name.mp4" which browsers
            # normalize against "/" to "/name.mp4". Accept either form.
            rel = Path(path.lstrip("/"))
            candidates = [rel, Path("review") / rel]
            target = next((c for c in (self.run_dir / p for p in candidates)
                           if c.is_file()
                           and str(c.resolve()).startswith(str(self.run_dir.resolve()))), None)
            if target is None:
                self.send_error(404)
                return
            body = target.read_bytes()
            size = len(body)
            start, end, status = 0, size - 1, 200
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    s = int(m.group(1))
                    e = int(m.group(2)) if m.group(2) else size - 1
                else:  # bytes=-N suffix range
                    s, e = max(0, size - int(m.group(2))), size - 1
                s, e = min(s, size - 1), min(e, size - 1)
                if s <= e:
                    start, end, status = s, e, 206
            chunk = body[start:end + 1]
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 32 * 1024 * 1024:
            self.send_error(413)
            return
        payload = self.rfile.read(length)
        try:
            data = json.loads(payload)
            annotator = "".join(c for c in str(data.get("annotator", "anon")) if c.isalnum() or c in "-_") or "anon"
        except json.JSONDecodeError:
            self.send_error(400)
            return
        inc_dir = self.run_dir / "review" / "blind" / "incoming"
        inc_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = inc_dir / f"blind-ratings-{annotator}-{ts}-{secrets.token_hex(3)}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(payload)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        tmp.rename(out)
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("work/sfx-pipeline/20260826-poc"))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force", action="store_true", help="overwrite existing assignment/page")
    parser.add_argument("--serve", action="store_true", help="serve the page + accept ratings POSTs")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    clips = discover_clips(run_dir)
    if not clips:
        print(f"error: no clip has >=2 rendered systems under {run_dir}/review", file=sys.stderr)
        return 1
    events = {clip: load_events(run_dir / "plan" / f"{clip}.placement.json") for clip in clips}

    out_dir = run_dir / "review" / "blind"
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = out_dir / "assignment.json"
    if assignment_path.exists() and not args.force:
        assignment = json.loads(assignment_path.read_text())
        for clip in clips:
            if clip not in assignment["clips"]:
                raise SystemExit(f"assignment.json exists but lacks clip '{clip}'; rerun with --force")
    else:
        assignment = build_assignment(clips, events, run_dir, args.seed)

    blob, n_labels = encode_config(assignment, run_dir.name)
    page = PAGE_TEMPLATE.replace("__CFG_B64__", blob)
    page = page.replace("__DIMS__", json.dumps(CLIP_RATING_DIMENSIONS))
    page = page.replace("__DIM_LABELS__", json.dumps(DIMENSION_LABELS))
    (out_dir / "index.html").write_text(page)
    (out_dir / "RATING_GUIDE.md").write_text(GUIDE.format(run_dir=run_dir))
    write_assignment = assignment_path.exists() and not args.force
    if not write_assignment:
        tmp = assignment_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(assignment, indent=2))
        tmp.rename(assignment_path)

    total_events = sum(len(v["events"]) for v in assignment["clips"].values())
    print(f"blinded review ready: {len(clips)} clips, "
          f"{total_events} events, labels up to {chr(ord('A') + n_labels - 1)}")
    for clip, info in assignment["clips"].items():
        print(f"  {clip}: {' / '.join(sorted(info['map']))}")
    if args.serve:
        SaveHandler.run_dir = run_dir
        server = ThreadingHTTPServer(("127.0.0.1", args.port), SaveHandler)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"serving {url}  (Ctrl-C to stop); ratings land in {out_dir/'incoming'}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
