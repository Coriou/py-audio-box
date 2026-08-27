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
DIMENSION_HELP = {
    "sync": "Would a viewer notice timing errors?",
    "semantic": "Right kind of sound for the action.",
    "believability": "Physically plausible for the scene.",
    "audio_quality": "Fidelity / artifact-free in isolation.",
    "palette": "Consistent sonic identity across events.",
    "artifacts": "Glitches, truncation, pumping.",
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
<meta name="viewport" content="width=1280">
<title>Blinded Foley Review</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101418;
  --bg-bar: #0c1014f2;
  --bg-clip: #151b21;
  --bg-inset: #1b2229;
  --line: #2a333c;
  --text: #d8dee6;
  --muted: #8b97a4;
  --faint: #6a7682;
  --accent: #5b93f5;
  --accent-dim: #5b93f526;
  --focus: #9cc0ff;
  --btn: #3d7ef0;
  --btn-off: #3a434c;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.45;
  padding: 0 0 64px;
}
::selection { background: #2d4d80; color: #fff; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.progress {
  position: sticky; top: 0; z-index: 8;
  background: var(--bg-bar);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(8px);
}
.progress-inner {
  max-width: 1280px; margin: 0 auto; padding: 10px 24px;
  display: flex; align-items: center; gap: 8px 14px; flex-wrap: wrap;
}
.progress-inner h1 { font-size: 1rem; font-weight: 650; margin: 0; letter-spacing: -0.02em; }
#prog { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.875rem; }
#saved {
  font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase;
  color: #8fbf93; min-height: 1em;
}
#saved:not(.show) { visibility: hidden; }
#saved.show::before {
  content: ""; display: inline-block; width: 0.45em; height: 0.7em;
  border-right: 2px solid #8fbf93; border-bottom: 2px solid #8fbf93;
  transform: rotate(40deg) translate(-1px, -1px); margin-right: 6px;
}
#missing { font-size: 0.8rem; color: var(--muted); display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; flex: 1 1 320px; min-width: 0; }
.missing-link {
  background: transparent; color: #8fb4ff; border: 1px solid #2f6fed55;
  border-radius: 999px; padding: 2px 9px; font-size: 0.72rem; font-weight: 500;
  margin-left: 4px; white-space: nowrap;
}
.missing-link:hover { background: #2f6fed22; filter: none; }
.progress-inner .sp { flex: 1; min-width: 8px; }
.progress-actions { display: flex; gap: 8px; flex-shrink: 0; }
button {
  background: var(--btn); color: #fff; border: 0; border-radius: 6px;
  padding: 8px 14px; font-size: 0.875rem; font-weight: 600; cursor: pointer;
  font-family: inherit;
}
button:hover { filter: brightness(1.08); }
button:disabled { background: var(--btn-off); color: #9aa3ab; cursor: not-allowed; filter: none; }
button.secondary { background: transparent; color: var(--text); border: 1px solid var(--line); }
button.play-ev {
  background: var(--bg-inset); color: var(--text); border: 1px solid var(--line);
  padding: 4px 8px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
  white-space: nowrap; display: inline-block; margin-bottom: 6px;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 18px 24px 0; min-width: 1100px; }
.lede { color: var(--muted); font-size: 0.875rem; max-width: 78ch; margin: 0 0 8px; }
.lede code { font-size: 0.8em; }
.clip {
  margin-top: 28px; padding: 18px 18px 8px;
  background: var(--bg-clip); border: 1px solid var(--line); border-radius: 12px;
}
.clip-h {
  display: flex; align-items: baseline; gap: 12px; margin: 0 0 14px;
  border-bottom: 1px solid var(--line); padding-bottom: 10px;
}
.clip-h h2 {
  margin: 0; font-size: 1.125rem; font-weight: 650; letter-spacing: -0.02em;
}
.clip-progress {
  margin-left: auto; font-variant-numeric: tabular-nums;
  font-size: 0.75rem; color: var(--muted); letter-spacing: 0.02em;
}
.ab-videos {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;
}
figure { margin: 0; }
figcaption {
  font-size: 0.8rem; font-weight: 650; color: var(--accent);
  margin: 0 0 6px; letter-spacing: 0.02em;
}
video {
  width: 100%; background: #000; border-radius: 8px; display: block;
  border: 1px solid #000;
}
video.seek-flash { outline: 2px solid var(--accent); outline-offset: 1px; }
.gnote-block { margin-top: 8px; }
.gnote-block label {
  display: block; font-size: 0.8rem; font-weight: 650; margin-bottom: 2px;
}
.gnote-block .hint { color: var(--muted); font-size: 0.75rem; margin: 0 0 6px; }
textarea {
  width: 100%; background: var(--bg-inset); color: var(--text);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px;
  font-family: inherit; font-size: 0.8125rem; resize: vertical;
}
.block { margin: 0 0 16px; }
.block h3 {
  margin: 0 0 8px; font-size: 0.8rem; font-weight: 650;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
}
.tier-legend { color: var(--muted); font-size: 0.75rem; margin: 0 0 8px; max-width: 90ch; }
table { border-collapse: collapse; width: 100%; font-size: 0.8125rem; }
table.events, table.clip-dims { table-layout: fixed; }
th, td { padding: 8px 8px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
thead th { font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); font-weight: 650; }
thead tr.sys th { text-transform: none; letter-spacing: 0; font-size: 0.8rem; color: var(--accent); padding-bottom: 2px; }
.ev-col { width: 18%; }
table.events col.pref { width: 16%; }
table.events col.keepc { width: 9%; }
table.events col.notesc { width: 16%; }
table.clip-dims th:first-child, table.clip-dims td:first-child { width: 28%; }
.event-row { cursor: pointer; }
.event-row:hover td { background: #1c242c; }
.event-row.inspecting td { background: var(--accent-dim); }
.ev-label { font-weight: 600; font-size: 0.875rem; }
.ev-meta { margin-top: 2px; color: var(--faint); font-size: 0.75rem; font-variant-numeric: tabular-nums; }
.ev-id { color: var(--faint); }
.ev-time { color: var(--muted); font-weight: 550; margin-right: 8px; }
.ev-tier { border-bottom: 1px dotted var(--faint); }
.dim-name { display: block; font-weight: 600; color: var(--text); text-transform: none; letter-spacing: 0; font-size: 0.875rem; }
.dim-help { display: block; font-weight: 400; color: var(--muted); font-size: 0.75rem; letter-spacing: 0; text-transform: none; margin-top: 2px; }
.scale { width: max-content; max-width: 100%; }
.scale-caps {
  display: flex; justify-content: space-between; font-size: 0.68rem;
  color: var(--muted); margin-bottom: 4px; gap: 8px;
}
.scale-opts { display: flex; gap: 4px; }
.scale-opt {
  position: relative; width: 32px; height: 32px; margin: 0; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
}
.scale-opt input { position: absolute; inset: 0; opacity: 0; margin: 0; cursor: pointer; }
.scale-opt span {
  width: 28px; height: 28px; border-radius: 6px; border: 1px solid var(--line);
  display: flex; align-items: center; justify-content: center;
  font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.8rem;
  background: var(--bg-inset); color: var(--text);
}
.scale-opt:hover span { border-color: var(--accent); }
.scale-opt input:checked + span { background: var(--accent); color: #fff; border-color: var(--accent); }
.scale-opt input:focus-visible + span { outline: 2px solid var(--focus); outline-offset: 2px; }
.keep { display: flex; flex-direction: column; gap: 4px; }
.keep-opt { display: flex; align-items: center; gap: 6px; cursor: pointer; white-space: nowrap; font-size: 0.8125rem; }
.keep-opt input { width: 16px; height: 16px; accent-color: var(--accent); }
.keep-hint { font-size: 0.75rem; font-weight: 400; text-transform: none; letter-spacing: 0; margin-top: 4px; color: var(--muted); }
#reveal {
  display: none; max-width: 1280px; margin: 24px auto 0; padding: 16px 24px;
  background: #18202a; border: 1px solid #2f6fed44; border-radius: 8px;
}
</style></head><body>
<div class="progress"><div class="progress-inner">
  <h1>Blinded Foley Review</h1>
  <span id="prog">...</span>
  <span id="missing"></span>
  <span id="saved">Saved</span>
  <span class="sp"></span>
  <span class="progress-actions">
    <button id="submit" disabled>Submit and reveal</button>
    <button id="download" type="button" class="secondary">Download ratings JSON</button>
  </span>
</div></div>
<div class="wrap">
<p class="lede">Rate each anonymous system honestly. Identities appear only after every required rating is submitted.
Per clip: six dimensions and keep/replace per system. Per visible event: preference (1&ndash;5) and keep/replace per system.
Click an event (or <strong>Play both</strong>) to cue both videos 0.75s before the moment and hear System A, then System B, for about 2.5s each.</p>
<div id="root"></div>
<div id="reveal"></div>
</div>
<script id="cfg" type="application/json">__CFG_B64__</script>
<script>
"use strict";
const CFG = JSON.parse(atob(document.getElementById("cfg").textContent));
const DIMS = __DIMS__;
const DIM_LABELS = __DIM_LABELS__;
const DIM_HELP = __DIM_HELP__;
const LS_KEY = "blind-ratings:" + CFG.run_id;
const PREROLL = 0.75;
const PLAY_WINDOW = 2.5;
const TIER_HELP = {
  "mirelo-fallback": "Auto-proposed by the Mirelo fallback detector, not hand-curated.",
  "direct": "Direct match from the SFX bank.",
  "dsp": "Bank match adapted with local DSP.",
  "sa3": "Regenerated when a bank match could not be timed."
};
const TIER_PHRASE = {
  "mirelo-fallback": "auto-proposed",
  "direct": "bank match",
  "dsp": "DSP-adapted",
  "sa3": "regenerated"
};
let annotator = localStorage.getItem("blind-annotator") || "";
while (!annotator) { annotator = prompt("Your name (stored locally, saved into the ratings file):") || ""; }
localStorage.setItem("blind-annotator", annotator);
const ratingsKey = (clip, scope, id, label) => clip+"\\u0000"+scope+"\\u0000"+id+"\\u0000"+label;
let R = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
function getRating(clip, scope, id, label){ return R[ratingsKey(clip,scope,id,label)] || {}; }
let savedTimer = 0;
function markSaved(){
  const el = document.getElementById("saved");
  if (!el) return;
  el.classList.add("show");
  clearTimeout(savedTimer);
  savedTimer = setTimeout(() => el.classList.remove("show"), 1200);
}
function setRating(clip, scope, id, label, field, value){
  const k = ratingsKey(clip,scope,id,label);
  const cur = R[k] || {}; cur[field] = value; R[k] = cur;
  localStorage.setItem(LS_KEY, JSON.stringify(R)); refreshProgress(); markSaved();
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
function completenessFor(clip){
  const info = CFG.clips[clip];
  const nSys = Object.keys(info.map).length;
  let total = DIMS.length * nSys + 2 * nSys, done = 0;
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
  return {done, total};
}
function missingItems(){
  const out = [];
  const fieldLabel = (f) => f === "keep" ? "Ship it?" : f === "pref" ? "Preference" : DIM_LABELS[f] || f;
  for (const [clip, info] of Object.entries(CFG.clips)){
    const nSys = Object.keys(info.map).length;
    for (let i=0;i<nSys;i++){ const lbl = CFG.labels[i];
      for (const dim of DIMS){ if (getRating(clip,"clip",dim,lbl)[dim] == null)
        out.push({clip, scope:"clip", id:dim, label:lbl, field:dim, short:`${clip} · ${lbl} · ${fieldLabel(dim)}`}); }
      if (getRating(clip,"clip","overall",lbl)["keep"] == null)
        out.push({clip, scope:"clip", id:"overall", label:lbl, field:"keep", short:`${clip} · ${lbl} · whole clip · Ship it?`});
      if (getRating(clip,"clip","overall",lbl)["pref"] == null)
        out.push({clip, scope:"clip", id:"overall", label:lbl, field:"pref", short:`${clip} · ${lbl} · whole clip · Preference`});
      for (const ev of info.events){
        if (getRating(clip,"event",ev.id,lbl)["pref"] == null)
          out.push({clip, scope:"event", id:ev.id, label:lbl, field:"pref", short:`${clip} · ${lbl} · ${ev.id} · Preference`});
        if (getRating(clip,"event",ev.id,lbl)["keep"] == null)
          out.push({clip, scope:"event", id:ev.id, label:lbl, field:"keep", short:`${clip} · ${lbl} · ${ev.id} · Ship it?`});
      }
    }
  }
  return out;
}
function refreshProgress(){
  const {done,total} = completeness();
  document.getElementById("prog").textContent = done + "/" + total + " required ratings recorded";
  document.getElementById("submit").disabled = done !== total;
  const miss = missingItems();
  const box = document.getElementById("missing");
  box.innerHTML = miss.length
    ? " · missing: " + miss.slice(0,16).map((m,i) => `<button type="button" class="missing-link" data-mi="${i}">${esc(m.short)}</button>`).join(" ")
      + (miss.length > 16 ? ` <span class="note">+${miss.length - 16} more</span>` : "")
    : "";
  document.querySelectorAll("[data-clip-progress]").forEach(el => {
    const c = completenessFor(el.getAttribute("data-clip-progress"));
    el.textContent = c.done + "/" + c.total;
  });
}
document.addEventListener("click", e => {
  const b = e.target.closest(".missing-link");
  if (!b) return;
  const m = missingItems()[Number(b.dataset.mi)];
  const el = document.querySelector(`input[type=radio][data-clip="${m.clip}"][data-scope="${m.scope}"][data-id="${m.id}"][data-label="${m.label}"][data-field="${m.field}"]`);
  if (el){ el.scrollIntoView({block:"center", behavior:"smooth"}); el.focus({preventScroll:true}); }
});
function collectRatings(){
  const out = { annotator, exported_utc: new Date().toISOString(), run_id: CFG.run_id,
    tool: "build_blind_review",
    cfg_b64: document.getElementById("cfg").textContent.trim(), ratings: {} };
  for (const [k,v] of Object.entries(R)){ out.ratings[k.replace(/\\u0000/g,"|")] = v; }
  return out;
}
function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function formatTime(s){
  s = Math.max(0, Number(s) || 0);
  const m = Math.floor(s / 60);
  let tenth = Math.round((s - m * 60) * 10);
  if (tenth === 600) { return (m + 1) + ":00.0"; }
  const sec = Math.floor(tenth / 10);
  const t = tenth % 10;
  return m + ":" + String(sec).padStart(2, "0") + "." + t;
}
function inspectWindow(timeS){
  const start = Math.max(0, Number(timeS) - PREROLL);
  return { start, end: start + PLAY_WINDOW };
}
function tierPhrase(tier){ return TIER_PHRASE[tier] || (tier || "unspecified"); }
function tierTitle(tier){ return TIER_HELP[tier] || ("Event source: " + (tier || "unspecified")); }
function radioName(clip, scope, id, label, field){
  return clip + "|" + scope + "|" + id + "|" + label + "|" + field;
}
function dsAttrs(clip, scope, id, label, field){
  return `data-clip="${esc(clip)}" data-scope="${esc(scope)}" data-id="${esc(id)}" data-label="${esc(label)}" data-field="${esc(field)}"`;
}
function scaleHtml(clip, scope, id, label, field, lo, hi, aria){
  const val = getRating(clip, scope, id, label)[field];
  const name = radioName(clip, scope, id, label, field);
  const ds = dsAttrs(clip, scope, id, label, field);
  let h = `<div class="scale" role="radiogroup" aria-label="${esc(aria)}">`;
  h += `<div class="scale-caps"><span>${esc(lo)}</span><span>${esc(hi)}</span></div><div class="scale-opts">`;
  for (let v = 1; v <= 5; v++){
    const checked = Number(val) === v ? "checked" : "";
    h += `<label class="scale-opt"><input type="radio" name="${esc(name)}" ${ds} value="${v}" ${checked}><span>${v}</span></label>`;
  }
  return h + `</div></div>`;
}
function keepHtml(clip, scope, id, label){
  const val = getRating(clip, scope, id, label)["keep"];
  const name = radioName(clip, scope, id, label, "keep");
  const ds = dsAttrs(clip, scope, id, label, "keep");
  return `<div class="keep" role="radiogroup" aria-label="Would you ship this sound without swapping it out?">` +
    `<label class="keep-opt"><input type="radio" name="${esc(name)}" ${ds} data-value="keep" ${val==="keep"?"checked":""}>keep</label>` +
    `<label class="keep-opt"><input type="radio" name="${esc(name)}" ${ds} data-value="replace" ${val==="replace"?"checked":""}>replace</label></div>`;
}
function notesHtml(clip, scope, id, label, field){
  const val = getRating(clip, scope, id, label)[field];
  const ds = dsAttrs(clip, scope, id, label, field);
  return `<textarea rows="2" ${ds}>${val==null?"":esc(val)}</textarea>`;
}
function gnoteHtml(clip, label){
  const ds = dsAttrs(clip, "clip", "general", label, "text");
  const val = (getRating(clip,"clip","general",label)).text ?? "";
  return `<textarea rows="2" class="gnote" ${ds}>${esc(val)}</textarea>`;
}
function labelsFor(info){
  return CFG.labels.slice(0, Object.keys(info.map).length);
}
function build(){
  const root = document.getElementById("root");
  let html = "";
  for (const [clip, info] of Object.entries(CFG.clips)){
    const labels = labelsFor(info);
    html += `<section class="clip" data-clip="${esc(clip)}">`;
    html += `<header class="clip-h"><h2>${esc(clip)}</h2><span class="clip-progress" data-clip-progress="${esc(clip)}">0/0</span></header>`;
    html += `<div class="ab-videos">`;
    for (const label of labels){
      html += `<figure><figcaption>System ${esc(label)}</figcaption>` +
        `<video controls preload="metadata" data-clip="${esc(clip)}" data-label="${esc(label)}" src="${esc(info.videos[label])}"></video>` +
        `<div class="gnote-block"><label>General media notes</label>` +
        `<p class="hint">Anything ratings cannot capture (noise, wind/ambience leaks, dropouts). Prefix timestamps like <code>0:47</code> when a problem is location-specific.</p>` +
        gnoteHtml(clip, label) + `</div></figure>`;
    }
    html += `</div>`;
    if (info.events.length){
      html += `<div class="block"><h3>Visible events</h3>`;
      html += `<p class="tier-legend">Each row is one sound-worthy moment. Click the row or <strong>Play both</strong> to hear both systems at that timestamp (0.75s preroll, ~2.5s each, A then B). ` +
        `<strong>auto-proposed</strong> = Mirelo fallback detector, not hand-curated. ` +
        `<strong>bank match</strong> / <strong>DSP-adapted</strong> / <strong>regenerated</strong> = how the moment was sourced.</p>`;
      html += `<p id="inspect-status-${esc(clip)}" class="tier-legend"></p>`;
      html += `<table class="events"><colgroup><col class="ev-col">`;
      for (const _l of labels) html += `<col class="pref"><col class="keepc"><col class="notesc">`;
      html += `</colgroup><thead>`;
      html += `<tr class="sys"><th class="ev-col">Event</th>`;
      for (const label of labels) html += `<th colspan="3">System ${esc(label)}</th>`;
      html += `</tr><tr><th class="ev-col"></th>`;
      for (const _l of labels) html += `<th>Preference</th><th>Ship it?</th><th>Notes</th>`;
      html += `</tr></thead><tbody>`;
      for (const ev of info.events){
        html += `<tr class="event-row" tabindex="0" data-clip="${esc(clip)}" data-ev="${esc(ev.id)}" data-time="${ev.time_s}">`;
        html += `<td class="ev-col"><button type="button" class="play-ev" data-action="inspect">Play both</button>` +
          `<div class="ev-label">${esc(ev.label)}</div>` +
          `<div class="ev-meta"><span class="ev-time">${formatTime(ev.time_s)}</span>` +
          `<span class="ev-id">${esc(ev.id)}</span> · ` +
          `<span class="ev-tier" title="${esc(tierTitle(ev.tier))}">${esc(tierPhrase(ev.tier))}</span>` +
          ` <span class="ev-id">(${esc(ev.tier)})</span></div></td>`;
        for (const label of labels){
          html += `<td>` + scaleHtml(clip,"event",ev.id,label,"pref","unusable","strongly prefer",
              `Preference for System ${label}, ${ev.label}. 1 unusable to 5 strongly prefer`) + `</td>`;
          html += `<td>` + keepHtml(clip,"event",ev.id,label) + `</td>`;
          html += `<td>` + notesHtml(clip,"event",ev.id,label,"notes") + `</td>`;
        }
        html += `</tr>`;
      }
      html += `</tbody></table></div>`;
    }
    html += `<div class="block"><h3>Whole-clip ratings</h3>`;
    html += `<table class="clip-dims"><thead><tr><th>Dimension</th>`;
    for (const label of labels) html += `<th>System ${esc(label)}</th>`;
    html += `</tr></thead><tbody>`;
    for (const dim of DIMS){
      html += `<tr><th><span class="dim-name">${esc(DIM_LABELS[dim])}</span>` +
        `<span class="dim-help">${esc(DIM_HELP[dim] || "")}</span></th>`;
      for (const label of labels){
        html += `<td>` + scaleHtml(clip,"clip",dim,label,dim,"poor","excellent",
            `${DIM_LABELS[dim]} for System ${label}. 1 poor to 5 excellent`) + `</td>`;
      }
      html += `</tr>`;
    }
    html += `<tr class="clip-keep" data-row="clip-keep"><th><span class="dim-name">Would you ship this sound without swapping it out?</span>` +
      `<span class="keep-hint dim-help">keep = ship as-is; replace = you would manually swap it.</span></th>`;
    for (const label of labels) html += `<td>` + keepHtml(clip,"clip","overall",label) + `</td>`;
    html += `</tr>`;
    html += `<tr class="clip-notes" data-row="clip-notes"><th><span class="dim-name">Clip notes</span></th>`;
    for (const label of labels) html += `<td>` + notesHtml(clip,"clip","overall",label,"notes") + `</td>`;
    html += `</tr></tbody></table></div></section>`;
  }
  root.innerHTML = html;
}
function cellHtml(clip, scope, id, label, field, kind){
  if (kind === "scale"){
    const lohi = field === "pref" ? ["unusable","strongly prefer"] : ["poor","excellent"];
    return `<td>` + scaleHtml(clip, scope, id, label, field, lohi[0], lohi[1], field) + `</td>`;
  }
  if (kind === "keep") return `<td>` + keepHtml(clip, scope, id, label) + `</td>`;
  return `<td>` + notesHtml(clip, scope, id, label, field) + `</td>`;
}
let inspectGen = 0;
function setInspectStatus(clip, msg){
  const el = document.getElementById("inspect-status-" + clip);
  if (el) el.textContent = msg || "";
}
function inspectEvent(row){
  const clip = row.dataset.clip;
  const timeS = Number(row.dataset.time);
  const { start, end } = inspectWindow(timeS);
  document.querySelectorAll(".event-row.inspecting").forEach(r => r.classList.remove("inspecting"));
  row.classList.add("inspecting");
  const videos = [...document.querySelectorAll(`video[data-clip="${clip}"]`)];
  const gen = ++inspectGen;
  videos.forEach(v => {
    v.classList.add("seek-flash");
    setTimeout(() => v.classList.remove("seek-flash"), 280);
    const cue = () => { try { v.currentTime = start; } catch (e) {} };
    if (v.readyState >= 1) cue();
    else v.addEventListener("loadedmetadata", cue, { once: true });
    v.pause();
  });
  const playOne = (i) => {
    if (gen !== inspectGen) return;
    if (i >= videos.length) { setInspectStatus(clip, ""); return; }
    const v = videos[i];
    const label = v.getAttribute("data-label") || String(i);
    setInspectStatus(clip, "Playing System " + label + " at " + formatTime(timeS));
    videos.forEach((other, j) => { if (j !== i) other.pause(); });
    const startPlay = () => {
      if (gen !== inspectGen) return;
      try { v.currentTime = start; } catch (e) {}
      const p = v.play();
      if (p && p.catch) p.catch(() => {});
      const tick = () => {
        if (gen !== inspectGen) { v.removeEventListener("timeupdate", tick); return; }
        if (v.currentTime >= end - 0.04) {
          v.pause();
          v.removeEventListener("timeupdate", tick);
          playOne(i + 1);
        }
      };
      v.addEventListener("timeupdate", tick);
    };
    if (v.readyState >= 1) startPlay();
    else v.addEventListener("loadedmetadata", startPlay, { once: true });
  };
  playOne(0);
}
const coerce = (t) => t.type === "radio"
  ? (t.dataset.value === "keep" || t.dataset.value === "replace" ? t.dataset.value : Number(t.value))
  : t.value.trim();
const persistInput = (t) => {
  const {clip, scope, id, label, field} = t.dataset;
  if (!field) return;
  if (t.type === "radio"){
    for (const sib of document.querySelectorAll(`input[type=radio][data-clip="${clip}"][data-scope="${scope}"][data-id="${id}"][data-label="${label}"][data-field="${field}"]`))
      if (sib !== t) sib.checked = false;
  }
  setRating(clip, scope, id, label, field, coerce(t));
};
document.addEventListener("change", e => { if (e.target.matches("input[type=radio],textarea[data-field]")) persistInput(e.target); });
document.addEventListener("input", e => { if (e.target.matches("textarea[data-field]")) persistInput(e.target); });
document.addEventListener("click", e => {
  const ignore = e.target.closest("input, textarea, label.scale-opt, label.keep-opt");
  if (ignore) return;
  const row = e.target.closest(".event-row");
  if (row) inspectEvent(row);
});
document.addEventListener("keydown", e => {
  if (/^[1-5]$/.test(e.key)) {
    const scale = e.target.closest(".scale");
    if (scale) {
      const radio = scale.querySelector(`input[type="radio"][value="${e.key}"]`);
      if (radio) { radio.checked = true; persistInput(radio); e.preventDefault(); }
    }
  }
  if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("event-row")) {
    e.preventDefault();
    inspectEvent(e.target);
  }
});
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
    rows += `<tr><td colspan="2"><strong>${esc(clip)}</strong></td></tr>`;
    for (const [lbl, sys] of Object.entries(info.map)){
      rows += `<tr><td>System ${esc(lbl)}</td><td>${esc(sys)}</td></tr>`;
      const r = getRating(clip,"clip","overall",lbl);
      if (r.keep){
        tally[sys] = tally[sys] || {keep:0, replace:0};
        tally[sys][r.keep] += 1;
      }
    }
  }
  rows += "</table><p>Keep-replace tallies: <code>" + esc(JSON.stringify(tally)) + "</code></p>" +
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
4. Per visible event: overall preference 1–5 + keep/replace. Click the event
   (or Play both) to hear System A then System B at that moment. Disagreement
   with your own taste is data — record what you hear.
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
    page = page.replace("__DIM_HELP__", json.dumps(DIMENSION_HELP))
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
