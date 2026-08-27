#!/usr/bin/env node
/** Stub-DOM runner: execute the blinded review page script and print DOM stats. */
import { readFileSync } from "node:fs";

const html = readFileSync(process.argv[2], "utf8");
const cfgMatch = html.match(/<script id="cfg"[^>]*>([^<]*)<\/script>/);
const scriptMatch = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/);
if (!cfgMatch || !scriptMatch) {
  console.error("could not extract cfg/script");
  process.exit(2);
}
const cfgText = cfgMatch[1].trim();
const script = '"use strict";' + scriptMatch[1];

const store = Object.create(null);
const localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => {
    store[k] = String(v);
  },
};

const elements = Object.create(null);
function makeEl(id) {
  const el = {
    id,
    textContent: id === "cfg" ? cfgText : "",
    innerHTML: "",
    disabled: true,
    style: {},
    className: "",
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {},
    removeEventListener() {},
    matches() { return false; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    click() {},
    focus() {},
    offsetTop: 0,
  };
  elements[id] = el;
  return el;
}
for (const id of ["cfg", "root", "prog", "submit", "download", "reveal", "saved"]) makeEl(id);
elements.cfg.textContent = cfgText;

const document = {
  getElementById: (id) => elements[id] || makeEl(id),
  addEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
  body: { appendChild() {} },
};

const window = {
  scrollTo() {},
  localStorage,
  location: { protocol: "file:" },
};
Object.assign(globalThis, {
  document,
  window,
  localStorage,
  prompt: () => "tester",
  atob: (s) => Buffer.from(s, "base64").toString("binary"),
  location: window.location,
  URL: globalThis.URL,
  Blob: class {
    constructor(parts) {
      this.parts = parts;
    }
  },
  fetch: async () => ({ ok: true }),
});

eval(script + "\n;globalThis.__blind={inspectWindow,formatTime,setRating,collectRatings,completeness,getRating,R};"); // eslint-disable-line no-eval

const root = elements.root.innerHTML;
const radios = [...root.matchAll(/<input\b([^>]*)>/g)].filter((m) => /type="radio"/.test(m[1]));
const textareas = [...root.matchAll(/<textarea\b([^>]*)>/g)];
const videos = [...root.matchAll(/<video\b([^>]*)>/g)];
const gnotes = textareas.filter((m) => /data-id="general"/.test(m[1]) && /data-field="text"/.test(m[1]));
const eventTables = [...root.matchAll(/<table\b([^>]*)>/g)].filter((m) => /events/.test(m[1]));

function countAttr(re) {
  return radios.filter((m) => re.test(m[1])).length;
}

const B = globalThis.__blind;
B.setRating("clip-a", "event", "ev_001", "A", "pref", 2);
B.setRating("clip-a", "event", "ev_001", "A", "pref", 3);
const collected = B.collectRatings();
const prefKey = Object.keys(collected.ratings).find((k) => k.includes("ev_001") && k.endsWith("|A"));
const out = {
  annotator: "tester",
  progressText: elements.prog.textContent,
  submitDisabled: elements.submit.disabled,
  root,
  radioCount: radios.length,
  textareaCount: textareas.length,
  gnoteCount: gnotes.length,
  videoCount: videos.length,
  eventTableCount: eventTables.length,
  eventPrefA: countAttr(/data-scope="event"/) && countAttr(/data-scope="event"[^>]*data-label="A"[^>]*data-field="pref"|data-label="A"[^>]*data-field="pref"/),
  namedRadios: radios.filter((m) => /\bname="/.test(m[1])).length,
  rows2: textareas.filter((m) => /rows="2"/.test(m[1])).length,
  inspect: B.inspectWindow(1.109),
  formatTime: B.formatTime(1.109),
  formatTimeLong: B.formatTime(65.0),
  prefKey,
  prefValue: collected.ratings[prefKey],
  payloadKeys: Object.keys(collected.ratings),
};
process.stdout.write(JSON.stringify(out) + "\n", () => process.exit(0));
