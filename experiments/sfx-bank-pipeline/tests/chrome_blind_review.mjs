#!/usr/bin/env node
/** Headless Chrome checks for the blinded review page (CDP, no extra deps). */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const BASE = process.argv[2] || "http://127.0.0.1:8899/";
const CDP = process.argv[3] || "http://127.0.0.1:9333";
const SHOT = process.argv[4] || ".impeccable/review/desktop.png";

let id = 0;
const pending = new Map();
let ws;

function send(method, params = {}, sessionId) {
  const msgId = ++id;
  const body = { id: msgId, method, params };
  if (sessionId) body.sessionId = sessionId;
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("cdp timeout " + method)), 15000);
    pending.set(msgId, {
      resolve: (v) => { clearTimeout(t); resolve(v); },
      reject: (e) => { clearTimeout(t); reject(e); },
    });
    ws.send(JSON.stringify(body));
  });
}

async function evalExpr(sessionId, expression, awaitPromise = false) {
  const r = await send("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
  }, sessionId);
  if (r.exceptionDetails) {
    throw new Error("eval: " + JSON.stringify(r.exceptionDetails.exception || r.exceptionDetails));
  }
  return r.result.value;
}

const version = await fetch(CDP + "/json/version").then((r) => r.json());
ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.addEventListener("open", res); ws.addEventListener("error", rej); });
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) reject(new Error(JSON.stringify(msg.error)));
    else resolve(msg.result);
  }
});

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
await send("Page.enable", {}, sessionId);
await send("Runtime.enable", {}, sessionId);
await send("Emulation.setDeviceMetricsOverride", {
  width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false,
}, sessionId);
await send("Page.setWebLifecycleState", { state: "active" }, sessionId);

ws.addEventListener("message", async (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.method === "Page.javascriptDialogOpening" && msg.sessionId === sessionId) {
    await send("Page.handleJavaScriptDialog", { accept: true, promptText: "tester" }, sessionId);
  }
});

const loaded = new Promise((resolve) => {
  const handler = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method === "Page.loadEventFired" && msg.sessionId === sessionId) {
      ws.removeEventListener("message", handler);
      resolve();
    }
  };
  ws.addEventListener("message", handler);
});
await send("Page.navigate", { url: BASE }, sessionId);
await Promise.race([loaded, new Promise((r) => setTimeout(r, 4000))]);
await new Promise((r) => setTimeout(r, 400));

const ready = await evalExpr(sessionId, `
  new Promise((res) => {
    const go = () => res({
      clips: document.querySelectorAll("section.clip").length,
      radios: document.querySelectorAll('input[type="radio"]').length,
      gnotes: document.querySelectorAll("textarea.gnote").length,
      events: document.querySelectorAll("tr.event-row").length,
      videos: document.querySelectorAll("video").length,
      eventTables: document.querySelectorAll("table.events").length,
      prog: document.getElementById("prog") && document.getElementById("prog").textContent,
    });
    if (document.querySelector("section.clip")) go();
    else setTimeout(go, 400);
  })
`, true);

const r2 = documentCreate => documentCreate;
void r2;

await evalExpr(sessionId, `
  const a = document.querySelector('input[type="radio"][data-scope="event"][data-label="A"][data-field="pref"][value="2"]');
  const b = document.querySelector('input[type="radio"][data-scope="event"][data-label="A"][data-field="pref"][value="3"]');
  a.click(); b.click();
  const clip = a.dataset.clip;
  const key = "blind-ratings:" + JSON.parse(atob(document.getElementById("cfg").textContent.trim())).run_id;
  const store = JSON.parse(localStorage.getItem(key) || "{}");
  const rk = clip + "\\u0000" + "event" + "\\u0000" + a.dataset.id + "\\u0000" + "A";
  const g = document.querySelector("textarea.gnote");
  g.value = "wind leak at 0:12";
  g.dispatchEvent(new Event("input", { bubbles: true }));
  const store2 = JSON.parse(localStorage.getItem(key) || "{}");
  const gk = g.dataset.clip + "\\u0000clip\\u0000general\\u0000" + g.dataset.label;
  window.__radioCheck = {
    checked3: b.checked,
    checked2: a.checked,
    pref: store[rk] && store[rk].pref,
    prefType: store[rk] && typeof store[rk].pref,
    gnote: store2[gk] && store2[gk].text,
    payloadSample: (() => {
      const btn = document.getElementById("download");
      return true;
    })(),
  };
`);

const radioCheck = await evalExpr(sessionId, "window.__radioCheck");

await evalExpr(sessionId, `
  (async () => {
    const row = document.querySelector("tr.event-row");
    const clip = row.dataset.clip;
    const videos = [...document.querySelectorAll('video[data-clip="' + clip + '"]')];
    await Promise.all(videos.map(v => v.readyState >= 1 ? Promise.resolve() : new Promise(r => v.addEventListener("loadedmetadata", r, { once: true }))));
    row.querySelector(".play-ev").click();
    await new Promise(r => setTimeout(r, 350));
    window.__seek = {
      clip,
      time: Number(row.dataset.time),
      inspecting: row.classList.contains("inspecting"),
      currents: videos.map(v => v.currentTime),
      paused: videos.map(v => v.paused),
    };
  })()
`, true);

const seek = await evalExpr(sessionId, "window.__seek");

const collected = await evalExpr(sessionId, `
  (function(){
    const keys = Object.keys(JSON.parse(localStorage.getItem("blind-ratings:" + JSON.parse(atob(document.getElementById("cfg").textContent.trim())).run_id) || "{}"));
    return { lsKeys: keys, piped: keys.map(k => k.replace(/\\u0000/g, "|")) };
  })()
`);

await send("Runtime.evaluate", { expression: "window.scrollTo(0,0)" }, sessionId);
await new Promise((r) => setTimeout(r, 200));
const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }, sessionId);
mkdirSync(dirname(SHOT), { recursive: true });
writeFileSync(SHOT, Buffer.from(shot.data, "base64"));

const mid = await evalExpr(sessionId, "document.querySelectorAll('section.clip')[1].scrollIntoView(); 1;");
void mid;
await new Promise((r) => setTimeout(r, 200));
const shot2path = SHOT.replace(/desktop\.png$/, "clip-2.png");
const shot2 = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, sessionId);
writeFileSync(shot2path, Buffer.from(shot2.data, "base64"));

console.log(JSON.stringify({ ready, radioCheck, seek, collected, shot: SHOT, shot2: shot2path }, null, 2));
await send("Target.closeTarget", { targetId });
ws.close();
process.exit(0);
