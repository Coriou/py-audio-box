import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import WaveSurfer from "wavesurfer.js";
import {
  fixtures,
  clone,
  current,
  initial,
  edit,
  travel,
  restoreSession,
  overlaps,
  audible,
  RevisionConflict,
  type Session,
  type Clip,
  type Operation,
  type Request,
} from "./adapter";
import { samples, mix } from "./media";
import "./style.css";
const key = (id: string) => `editor-0b:1.3.0:${id}`;
function read(id: string): Session {
  const raw = localStorage.getItem(key(id));
  return raw ? restoreSession(raw, fixtures[id]) : initial(fixtures[id]);
}
// Undo belongs to this tab. A foreign revision clears the local undo cursor;
// it never turns another writer's edit into this client's undo target.
function readForTab(id: string): Session {
  const s = read(id);
  const known = sessionStorage.getItem(key(id) + ":known");
  sessionStorage.setItem(key(id) + ":known", String(current(s).revision));
  return known === String(current(s).revision)
    ? s
    : { ...s, undo: [], redo: [] };
}
function Waveform({ clip }: { clip: Clip }) {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let dead = false;
    let wave: WaveSurfer | undefined;
    setError("");
    samples(clip.asset_id)
      .then((data) => {
        if (dead) return;
        const source = data.slice(clip.in_tick, clip.out_tick);
        wave = WaveSurfer.create({
          container: host.current!,
          height: 46,
          waveColor: "#557f7c",
          progressColor: "#145c57",
          cursorWidth: 0,
          interact: false,
          normalize: false,
          peaks: [source],
          duration: source.length / 48000,
        });
      })
      .catch((e) => {
        if (!dead) setError(String(e));
      });
    return () => {
      dead = true;
      wave?.destroy();
    };
  }, [clip.asset_id, clip.in_tick, clip.out_tick]);
  return (
    <div aria-hidden="true" className="wave" ref={host}>
      {error}
    </div>
  );
}
function App() {
  const [fixture, setFixture] = useState("30");
  const [session, setSession] = useState<Session>(() => {
    try {
      return readForTab("30");
    } catch {
      return initial(fixtures["30"]);
    }
  });
  const [selected, setSelected] = useState("c0");
  const [status, setStatus] = useState("Saved locally · no service");
  const [conflict, setConflict] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [scroll, setScroll] = useState(0);
  const [scriptScroll, setScriptScroll] = useState(0);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audio = useRef<AudioContext | null>(null);
  const source = useRef<AudioBufferSourceNode | null>(null);
  const playGeneration = useRef(0);
  const p = current(session),
    clip = p.clips.find((c) => c.id === selected) ?? p.clips[0];
  const [draft, setDraft] = useState<Clip>(clone(clip));
  const [take, setTake] = useState("candidate-long");
  const [adjust, setAdjust] = useState(false);
  const block = p.blocks.find(
    (b) => p.takes.find((t) => t.id === clip.take_id)?.block_id === b.id,
  );
  useEffect(() => {
    setTake(
      p.takes.find((t) => t.block_id === block?.id && t.id === "candidate-long")
        ?.id ??
        block?.selected_take_id ??
        "",
    );
    setAdjust(false);
  }, [block?.id]);
  const collisions = overlaps(p);
  const px = (600 * zoom) / 48000;
  const duration = Math.max(
    ...p.clips.map((c) => c.start + c.out_tick - c.in_tick),
  );
  useEffect(() => setDraft(clone(clip)), [clip]);
  useEffect(() => {
    const fn = (e: StorageEvent) => {
      if (e.key === key(fixture)) {
        setConflict(true);
        setStatus(
          "Conflict · another tab changed this fixture. Reload current.",
        );
        stop();
      }
    };
    window.addEventListener("storage", fn);
    return () => window.removeEventListener("storage", fn);
  }, [fixture]);
  function stop() {
    playGeneration.current++;
    source.current?.stop();
    source.current = null;
    setPlaying(false);
  }
  function save(next: Session) {
    localStorage.setItem(key(fixture), JSON.stringify(next));
    sessionStorage.setItem(
      key(fixture) + ":known",
      String(current(next).revision),
    );
    setSession(next);
    setStatus("Saved locally · no service");
    setConflict(false);
    stop();
  }
  function submit(op: Operation, fields: Record<string, unknown>) {
    try {
      const stored = {
        ...read(fixture),
        undo: session.undo,
        redo: session.redo,
      };
      const r: Request = {
        project_id: p.id,
        expected_revision: p.revision,
        ...fields,
      };
      const next = edit(stored, op, r);
      save(next);
    } catch (e) {
      if (e instanceof RevisionConflict) setConflict(true);
      setStatus(String(e instanceof Error ? e.message : e));
    }
  }
  function history(way: "undo" | "redo") {
    try {
      save(
        travel(
          { ...read(fixture), undo: session.undo, redo: session.redo },
          p.revision,
          way,
        ),
      );
    } catch (e) {
      if (e instanceof RevisionConflict) setConflict(true);
      setStatus(String(e instanceof Error ? e.message : e));
    }
  }
  function load(id: string) {
    stop();
    try {
      const s = readForTab(id);
      setFixture(id);
      setSession(s);
      setSelected(current(s).clips[0].id);
      setConflict(false);
      setScroll(0);
      setScriptScroll(0);
      setStatus("Reloaded current · saved locally");
    } catch (e) {
      setStatus(
        `Local history invalid: ${String(e)}. Use Reset fixture to recover.`,
      );
    }
  }
  function reset() {
    stop();
    const s = initial(fixtures[fixture]);
    localStorage.setItem(key(fixture), JSON.stringify(s));
    sessionStorage.setItem(
      key(fixture) + ":known",
      String(current(s).revision),
    );
    setSession(s);
    setSelected("c0");
    setConflict(false);
    setStatus("Fixture reset locally");
  }
  async function play() {
    if (playing) {
      stop();
      return;
    }
    const generation = ++playGeneration.current;
    setStatus("Preparing synthetic preview…");
    try {
      const data = await mix(p);
      if (generation !== playGeneration.current) return;
      audio.current ??= new AudioContext({ sampleRate: 48000 });
      await audio.current.resume();
      if (generation !== playGeneration.current) return;
      const b = audio.current.createBuffer(1, data.length, 48000);
      b.copyToChannel(data, 0);
      const node = audio.current.createBufferSource();
      node.buffer = b;
      node.connect(audio.current.destination);
      source.current = node;
      node.onended = () => {
        if (generation === playGeneration.current) {
          setPlaying(false);
          source.current = null;
        }
      };
      node.start(0, Math.min(playhead / 48000, (data.length - 1) / 48000));
      setPlaying(true);
      setStatus(
        `Synthetic preview · device ${audio.current.sampleRate} Hz · no acoustic qualification`,
      );
    } catch (e) {
      setStatus(String(e));
      setPlaying(false);
    }
  }
  function replace() {
    if (!block) return;
    const t = p.takes.find((t) => t.id === take);
    if (!t) {
      setStatus("Select a candidate for this block.");
      return;
    }
    const a = p.assets.find((a) => a.id === t.asset_id)!;
    submit("take_select", {
      block_id: block.id,
      take_id: t.id,
      adjusted_clip: adjust
        ? {
            ...clip,
            asset_id: t.asset_id,
            take_id: t.id,
            in_tick: 0,
            out_tick: a.frames,
            fade_in: 0,
            fade_out: 0,
          }
        : null,
    });
  }
  return (
    <>
      <header>
        <div>
          <h1>Production editor</h1>
          <span className="sub">Synthetic fixture lab</span>
        </div>
        <div className="actions">
          <label>
            Fixture{" "}
            <select
              aria-label="Fixture"
              value={fixture}
              onChange={(e) => load(e.target.value)}
            >
              <option value="30">30 blocks / 31 clips</option>
              <option value="100">100 blocks / 200 clips</option>
            </select>
          </label>
          <button onClick={() => load(fixture)}>Reload current</button>
          <button onClick={reset}>Reset fixture</button>
        </div>
      </header>
      <div className="statusbar">
        <span role="status" className={conflict ? "error" : ""}>
          {status}
        </span>
        <span>
          Revision <b data-testid="revision">{p.revision}</b> ·{" "}
          {p.blocks.length} blocks · {p.clips.length} clips
        </span>
      </div>
      <main>
        <aside className="script">
          <div className="panel-heading">
            <h2>Script</h2>
            <span>{p.speakers.length} speakers</span>
          </div>
          <div
            className="script-list"
            onScroll={(e) => setScriptScroll(e.currentTarget.scrollTop)}
          >
            <div style={{ height: p.blocks.length * 96, position: "relative" }}>
              {p.blocks
                .slice(
                  Math.max(0, Math.floor(scriptScroll / 96) - 2),
                  Math.floor(scriptScroll / 96) + 12,
                )
                .map((b, i) => {
                  const index =
                    Math.max(0, Math.floor(scriptScroll / 96) - 2) + i;
                  const c = p.clips.find(
                    (c) =>
                      p.takes.find((t) => t.id === c.take_id)?.block_id ===
                      b.id,
                  );
                  return (
                    <button
                      key={b.id}
                      className={`block ${block?.id === b.id ? "selected" : ""}`}
                      style={{ top: index * 96 }}
                      onClick={() => c && setSelected(c.id)}
                    >
                      <span>
                        {b.id} · {b.speaker_id} · {b.selected_take_id}
                      </span>
                      <pre>{b.text}</pre>
                    </button>
                  );
                })}
            </div>
          </div>
          <p className="hint">
            Exact source text. Selecting a block never generates or replaces a
            take.
          </p>
        </aside>
        <section className="workspace">
          <div className="toolbar">
            <h2>Timeline</h2>
            <div className="actions">
              <button className="primary" onClick={play}>
                {playing ? "Stop preview" : "Play preview"}
              </button>
              <button
                disabled={!session.undo.length}
                onClick={() => history("undo")}
              >
                Undo
              </button>
              <button
                disabled={!session.redo.length}
                onClick={() => history("redo")}
              >
                Redo
              </button>
            </div>
          </div>
          <div className="timeline-options">
            <label>
              Zoom{" "}
              <select
                aria-label="Zoom"
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
              >
                <option value="0.5">50%</option>
                <option value="1">100%</option>
                <option value="2">200%</option>
              </select>
            </label>
            <label>
              Cursor (ticks){" "}
              <input
                aria-label="Cursor ticks"
                type="number"
                min="0"
                max={duration - 1}
                value={playhead}
                onChange={(e) =>
                  setPlayhead(
                    Math.max(0, Math.min(duration - 1, Number(e.target.value))),
                  )
                }
              />
            </label>
            <span>Fixed placement · 48,000 ticks/s</span>
          </div>
          <div
            className="timeline-scroll"
            onScroll={(e) => setScroll(e.currentTarget.scrollLeft)}
          >
            <div
              className="timeline"
              style={{ width: Math.max(800, duration * px + 160) }}
            >
              <div className="ruler">
                {Array.from(
                  { length: Math.ceil(duration / 4800) + 1 },
                  (_, i) => (
                    <span key={i} style={{ left: 140 + i * 4800 * px }}>
                      {(i / 10).toFixed(1)}s
                    </span>
                  ),
                )}
              </div>
              {p.tracks.map((t) => (
                <div className="track" key={t.id}>
                  <div className="track-controls">
                    <b>{t.id}</b>
                    <div>
                      <button
                        aria-label={`Mute ${t.id}`}
                        aria-pressed={t.mute}
                        onClick={() =>
                          submit("track_put", {
                            track: { ...t, mute: !t.mute },
                          })
                        }
                      >
                        Mute
                      </button>
                      <button
                        aria-label={`Solo ${t.id}`}
                        aria-pressed={t.solo}
                        onClick={() =>
                          submit("track_put", {
                            track: { ...t, solo: !t.solo },
                          })
                        }
                      >
                        Solo
                      </button>
                    </div>
                  </div>
                  <div className="lane" style={{ width: duration * px + 20 }}>
                    {p.clips
                      .filter(
                        (c) =>
                          c.track_id === t.id &&
                          c.start * px < scroll + 1400 &&
                          (c.start + c.out_tick - c.in_tick) * px >
                            scroll - 200,
                      )
                      .map((c) => (
                        <button
                          key={c.id}
                          aria-label={`Clip ${c.id}`}
                          aria-pressed={clip.id === c.id}
                          className={`clip ${clip.id === c.id ? "selected" : ""} ${!audible(p, c) ? "silent" : ""}`}
                          style={{
                            left: c.start * px,
                            width: Math.max(12, (c.out_tick - c.in_tick) * px),
                            top: collisions.includes(c.id)
                              ? (p.clips.indexOf(c) % 2) * 38 + 12
                              : 12,
                          }}
                          onClick={() => setSelected(c.id)}
                          onKeyDown={(e) => {
                            if (
                              e.key === "ArrowRight" ||
                              e.key === "ArrowLeft"
                            ) {
                              e.preventDefault();
                              submit("clip_put", {
                                clip: {
                                  ...c,
                                  start:
                                    c.start +
                                    (e.key === "ArrowRight" ? 1 : -1) *
                                      (e.shiftKey ? 4800 : 480),
                                },
                              });
                            }
                          }}
                        >
                          <span>
                            {c.id}{" "}
                            {collisions.includes(c.id) ? "· overlap" : ""}
                          </span>
                          <Waveform clip={c} />
                        </button>
                      ))}
                  </div>
                </div>
              ))}
              <div className="playhead" style={{ left: 140 + playhead * px }} />
            </div>
          </div>
          <div className="timeline-note">
            <b>
              {collisions.length
                ? `${collisions.length} overlapping clips`
                : "No same-track overlaps"}
            </b>
            <span>
              Arrow keys move a focused clip 480 ticks; Shift moves 4,800.
              Neighbours stay fixed.
            </span>
          </div>
          <details className="history">
            <summary>
              Revision history · {session.entries.length} snapshots
            </summary>
            <ol>
              {session.entries.slice(-15).map((e) => (
                <li key={e.project.revision}>
                  Revision {e.project.revision} · {e.operation} ·{" "}
                  {e.changed_ids.join(", ") || "original fixture"}
                </li>
              ))}
            </ol>
            <button
              onClick={() => {
                try {
                  const stored = read(fixture);
                  const c = current(stored).clips[1];
                  const next = edit(stored, "clip_put", {
                    project_id: p.id,
                    expected_revision: current(stored).revision,
                    clip: { ...c, start: c.start + 480 },
                  });
                  localStorage.setItem(key(fixture), JSON.stringify(next));
                  setStatus(
                    "Second writer simulated. Next edit or undo checks the stale revision.",
                  );
                } catch (e) {
                  setStatus(String(e));
                }
              }}
            >
              Simulate another writer
            </button>
            <p>
              Local browser storage only. Reload current resolves a conflict
              without overwriting the other edit.
            </p>
          </details>
        </section>
        <aside className="inspector">
          <div className="panel-heading">
            <h2>Clip properties</h2>
            <b>{clip.id}</b>
          </div>
          <p className="hint">
            {clip.asset_id} · {clip.take_id ?? "imported bed"}
            <br />
            {audible(p, clip) ? "Audible routing" : "Silent routing"} · original
            media preserved
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit("clip_put", { clip: draft });
            }}
          >
            <div className="fields">
              {(
                [
                  "start",
                  "in_tick",
                  "out_tick",
                  "gain",
                  "fade_in",
                  "fade_out",
                ] as const
              ).map((k) => (
                <label key={k}>
                  {
                    {
                      start: "Start · ticks",
                      in_tick: "Source in · ticks",
                      out_tick: "Source out · ticks",
                      gain: "Linear gain",
                      fade_in: "Fade in · ticks",
                      fade_out: "Fade out · ticks",
                    }[k]
                  }
                  <input
                    aria-label={k}
                    type="number"
                    step={k === "gain" ? "any" : 1}
                    min="0"
                    value={draft[k]}
                    onChange={(e) =>
                      setDraft({ ...draft, [k]: Number(e.target.value) })
                    }
                  />
                </label>
              ))}
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={draft.mute}
                onChange={(e) => setDraft({ ...draft, mute: e.target.checked })}
              />{" "}
              Mute clip
            </label>
            <button className="primary" type="submit">
              Apply clip edit
            </button>
          </form>
          <button
            className="wide"
            onClick={() =>
              submit("clip_split", {
                clip_id: clip.id,
                at_tick: Math.floor((clip.in_tick + clip.out_tick) / 2),
                new_clip_id: `split-${crypto.randomUUID()}`,
              })
            }
          >
            Split at source midpoint
          </button>
          <p className="hint">
            Split requires zero fades. Trim with source in/out. Ranges are
            end-exclusive.
          </p>
          {block && (
            <section className="takes">
              <h2>Replace take</h2>
              <p>Selected: {block.selected_take_id}</p>
              <label>
                Candidate{" "}
                <select
                  aria-label="Candidate"
                  value={take}
                  onChange={(e) => setTake(e.target.value)}
                >
                  {p.takes
                    .filter((t) => t.block_id === block.id)
                    .map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.id} · {t.asset_id}
                      </option>
                    ))}
                </select>
              </label>
              <label className="check">
                <input
                  aria-label="Adjust trim explicitly"
                  type="checkbox"
                  checked={adjust}
                  onChange={(e) => setAdjust(e.target.checked)}
                />{" "}
                Explicitly fit full source and clear fades
              </label>
              <button onClick={replace}>Replace selected take</button>
              <p className="hint">
                Longer takes expose overlap. Shorter takes require explicit trim
                adjustment. Other starts never move.
              </p>
            </section>
          )}
        </aside>
      </main>
      <footer>
        Fixture contract {p.schema_version} · browser evaluation only · no
        server or generation
      </footer>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
