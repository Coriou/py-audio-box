import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  fixtures,
  initial,
  current,
  edit,
  travel,
  restoreSession,
  clone,
  overlaps,
  audible,
  fade,
  validate,
  RevisionConflict,
  type Session,
  type Clip,
} from "./adapter";
const request = (s: Session, extra: Record<string, unknown>) => ({
  project_id: current(s).id,
  expected_revision: current(s).revision,
  ...extra,
});
const put = (s: Session, c: Clip) =>
  edit(s, "clip_put", request(s, { clip: c }));
function oracle(s: Session) {
  const result = spawnSync(
    process.env.CONTRACT_PYTHON ?? "python3",
    [
      "-c",
      `import sys,json\nfrom contracts.validate import validate,edit_semantics\ns=json.load(sys.stdin)\nfor e in s['entries']: validate('project',e['project'])\nfor a,b in zip(s['entries'],s['entries'][1:]):\n validate(b['operation']+'_request',b['request'])\n edit_semantics(a['project'],b['project'],b['operation'])\nprint('canonical Python oracle passed')`,
    ],
    { cwd: "..", input: JSON.stringify(s), encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
}
test("shared fixtures remain verbatim and satisfy canonical Python schema/semantics", () => {
  for (const p of Object.values(fixtures)) {
    const s = initial(p);
    assert.deepEqual(current(s), p);
    oracle(s);
  }
});
test("fixed move overlaps without ripple; zero gain, mute, solo and exact fade endpoints", () => {
  let s = initial(fixtures["30"]);
  const c = current(s).clips[0];
  s = put(s, { ...c, start: 480, gain: 0, fade_in: 100, fade_out: 100 });
  assert.equal(current(s).clips[1].start, 4800);
  assert.ok(overlaps(current(s)).includes(c.id));
  assert.equal(audible(current(s), current(s).clips[0]), false);
  assert.equal(fade(current(s).clips[0], 0), 0);
  assert.equal(fade(current(s).clips[0], 4799), 0);
  assert.equal(fade(current(s).clips[0], 99), 0.99);
  s = put(s, { ...current(s).clips[0], gain: 1, mute: true });
  assert.equal(audible(current(s), current(s).clips[0]), false);
  s = edit(
    s,
    "track_put",
    request(s, { track: { ...current(s).tracks[1], solo: true } }),
  );
  assert.equal(audible(current(s), current(s).clips[1]), false);
  oracle(s);
});
test("trim/split preserves source, gain, fixed offsets; rejects boundary, duplicate ID and faded split atomically", () => {
  let s = initial(fixtures["30"]);
  s = put(s, {
    ...current(s).clips[0],
    in_tick: 100,
    out_tick: 4700,
    gain: 0.5,
  });
  const before = clone(s);
  for (const extra of [
    { at_tick: 100, new_clip_id: "right" },
    { at_tick: 200, new_clip_id: "c1" },
  ])
    assert.throws(() =>
      edit(s, "clip_split", request(s, { clip_id: "c0", ...extra })),
    );
  assert.deepEqual(s, before);
  s = edit(
    s,
    "clip_split",
    request(s, { clip_id: "c0", at_tick: 2400, new_clip_id: "right" }),
  );
  const [left, right] = ["c0", "right"].map(
    (id) => current(s).clips.find((c) => c.id === id)!,
  );
  assert.equal(left.out_tick, right.in_tick);
  assert.equal(right.start, left.start + left.out_tick - left.in_tick);
  assert.equal(right.gain, 0.5);
  assert.equal(current(s).clips[1].start, 4800);
  oracle(s);
  const faded = put(s, { ...left, fade_in: 1 });
  assert.throws(
    () =>
      edit(
        faded,
        "clip_split",
        request(faded, { clip_id: "c0", at_tick: 200, new_clip_id: "new" }),
      ),
    /split_fades/,
  );
});
test("explicit long replacement exposes overlap; short fails then succeeds only with explicit adjusted clip", () => {
  let s = initial(fixtures["30"]);
  const original = clone(current(s));
  s = edit(
    s,
    "take_select",
    request(s, {
      block_id: "b0",
      take_id: "candidate-long",
      adjusted_clip: null,
    }),
  );
  assert.equal(current(s).clips[0].out_tick, 9600);
  assert.equal(current(s).clips[1].start, 4800);
  assert.ok(overlaps(current(s)).includes("c0"));
  const before = clone(s);
  assert.throws(
    () =>
      edit(
        s,
        "take_select",
        request(s, {
          block_id: "b0",
          take_id: "candidate-short",
          adjusted_clip: null,
        }),
      ),
    /short_take/,
  );
  assert.deepEqual(s, before);
  s = edit(
    s,
    "take_select",
    request(s, {
      block_id: "b0",
      take_id: "candidate-short",
      adjusted_clip: {
        ...current(s).clips[0],
        take_id: "candidate-short",
        asset_id: "short",
        in_tick: 0,
        out_tick: 2400,
        fade_in: 0,
        fade_out: 0,
      },
    }),
  );
  assert.equal(current(s).blocks[0].selected_take_id, "candidate-short");
  assert.deepEqual(current(s).takes, original.takes);
  assert.deepEqual(current(s).assets, original.assets);
  oracle(s);
});
test("reload preserves recorded operations, undo/redo create revisions; conflict never overwrites another edit", () => {
  let s = initial(fixtures["30"]);
  s = put(s, { ...current(s).clips[0], start: 480 });
  s = restoreSession(JSON.stringify(s), fixtures["30"]);
  const stale = current(s).revision;
  s = travel(s, stale, "undo");
  assert.equal(current(s).revision, 3);
  assert.equal(current(s).clips[0].start, 0);
  s = travel(s, 3, "redo");
  assert.equal(current(s).clips[0].start, 480);
  const before = clone(s);
  assert.throws(() => travel(s, stale, "undo"), RevisionConflict);
  assert.deepEqual(s, before);
  assert.equal(s.entries.length, 4);
  oracle(s);
});
test("invalid outbound edits and malformed/replayed local state fail with no partial mutation", () => {
  const s = initial(fixtures["100"]),
    before = clone(s),
    c = current(s).clips[0];
  for (const clip of [
    { ...c, start: -1 },
    { ...c, out_tick: 999999 },
    { ...c, fade_in: 4800, fade_out: 1 },
    { ...c, start: Number.MAX_SAFE_INTEGER },
    { ...c, asset_id: "missing" },
  ])
    assert.throws(() => put(s, clip));
  assert.deepEqual(s, before);
  assert.throws(() =>
    validate("clip_put_request", {
      ...request(s, { clip: c }),
      arbitrary: true,
    }),
  );
  const next = put(s, { ...c, start: 480 });
  next.entries[1].project.clips[0].start = 999;
  assert.throws(
    () => restoreSession(JSON.stringify(next), fixtures["100"]),
    /snapshot_mismatch/,
  );
  assert.equal(current(s).blocks.length, 100);
  assert.equal(current(s).clips.length, 200);
});
