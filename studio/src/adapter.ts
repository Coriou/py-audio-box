import Ajv2020 from "ajv/dist/2020.js";
import schema from "../../contracts/v1/schema.json";
import production30 from "../../contracts/fixtures/v1/production-30.json";
import production100 from "../../contracts/fixtures/v1/production-100.json";
// Types derive from the shared fixture, never a competing wire schema.
export type Project = typeof production30;
export type Clip = Project["clips"][number];
export type Track = Project["tracks"][number];
export const fixtures: Record<string, Project> = {
  "30": production30,
  "100": production100,
};
export const clone = <T>(value: T): T => structuredClone(value);
const ajv = new Ajv2020({ strict: false, allErrors: true });
ajv.addSchema(schema, "canonical");
export function validate(kind: string, value: unknown) {
  const check =
    ajv.getSchema(`canonical#/$defs/${kind}`) ??
    ajv.compile({ $ref: `canonical#/$defs/${kind}` });
  if (!check(value))
    throw new Error(
      `invalid_request: ${ajv.errorsText(check.errors).slice(0, 350)}`,
    );
}
function requireValue(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(`invalid_request: ${message}`);
}
export function validateProject(p: Project) {
  validate("project", p);
  const maps = Object.fromEntries(
    [
      "blocks",
      "takes",
      "tracks",
      "assets",
      "speakers",
      "voices",
      "voice_versions",
    ].map((k) => {
      const values = p[k as keyof Project] as { id: string }[];
      requireValue(
        new Set(values.map((x) => x.id)).size === values.length,
        "duplicate_id",
      );
      return [k, new Map(values.map((x) => [x.id, x]))];
    }),
  );
  requireValue(
    new Set(p.clips.map((c) => c.id)).size === p.clips.length,
    "duplicate_id",
  );
  for (const b of p.blocks) {
    requireValue(maps.speakers.has(b.speaker_id), "speaker_missing");
    const t = p.takes.find((t) => t.id === b.selected_take_id);
    requireValue(
      b.selected_take_id === null || t?.block_id === b.id,
      "selection_invalid",
    );
  }
  for (const t of p.takes)
    requireValue(
      maps.blocks.has(t.block_id) &&
        maps.assets.has(t.asset_id) &&
        maps.voice_versions.has(t.voice_version_id),
      "take_reference_missing",
    );
  for (const c of p.clips) {
    const a = p.assets.find((a) => a.id === c.asset_id);
    requireValue(a && maps.tracks.has(c.track_id), "missing_media");
    requireValue(
      c.in_tick < c.out_tick &&
        c.out_tick <= Math.floor((a.frames * 48000) / a.sample_rate + 0.5),
      "invalid_trim",
    );
    requireValue(
      c.fade_in + c.fade_out <= c.out_tick - c.in_tick,
      "invalid_fades",
    );
    requireValue(
      Number.isSafeInteger(c.start + c.out_tick - c.in_tick),
      "tick_overflow",
    );
    requireValue(
      c.take_id === null ||
        p.takes.find((t) => t.id === c.take_id)?.asset_id === c.asset_id,
      "take_asset_mismatch",
    );
  }
}
export type Operation =
  | "clip_put"
  | "track_put"
  | "clip_split"
  | "take_select"
  | "project_restore";
export type Request = Record<string, unknown> & {
  project_id: string;
  expected_revision: number;
};
export class RevisionConflict extends Error {
  constructor(
    public current_revision: number,
    public changed_ids: string[],
  ) {
    super(
      `revision_conflict: current revision ${current_revision}. Reload current before editing.`,
    );
  }
}
export interface Entry {
  project: Project;
  operation: string;
  request: Request | null;
  changed_ids: string[];
}
export interface Session {
  version: 1;
  entries: Entry[];
  undo: number[];
  redo: number[];
}
export function initial(p: Project): Session {
  validateProject(p);
  return {
    version: 1,
    entries: [
      {
        project: clone(p),
        operation: "fixture",
        request: null,
        changed_ids: [],
      },
    ],
    undo: [],
    redo: [],
  };
}
export const current = (s: Session) => s.entries[s.entries.length - 1].project;
export function apply(
  s: Session,
  operation: Operation,
  request: Request,
): Session {
  validate(`${operation}_request`, request);
  const before = current(s);
  if (request.expected_revision !== before.revision)
    throw new RevisionConflict(
      before.revision,
      s.entries
        .filter((e) => e.project.revision > request.expected_revision)
        .flatMap((e) => e.changed_ids)
        .filter((x, i, a) => a.indexOf(x) === i)
        .slice(0, 20),
    );
  requireValue(request.project_id === before.id, "project_mismatch");
  const after = clone(before);
  let ids: string[] = [];
  if (operation === "clip_put") {
    const c = clone(request.clip as Clip);
    requireValue(
      before.clips.some((x) => x.id === c.id),
      "clip_missing",
    );
    after.clips = after.clips.map((x) => (x.id === c.id ? c : x));
    ids = [c.id];
  } else if (operation === "track_put") {
    const t = clone(request.track as Track);
    requireValue(
      before.tracks.some((x) => x.id === t.id),
      "track_missing",
    );
    after.tracks = after.tracks.map((x) => (x.id === t.id ? t : x));
    ids = [t.id];
  } else if (operation === "clip_split") {
    const c = after.clips.find((c) => c.id === request.clip_id);
    requireValue(c, "clip_missing");
    // The pack has one new ID and no fade-distribution fields. Faded split is
    // deliberately refused; see EDITOR-0B contract-gap report.
    requireValue(!c.fade_in && !c.fade_out, "split_fades_need_contract");
    const at = request.at_tick as number;
    requireValue(
      at > c.in_tick && at < c.out_tick,
      "split_must_be_interior_source_tick",
    );
    requireValue(
      !after.clips.some((c) => c.id === request.new_clip_id),
      "duplicate_id",
    );
    after.clips.push({
      ...c,
      id: request.new_clip_id as string,
      start: c.start + at - c.in_tick,
      in_tick: at,
    });
    c.out_tick = at;
    ids = [c.id, request.new_clip_id as string];
  } else if (operation === "take_select") {
    const b = after.blocks.find((b) => b.id === request.block_id),
      t = after.takes.find((t) => t.id === request.take_id);
    requireValue(b && t && t.block_id === b.id, "selection_invalid");
    const affected = after.clips.filter(
      (c) =>
        c.take_id !== null &&
        after.takes.find((t) => t.id === c.take_id)?.block_id === b.id,
    );
    requireValue(
      affected.length === 1,
      "replacement_requires_single_clip_contract",
    );
    const c = affected[0],
      a = after.assets.find((a) => a.id === t.asset_id)!;
    const adjusted = request.adjusted_clip as Clip | null;
    const next = adjusted
      ? clone(adjusted)
      : {
          ...c,
          asset_id: t.asset_id,
          take_id: t.id,
          out_tick: Math.floor((a.frames * 48000) / a.sample_rate + 0.5),
        };
    requireValue(
      next.id === c.id &&
        next.track_id === c.track_id &&
        next.start === c.start &&
        next.take_id === t.id &&
        next.asset_id === t.asset_id,
      "replacement_identity_or_fixed_placement",
    );
    requireValue(
      adjusted || (a.frames * 48000) / a.sample_rate >= c.out_tick,
      "short_take_requires_explicit_trim",
    );
    after.clips = after.clips.map((x) => (x.id === c.id ? next : x));
    b.selected_take_id = t.id;
    ids = [b.id, c.id];
  } else {
    const source = s.entries.find(
      (e) => e.project.revision === request.source_revision,
    );
    requireValue(source, "revision_missing");
    Object.assign(after, clone(source.project));
    ids = ["project"];
  }
  after.revision = before.revision + 1;
  validateProject(after);
  for (const group of ["takes", "assets", "voice_versions"] as const)
    for (const old of before[group])
      requireValue(
        after[group].some(
          (x) => x.id === old.id && JSON.stringify(x) === JSON.stringify(old),
        ),
        "immutable_identity",
      );
  return {
    ...s,
    entries: [
      ...s.entries,
      { project: after, operation, request: clone(request), changed_ids: ids },
    ],
  };
}
export function edit(s: Session, op: Operation, r: Request): Session {
  const next = apply(s, op, r);
  return { ...next, undo: [...s.undo, current(s).revision], redo: [] };
}
export function travel(
  s: Session,
  expected: number,
  way: "undo" | "redo",
): Session {
  const list = s[way];
  requireValue(list.length, "history_empty");
  const target = list[list.length - 1];
  const next = apply(s, "project_restore", {
    project_id: current(s).id,
    expected_revision: expected,
    source_revision: target,
  });
  return {
    ...next,
    [way]: list.slice(0, -1),
    [way === "undo" ? "redo" : "undo"]: [
      ...s[way === "undo" ? "redo" : "undo"],
      current(s).revision,
    ],
  };
}
export function overlaps(p: Project): string[] {
  return p.clips
    .filter((c) =>
      p.clips.some(
        (d) =>
          c.id !== d.id &&
          c.track_id === d.track_id &&
          c.start < d.start + d.out_tick - d.in_tick &&
          d.start < c.start + c.out_tick - c.in_tick,
      ),
    )
    .map((c) => c.id);
}
export function audible(p: Project, c: Clip): boolean {
  const t = p.tracks.find((t) => t.id === c.track_id)!;
  return (
    !c.mute &&
    !t.mute &&
    c.gain !== 0 &&
    (!p.tracks.some((t) => t.solo) || t.solo)
  );
}
// Exact linear multiplier in source-trim-relative 48k samples.
export function fade(c: Clip, i: number): number {
  const n = c.out_tick - c.in_tick;
  return (
    (c.fade_in && i < c.fade_in ? i / c.fade_in : 1) *
    (c.fade_out && i >= n - c.fade_out ? (n - 1 - i) / c.fade_out : 1)
  );
}
export function restoreSession(raw: string, fixture: Project): Session {
  const s = JSON.parse(raw) as Session;
  requireValue(
    s.version === 1 && s.entries.length > 0,
    "local_session_version",
  );
  requireValue(
    JSON.stringify(s.entries[0].project) === JSON.stringify(fixture),
    "fixture_changed",
  );
  let replay = initial(fixture);
  for (const e of s.entries.slice(1)) {
    requireValue(e.request, "history_request_missing");
    replay = apply(replay, e.operation as Operation, e.request);
    requireValue(
      JSON.stringify(current(replay)) === JSON.stringify(e.project),
      "history_snapshot_mismatch",
    );
  }
  requireValue(
    [...s.undo, ...s.redo].every((r) =>
      s.entries.some((e) => e.project.revision === r),
    ),
    "history_revision_missing",
  );
  return s;
}
