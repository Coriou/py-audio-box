import { test } from "node:test";
import assert from "node:assert/strict";
import { fixtures, initial, edit, current, travel } from "./adapter";
import { FIXTURE_END, extent, requireSupported, recoverRaw, readStoredSession, recover } from "./recovery";
const fixture = fixtures["30"];
function move(start: number) {
  const s = initial(fixture);
  return edit(s, "clip_put", {
    project_id: fixture.id,
    expected_revision: 1,
    clip: { ...fixture.clips[0], start },
  });
}
test("schema-valid huge timings are refused by fixture policy; canonical history survives and undo recovers", () => {
  for (const start of [9007199254700000, 1e9, Number.MAX_SAFE_INTEGER - 4800]) {
    const s = move(start); // Must pass canonical validation, not count schema errors as a fix.
    const bytes = JSON.stringify(s);
    assert.throws(() => requireSupported(s), /Fixture extent unsupported/);
    const recovered = recoverRaw(bytes, fixture);
    assert.equal(recovered.invalid, false);
    assert.match(recovered.status, /Timeline is bounded/);
    assert.equal(current(recovered.session).clips[0].start, start);
    assert.equal(JSON.stringify(s), bytes);
    const undone = travel(recovered.session, 2, "undo");
    requireSupported(undone);
    assert.equal(current(undone).clips[0].start, 0);
  }
  requireSupported(move(FIXTURE_END - 4800));
  assert.throws(
    () => requireSupported(move(FIXTURE_END - 4799)),
    /unsupported/,
  );
  assert.equal(extent(move(FIXTURE_END - 4800)), FIXTURE_END);
  requireSupported(initial(fixtures["100"]));
});
test("startup exposes malformed JSON and replay-invalid snapshot without rewriting bytes", () => {
  const s = move(480);
  s.entries[1].project.clips[0].start = 999;
  const key = "editor-0b:1.3.0:30";
  const storage = new Map<string, string>();
  const entry = { getItem: (name: string) => storage.get(name) ?? null };
  const read = () => readStoredSession(entry, key, fixture);
  assert.deepEqual(read(), initial(fixture)); // Legitimate missing-key startup.
  for (const raw of ["", " ", "null", "{}", "{broken", JSON.stringify(s)]) {
    storage.set(key, raw);
    assert.throws(read); // Exercise the same entry boundary used by startup/reload/edit.
    const recovered = recover(read, fixture);
    assert.equal(recovered.invalid, true);
    assert.match(recovered.status, /unsaved fixture fallback/);
    assert.match(recovered.status, /Original bytes retained/);
    assert.equal(current(recovered.session).revision, 1);
    assert.equal(storage.get(key), raw);
  }
  const restored = recoverRaw(JSON.stringify(move(480)), fixture);
  assert.equal(restored.invalid, false);
  assert.equal(current(restored.session).clips[0].start, 480);
  assert.equal(current(travel(restored.session, 2, "undo")).clips[0].start, 0);
});
