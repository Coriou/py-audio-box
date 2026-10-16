import {
  current,
  initial,
  restoreSession,
  type Project,
  type Session,
} from "./adapter";

// Engineering ceiling for this synthetic UI fixture, not a canonical/API limit.
export const FIXTURE_END = 60 * 48000;
export const extent = (s: Session) =>
  current(s).clips.reduce(
    (end, c) => Math.max(end, c.start + (c.out_tick - c.in_tick)),
    0,
  );
export function requireSupported(s: Session) {
  if (extent(s) > FIXTURE_END)
    throw new Error(
      "Fixture extent unsupported: clip end exceeds 2,880,000 ticks (60 seconds). Edit was not saved. Reduce Start or use Undo / Reset fixture. This is a synthetic fixture ceiling, not an API or hardware limit.",
    );
}
export function recover(read: () => Session, fixture: Project) {
  try {
    const session = read();
    return {
      session,
      invalid: false,
      status:
        extent(session) > FIXTURE_END
          ? "Saved history exceeds this fixture’s 60-second ceiling. Timeline is bounded; use Undo, reduce Start, or Reset fixture. Stored timing is unchanged."
          : "Saved locally · no service",
    };
  } catch (error) {
    return {
      session: initial(fixture),
      invalid: true,
      status: `Local history invalid: ${String(error)}. Showing unsaved fixture fallback. Original bytes retained; use Reset fixture to replace them. Edits are blocked until recovery.`,
    };
  }
}
export function recoverRaw(raw: string, fixture: Project) {
  return recover(() => restoreSession(raw, fixture), fixture);
}

// Only a missing key is initial state; every stored string must be validated.
export function readStoredSession(
  storage: Pick<Storage, "getItem">,
  key: string,
  fixture: Project,
): Session {
  const raw = storage.getItem(key);
  return raw === null ? initial(fixture) : restoreSession(raw, fixture);
}
