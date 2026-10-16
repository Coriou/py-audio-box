# Published editor engine probe

Executed 2026-10-15 on the Intel Mac host, Node v25.2.1. Packages: `@waveform-playlist/engine@13.6.0` and `@waveform-playlist/core@12.6.1`, installed only in a temporary task-owned directory with `--ignore-scripts --no-audit --no-fund`; two packages, no lifecycle scripts. See package-lock.json for registry integrity and result.json for imported distribution hashes.

The 11 checks passed as observations/assertions against the current published headless metadata engine. They cover ingestion/IDs, move, split source intervals, undo/redo, transaction abort, cross-track state, overlap behavior and state ownership. This is not 11 release gates passing. Some checks intentionally confirm contract differences.

Findings requiring integration decisions:

- A same-track move requesting overlap is clamped; loading an existing overlap is accepted. Our server must explicitly allow/reject overlap, not silently accept an edit and change its meaning in the browser.
- setTracks refresh clears local undo. Persistent revision history belongs in the service and a client adapter must handle refresh/rebase deliberately.
- getState copies arrays but shares clip objects; callers need immutable handling/cloning at the boundary.
- Splitting creates fresh client IDs. The authoritative server must reconcile/adopt returned IDs or return its own mapping; unrelated clients cannot assume generated local IDs are canonical.

No AudioContext, decoded audio, browser UI, FFmpeg render, sample playback, accessibility, RPC/CAS, memory workload or model was exercised. Those remain separate qualification gates. The inspected upstream source snapshot is `772328046f7f3f191e82e43b56a4b3025ae417a0`; source and published distributions are separate evidence identities.

Reproduce with an isolated temporary prefix (not the project application): install the two exact packages with scripts disabled, then run `node probe.mjs <prefix> <output-json>`. Never treat a registry current/latest tag as the tested identity.
