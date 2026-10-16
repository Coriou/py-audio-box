# 0B round 1 correction response

Implementation candidate awaiting independent Sol-low recheck; **no self-GO**. No prior objections waived.

- Start: `1018f4a1bdc48e98601a78d4de2c2710bf50fbd5`, clean app-owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-bdf5bf95`, branch `t3/bound-timeline-duration`.
- Requested candidate `a7b7435fa1819a100f33c5f5c08e9d0dd8b2a530` cherry-picked only here as `e43f773`. Final candidate SHA is the commit containing this response, returned in the handoff.
- Correction scope: `studio/src/main.tsx`, new `studio/src/recovery.ts` and `recovery.test.ts`, EDITOR-0B, this response, and `editor-0b/correction-*` evidence. Lock and canonical adapter/contracts unchanged. No ledger/ticket/review/backend/research edits.

## R1

Fixture-only policy rejects resulting clip ends above 60 seconds / 2,880,000 canonical ticks **after canonical validation and before storage writes**, including undo/redo and simulated-writer saves. It never rescales or clamps persisted timing. Error explicitly names unsupported fixture extent, unsaved edit, recovery actions and engineering-only scope. Render duration is independently capped: at most 601 ruler nodes and 72,160px at 200% zoom, with off-window clips omitted and cursor coordinates bounded. There is no duration-proportional huge array/CSS allocation for an existing unsupported history.

Existing valid huge history replays unchanged, shows an explicit bounded-timeline warning, and retains numeric repair, Reset and this tab's existing valid Undo history. Foreign-revision undo clearing remains the prior safety policy. Undo whose target is itself unsupported is refused with Reset/numeric recovery available.

E4 native T3 browser: actual Apply action with start `9007199254700000` leaves revision 1 and a usable page. Injected a canonical adapter-created history with that exact start, full navigation reload showed 601 ruler spans / 36,160px at 100%, exact stored start preserved, enabled Undo; actual Undo restored start 0. Lesser start `1000000000`, safe-max-end start `9007199254736191`, and one-tick-over-ceiling start `2875201` all refused without advancing revision 3. Exact-ceiling start `2875200` saved revision 4. Schema rejection is not counted as fixing this failure: regression tests first construct these histories through the unchanged adapter successfully, then assert the fixture policy refusal.

## R2

Shared startup/reload recovery returns an explicit unsaved fixture fallback on malformed or replay-invalid bytes. A persistent alert remains even when another action changes transient status. Submit blocks saving in recovery. Deliberate Reset replaces the corrupt history; a failed Reset reports its failure. No automatic repair/rewrite occurs during recovery.

E4 full navigation reloads with `{broken` and a snapshot whose recorded start is 999 versus replayed 480 both showed explicit invalid-history/unsaved status. Actual Apply left malformed bytes / invalid snapshot intact and reported edits blocked. Deliberate Reset recovered; focus c0 + ArrowRight saved start 480, full reload retained it, actual Undo saved start 0. The browser report retains bounded observations, not full fixture dumps.

## Verification and evidence

Read PRODUCT, IMPLEMENTATION-START, orchestration protocol, canonical plan including 0B/evidence tiers, CONTRACTS, EDITOR-0B and full independent `reviews/0B-r1-sol.md`. No AGENTS.md found in checkout. Applied Impeccable context once, studio surface brief, harden/new-work/craft-floor; preserved incumbent Operate visual world. Detector returned `[]`. Preexisting design-sidecar drift and unset build-path were reported by context; no out-of-scope repair. Independent visual review remains assigned to the separate reviewer, not self-certified here.

Inspected `studio/package.json` before runs. `npm ci --prefix studio --ignore-scripts`; disposable Python venv installed exact `evaluation/requirements-oracle.txt`. No lock change. Final source tested/built once:

- **E1:** `CONTRACT_PYTHON="$PWD/studio/.test-venv/bin/python" npm test --prefix studio`: **8/8 pass**, including the unchanged canonical Python schema/edit-semantics oracle and new valid-huge/recovery tests. [correction-tests.txt](editor-0b/correction-tests.txt).
- **E1:** `npm run build --prefix studio`: TypeScript/Vite pass; known bundle warning, JS 615.12 kB / gzip 140.99 kB. [correction-build.txt](editor-0b/correction-build.txt).
- **E4:** native `preview_status` → `preview_open`, click/type/press/select/evaluate/navigation/snapshot; no alternate browser or standalone Playwright. [correction-browser.json](editor-0b/correction-browser.json). All injected histories are disposable synthetic localStorage states. Normal 100 fixture scroll reaches b99/bed99 with 25 mounted clips, 102 ruler marks and 6,190px timeline; document width 1280 equals viewport. Corrupt-state narrow document width 390 equals viewport.
- Captures: [huge refusal](editor-0b/correction-huge-refused.png), [100 scroll](editor-0b/correction-100-scroll.png), [corrupt desktop](editor-0b/correction-corrupt-desktop.png), [corrupt narrow](editor-0b/correction-corrupt-mobile.png). Captured through T3 snapshot save; desktop/narrow recovery and huge-refusal images visually inspected. No new design assets.
- Sole task-owned loopback Vite started with `npm run dev --prefix studio` on 127.0.0.1:5178 and stopped after browser verification. No unrelated process stopped.

No models, cloud, private audio, STT, Voicebox, Compose, remote e2e, push, PR, deployment or subagents. E2 service integration, E3 runtime/hardware and E5 listening are not established. Prior library/split/replacement/persistence/storage-CAS/media/production-scale limitations remain. Independent reviewer must rerun both original counterexamples and assess the actual final diff; these implementation observations do not close the review by themselves.
