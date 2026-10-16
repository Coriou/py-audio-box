# 0B round 3 — independent Sol-low re-review

**GO for bounded 0B candidate `b62261e4ebb37fd9e0d1840c599b92846fcadb96`**, branch `t3/empty-string-restoration`, reviewed read-only in `/Users/ben/.txcode/worktrees/py-audio-box/t3-badfaf5f` on 2026-10-16. R2 is closed; R1 remains closed. No finding waived and no new blocking finding. This is candidate-level 0B approval only; integration verification and final Astra global review remain separate.

## R2 closure

Inspected the complete small correction and its callers. `studio/src/main.tsx:23-25` calls `readStoredSession`; `studio/src/recovery.ts:46-52` distinguishes only `raw === null` as absence and restores every present string. Startup passes readForTab through recover; Reload current uses the same path. Failed restoration supplies an invalid unsaved fallback, persistent alert and submit guard. The storage-entry test calls the exact exported function used by UI read, with a getItem-shaped storage double, rather than bypassing that boundary through recoverRaw. It covers missing key, empty string, whitespace, JSON null, invalid object, malformed JSON and replay-invalid history. The browser checks below supply the actual UI wiring evidence beyond that unit boundary.

Independent native T3 E4 reproduction on the exact candidate:

- Set `editor-0b:1.3.0:30` to exact empty string and call full `location.reload()`. Snapshot showed revision 1, explicit unsaved fixture fallback, persistent recovery alert and no false Saved status. Actual Clip c0 click + ArrowRight and Apply clip edit retained exactly `''`, revision 1, and reported edits blocked. Actual Reload current followed by Apply again retained exactly `''` and the alert. Actual Reset fixture replaced the empty string and cleared the alert.
- For each of `' '`, `'null'`, `'{}'`, `'{broken'`, inject exact bytes, click Reload current and Apply. Every case showed persistent recovery, blocked editing, and exact string equality after the attempted edit. These supplemental cases used in-page Reload current; the full-reload empty-string check exercises startup independently.
- Construct replay-invalid history through browser-imported canonical adapter: valid move to 480, then alter recorded snapshot start to 999. Reload current and Apply retained byte-identical JSON, persistent alert and blocked-edit status. Original malformed/replay-invalid startup evidence from round 2 is preserved; this round does not relabel that older browser run as fresh evidence.
- Remove the key and fully reload: raw remains null, no alert, ordinary initial status. Actual c0 ArrowRight saves revision 2/start 480; full navigation retains revision 2/start 480; actual Undo saves revision 3/start 0.

The prior empty-string overwrite reproduction no longer succeeds. Deliberate Reset is the exercised recovery action. No production offline-draft or service-store acceptance is implied.

## R1 and ordinary-path preservation

Fresh E4 checks: actual Apply with start `9007199254700000` was refused at revision 1, with exact stored bytes unchanged and explicit fixture-ceiling message. Independently construct a valid canonical history containing that start, store it with known revision 2, and fully navigate: exact start survives, warning remains visible, ruler has **601** spans and timeline width **36,160px**. Actual Undo recovers revision 3/start 0. Full tests retain the lesser huge, safe-max-end and exact-ceiling boundaries. The rendering cap and pre-write policy remain unchanged by this correction.

Fresh 100 fixture check: **47** mounted clips, **102** ruler spans, **6,190px** timeline initially. Actual keyboard edit, Reload current and Undo return revision 3/start 0. Scroll reaches **c99, bed99 and b99**, with **25** mounted clips. The first attempted native select-key sequence stayed on fixture 30; its results were not counted as 100 evidence. A corrected DOM select change event selected 100 before these checks.

The accepted custom timeline + WaveSurfer decision and safe faded-split/split-block-replacement restrictions remain accepted. The latest delta does not touch adapter/media equations or those controls. The full canonical-oracle tests rerun overlap/fixed placement, trim/split, take replacement, gain zero/fades/routing and revision conflict. Prior actual-browser evidence in the full round 1/2 reviews remains supporting evidence for those unchanged paths; no needless rival implementation or new library-currentness claim was made.

## Verification and limits

Read PRODUCT, start/protocol, canonical plan shared architecture/Section 3/0B/evidence tiers, CONTRACTS 1.3.0, EDITOR-0B, both correction responses and both **full** prior Sol reviews. Inspected source recovery/save/load/submit/history/render paths, tests, package scripts and correction diff. No AGENTS.md found in candidate tree. No source/UI polish or repairs.

E1 independently executed using existing dependencies and oracle venv, Node **25.2.1**, npm **11.6.2**:

- `CONTRACT_PYTHON="$PWD/studio/.test-venv/bin/python" npm test --prefix studio`: **8/8 pass**, including canonical Python schema/edit-semantics oracle and both recovery tests. Focused tests are included in this full run; no separate focused invocation claimed.
- `npm run build --prefix studio`: TypeScript/Vite pass; JS **615.16 kB / 141.01 kB gzip**, existing chunk-size warning.
- `git diff --exit-code HEAD^ HEAD -- contracts studio/package-lock.json studio/src/adapter.ts PRODUCT.md docs/agent-interface/IMPLEMENTATION-PLAN.md docs/agent-interface/IMPLEMENTATION-START.md docs/agent-interface/ORCHESTRATION-PROTOCOL.md`: empty. Candidate status clean before and after verification.

E4 used native T3 preview_status first (unavailable), then preview_open successfully, task-owned documented loopback Vite at 127.0.0.1:5178, actual click/type/press/reload/navigation and DOM/storage diagnostics. No alternative browser or standalone Playwright. Snapshot inspected recovery semantics; no fresh screenshot visual, mobile, touch, screen-reader or comprehensive accessibility qualification claimed. The diagnostic server was restarted once to correct the 100-fixture selection check; both owned runs stopped with Ctrl-C/exit 130, both owned tabs closed, fixtures reset and known revision keys removed.

E2 service integration, E3 runtime/hardware and E5 acoustic acceptance remain unproven. Retain documented nontransactional localStorage CAS, full-snapshot history growth, tiny synthetic media/60-second preview, no general pointer editing and later service/general arrangement/production-scale gates. These known bounded limitations do not reopen later phases or convert this fixture into release acceptance.

## Preservation and handoff

Only this report was written in the root checkout. Candidate source/contracts/lock/ledger remain untouched and clean; root research and pre-existing STT artifacts preserved. No install/autofix, commit, backend/Compose/model/private audio/cloud/STT/Voicebox/remote-e2e, push/PR/deployment or subagents. Owner may proceed to integration verification of this exact 0B candidate; final Astra global decision is separate.
