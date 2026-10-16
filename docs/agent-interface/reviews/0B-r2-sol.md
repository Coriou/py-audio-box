# 0B round 2 — independent Sol-low re-review

**NO-GO for exact candidate `1a6bdd508b785cbe64200bd5762244494fe76f54`**, branch `t3/bound-timeline-duration`, inspected in `/Users/ben/.txcode/worktrees/py-audio-box/t3-bdf5bf95` on 2026-10-16. R1 is closed. R2's two originally specified inputs are fixed, but one closely related malformed-history case remains. No objections are waived. This is only the bounded 0B component decision; integration and final Astra global review remain separate.

## Required correction: R2 remains partially unresolved — P2

**Location:** `studio/src/main.tsx:24-26`, especially `return raw ? restoreSession(raw, fixtures[id]) : initial(fixtures[id]);`.

The storage boundary treats an existing empty string as an absent key. An empty string is invalid JSON, yet it bypasses `restoreSession` and the new recovery handler. This reproduces the false saved fallback and additionally permits replacing the invalid value without deliberate Reset.

Actual native T3 browser reproduction on this candidate:

1. In the disposable loopback fixture origin, execute `localStorage.setItem('editor-0b:1.3.0:30', '')`.
2. Fully navigate/reload `http://127.0.0.1:5178/`.
3. Observe **Saved locally · no service**, revision 1, no recovery alert, and the key still contains exactly `''`.
4. Click **Clip c0**, press **ArrowRight**. Observe revision 2 and **Saved locally · no service**; the previously empty value is now a 72,721-character history, without Reset.

This violates the correction's invalid-history preservation/explicit recovery behavior. It is a small storage-entry distinction, not a request for production offline drafts or Phase 5 expansion. Distinguish `raw === null` (missing) from any present string; route every present string through restoration. Verify empty-string startup and Reload current show the unsaved recovery alert, retain exact bytes after attempted edits, and recover only through explicit Reset. Also retain the existing missing-key behavior. The new `recoverRaw` tests do not cover this UI boundary because they bypass `read()`.

## R1 closed: bounded extents and usable recovery

Inspected actual correction diff (`HEAD^`/`e43f773` to candidate), `recovery.ts`, tests, adapter/callers and rendering. `requireSupported` executes after canonical edit validation and before storage writes in save and simulated writer. Canonical adapter/contracts remain unchanged. The 60-second ceiling is visibly fixture engineering policy, not a canonical/API/hardware restriction. Rendering independently caps extent, ruler and cursor; no silent timing clamp or data rewrite was found.

Independent actual browser checks:

- Apply starts **9007199254700000**, **1000000000**, **9007199254736191** (safe-max end), and **2875201**: all refused at revision 1 with start 0 and explicit unsupported-fixture message. Separate exact-byte comparison for the original huge input confirmed localStorage unchanged.
- Start **2875200** (exact 2,880,000 end) saved revision 2; ruler contained **601** spans.
- Constructed the original huge history through the unchanged browser-imported canonical adapter, persisted it and set this tab's known revision to 2. Full navigation retained identical stored bytes and exact start **9007199254700000**, displayed the bounded warning, **601** ruler spans and **36,160px** timeline. Actual Undo saved revision 3/start 0. Redo targeting the oversized snapshot was explicitly refused. Recovery controls remained usable.
- A separately adapter-created safe-max-end history with no known tab revision survived full reload; numeric Start reduction to 0 saved revision 3. This covers recovery when foreign/new-tab policy clears Undo. Reset remained usable and was exercised throughout the diagnostic sequence.

The bounded rendering and refusal close the persistent blank-editor/allocation finding. No arbitrary-history length, real-media memory or production performance qualification is inferred.

## Original R2 cases now pass

For **`{broken`** and a replay-invalid history (recorded snapshot start 999 versus request/replay start 480), independently injected exact bytes and fully navigated the page. Both showed explicit **unsaved fixture fallback**, persistent alert, correct invalid-history detail, and no false saved status. Actual Apply reported edits blocked and retained byte-identical storage. Deliberate Reset recovered. Then actual c0 ArrowRight saved start 480; full reload retained 480 and Saved locally; actual Undo saved start 0 at revision 3. The empty-string finding above prevents closing R2 as a whole.

## Verification actually run

Read PRODUCT, IMPLEMENTATION-START, orchestration protocol, canonical plan shared boundaries/Section 3/0B/evidence tiers, CONTRACTS 1.3.0, EDITOR-0B, RESPONSE-0B-r1 and the **full** prior `reviews/0B-r1-sol.md`. No AGENTS.md was found in candidate checkout. Candidate began and ended clean. Inspected package scripts before running; used existing installed dependencies and existing oracle venv, no install/autofix/lock mutation.

**E1**, Node **25.2.1**, npm **11.6.2**:

- `CONTRACT_PYTHON="$PWD/studio/.test-venv/bin/python" npm test --prefix studio`: **8/8 pass**, including unchanged Python contract/schema/edit-semantics oracle and new recovery/extents tests.
- `npm run build --prefix studio`: TypeScript/Vite pass; JS **615.12 kB / 140.99 kB gzip**, existing chunk-size warning.
- `git diff --exit-code e43f773 HEAD -- contracts studio/package-lock.json PRODUCT.md docs/agent-interface/IMPLEMENTATION-PLAN.md docs/agent-interface/IMPLEMENTATION-START.md docs/agent-interface/ORCHESTRATION-PROTOCOL.md`: empty.

**E4**, actual native T3 browser, task-owned loopback Vite via `npm run dev --prefix studio`:

- `preview_status` first returned unavailable. Initial open before server startup returned `net::ERR_CONNECTION_REFUSED`; after starting the documented server, native open succeeded. No alternate browser or standalone Playwright.
- Ordinary long replacement persisted out 9600 with neighbour start 4800 unchanged and overlap exposed. Unadjusted short refused; explicit trim adjustment succeeded. Midpoint split succeeded. Gain 0/fade-in 100 persisted with Silent routing; faded split refused at unchanged revision 5. Speech mute and bed solo persisted their exact flags.
- Simulated second writer advanced storage to revision 8; stale Undo kept UI revision 7/storage 8 and reported revision_conflict. Reload recovered.
- 100-block/200-clip fixture initially mounted **47** clips, **102** ruler spans, **6,190px** timeline. Scrolling reached **c99, bed99 and b99**, with **25** mounted clips. Document width **1280** equalled viewport width.
- Inspected an actual desktop invalid-history screenshot from this run: alert/status and Reset are legible and available, with the incumbent Operate workbench preserved. Inspected retained candidate `correction-corrupt-mobile.png` as supporting narrow-layout evidence; no fresh mobile viewport/touch or screen-reader qualification is claimed.
- Impeccable context ran once; read existing studio surface brief and audit guidance, preserved the existing world with no UI edits. Mechanical detector over `studio/src/main.tsx` returned `[]`. Bounded inspection found no additional recovery-layout blocker; this is not a comprehensive WCAG/theming score or production design approval.

The R1 library decision (custom timeline + WaveSurfer) and safe bounded split/replacement restrictions remain accepted; they were not reopened or silently expanded. Existing localStorage nontransactional CAS, full-snapshot history, synthetic PCM/media limits and later service/pointer/general editing gates remain as documented. No new library-currentness or hardware claim is made. E2 service integration, **E3 runtime/hardware** and **E5 acoustic** acceptance remain unproven.

## Preservation and handoff

Only `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/0B-r2-sol.md` was written. Candidate source, contracts, lock, ledger and root research were untouched. Build outputs are ignored. Disposable fixtures were reset and the diagnostic sessionStorage key removed. Task-owned Vite stopped with Ctrl-C (exit 130); both task-owned preview tabs closed. No commit, push, PR, deployment, backend/runtime/Compose/model/private audio/cloud/STT/Voicebox/remote-e2e or subagents.

Return the one bounded R2 storage-boundary correction for independent recheck. Preserve R1 and ordinary-path evidence. No 0B GO or final Astra acceptance is implied by the passing unit/build/browser subsets.
