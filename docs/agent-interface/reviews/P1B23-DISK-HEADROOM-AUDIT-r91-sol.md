# P1B23 disk headroom audit r91 Sol

2026-10-24. **HOLD remains; no cleanup performed; no build admission GO.** Read the full `P1B23-DIRECTED-IMAGE-BUILD-r90-sol.md`. r90 remains build=0, prepare=0, mutex mutation=0. This audit proposes one finite cache cleanup, covered by the earlier cache allowlist as restated by the user, but does not execute it or authorize a lane retry.

## Current verified host observation

Cwd `/Users/ben/Projects/py-audio-box`; HEAD `bd05e2b59a27a3257154a9890daf37441bb2d216`. Existing untracked STT evidence was observed and left untouched. No applicable ancestor AGENTS.md was found in the checked user/project ancestors; r90 also records its scoped instruction check.

At **2026-10-24T03:07:17.732306Z**, stdlib-only `python3 -I -S -B` `os.statvfs('.')` returned f_frsize=4096, f_bavail=f_bfree=1531811: **6,274,297,856 available bytes** (5.843 GiB). Required 6 GiB = **6,442,450,944 bytes**; current shortfall **168,153,088 bytes**. Earlier `df -k .` returned 6,129,100 available 1024-byte blocks = 6,276,198,400 bytes on `/dev/disk1s2`, mounted `/System/Volumes/Data`. Small changes between observations are real sampling differences.

Read-only `diskutil info /System/Volumes/Data` reported APFS container disk1, total 499,963,174,912 bytes, container free 6,275,256,320 bytes, volume used 478,224,592,896 bytes. Container shared free and per-filesystem statvfs available are distinct observations; neither is a promise of future admission. No purgeable-space amount was obtained or counted. APFS clones/snapshots and concurrent allocation can make measured cache allocation differ from reclaimed container space.

r90's later post-STOP free was **6,276,456,448**, required **6,442,450,944**, short **165,994,496**. Its exact failure-time free remains unknown; neither that later value nor this audit reconstructs it.

## Finite recommended cleanup manifest — recommendation only

All paths below are literal directories owned by ben, with real directory roots. Sizes are `du -sk` allocation multiplied by 1024 (not decimal KB). Remove only these six directory trees in a separately authorized cleanup action, after a fresh no-active-use check; no wildcard or parent-directory removal. Preserve Playwright `.links`, ffmpeg and all other siblings.

| Priority | Exact directory | Allocated bytes | Rebuildability and specific risk |
|---|---|---:|---|
| 1 | `/Users/ben/Library/Caches/pnpm/v11` | 853,966,848 | Only metadata, metadata-full and metadata-full-filtered cache (297168, 127700, 409084 KiB). Registry metadata is fetched again; next resolution needs network and can be slower. This is separate from the pnpm package store and installed environments. |
| 2 | `/Users/ben/Library/Caches/electron/bfa62fe1b54e5e709b522b5f731c7644136b8ab3a8123f8d2facc4ecf2b925fe` | 134,205,440 | Contains downloaded `electron-v44.4.5-darwin-x64.zip`, not an installed application/profile. Re-download required for a future matching install; offline availability is lost. |
| 3 | `/Users/ben/Library/Caches/pip/http-v2` | 14,323,712 | HTTP response/download cache, not site-packages, a venv, configuration or credentials. Future installations may need network. |
| 4 | `/Users/ben/Library/Caches/ms-playwright/chromium-1228` | 373,129,216 | Downloaded browser distribution; reinstall with the matching project's pinned Playwright installer when next needed. Browser automation requiring this revision will fail until restored. |
| 5 | `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1223` | 204,623,872 | Downloaded browser distribution; same restoration requirement. Older revision does not prove unused by a pinned project. |
| 6 | `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1228` | 206,635,008 | Downloaded browser distribution; same restoration requirement. |

**Total measured allocation: 1,786,884,096 bytes (1.664 GiB).** Arithmetic-only projection from the statvfs sample: 8,061,181,952 available bytes (7.508 GiB), **1,618,731,008 bytes (1.508 GiB) above the 6 GiB threshold**. This provides practical planned headroom exceeding 1 GiB, contingent on actual reclamation and competing allocation. The first three entries alone are only 1,002,496,000 bytes and would not provide a full 1 GiB margin above admission; the browser distributions are included for that reason.

**Authority:** the user's earlier selected pnpm/pip/Electron/Playwright cache allowlist clearly covers these identified cache trees. No new category approval is needed for this recommendation. This READONLY task grants no actual cleanup authority, so nothing was removed. If the matching Playwright distributions must remain immediately available/offline, retain them; do not substitute unknown folders or a broader prune. That would leave the proposed >1 GiB margin unmet.

## Age and active-use evidence / limits

Metadata-only traversal (no credential/token file contents) found file mtime ranges in UTC:

- pnpm whole cache: Oct 16 08:27:26 through Oct 20 07:39:31; pip whole cache: Oct 16 09:32:42 through 10:55:25; Electron ZIP: Oct 16 08:31:12.
- Chromium 1228: Oct 16 09:20:41–09:21:05; headless 1228: Oct 16 09:21:04–09:21:05; headless 1223: Oct 20 07:41:42–07:41:44. These are roughly 4–8 days old, not proof of last use.

Whole-cache regular-file inventories: pnpm 3960 files / 853,995,520 allocated file bytes; pip 120 / 14,340,096; Electron 1 / 134,205,440; Playwright 378 / 787,820,544. No regular files with multiple hardlinks were observed in these four trees. Playwright had five symlinks; traversal did not follow directory symlinks. Directory blocks explain differences from du. APFS clone sharing was not established.

One `lsof -nP +D` invocation covering these four cache roots returned no output, exit 1; stderr was suppressed, so this is **no observed open-file match**, not exhaustive proof of inactivity or absence of permission limitations. A bounded process executable-name observation showed active installed Chrome and Electron-based applications elsewhere; it did not establish use of these cached downloads. Do not stop those apps or remove their profiles. Fresh open-file/process checks immediately before cleanup remain necessary; if any manifest entry is in use, defer it without killing processes or widening scope.

## Other bounded observations — excluded from cleanup

- `/Users/ben/Library/pnpm/store`: 8,887,800 KiB (v10 1,381,384; v11 7,506,416; v3 zero). Preserve: package store may support project environments/hardlinks/offline installs; no dependency ownership audit performed.
- `/Users/ben/.npm/_cacache`: 2,542,404 KiB. Content-addressed npm cache, but npm was not explicitly in the supplied prior selected allowlist. Excluded; would require new category authority rather than assuming pnpm authority extends to npm.
- `/Users/ben/.cache/uv`: 263,448 KiB. Excluded: user specified managed UV cleanup; no managed cleanup executed or environment/link analysis attempted.
- Playwright total 769,356 KiB; ffmpeg-1011 3,340 KiB and `.links` preserved. Electron total 131,060 KiB. pip total 14,004 KiB; selfcheck preserved. Trivy cache measured zero.

No Chrome profile/cache, Downloads, models/weights, datasets, package environment, Xcode/simulator or Docker storage audit was needed. No Docker daemon command was issued. No proof/image/volume is classified disposable here, even if formerly called cache.

## Bounded cleanup handoff and postcheck

A later cleanup operator should revalidate the six exact roots (real directories, same owner/content class, no new active use; do not follow replacement symlinks), remeasure the finite manifest, and remove only that manifest if separately authorized. Then obtain a fresh UTC statvfs available-byte measurement at the repo and `df -k`, with APFS container free if needed. Report actual before/after delta; require preferably **at least 7 GiB available (7,516,192,768 bytes)** for a full 1 GiB margin above the 6 GiB admission gate. If reclamation falls short, STOP and report; no automatic fallback deletion or new research campaign. Even a successful space postcheck is not operational permission to retry r90.

Preserve the entire root repo, every `/private/tmp/pab-*` tree and `/Users/ben/.txcode/worktrees/py-audio-box/*`, including all proof/preparation/checkouts. All old Docker images, volumes, objects and lane markers remain preserved. No lane file was opened for mutation, no cleanup, source change, commit, delegation, Docker operation, test, runtime/product import, prepare or build occurred. The only write is this report. Held lane unchanged by this audit; no fresh quiescence or preservation rehash certification is claimed. **STOP / HOLD.**
