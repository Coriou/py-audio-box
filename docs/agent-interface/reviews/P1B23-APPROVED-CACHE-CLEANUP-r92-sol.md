# P1B23 approved cache cleanup r92 Sol

**Finite cleanup complete; space observation clears the preferred 7 GiB level. HOLD remains for lane/build activity; this is not r90 retry authority.**

Read the full `P1B23-DISK-HEADROOM-AUDIT-r91-sol.md`. Parent/user explicitly authorized one cleanup of exactly its six cache trees, including loss of offline browser distributions. No further confirmation was required. Checkout HEAD: `d3f03a79efcc32ec379c520e6869622285830637`.

## Fresh preflight and deletion

Stdlib-only host checks verified all six literal roots were real directories owned by ben, and all inventoried descendants were ben-owned. Fresh content classes and allocations matched r91. The private manifest records each descendant's relative path, device/inode/mode/owner/mtime/ctime, size and allocated blocks without file contents. No symlink traversal was used.

| Exact removed directory | Descendant entries | Allocated bytes |
|---|---:|---:|
| `/Users/ben/Library/Caches/pnpm/v11` | 4245 | 853,966,848 |
| `/Users/ben/Library/Caches/electron/bfa62fe1b54e5e709b522b5f731c7644136b8ab3a8123f8d2facc4ecf2b925fe` | 1 | 134,205,440 |
| `/Users/ben/Library/Caches/pip/http-v2` | 360 | 14,323,712 |
| `/Users/ben/Library/Caches/ms-playwright/chromium-1228` | 653 | 373,129,216 |
| `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1223` | 24 | 204,623,872 |
| `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1228` | 24 | 206,635,008 |

Total allocation: **1,786,884,096 bytes**. pnpm contained metadata, metadata-full and metadata-full-filtered; Electron contained the downloaded ZIP; pip was the HTTP cache; Playwright roots contained matching browser distributions and installation markers.

Each root had a bounded `lsof -nP +D` check during preflight and again before removal: all twelve calls returned 1 with empty stdout and stderr. No errors were suppressed. A bounded PID-only process enumeration plus macOS proc_pidpath resolved actual executable paths for every still-observed process; no unresolved paths or executable within the six roots was found. No command lines or credentials were collected. These are point-in-time observations, not a guarantee against later activity.

Immediately before each removal, the full descendant identity inventory and root identity were compared with preflight. Ancestor components were checked as real directories. Removal pinned parent/root descriptors with O_NOFOLLOW, checked root identity, recursively opened directories with O_NOFOLLOW and matched descriptor identity, unlinked symlinks themselves, and checked directory identity before rmdir. No wildcard, fallback or failed-entry retry was used. This prevents following replacement symlinks outside the trees; concurrent mutation cannot be universally excluded by point-in-time checks.

**Deleted: 6; deferred: 0; errors: 0.** All six exact paths were absent afterward. Immediate excluded siblings retained their preflight device/inode/mode/owner/mtime/ctime identities: Playwright `.links` and `ffmpeg-1011`, pip `selfcheck`, pnpm `lockfile-verified.jsonl`. Electron had no excluded immediate siblings. This is a sibling metadata check, not a recursive preservation audit.

## Observed space and handoff

- Before at 2026-10-24T03:10:24.889981+00:00: **6,265,450,496 available bytes**, f_frsize=4096, f_bavail=1529651, repo statvfs.
- After at 2026-10-24T03:10:28.976412+00:00: **8,054,173,696 available bytes**, f_frsize=4096, f_bavail=1966351, repo statvfs.
- Observed available-byte delta: **+1,788,723,200**.
- After: **7.501034 GiB**, **537,980,928 bytes above 7 GiB (7,516,192,768)** and **1,611,722,752 bytes above 6 GiB (6,442,450,944)**.
- `df -k .` at 2026-10-24T03:10:28.980326+00:00 returned 0, empty stderr: `/dev/disk1s2`, 7,865,404 available 1024-byte blocks (8,054,173,696 bytes), mounted `/System/Volumes/Data`.

The earlier preflight statvfs sample at 2026-10-24T03:10:02.038052+00:00 was 6,269,153,280 bytes; the immediately preceding deletion sample above is used for the actual delta. Allocation is not guaranteed APFS reclaim: snapshots/clones and concurrent allocation affect free space. The observed delta exceeds the measured tree allocation by 1,839,104 bytes, so it is a host free-space change, not an attribution of every byte to this removal.

Private evidence: `/private/tmp/r92-cache-evidence-nd_4qo1k`, created exclusively, mode 0700; all JSON leaves created exclusively, mode 0600. Contains preflight manifest/identities, executable-path observation, UTC measurements, deletion start, six individual outcomes and final checks (nine leaves total). No secrets or file contents were captured.

Only this report was written inside the repository. No source edit, commit, delegation, build, prepare, Docker, pytest, product import, process kill, lane action, old-proof mutation, image/volume removal or fallback cleanup occurred. No package store, npm, uv, Chrome profile, Downloads, Xcode, model, proof/cache volume, interpreter environment, `/private/tmp/pab-*`, or project worktree was deleted. Lane preservation hashes were not needed and no new preservation audit is claimed. Parent owns the next separately authorized build dispatch; no automatic LOCKONLY/build resume occurred.
