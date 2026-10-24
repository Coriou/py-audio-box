# P1B23 managed cache headroom — r110 Sol

2026-10-24. **ONE authorized finite cleanup sweep complete; STOP / HOLD for build and lane activity.** Observed free space clears 6 GiB, but remains below the preferred 7 GiB. No build retry or operational admission is issued.

Read r109/r91/r92 in full. Parent explicitly authorized this cleanup after r109's disk STOP (6,420,115,456 free versus 6,442,450,944 required). Root checkout HEAD observed: `13e6b0a9e4360712ed96c5e216226a8b89d8b488`. No applicable ancestor AGENTS.md was found; no docs AGENTS.md was listed. Memory registry search returned no relevant hit and supplied no technical premise.

## Exact manifest and outcome

The six literal r92 paths were freshly inspected and all remain absent:

- `/Users/ben/Library/Caches/pnpm/v11`
- `/Users/ben/Library/Caches/electron/bfa62fe1b54e5e709b522b5f731c7644136b8ab3a8123f8d2facc4ecf2b925fe`
- `/Users/ben/Library/Caches/pip/http-v2`
- `/Users/ben/Library/Caches/ms-playwright/chromium-1228`
- `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1223`
- `/Users/ben/Library/Caches/ms-playwright/chromium_headless_shell-1228`

Actual macOS Trivy cache `/Users/ben/Library/Caches/trivy` is a real ben-owned empty directory, allocated bytes 0. `/Users/ben/.cache/trivy` was absent (literal ls diagnostic retained in tool output). No Trivy removal was useful or performed. No non-UV deletion was attempted, so active-use/deletion admission for those absent/empty roots was unnecessary. No widened search or fallback cleanup occurred.

`/usr/local/bin/uv cache dir` returned exactly `/Users/ben/.cache/uv`. Local `uv cache clean --help` and `uv cache prune --help` were inspected; clean supports an exact cache directory and native in-use checks. The selected command used neither force nor manual removal:

```text
/usr/local/bin/uv --no-config --offline cache clean --cache-dir /Users/ben/.cache/uv
```

Before execution, all ancestor components were verified real directories, the UV root was ben-owned, and a no-follow metadata inventory checked ownership of descendants and recorded identities/sizes/blocks. Allocation was **269,770,752 bytes**. A fresh bounded PID enumeration resolved executable paths using macOS proc_pidpath: no unresolved live PID and no uv/uvx executable or executable inside the UV cache was observed. No process command lines were captured. Fresh `lsof -nP +D /Users/ben/.cache/uv` returned 1 with empty stdout and unsuppressed empty stderr. Immediately before the managed call, ancestors were rechecked and the full metadata inventory matched. These are point-in-time observations; they cannot exclude every concurrent race. Native UV in-use checks remained enabled.

**Managed command exit 0; stdout empty; exact stderr:**

```text
Clearing cache at: /Users/ben/.cache/uv
Removed 3129 files (249.4MiB)
```

The UV cache root was absent afterward (allocation 0); that removal was performed by UV itself. The managed call was bounded to 60 seconds and completed without timeout. No cleanup failure, deferred occupied cache, retry, tool installation, or manual UV/environment deletion occurred. Six prior roots absent; empty Trivy retained. Individual UV entries are documented by the private before inventory and managed outcome, rather than an unsupported claim of per-entry deletion receipts.

## Measured headroom

| Observation | UTC | statvfs available bytes | df available 1024-byte blocks |
|---|---|---:|---:|
| Before | 2026-10-24T05:06:00.374240+00:00 | 6,462,390,272 | 6,310,928 |
| After | 2026-10-24T05:06:01.724206+00:00 | 6,717,054,976 | 6,559,624 |

Both statvfs samples used the root repo, f_frsize=4096; f_bavail was 1577732 before and 1639906 after. Both `df -k .` calls returned 0, empty stderr, `/dev/disk1s2` mounted at `/System/Volumes/Data`. Samples are sequential, not simultaneous.

- Actual observed available-byte delta: **+254,664,704 bytes**.
- Final available: **6.255745 GiB**.
- Above unchanged 6 GiB gate (6,442,450,944): **274,604,032 bytes**.
- Below preferred 7 GiB (7,516,192,768): **799,137,792 bytes**.

The fresh before sample already exceeded the hard gate as host allocation changed since r109. Cleanup increased observed headroom, but supplies only about 262 MiB above the hard gate, not the preferred full 1 GiB margin. APFS snapshots/clones and concurrent allocations prevent equating inventoried allocation or UV's reported size with actual reclaimed bytes. No assumed APFS reclamation is counted. No remaining eligible space was identified in this finite allowlist; STOP without widening.

## Evidence and preservation boundary

Private evidence: `/private/tmp/r110-cache-evidence-5_c1dlz4`, exclusively created mode 0700, nine exclusively created JSON leaves mode 0600. Contains exact manifest, UV CLI/help, before/after metadata inventories, process executable observation, unsuppressed active-use result, raw managed outcome, and UTC statvfs/df. No large history was copied; no file contents, secrets or command-line credentials were captured.

Only this report was written inside the root repository. All `/private/tmp/pab-*` proof/preparation, all project repos/worktrees, Docker images/containers/volumes, models and environments were left untouched by this task. No preservation rehash is claimed. No source edit, commit, delegation, test, Docker command, build/prepare, lane mutation, process kill, tool installation, parent/wildcard/sibling deletion, npm/package-store cleanup, browser-profile/Chrome cleanup, Downloads/Xcode cleanup, new category, fallback cleanup, threshold reduction or automatic retry occurred. Parent owns any next separately authorized operational decision and fresh preflight. **STOP / HOLD.**
