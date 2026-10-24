# Approved cache headroom — r236 sol

**NO-GO: available host space remains below 6 GiB.** Bounded authorized cache cleanup completed; no scope expansion. Observed 2026-10-24T18:52:54.807948+00:00.

Read `docs/agent-interface/reviews/CPU-SMOKE-PREFLIGHT-r234-sol.md` (the requested preflight is under reviews, not repository root). Required floor: 6,442,450,944 bytes. Initial `df -k .`: 4,029,540 KiB = 4,126,248,960 bytes available. Immediately before cache mutation audit: 4,122,652,672 bytes. Final available: **4,117,741,568 bytes (3.834946 GiB)**. Remaining need: **2,324,709,376 bytes**. Net free-space change versus initial df: -8,507,392 bytes. Concurrent filesystem activity means this delta is not attributable cache recovery.

## Commands and retained results

- `uv cache prune --help`: supported managed prune removes unreachable objects; `--force` ignores in-use checks and was deliberately not used.
- `uv cache dir`: `/Users/ben/.cache/uv`; `uv --version`: 0.11.15.
- `uv cache prune --offline --no-config --no-python-downloads --cache-dir /Users/ben/.cache/uv`: exit 0; stdout/stderr: `Pruning cache at: /Users/ben/.cache/uv` / `No unused entries found`. **Tool-reported recovered bytes: zero.** Post-prune df: 4,025,844 KiB = 4,122,464,256 bytes available.
- `/usr/local/bin/python3 -m pip cache info`: recognized HTTP cache `/Users/ben/Library/Caches/pip/http-v2` and older `http`, both zero HTTP files / zero bytes; wheels zero. No purge needed.
- `COREPACK_ENABLE_NETWORK=0 pnpm store path --offline`: rejected unknown `offline` option; read-only discovery corrected to `COREPACK_ENABLE_NETWORK=0 pnpm store path`, which returned `/Users/ben/Library/pnpm/store/v10`.
- `COREPACK_ENABLE_NETWORK=0 pnpm help store`: standard prune removes unreferenced packages; force not used. No pnpm mutation.
- `lsof -nP +d /Users/ben/Library/pnpm/store/v10`: exit 1, no open root-directory entries reported; this is not recursive proof of inactivity.

## Bounded audit and skips

No-follow `lstat` and `os.walk(..., followlinks=False)` accounting, capped at 300,000 entries / 15 seconds per root; all scans completed within caps. Allocated byte totals include each directory entry's blocks and are not unique physical reclaim estimates. All inspected roots owned by uid 501, directory mode 0755, not symlinks; no foreign-owned descendants. Group/other write absent. Symlinks were counted, not traversed. No direct deletion was performed.

- `/Users/ben/.cache/uv`: 51,638,272 allocated bytes, 1,182 entries, two symlinks. Standard managed prune only; no force, CI mode, environment removal or package deletion.
- `/Users/ben/Library/Caches/pip`: 16,384 allocated bytes, five entries, no symlinks; empty recognized download cache, skipped.
- `/Users/ben/Library/pnpm/store/v10`: 1,414,537,216 allocated bytes, 58,241 entries, 58 symlinks. Temporary `tmp/_tmp_93271_6b563e1a17d224a4e7a01e88839fec67/pnpm-lock.yaml` observed. No active pnpm process found, but lock/inactivity clearance incomplete; prune skipped conservatively. Package names containing “lock” were not treated as operational locks.
- `/Users/ben/Library/Caches/ms-playwright`: 3,432,448 allocated bytes, nine entries, no symlinks; visible `ffmpeg-1011` directory. Staleness and non-use not established; skipped.
- `/Users/ben/Library/Caches/trivy` and `/Users/ben/Library/Caches/electron`: empty, zero allocated descendant bytes; skipped.
- Chrome skipped entirely. No active browser profile/current state touched.

Narrow process-name checks found no running uv, pip, Python installer, pnpm, Playwright or Trivy process; existing application Node/Electron processes remained running. No process environment or secrets read. `/private/tmp/pab-0c-venv` remained owned by uid 501, nonsymlink; interpreter checks before and after cleanup returned Python 3.14.7 (final exit 0, output `Python 3.14.7`). Installed venv/tools untouched; only uv's supported cache operation ran, retaining in-use checks.

No Docker commands/operations, network, installation, build, model import/inference, HELD access, model/research/STT/proof/worktree/archive/source-output deletion, simulator cleanup, broad home cleanup, direct rm, delegation or commit. Sole report write: this file, mode 0600. Stop here; headroom admission remains blocked and no runtime authority is granted.
