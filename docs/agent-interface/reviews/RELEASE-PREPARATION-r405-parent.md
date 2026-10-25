# Release preparation r405 — not ready to publish

2026-10-25. Headless API/CLI/MCP remains the release scope; no web UI.
Source checkpoint: `58171c1`, following `d8b05c8` and `4fc23ab`.

## Verified progress

- Actual Kokoro French CPU synthesis and actual product WAV export passed under
  UID 10001, a read-only filesystem, no network and bounded resources. Exact
  measurements and converter/library identities are retained in
  `../evaluation/cpu-export-r405/engine-qualification.json` and
  `../evaluation/cpu-export-r405/converter-qualification.json`.
- The installed Docker path imported the pinned model and completed split
  initialization on fresh volumes. It has not completed live activation.
- Actual Linux volume testing exposed stale directory descriptor offsets after
  namespace creation. Independent descriptor scans fix the observation while
  retaining exact root identities and create-only ownership transitions. A real
  named-volume probe then reached the deliberate stop immediately before Core
  creation; the subsequent real installer completed Core initialization.
- Docker reports the volume root in `Mounts.Source` for a subpath mount. The
  validator now requires that representation, while still requiring exactly
  `Subpath=core/assets`, read-only/no-copy options and matching live asset inode
  observations. There is no whole-Core fallback mount.
- The full `tests/release` run passed: **522 tests, 147 subtests**. It preceded
  the final directory/subpath/diagnostic fixes; focused checks were run after
  those changes. Package checks after the directory fix passed **59 tests,
  74 subtests**. These are source/package results, not installed qualification.

## Open release blockers

1. Actual installed control activation fails after worker startup and private
   notification. The existing critical-only Uvicorn logging hid the underlying
   lifespan error. A sanitized source-location diagnostic has been added and
   tested; a fresh attempt with that image is still needed. The stopped
   container's generic `SystemExit(3)` is not a diagnosis. The fifth attempt's
   worker was stopped normally; no test job was submitted.
2. The broader service run stopped with **604 passes, 353 passing subtests and
   three failures**: the Pocket public journey, Pocket discovery's new export
   command, and a private-copy assertion. Recovery stopped with **861 passes,
   two skips, two failures and one teardown error**. The frozen-snapshot test
   now targets recapture rather than legitimate private validation, and missing
   retained descriptors have an explicit refusal; their focused checks pass.
   Remaining cases and the full service run are not cleared.
3. Maintenance evidence controls required CPython 3.11 fixes: the real trace
   event supplies a traceback before `exception.__traceback__` is attached;
   pre-PEP-667 frame mutation also requires explicit synchronization in labelled
   test probes. Thirty-three focused P controls passed, then the remaining
   bad-slice probe passed after using the observed event traceback. A complete
   fresh guard run, including the separately held B diagnostics, remains open.
4. Installed API/CLI/MCP generation, download, cancellation, idempotent replay
   and restart have not been qualified end to end. Actual cloning, CUDA and
   Apple Metal are also unqualified. No GitHub publication was performed.

## Resource retention and next attempt

The user authorized deletion of only attempts 1 and 2. Exactly their two stopped
or never-started helpers and 16 named volumes were removed after identity and
consumer checks; private receipts/logs remain. One explicitly rebuildable
Next.js BuildKit cache record was also removed under the existing build-cache
authorization. No broad Docker/image/volume prune was used.

Attempts 3, 4 and 5 remain retained. Their exact volume names and private handle
locations are listed in `../evaluation/cpu-export-r405/verification-summary.json`.
Attempt 3 has no Core or jobs; deleting its Docker resources awaits the user's
separate response. Attempts 4 and 5 have initialized stores and must not be
silently repaired, rebound to another image, restarted through a bypass or
removed. No credential or initialization seed belongs in Git.

The user has a credited Vast.ai account and prefers total qualification spend
at or below USD 5. The shared browser is not authenticated yet. No paid instance
has been rented. No Apple Silicon test machine is currently available.

Next: obtain the actual sanitized startup refusal on a fresh installation,
resolve that cause without relaxing ownership/publication fences, complete the
public installed journey and remaining tests, then build final archives and
checksums. The passing package suite alone does not authorize release.
