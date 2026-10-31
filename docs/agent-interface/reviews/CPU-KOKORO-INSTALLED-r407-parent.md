# Actual installed Kokoro CPU journey

Bounded GO for the French Kokoro/export-v2 journey below. This is not a complete
CPU release GO: managed Pocket, cancellation and final broad regressions remain
separate checks. CUDA and Metal remain unqualified.

On 2026-10-25, the public `scripts/cpu-kokoro-export.py` recipe completed from
source `8b2e024` on this Intel Mac's OrbStack Linux/amd64 daemon. It reused only
model downloads that were independently checked against the pinned expected
manifest. Builds, model execution and installation ran serially; no host ML ran.

Finished images:

- Control: `sha256:e97c22a46e65b2dac9c76829ac849de3e3696137a1adf39c0638fa4fe8bc2847`.
- Worker: `sha256:48c681c67deff15018597d552983b83ddf2233403ecc3d16d6cec7d368b093b6`.

The contained engine/converter probe completed with actual exit 0 and no OOM.
The fresh installed service then completed the following canonical operations:

1. Healthy activation and preset registration.
2. Idempotent HTTP speech submission, succeeded/reaped job, PCM16 mono 24 kHz,
   79,360 frames.
3. Idempotent HTTP export submission, succeeded/reaped job, PCM24 stereo 48 kHz,
   158,720 frames.
4. Chunked download with complete SHA256 verification:
   `20e881aec3efdbb4c3f2badfc857de545b2e6f1248d9fc0f6e44d177cea8fb0e`.
5. Checked stop (`stopped`) and same-container restart (`ready`), with both
   `activation_ready` and `ready_for_speech` true afterward.
6. Actual CLI status and exclusive export download, matching the same hash.
7. Actual MCP stdio initialization, tool enumeration, authenticated activation
   status and the retained succeeded speech job after restart. The MCP adapter
   ran under the finished control image's pinned SDK, with no Docker socket
   mounted. It exited normally and its client process was reaped.

Private installation 14 retains configuration, handles, actual IDs, receipts,
logs, downloaded WAVs and MCP records. Credentials are excluded from this report.
Listening/voice-quality assessment is not implied by machine media checks.

Earlier failed trials remain retained. Trial 13 exposed SQLite's implicit empty
temp namespace after `integrity_check`; the owned backup correction passed an
independent source review and 48 tests/10 subtests before trial 14. The fix keeps
the original owner connection and rejects attached databases or nonempty temp
schemas. No live original SQLite files were opened for diagnostic inspection.

The public recipe's latest focused suite passed 16 tests. Actual probe corrections
also preserve Linux-owned sealed copies for macOS bind inputs, separate bounded
diagnostic output from JSON, and continue exact wait/inspection across a TERM/exit
race. Source review alone was not used to declare this installed journey passed.
