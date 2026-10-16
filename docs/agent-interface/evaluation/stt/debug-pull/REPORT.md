# whisper.cpp pull diagnosis — 2026-10-15

**The pinned linux/amd64 image is available and verified.** The final pull completed in 234.448 seconds within its 300-second limit, with no retry status. [Final watchdog](final-watchdog.json), [verified image](final-state.json).

The inherited Perl alarm is a confirmed harness defect. The original silent pull's precise cause remains unknown. New transfer evidence shows that 120- and 180-second caps were inadequate at the observed speed, and cancellation discarded the newly downloaded layers. No inference ran here; no model/default, global Docker configuration, or unrelated container was changed.

## Confirmed findings

- `perl -e 'alarm 1; exec @ARGV' docker events` survived its alarm until the external parent terminated it at 3.068 seconds; the same wrapper around `sleep 30` died with SIGALRM at 1.039 seconds. Receipts: [Docker](alarm-docker-events.json), [sleep](alarm-sleep.json). An alarm inherited by this Docker client is not a usable deadline.
- `watchdog.py` owns a direct child in a new session, sends group TERM then KILL after the grace period, directly waits/reaps its child, and inventories the group separately. TERM and forced-KILL fixtures passed: [TERM](repaired-term.json), [KILL](repaired-escalation.json). Late group KILL can return EPERM on this macOS host after the client has exited. EPERM is recorded, never treated as proof of group absence. This helper does not guarantee containment of descendants left after a normal leader exit, nor reap arbitrary descendants or stop daemon work.
- The first real 180-second pull exposed that EPERM edge in an earlier version of this new helper: the client was terminated, but cleanup crashed before writing its final receipt. This deficit is preserved in [failure](first-watchdog-failure.json), with [independent state](first-pull-state.json); no successful final receipt was reconstructed. The helper was repaired and both fixtures rerun before continuation.
- Normal Docker context is OrbStack. Anonymous isolated config selects Docker's default socket; `/var/run/docker.sock` currently resolves to that same OrbStack socket. Both isolated and explicit-socket version calls succeeded. Credential isolation does not currently select a different daemon. The pull itself uses the explicit Engine socket and no credential helper. Config structure/counts were inspected, but no credential values were used or printed.
- Host anonymous token, pinned manifest, config HEAD and largest-layer HEAD all returned 200 in roughly 0.2–0.3 seconds. A [1 MiB blob range](host-blob-range.json) returned 206 in 0.59 seconds. [Registry probes](registry-probes.json), [manifest](manifest.json).
- The pinned manifest declares four compressed layers totaling 504,943,686 bytes. The first new pull streamed 354,578,730 bytes of compressed progress, with three layers reporting Download complete, before the 180-second limit. No retry status was observed. At that throughput a 120-second full-image allowance is too short. Progress counters are not wire-byte measurements.
- The daemon's read-only goroutine snapshot after cancellation showed no pull/download/distribution/xfer frames. A [positive control during continuation](daemon-during-continuation.json) did show those frames. This supports that the first request was no longer active before continuation; it does not reconstruct historical network transfer.
- Host free space at preflight was 42,229,972,992 bytes. The former 8.7 GiB free state and later cleanup do not prove that disk space caused the original stall. The new stream reports no disk error.

## Transfer windows and revised budget

The separately authorized 120-second continuation also expired: 387,809,002 bytes of compressed progress, three Download complete layers, no retry statuses, image absent. Its [watchdog receipt](continuation-watchdog.json) records TERM, direct-child reap, and an independent empty group inventory. Late KILL EPERM is retained as a signal error. [State and post-cancellation daemon snapshot](continuation-state.json).

That stream started all four layers from zero: completed layers from the canceled transaction were not reused. Thus stopping and restarting short windows causes repeated transfer. The measured results support increasing the window, rather than treating the unchanged short deadline as a test of runtime suitability.

After explicit revised authorization, exactly one final 300-second pull of the same pinned image completed successfully in 234.448 seconds, after the preceding client and daemon activity were checked. Its client exited 0 and was reaped; the post-reap group inventory was empty. All four layers completed, with 504,943,686 compressed-progress bytes and no retry status. The maximum authorized client transfer windows were 180 + 120 + 300 seconds, plus bounded termination grace. No further pull was made.

Three full declared manifests plus the one 1 MiB host range total 1,515,879,634 payload bytes, below the revised 1.6 GiB intended-payload ceiling. This is a conservative payload calculation, not measured wire accounting: TLS, HTTP, retries, and buffered in-flight data are not counted. The final client aborts on a retry status or compressed-progress budget crossing. The original failed pull's unknown transfer is separate from these new attempts.

Actual observed compressed-progress totals across the three new requests: 354,578,730 + 387,809,002 + 504,943,686 = **1,247,331,418 bytes**, plus the 1,048,576-byte host probe. These remain progress/body counts, not measured wire bytes.

Verified image ID: `sha256:8149282c7b253d831f9589f546e364f43b6016901eac578719f6f4c5bc63ba11`. Its config digest matches the pinned manifest; RepoDigests contains exactly the requested pinned reference. Architecture is `amd64`, OS `linux`, unpacked image size 1,126,515,224 bytes. No image/model/default promotion was performed.

## Disposition

The original CLI stall is **unresolved as a historical cause**, not evidence against the machine or runtime. Current credential-isolated socket selection, anonymous registry access, and daemon body download all work. Current evidence establishes two setup defects: the inherited-alarm deadline did not work, and repeated short pull windows prevented completion while discarding partial work. The external parent helper plus an evidence-based 300-second allowance obtained the exact pinned image.

The image is ready for the separately coordinated inference diagnostic. The downloaded image alone is not inference, model-quality, or CPU-performance evidence. All new files remain within this directory; historical receipts and `run-cpp.sh` were preserved.
