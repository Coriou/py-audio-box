# CPU quarantined stop and artifact response source gate r407

2026-10-25. Source commits811ae09 andc8c48d4. Independent Sol low and Astra low SOURCE GO for both lots. No release authority is implied.

Quarantined stop retains startup incarnation/gate/registration validation, checks the current worker/control observation before TERM, waits control exit0 then worker exit0, and checks terminal bound reaped receipts under the lane lock. It writes no clean-stop receipt, clears no faults/reservations, and grants no restart authority. Healthy clean-history verification now occurs after both containers have exited. An unhealthy worker hello still refuses before TERM.

Bounded Linux tests:131 tests106 subtests passed in137.16 seconds, covering daemon volumes, export daemon, export owner activation and package readiness reader. One CPU,1GiB memory,network disabled,readonly checkout; no host pytest.

Artifact reads alone allow64KiB; all other production responses remain12KiB and asset metadata2KiB. Both immutable converter registrations (each already bounded16KiB) remain present. Core/ASGI full214-path converter closure, >64KiB rejection and unchanged jobs response limits are covered.23 audio-export tests passed in18.79 seconds under the same bounded Linux test environment.

Operational evidence: fresh install11 source6454292 generated audio and exported48kHz stereo successfully; job receipts report completed/succeeded/reaped. Public artifact_get then refused its valid metadata at the former12KiB limit. It remains operationally unqualified until a fresh corrected image completes retrieval/download and restart.

A later direct read-only SQLite inspection overlapped an unexpected control SIGBUS exit135. Exact stopped-container evidence preserved privately shows a three-byte SQLite SHM file and no OOM. A fresh isolated SQLite3.40.1 probe confirms raw open/read/close of the owner process database/WAL/SHM removes its POSIX locks. Sequential two-process probes survived; SIGBUS has not been reproduced. The active-owner postcheck raw snapshot is a confirmed lock-lifetime defect and is being corrected separately. Install11 is retained; worker exited0 after bounded exact-ID TERM. No restart, repair or clean-stop receipt was fabricated.
