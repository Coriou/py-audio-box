# Installed owner snapshot lock preservation r407

2026-10-25. Source bccf992. Independent Sol low and Astra low SOURCE GO; Sol also reviewed the private destination page cap and portable filesystem fixture follow-up.

Confirmed defect: installed postcheck raw-copied DB/WAL/SHM after Core opened SQLite. A fresh isolated probe in the actual control image with SQLite3.40.1 showed both POSIX locks present before raw open/read/close, and none afterward. This matches SQLite documentation on [same-process descriptor close and POSIX locks](https://www.sqlite.org/howtocorrupt.html#_posix_advisory_locks_canceled_by_a_separate_thread_doing_close_). The earlier observed SIGBUS race remains an inference: sequential two-process probes survived and did not reproduce that signal.

Postcheck now supplies the existing owner connection to the shared startup reader. The owned route uses SQLite backup into a private snapshot and never raw-opens/closes original SQLite files. It requires an exact SQLite connection, sole canonical main database, no active transaction, stat-only source identity/ownership/namespace checks, aggregate physical and logical16MiB limits, 64-page steps, a two-second deadline and private destination page cap. Existing read-only validation and activation binding checks remain downstream; pre-owner/offline copying is unchanged. Failure leaves the owner connection and lifetime exclusion intact.

Bounded Linux suite:111 tests57 subtests passed in125.32 seconds. Selection: owned startup snapshot, installed activation, export retained history, installed registration and installed host policy. Includes prohibited original descriptor reads, actual Linux lock preservation, identity/size/deadline refusal and continued owner usability after failure. One CPU,1GiB memory,network disabled,readonly checkout; no host pytest.

The initial Linux fixture failure was a real synthetic filesystem mismatch between /private/tmp tmpfs and /tmp. Fixtures now co-locate their input trees and split roots; production device validation is unchanged.

Installed qualification on fresh images containing this fix remains required. Prior failed installations and their databases were retained without repair or adoption. No direct SQLite inspection of an active installation should be used during the remaining operational checks; use the authenticated API and fixed observation protocol.
