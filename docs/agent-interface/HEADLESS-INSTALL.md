# Headless install source checkpoint

The installed `speech-box init/up/stop/status/complete` product is **not yet
available**. This checkpoint implements the private waiting-control notification
and pre-owner reader boundary. Do not run the fixture release Compose or
`service.api:app` as an installed service. No installed activation is enabled.

`service.entrypoint.WaitingControl` creates one private 0700 directory in the
host-checked bounded `/tmp` tmpfs. Existing directories refuse. It waits at most
30 seconds without constructing Core, Dispatcher, a DB connection or a listener.
The fixed host exec helper is the only packaged notification operation currently
implemented: `python -m service.entrypoint notify`. Its stdin is a canonical,
closed, nonsecret gate document, at most 16 KiB. It has no path/action argument.
This internal operation is not an HTTP/MCP endpoint or a user installation tool.

The root-held `register_and_notify` adapter first completes the existing protected
registration transaction and all its postbrackets. It retains provenance on the
host, reopens the exact retained registration/provenance and rechecks the live
observation before sending to the same control ID with fixed Docker exec user
10001:10001. Failed or unknown outcomes never become activation success.
Successful notification delivery is not owner/listener readiness.

The projection contains only version, exact registration and ID, configuration
and enrollment digests, provenance digest, and two current process identities.
No key, seed, credential contents, protected receipt or raw host topology crosses
this boundary. The provenance digest is a label, not evidence control can verify
by reading the root-protected provenance. Authority relies on the trusted host
launch channel plus independent copied-DB and live checks.

`checked_waiting_gate` holds the existing control exclusion, reads only a stable
private SQLite copy, validates the split enrollment/configuration and exact
stored registration, rejects recovery obligations, and brackets authenticated
hello, local PID1 identity and notification identity. It reads no receipts or
host authority. The parent observes the host cgroup associated with the exact
PID namespace/start identity; control corroborates that identity and its private
cgroup view `0::/`. Control cannot directly read the host cgroup path from its
private cgroup namespace. These are source adapters, not actual Linux evidence.

The caller still needs an integrated owner startup that repeats its binding
after Core independently acquires exclusion, then reconciles before permitting
admission. This checkpoint deliberately does not change Dispatcher's enrolled
inspect-only guards. Retain failed notification state; no deletion, replacement,
reminting, provisioning or restart repair is performed here.

The remaining host deliverable must be a separately installed `speech-box`
command with the trusted Python interpreter, pinned dependencies, its transitive
`service`, `worker`, `contracts` and `lib` modules/data, and pinned configuration.
A Docker build context or a script requiring a checkout is not that deliverable.
The release image recipe must reuse the pinned Python 3.11 CPU base and existing
Kokoro lock; control-lock compatibility with that interpreter still needs
verification. No dependency hash or image-build receipt has been generated here.

Remaining implementation includes create-only protected authority provisioning,
the finite host command/container lifecycle, installed historical-incarnation
handling, checked Dispatcher activation, validated worker model mapping,
factory-only authenticated API, truthful installed status, graceful reap-aware
stop, and release packaging. Ordinary restart must retain every historical
registration and provenance, reject unresolved old attempts, and bind a fresh
current incarnation. The legacy union currently refuses a changed start time
for a retained container ID; do not remove that guard as a workaround.

Independent source review and separately authorized actual Linux/installed
CPU Kokoro audible-speech qualification remain required. No Docker build/run/
pull/exec, UID transition, mount, listener, model execution or actual restart was
performed for this checkpoint.
