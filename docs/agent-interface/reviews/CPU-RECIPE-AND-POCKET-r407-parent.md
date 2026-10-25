# CPU recipe and managed Pocket checkpoint

Source commits: `91b144d` (Pocket namespace/deadline) and `3c2a133`
(public CPU worked recipe). This is a source/test checkpoint, not a release GO.

The Pocket namespace is created by the stage owner and validated by the
allocated engine before use. Its explicit Torch cache location avoids the
passwd lookup for an intentionally unnamed engine UID. Post-reap cleanup
removes only the literal empty namespace. Pocket admission and execution now
have a fixed 180-second ceiling, bounded by the selected profile and original
job deadline; the child also preserves the supervisor's monotonic anchor.
Other engines retain their existing limits.

The public recipe builds exact images, acquires a pinned bundle, measures actual
engine and converter output, then uses the existing fresh export-v2 installer.
It does not promote a profile from source tests. Sol and Astra source reviews
identified permission and subprocess-ownership failures before execution.
Corrections open the readonly source capability before transferring ownership,
make only the nonsecret probe input readable inside a private host directory,
retain the exact subprocess through actual wait despite receipt-write failures,
and allow the setup supervisor to TERM its permanently unprivileged children.
Children verify cleared effective/permitted/inheritable/ambient capabilities.

Bounded, serialized Linux Docker checks (2 CPUs, 1 GiB RAM, no network, readonly
checkout, no capabilities, tmpfs scratch):

- Pocket namespace/engine/daemon/install selection: 114 passed, 86 subtests.
- Pocket deadline and lifecycle selection: 49 passed, 35 subtests.
- Public recipe finite subprocess/context tests: 14 passed.
- Release package tests: 34 passed, 13 subtests.

The first Pocket lifecycle rerun exposed a test fixture race: process-wide fake
filesystem metadata could reach Core's owner thread. The injection is now
restricted to the helper thread; production metadata policy was not relaxed.

Private raw logs are retained under the operator's release-r407 state directory.
The fresh public recipe installation is the next operational gate. Actual
managed Pocket preparation, saved-clone generation, restart, and final broad
regressions remain open. No CUDA/Metal or complete-release qualification is
claimed. No GitHub push was performed.
