# Fresh CPU export staging policy

Scope: source commit `6d80850`, following registrar handoff fix `fa13f15`.
CPU is the release priority; CUDA and Metal qualification are deferred.

The installed control process could not enumerate the legacy staging root
`0:0/1733`. Its integrity monitor correctly refused activation. Fresh
`docker-kokoro-export-split-v2` installations now use `0:10001/1753`: the control
group can enumerate staging through its read-only mount, while isolated executor
groups retain write/traverse access without directory-read access. Legacy policy
and retained roots are unchanged. An old export root is refused, not repaired.

Initialization, public role validation, startup, recovery and reconciliation
select the distinct policy only with the explicit export family. Negative checks
cover malformed records, scalar aliases, policy identity, family mismatches and
incorrect ownership/modes. Generic paths reject familyless export-policy use.

Validation: 152 tests and 92 subtests passed in the focused policy suite; two
additional tests and five subtests passed for final negative cases. Independent
Sol low source review and a batched Astra low review found no source blocker.
These reviews do not establish installed-runtime qualification.

A Linux Docker permission probe used the existing control image, no network,
read-only root and a disposable tmpfs. After clearing supplementary groups and
dropping to the tested identities, UID/GID 10001 could enumerate the new root;
UID/GID 20000 received permission denial. The container was removed automatically.
This proves the tested kernel permissions, not named-volume activation or speech.

Remaining operational gate: build the corrected control image, initialize fresh
volumes, activate the service and complete authenticated API/CLI/MCP speech,
export, replay, cancellation and retained restart checks. Pocket cloning is a
separate installation and requires its own source and actual CPU qualification.

Private test logs and the Linux probe receipt are retained under
`/private/tmp/pab-release-r406-staging-policy-tests4.log` and
`/private/tmp/pab-release-r407-linux-staging-permission.json`.
