# P1B2.3a independent adversarial review — round 2

2026-10-17 · gpt-6.1-sol / low. **GO for bounded B2.3a source/E1 acceptance only** at exact candidate `ee650cc7920d1e237c21c6b795cb4cc6ac4573e5`. R1 F1 and F2 are closed within the assessed envelope; no new concrete blocker was established. Apply remains unconditionally unavailable. This review does not qualify actual-host authentication, installed operation, integration, or public/unattended use.

## Identity, scope and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-c77b484f` read-only at exact HEAD `ee650cc7920d1e237c21c6b795cb4cc6ac4573e5`. Entry candidate status is empty. Actual baseline `aa72485` to HEAD changes exactly `scripts/p1-operator.py`, `service/recovery.py`, `tests/service/test_recovery.py`, and the appended `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Frozen source/tests are `37d2457c73de6712534d237017336b37a9d6531e`; frozen-to-HEAD is handoff-only. `git diff --check aa72485 HEAD` passes.

Independently compared baseline recovery bytes with the current prefix: **131,362 bytes, byte-for-byte unchanged**, SHA-256 `e4206e1a719dfa969fe73962cae8e67f98e6c1d7064dd0193c22e5cfa5ed6e64`. No preparation, kernel/apply wiring, packaging, contract, dependency, ledger or existing-review mutation. `git range-diff 38b6221..05afbaec aa72485..533aa67` reports all three imported patches equal. Correction-only production changes are the conservative storage refusal and the repeated fixed-role submount guard. No applicable ancestor/scoped AGENTS.md was found. Root has preexisting untracked research, preserved.

Read FULL R1 review, including exact reproducer, minimal fixes and provenance; FULL kernel Astra gate and original boundary architecture; FULL candidate RECOVERY-P1B2 original contracts, tranches, corrections, B2.3a response, clock correction and R1 response; P1B22d-r5, orchestration protocol and R1 fault-authority context. Inspected actual producer, content validators, protected transport, registration/proc/hello association, consumer/lifecycle sequence, authenticated runtime adapter and source/test delta. Memory registry supplied only general exact-checkout/evidence-tier guidance, not B2 authority or current acceptance.

## F1 — high — closed within the conservative source/E1 envelope

Locations: `scripts/p1-operator.py:126-142`, `173-192`, `377-428`; regressions `tests/service/test_recovery.py:3207-3245`.

The correction does not claim that lexical realpath, unequal root inodes or unequal devices prove disjoint storage. Exact-name, lexical ancestor/descendant and root-identity matches remain positive consumer detections, followed by strict owner/incarnation/mount validation. **Every remaining bind/volume source raises `unsupported_host_storage_disjointness`.** Missing/unusable source observations also refuse. There is no example-path blocklist, label/hash exception, filename wildcard or guessed separate-filesystem exemption. This intentionally refuses unrelated foreign bind/volume storage as unsupported.

All enumerated containers, including stopped ones, reach this rule. The complete inventories run during preflight and again before any stop, then repeatedly before capsule publication. A known mount short-circuit does not waive extra storage: the complete consumer mount list must still satisfy the exact role/count/type/name/source/mode validator. Unknown consumers detected by a positive match are rejected by owner coverage rather than omitted.

Independently executed the FULL exact R1 embedded Python setup in memory, replacing only its incorrect-acceptance assertions with mandatory specific refusal and zero-stop/no-capsule assertions. Both LIVE and STOPPED `/alias/staging/_data/retained-subdir`, device 7/inode 999 variants now refuse before any stop/wait. Logical SQL and the unknown consumer's running state remain unchanged; no invocation directory is published. Exit 0. This retains R1's modeled Linux stat/inspect boundary and canonicalized temporary-root setup; it is not an actual Linux bind-mount receipt.

A separate 24-case extension varies bind versus volume type, live versus stopped, first versus second inventory, and three sources: exact R1 alias descendant, arbitrary unlisted same-device/different-inode path, and different-device descendant. Every case requires the specific unsupported-disjointness reason, zero stop/wait, no capsule, unchanged SQL and all three retained consumers. All pass, exit 0. Five further independent controls pass: unchanged two-owner capsule; complete retained-registration union including old stopped owners; unknown exact-name consumer refusal; direct root-inode alias refusal; and an extra unproved storage mount on a known owner refusal. The three negatives preserve zero stop/wait/no capsule.

The original consequence—omitting storage access and stopping registered owners before obtaining complete consumer coverage—is closed by refusal rather than an unqualified topology algorithm. No further minimal fix is requested for F1. Broader foreign-storage availability requires a separately reviewed bounded topology/disjointness proof.

## F2 — medium — closed at every repeated observation

Locations: `service/recovery.py:2217-2224`, `2235`, `2262-2264`, `2273`; regressions `tests/service/test_recovery.py:3248-3263`.

`_host_authority_mount_observation` wraps the actual bounded `/proc/self/mountinfo` reader and rejects every mount target beneath `/control/`, `/staging/`, `/receipts/` or `/ipc/` with `host_role_nested_mount`. There is no reviewed submount exception. All THREE topology observations use this guard: initial topology, after protected capsule read, and after content plus role-root identity checks. Initial list equality alone cannot bypass it. Separate fixed proof-mount presence/RO/root/alias checks, owner/mode/link/no-follow/file stability and actual role root/device/inode checks remain.

The exact R1 reproducer's stable foreign writable mount below each of the four roles now receives the specific refusal, with valid unchanged fake RO transport established first and capsule bytes/SQL unchanged. Exit 0. A separate 24-case extension uses the exact managed destinations `/control/core`, `/receipts/journal.json`, `/staging/foreign`, `/ipc/lifecycle.sock` and arbitrary deeper `unlisted/deep/managed.file` destinations, using foreign READ-ONLY mounts as well. Each first appears at observation 1, 2 or 3. Every case refuses at exactly that observation with `host_role_nested_mount`, retaining evidence and SQL. All pass, exit 0. Candidate regressions additionally cover directory destinations and foreign writable mounts with unchanged base identities.

The original consequence—foreign authority files beneath apparently matching role roots—is closed in the assessed launch topology. No further minimal fix is requested for F2. These checks are repeated OS observations, not an atomic mount-namespace lock or protection against a malicious trusted administrator changing mounts between observations. C1 must qualify the pinned launch tuple and maintenance isolation; this lot adds no packaging or administrator-defense claim.

## Preserved boundaries

`recovery.apply:414-415` remains an unconditional `P1B2_apply_unavailable`; host/CLI apply reaches it before request interpretation. The host helpers never invoke the four private kernel continuations. Enrolled activation remains closed. No hash or `actual-host-inspection` string authenticates origin; E1 documents exercise internal validators only.

The producer retains ALL retained-registration union, full all-container enumeration, full container/image/start/finish identities, restart-no/private PID topology and local Linux cgroup/IPC association bracketing authenticated hello. Ambiguous/missing/unknown owners refuse. Desktop/remote and unsupported namespace/cgroup provenance refuse. Fixed context/inspect/stop/wait arguments, shell disabled, bounded stdout/stderr/process/global duration and sanitized subprocess errors remain. Control stop plus successful wait precede worker stop. Container exit 137 is acceptable only with successful wait-command execution. Final stopped/nonrestarting/PID-zero/meaningful-finish validation and repeated daemon/volume/all-consumer/registration checks precede publication. No remove/prune/restart/cleanup path is introduced.

The trusted reader retains the fixed separate RO proof path, protected 0700 directory and 0600 owned regular single-link no-follow bounded stable file. There is no caller UID, force, environment, label/hash or E1 authorization bypass. Independently verified on valid fake RO transport: OS time after expiry raises `host_capsule_expired`; `read_host_capsule(now=issued)` raises TypeError; `apply(capsule)` raises the unconditional unavailable error. Exit 0. Historical content-only original-capsule validation remains distinct from trusted reading.

The 14-failed/18-passed RED, 157-targeted, 850-focused and 1002-passed/18-skipped implementation receipts are attributed claims, not independent pre-correction runs. The fresh final suites below determine current preservation evidence. Passing tests do not by themselves establish either closure; the independent counterexample/converse checks and source algorithm above do.

## Fresh verification

Existing interpreter only; no installs. Bytecode/cache disabled. Focused and full suites run sequentially against unchanged exact candidate source/tests. Independent stdin probes use separate disposable synthetic roots; all Docker/proc/mount/UID boundaries remain explicitly mocked.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Focused result: **850 passed in 546.38s (9:06), exit 0**. This freshly includes all 157 host cases (117 inherited plus 40 correction regressions) and the complete inherited recovery/packaged lifecycle/Admission/Dispatcher population. Full service/release began afterward on the same unchanged exact candidate.

Full result: **1002 passed, 18 skipped in 589.41s (9:49), exit 0**. Sequential after focused verification on the same unchanged source/tests. All 18 skips remain unqualified. The source skip guards are the Linux peer-credential cases in `test_worker_transport.py`: authenticated exchange (10), forged control proof (1), replayed hello (1), and authenticated wrong execute (6), reason `Linux peer credentials required`. No installed or actual-host qualification follows. Both fresh suite populations match the final candidate's claimed counts. No candidate test failed, no source/test edit occurred during or between the suites, and no redundant broad campaign ran.

Probe reproduction: use R1's full embedded `PY` body with its complete fixture setup. Replace the F1 `freeze_host` success block with a try/except requiring `unsupported_host_storage_disjointness`, zero stop/wait, no invocation directory, unchanged SQL and unchanged unknown running state. Replace only the nested-loop reader acceptance assertion with try/except requiring `host_role_nested_mount`, retaining the initial unchanged-transport positive and all byte/SQL assertions. Retain the original expiry/no-caller-clock checks. Executed via `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -` from the exact candidate, extracting/compiling in memory; no candidate edits or saved probe file. The separate extensions use the same fixture and explicitly fake only their described OS boundaries. All probe invocations completed without setup/test failure.

Final HEAD remains exactly `ee650cc7920d1e237c21c6b795cb4cc6ac4573e5`; candidate status is empty. Rechecked exact four-path baseline scope, byte-identical 131,362-byte kernel prefix/hash, handoff-only frozen-to-final delta and `git diff --check aa72485 HEAD`: all pass. Root has no tracked diff; the designated new review is untracked alongside preserved preexisting research. No candidate edit or commit occurred.

## Evidence limits and write boundary

**E1 mocked host evidence is NOT actual-host authentication.** No actual Docker operation, network, other-user proc read, mount operation, audio/model/paid runner, run-e2e-test, install, push/deploy, prune or user cleanup ran. Disposable local fixture evidence is not installed Linux mount/teardown acceptance. No malicious-host-administrator defense, physical ENOSPC/power-loss/daemon/VM/controller durability, clean-host, engine/audio/hardware or public/unattended qualification is inferred.

C1 must still qualify the actual image/entrypoint, host-admin identity, mapped UID 0, compatible NEW-installation ownership model, protected proof mount and exact launcher-created maintenance-consumer exception. No host/container UID equality or chown of existing data is assumed; no packaging is improvised. B2.3b preparation and its conservative existing-high-water footprint proof, B2.3c wiring/resume/startup binding, B3, C1/C2, parent integration and final global Astra remain pending. All 18 installed/platform skips remain unqualified. **A1 global Docker-inventory drift remains unexplained and unqualified; no cause or waiver is inferred.**

Only `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23a-r2-sol.md` is written as a repository artifact. No candidate/source/test edit, commit, subagent, push or deployment. Preexisting root research and other worktrees are preserved.
