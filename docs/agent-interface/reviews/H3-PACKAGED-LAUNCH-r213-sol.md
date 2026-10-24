# H3 r213 Sol — LotB private handoff/startup-reader checkpoint

2026-10-24. **PARTIAL SOURCE CHECKPOINT; full LotB and installed activation remain NO-GO.** Working source/tests are retained for independent review. No self-adoption or installed acceptance is claimed. The packaged ordinary command, production activation and clean restart are not delivered by this checkpoint. Parent explicitly directed preservation of this checkpoint and commissioning of the historical-incarnation successor after reading the handoff.

## Exact checkout and pins

Implementation checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-6073a1b5`, branch `t3/packaged-headless-activation`. Supplied worktree was clean at `388f575b856d3c7d941d2e4be9d5c75d596f0d7b`; fast-forwarded without merge/cherry-pick to then-current root `ed36a42d7601edfd0b5a87ce6ea7348300145662`. Worktree was clean after FF. Root's existing untracked STT evidence was inspected only through status and left untouched.

Verified these integrated ancestry pins:

- Registration identity/report `339f09465b4d7aede9441b77d7f09d63f44f704e`.
- Observer bounds/report `d0fe258fb5bb40b5909bd7bab2d39b49bc7d3b48`.
- Split-init path/report `d9feecc3124dba3553f2f25cf3dbe57e01ac1f84`.
- Checkpoint source `c6d71b725052b6e772634d5981e8ef1574db3e0c`, direct parent the FF base above.

Read full normative root r198, including six launch steps, LotB and failure/restart rules; read r212, r208 and r204. No applicable repository/ancestor AGENTS.md found. Bounded memory-registry search returned no relevant hit and supplied no premise. No skill or delegation was used.

## Exact source delta

Source commit changes exactly:

1. New `service/entrypoint.py`: closed notification projection, bounded canonical parser, fixed private-file publisher, finite waiting control, local PID1 observation and bracketed pre-owner gate.
2. `service/host_install.py`: held-authority `register_and_notify`, successful-retained-registration handoff and fixed Docker exec helper. Existing registration implementation remains the writer.
3. `service/recovery.py`: append-only `installed_startup_read`, explicitly selecting SPLIT, with no edits to legacy startup/enrollment/union/writer/maintenance/B2 paths.
4. New `tests/release/test_headless_install.py`: 25 source-only tests using real disposable files/SQLite and injected privilege/host/runtime observations.
5. New `docs/agent-interface/HEADLESS-INSTALL.md`: truthful checkpoint interface, privilege seam and missing product/package scope.

No helper package files were added. No Dispatcher, Core, persistence, engine, auth, API, Compose, Dockerfile, dependency lock or existing test file was changed. There is no `scripts/headless-install.py` in this checkpoint. The source delta is 677 inserted lines across these five files. This report is a separate documentation commit and the sole file copied to root.

## Authorized privilege seam and full gate delta

The parent authorized the reader-only recovery extension, then specifically authorized a bounded public gate projection through the fixed host-coordinator exec channel. Control UID 10001 cannot read protected receipts or provenance; it is never granted that access here.

`HeldAuthority.register_and_notify` first calls the complete existing protected registration transaction. Failure or unknown child outcome prevents the handoff call. Handoff opens the retained provenance with the existing descriptor/metadata/byte identity reader, obtains the healthy private-copy registration snapshot, invokes the existing closed installed-provenance validator, compares the exact selected DB registration/ID and live host/process observation, and checks provenance continuity before publication. It checks provenance and host observation again after helper completion. This returns notification delivery only, not owner readiness. Provenance and DB rows are never erased/rewritten to repair a handoff failure.

The closed projection has exactly `version`, `registration_id`, `registration`, `configuration_digest`, `enrollment_digest`, `provenance_digest`, `processes`. It carries two current process records with exact registered container/image/start and parent-validated host PID/start ticks/PID namespace/cgroup. It contains no key, seed, credential content, protected receipt, authority path or full topology. Canonical bytes and a 16 KiB limit are required. Duplicate/unknown fields, invalid process bindings and malformed digests refuse.

The fixed helper uses `docker --context <closed context> exec --user 10001:10001 -i <exact control ID>` with a fixed isolated Python invocation/import path and bounded stdin. No caller-selected command, file or action is accepted. It writes only `/tmp/speech-box-activation/gate.json` with exclusive/no-follow creation, private owner/mode/single-link validation, fsync and descriptor-to-path identity checks. It has finite communicate/kill/reap handling; unreaped uncertainty cannot return success. Raw helper errors and credential/request contents are not logged.

`WaitingControl` creates its private directory exclusively, keeps its descriptor, refuses stale reuse/replacement/changed mode, and waits at most 30 seconds. Waiting constructs no Core, Dispatcher, DB connection or listener. The notification remains on failure; no cleanup/remint/repair is attempted. File identity and exact bytes become the saved baseline on the first complete checked read and must remain identical thereafter. A partial/malformed notification refuses; there is no retry-over-corruption or overwrite path. Serial publication is tested; concurrent timing/actual filesystem behavior remains unqualified.

`checked_waiting_gate` holds the existing role-checked control exclusion while it captures the current local incarnation, obtains authenticated runtime hello, invokes the installed private-copy reader, and brackets a second hello, local observation and exact notification reread. It is pre-owner only: integration must repeat the relevant binding after Core independently acquires exclusion.

`installed_startup_read` uses the existing stable WAL-aware private copy and SPLIT role validators, query-only SQLite with trusted schema off, bounded schema/row/byte/time checks, and propagated DB/copy teardown failures. It validates exact installation record/digest plus stored scope/revision columns, configuration/digest, control root inode binding, every retained registration's identity/origin, the selected exact registration bytes/ID, configured images and healthy lane/boot hello. Any B2 table presence or pending recovery-plan row refuses. No original SQLite connection or protected receipt/provenance read occurs.

The provenance digest is **not proof** inside control. Authority is the trusted host launch channel plus independently checked DB/live bindings. Under private cgroup namespaces, control sees `0::/`, not the host cgroup path. Parent's checked association links that host cgroup to the exact PID namespace/start ticks; control verifies its own PID1 start ticks/PID namespace and private cgroup view. The host path is validated in the projection and parent observer, not independently reopened from control.

**No production activation gate was removed or enabled.** Legacy synthetic startup/enrollment and all `self.enrolled` guards remain unchanged. The reader/handoff adapters are not reachable through an HTTP/MCP publishing endpoint or untrusted configuration/environment. Existing integrity, orphan, resource, persistence, recovery and reap guards are untouched. This is a reviewable boundary unit, not the single installed-activation predicate or its final integration.

## Concrete restart dependency retained

`recovery.host_registration_union` indexes owners by container ID and rejects a differing retained `started` value with `host_ambiguous_incarnation`. `_register_installed` also rejects a selected changed incarnation for that ID. Therefore the required same-container clean stop/up cannot currently create its next retained registration. No guard was weakened to make the product appear complete.

Parent authorized a **separate installed-only historical-incarnation adapter** in recovery/host_install plus focused tests, while explicitly directing that it not be implemented hastily in this retained checkpoint. The successor must preserve the legacy union exactly, retain complete provenance/history, distinguish immutable historical registration from one currently observable incarnation, and prove old clean terminal/reap obligations rather than infer death from a new StartedAt. Dirty, active, uncertain, quarantined, missing-provenance or unaccounted old ownership must refuse. New boot/start/PID namespace/cgroup and exact volume/config/image bindings require fresh registration brackets. No actual restart authority was granted.

## Checks

Both existing conftests were inspected. No release conftest exists. Only isolated unittest loading was used, with explicit absolute candidate and candidate/tests paths inserted into `sys.path`:

`/private/tmp/pab-0c-venv/bin/python -I -B -c ...`

The required interpreter reports **Python 3.14.7**. These tests do not prove Python 3.11 lock/package compatibility. No dependency hashes or build receipts were invented.

**Final pinned-source result: 141 tests PASS, exit 0, 26.998 seconds** (25 new release tests plus 116 existing LotA tests). `git diff --check` passed. Named modules are exactly:

- `tests.release.test_headless_install`
- `tests.service.test_installed_registration`
- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

The new tests cover actual private-copy SQLite connections, exact enrollment/registration, scope/config/digest/boot/health/image/process/cgroup negatives, recovery namespace/plan refusal, competing control exclusion before hello, finite waiting, exclusive notification/no stale reuse, symlink/hardlink refusal, directory-mode change, byte-identical inode replacement, second-hello mismatch, successful retained host projection and changed/unknown host-record refusal before notify. Privilege/host/namespace/channel observations are injected; no test claims actual Docker, peer or multi-UID proof.

## Missing work and acceptance limits

Remaining LotB implementation: separate historical-incarnation adapter; create-only protected host authority/input provisioning; complete packaged `speech-box init/up/stop/status/complete`; final waiting-control then PID1-worker container coordination and final startup result; validated immutable worker mapping; checked Dispatcher predicate with repeated post-Core binding/reconciliation and preserved resource/orphan/integrity guards; factory-only authenticated API with explicit secret and truthful installed status; clean stop/reap and retained restart; pinned Python 3.11 release image/dependency/build-context work; separately installable host distribution with dependencies/data and pinned configuration, requiring no checkout. Docker context alone is not that distribution. The checkpoint documentation states these absences rather than presenting an incomplete command as usable installation.

No Docker build/run/pull/exec, actual setuid/setgid/chown/mount/network/listener, model download/provisioning/inference, ML/hardware, Linux proc qualification, pytest guard/full suite, maintenance/HELD operation, delegation, self-adoption or unrelated cleanup was performed. Only ordinary test-owned fixture cleanup occurred. The production notification subprocess exists in source but was executed only with an injected Popen in tests.

Independent source/security review is next. Actual installed Linux and audible CPU Kokoro speech, authenticated user journeys, real permissions/peer/mount/process identity, terminal publication/cancellation/reap and restart acceptance remain separately parent-owned. No operational or release GO is inferred from this checkpoint.
