# P1B23 B100 proof recovery ACTUAL r79 Sol

2026-10-24. **One authorized recovery completed; STOP for independent ACTUAL audit. acceptance=false.** Supervisor OS exit0; worker OS exit0; transport_verified=true and content_verified=true. This is the executing test sub-agent's handback, not an independent audit or self-GO. Historical r73 retained=false and node017 FAIL remain unchanged. No campaign/product/release acceptance follows.

## Authority and exact entry

Read full r78/r77, selected future-entry-FINAL.json and launch-boundary.txt before dispatch. Parent explicitly accepted r78 bounded SOURCE+HOST GO and authorized this one operation. Stdlib-only checks verified the exact selected root `/private/tmp/pab-r77-recovery-prep-20261024/selected-READY`, all16 selected length/hash records, exact independent GO markers and manifest equality before use.

- Manifest: `a84c0deaf693b431c73a29a94e2594e4e25446d5793f8a5ac74a99974eef4f49`.
- Bootstrap: `457c959d9405d1e78e212c1f53febf2ae79b17ce41a6284b7806135886c07614`.
- Literal:1439 bytes, `9e594accc9b8945d6eb5515790a55b0ead628e89570e7b25ea077baccc14da4c`.
- Actual accepted r78 report: `2087ef195bf10a45308ba64ce9d088998161b8503c37390c308494b07aaf7766`.

Fresh authority root `/private/tmp/pab-r79-authority-20261024` is0700, created with umask077; all newly created authority files are exclusive0600. `accepted-r78.md` is an exact private copy of the accepted report, digest verified unchanged; original preserved. `approval.json` has exactly the nine r77 required fields, including explicit parent acceptance/one-recovery authorization, acceptance=false, both manifest pins, absolute private report path/digest, verdict GO and synthetic_fixture=false. Approval bytes SHA256 `a80c4e5b7b6b64426fc29135075573874f45be4046a444fc5e1f3f5d5e47ef2b`. `parent-authority.json` binds this parent authority, pins and concrete argv; `preflight.json` retains successful preflight. Dispatch wrapper SHA256 `aa9407d511874b449cc1896844f7edeef26e86f6c177f84a03564fc2119eb517`; its metadata and pinned approval/literal were checked before dispatch.

Invoked ONCE `/usr/local/bin/python3 -I -S -B -c LITERAL ROOT MANIFEST_SHA APPROVALPATH APPROVALSHA BOOTSTRAPSHA`, matching the reviewed argv with only its approval placeholders substituted. LITERAL came from already verified bytes. No mutable bootstrap pathname execution. Wrapper only retained stdout/stderr and OS outcome; no additional watchdog or capture variant. Reviewed tooling owned360s supervisor/345s capture/330s worker deadlines. Whole dispatch3.750s; worker3.376s. Dispatch-result.json records invocations=1 and OS exit0.

## Actual lifecycle and transport

Retained supervisor root `/private/tmp/pab-r77-recovery-supervisor-20261024`; operation root `/private/tmp/pab-r77-recovery-operation-20261024`. No earlier operation root existed at preflight. Selected admission verified candidate HEAD `7a53112d1c6c757fea51b20721b9fc6200dd9bc0`, clean tracked status, provenance and197 AUTH mappings before private operational loading.

Same image `sha256:14b0cf52e7337b449938701c0256fe1f32fe846fa25d62f8d18413a190edbbcb`, volume `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce`, old stopped writer `82ccb71b8e178e06458c894849ed185db6f6ddb5da2d555ff5897fb1ab8dae93` and old archive `ba057c4428408f5367201d7da751c2e4378dfcaa4069b98a2e9a833653864eaa` were retained. Fresh image/volume/full-attacher observations, fixed-name absence, strict Config.Cmd/security/RO mount bindings and created/prestart/stopped states passed selected checks.

Exactly one export helper then, after successful export parsing, one independent hash helper:

- `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce-r74-export`: `e947371bdd44da83f03c8b79d66d0abb879ca0ba0592157bed417b77aa977250`.
- `pab-b-e49b684f-65aa-4deb-b350-39767210c3ce-r74-hash`: `986142ac63791b699d9f1cfcfa64b96e6753af09cd050af6764d864d486b72a8`.

Both create/start mutations are confirmed, starts TERMINAL; local capture exit0, docker wait0 and stopped remote ExitCode0 reconcile. Both helpers remain exited. Final daemon observation completed with all four expected stopped attachers and unchanged image/volume/owner. No suffix collision or retry.

Export stdout10864640 bytes SHA256 `c84612a738cadd873ffb1c4b6b77526912a09891435f3af4f0710d2c99f752f0`; hash stdout52638 bytes `53de6bbd8e2a31b0d90d5b930c589c1fad10a012f0e0b9f32f27ddede4b0e261`. Both stderr empty. Raw stream sizes/hashes independently rechecked against capture receipts; leader_reaped/descriptors_closed=true, group_kill_attempted=false. Selected physical/logical PAX parsing and independent reread-map reconciliation succeeded across212 paths, preserving metadata, raw paths/link targets and regular payload hashes. No extraction occurred.

## Recovered historical evidence

Recovered root intent PENDING and terminal STOP/UNKNOWN ownership, seed/sentinel, selections, results, receipts and stream hashes passed selected reconciliation. Node000–016:17 PASS/exit0 receipts with collected=true, setup/call/teardown passed and accepted=true. Node017 remains STOP/enclosing exit1, pytest_exit1, collected=true, setup passed/call failed/teardown passed, accepted=false, exact selected `tests/service/test_child_wait_guard.py::test_B_directed_isolated_seams[B-S-original74-real-observer]`. Nodes018–099 are absent from the complete map:82 UNRUN, not missing successful receipts.

Bounded stdlib reads of retained export offsets independently hash-checked raw017 stdout4924 bytes (`ea37f1ce4d9c6ca25843e271535b6d1d91cf23b51b596a63eb96b201f903c22b`) and stderr616 bytes (`afea6c3828a0c7804438ad1ac0852317d7b992988a85f1f86e9c51306d202f8d`). No private payload/token is reproduced here. Stdout establishes nested `test_t1_real_blocked_pipe_backpressure_deadline` teardown AssertionError at `<string>:358`, followed by the enclosing assertion “target assertions must complete even for guard negatives”; stderr records `RuntimeError: STOP B pytest exit/collection/phases`.

Source-only AST/text correlation maps that generated line, including its one-line prefix, to `_B_SEAM_PROGRAM`'s assertion that a guard failure ends with `incomplete or invalid armed B contract`. Thus a guard failure existed but did not satisfy that expected suffix. The recovered output does not reveal its exact failure string; deeper native/root cause remains unresolved. Nested34/35 evidence records original-assertions-completed/hooks-restored and actual-child558-timeout, respectively; those steps do not override enclosing FAIL. No test or product code was rerun/imported to diagnose it.

The recovered map proves a103-byte absolute symlink target at `b-controls/017/pytest/test_B_directed_isolated_seams0/proof/pytest/test_t1_real_blocked_pipe_backcurrent`, targeting `/tmp/b-controls/017/pytest/test_B_directed_isolated_seams0/proof/pytest/test_t1_real_blocked_pipe_back0`. It is the sole recovered target exceeding USTAR's100-byte linkname bound and matches the old export's last-directory boundary and linkname-too-long traceback. This establishes the representability offender; it does not explain node017's call failure. Links remained inert.

Historical identity-before/after semantic records match: CPython3.14.8, assertions enabled, optimize0, SQLite3.40.1; library SHA256 `2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8`. These are recovered historical runtime observations, not a new test campaign. Actual helper pre/post observations establish the reviewed Linux image and stopped reader lifecycle separately.

## Publication and preservation reconciliation

Supervisor COMPLETE/error=null, worker_exit0/reaped=true. Worker summary RECOVERY-TRANSPORT-VERIFIED/error=null, transport_verified=true/content_verified=true, final_daemon_observation_complete=true, acceptance=false/lane_released=false; old_retained=false/old_test_failure=true. Independently decoded both524288-byte final slots and verified checksum, canonical JSON/padding and equality to their summary/completion JSON. No unknown.slot exists in either complete path set; no UNKNOWN final publication. supervisor-pending.json is the explicitly superseded FINALIZATION-PENDING precursor, not the final verdict.

- Worker final inventory:266 entries, SHA256 `52be098b86998dbb78744753d331019c2759d0deef871345423b731cce552fa5`; final.slot `ff1ef7e8b7b5c7c516dd8bc38831c52d1723e15afdf941e32e10a2672026f7d0`.
- Combined supervisor inventory:276 entries, SHA256 `7516cbba87fde23a872f7ead19c5bc290418f37731b90ed92aa7dd6b93470ad2`; supervisor final.slot `c72ada2c147a3bc6685731e841675e8ba1961109940e68cfa2415baaa3c160a1`.

Every inventoried length/hash and complete path set reconciled with explicit publication exclusions. Worker reported11669182 host bytes. Worker capture exit0/COMPLETE stdout81 bytes hashes reconcile; stderr empty. Supervisor OS outcome is independently retained in authority dispatch-result.json, rather than inferred from completion-looking JSON.

Old-proof-pre, worker old-proof-post and supervisor old-proof-post agree:105 leaves, unchanged=true, baseline SHA256 `d84e2834457109faebc5b8291316e614657f5e21ee426395b10d1b9f4c79c892`. Independently reconstructed current old-file metadata/hashes and directory identities match that baseline. Lane owner bytes/digest and current-operation inode observation agree pre/post; no lane acquisition/release/replacement.

Post-operation checks again match all16 selected inputs, all8 external provenance hashes and all197 candidate source mappings. Entire preparation inventory matches459 retained files/55 directories, file hashes/metadata and directory identities; inventory `768eeb87a5e34a1d6b5da963d437535bf6314673909a9a8a57ec3fa7d3a9881d` and preparation-final slot `28bf864e6436977b1c28d3d811222c222ca797e9862f15c02a448be273064d76` unchanged. Source admission provides the before check; these retained post checks establish preservation without weakening freeze.

## Stop boundary

Cleanup NONE. All old/new helpers, image, volume, proofs, preparation history and lane remain held. No source edit, commit, delegation, test retry, guard/product/pytest import on Darwin, network/build/pull/provisioning, DB access, extraction, docker cp, fix or follow-on operation. Sole repository write is this report; private authority and selected-tooling operation proofs are retained at the paths above. Fresh independent ACTUAL-receipt audit and parent disposition are required next. No self-acceptance or new GO is issued.
