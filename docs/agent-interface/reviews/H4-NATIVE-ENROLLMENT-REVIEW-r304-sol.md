# Independent bounded native enrollment SOURCE review — r304 Sol

2026-10-25. **NO-GO for completed independent validation/adoption at this handoff:** the required fresh finite execution was not performed because available disk was below the requested 6 GiB gate. Static review found no material security or activation bypass within the expressly bounded slice, and one P3 record-format issue below. This is a validation hold, not a demand to implement deferred Lot B work. Parent owns adoption; no self-adoption occurred.

## Pins and authority

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-b05cd780`, branch `t3/native-enrollment-startup`. Verified clean porcelain status before and after inspection. Exact chain:

- base `a2e07de7ce91d7503adf2d61398b22cb2f48ea3c`;
- SOURCE `8a402214afca47528af8670198f52ab18c8114ff`, direct child of base;
- report HEAD `e37385ea8c433eebd0fa1e4c7759ba942a453f8a`, direct child of SOURCE.

Base-to-SOURCE diff contains only native_package.py, enrollment tests and NATIVE-PACKAGE.md; `git diff --check` passed. SHA256 matches r303: native_package.py `cea1ed22f3515ce02a56842baa3d7e3318f7ec19ed52e3c4c840de6ddd3dbda3`; enrollment tests `81c0cefc0ef9c519244085c7c5feb86d8d0268752f2d96a80480932f321eaee5`; unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known legacy digest RED remains unchanged/unwaived.

Read r303, PRODUCT.md, NATIVE-PACKAGE.md, normative r277 and r296/r302; inspected actual context, runtime, admission and no-follow path consumers, installed startup seams and test fixtures. No candidate AGENTS.md found. Memory supplied evidence-boundary/review conventions only; current technical conclusions use candidate source. No candidate edits, commits, merge, push, delegation, cleanup, network, listener, Docker, installation, download, actual child, ML or crash work. Sole lasting write is this root-checkout report. HELD resources and untracked STT were untouched.

## Finding

**F1 — P3: saved record validation uses value equality rather than exact canonical/type identity.** `service/native_package.py:535–537` and `:540` parse the saved record and compare dictionaries. `document` (`:55–65`) enforces canonical JSON syntax, duplicates and nonfinite rejection, but does not enforce the enrollment record schema. Python considers `True == 1` and `1.0 == 1`. Consequently, after creating a valid enrollment, changing only its canonical JSON `version` from `1` to `true` (or `1.0`) still passes both saved-record comparisons and can return startup inputs after healthy hello. Likewise numerically equivalent float owner device/inode/UID fields can compare equal. The first snapshot's `version` remains integer 1, so there is no later schema check that closes this gap.

Static reproduction recipe: retain all fields of an enrolled canonical record, set `record['version'] = True`, write canonical bytes with the same private ownership/mode, then invoke `native_startup_inputs` with unchanged inputs and the existing fake hello connector. At line 537 the dictionaries compare equal; at line 540 the reloaded malformed record equals saved. This reproduction was reasoned from source, **not executed**. Existing malformed tests cover broken syntax, empty shape and duplicate keys, not this case.

This violates the broad malformed-state refusal claim but does not bypass config/template/trust hashes or substitute another native tuple. Under immutable operator inputs and no hostile same-UID writer, it is a nonblocking format correctness issue, not an attacker enrollment primitive. Compare `canonical(saved)` with `canonical(before)` (and retain final record byte comparison), or explicitly validate closed record fields and exact primitive types. Add one focused bool/float regression when execution is authorized.

## Bounded source assessment

`_enrollment_snapshot:462–501` binds literal owner path/device/inode/UID, control/template document hashes, individual trust byte hashes and exact native identity. Actual NativeQwenContexts (`service/execution_context.py:287–306`) validates registration native/execution/version/profile/hardware/frozen-plan/admission relations; duplicate keys and bounded row counts refuse. finite_budget validates the closed finite budget. No fixture-generated readiness is minted by these functions.

`enroll_native:510–524` brackets the actual enrolled_runtime hello with independently reconstructed snapshots, refuses observed changes, then creates an exclusive no-follow 0600 file and fsyncs file and parent. Existing destinations cannot be overwritten; partial records remain blocking. `native_startup_inputs:534–543` checks retained binding before TLS, reconstructs runtime, and repeats snapshot/record checks afterward. Private no-follow, mode, owner and single-link readers remain in use. Wrong boot/quarantine/uncertain import state refuses through enrolled_runtime/NativeRuntime. TLS trust is reopened by path under the explicit immutable-input/no-hostile-same-UID premise; brackets do not establish hostile-writer/ABA isolation or an atomic filesystem transaction.

No changed source constructs Dispatcher/Core, passes SQLite/Core callbacks into TLS, changes installed policy/history, publishes artifacts or releases reservations. Existing Linux policy/native-history refusal is untouched. NativeRuntime captures its constructing thread, so future integration must construct/use it on the intended owner path; a prepared return dictionary is not an activation credential. Parent must repeat native binding under Core exclusion and supply distinct native history validation before installing it into owner startup. Mutable returned profiles/budget are trusted private integration inputs, not public configuration.

Owner incarnation uses directory device/inode/UID, not an unforgeable lifetime token. Future activation must combine this preparation with existing owner exclusion and installed identity/history checks, and must not extrapolate a filesystem snapshot into process ownership. Boot/trust/config renewal intentionally refuses here and requires stopped reconciliation; current code must not learn a new boot from hello. Supervisor model/registration verification remains independent of control templates. These are precise integration obligations, not current slice blockers.

r302 package metadata/ZIP findings were not reopened: no new package counterexample was established. No installation, ABI, wheel provenance, topology, authenticated real TLS reachability or hardware qualification follows from source metadata or seeded fixtures.

## Independent validation receipt

`df -h /private/tmp` reported **3.0 GiB available** on the shared Data filesystem at entry, below 6 GiB. No space was freed and no retry occurred. I conservatively treated the task's disk gate as applying before the requested tests/probes; therefore **0 independent tests and 0 probes executed**. No Python interpreter was invoked, so I do not claim an independently observed Python version. The authorized interpreter remains `/private/tmp/pab-0c-venv/bin/python -I -B`, expected Python 3.14.7, not target 3.11.

Inspected tests/conftest.py maintenance autouse guard and tests/service/conftest.py listener/subprocess fixture; broad pytest would be inappropriate. Inspected enrollment tests and relevant package/archive/InstalledHost fixture code. The pending exact finite selection is 8 enrollment + 27 package + 20 InstalledHost + 1 archive manifest test via direct unittest with explicit candidate and tests/service paths. r303 reports 56 PASS / 4.833s; that is the implementer's receipt, **not independently reproduced here**. Static diff whitespace verification passed and candidate remained clean.

## Remaining parent gates

Obtain the missing once-only independent finite test receipt under an admitted disk condition (or explicit clarification that these tiny inert fixtures are exempt); assess F1; then separately decide exact SOURCE adoption. No merge or adoption is performed by this review.

Deferred B remains distinct: native owner startup/precheck/post-acquisition binding, native installation/completed-history schema and publication/import-intent obligations, entrypoint/lifespan/credential wiring, topology/mount policy, stopped boot/trust renewal and supported empty-catalogue onboarding. These are not demanded as blockers to this finite enrollment implementation.

Actual Darwin artifact acquisition/install provenance/import/ABI, real TLS/denied peers, model mapping/resources/process/FD/death observations, Apple Silicon probe/load/generate/cancel/restart, clone lifetime/audio/performance, and CPU/CUDA release rows remain parent-owned qualification gates. Native Metal remains UNAVAILABLE / UNQUALIFIED. Headless HTTP/CLI/MCP voice management remains scope; no UI or public STT is added or required.
