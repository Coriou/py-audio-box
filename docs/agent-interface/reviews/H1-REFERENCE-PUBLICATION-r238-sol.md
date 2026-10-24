# H1 reference publication — r238 Sol implementation

2026-10-24. Implementation ready for independent review; **no self-adoption, actual Pocket adapter, engine qualification or release approval**. This is a synthetic domain/publication migration to the independently reviewed reference-roles-v1 contract.

## Pins and authority

Owning task worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-af525d56`, branch `t3/reference-roles-publication`, was clean at entry `04df5036f2589fb10f39d7bdbe0768628afa9842`. Fast-forwarded cleanly to then-current root **`17df4a437a690512f3bb32017b1f9510da04bd49`**, containing reviewed contract report `3879e3e2a630b2020219c4f882ea2670a50461f9`. Source commit: **`41460ba3ae37736fdf11de7854f7b720f61df974`**. This report is a separate subsequent commit. Root advanced independently to `aa0c8586a353c470b406ed6f04a553645c11203e` with the two authorization notes; no adoption/rebase was performed. Sole root write is a byte-identical copy of this report.

Read r188 blocker 2/assignment 3, r235, r237, PRODUCT, implementation-plan preservation/freeze provisions, actual contracts/oracle, production/worker/Core publication, reconciliation and recovery consumers. No applicable AGENTS.md found. Bounded memory registry search found no relevant entry and supplied no implementation premise.

Parent explicitly extended the allowlist for `service/persistence.py` and then minimal `service/recovery.py` shared decoding. No DDL, migration, schema identity, installer activation, public API/client contract or canonical fixture edit. No delegation.

Exact source files (10):

- `service/production.py`, `service/core.py`, `service/persistence.py`, `service/reconciliation.py`, `service/recovery.py`.
- `worker/synthetic.py`, `worker/isolated_synthetic.py`.
- `tests/service/test_reference_publication.py`, `tests/service/test_production_api.py`, `tests/worker/test_production_synthetic.py`.

`service/execution_context.py`, runtime protocol/transport and actual engine files needed no edit. The API test change migrates its inert producer and expected reusable dependency; no API implementation changed.

## One-output inventory and implementation

Before edits, reported Core's single-output assumptions in `_verified` cardinality/file selection, artifact membership/hash map, `_publish` adoption intent/file adoption, and `_publication_barrier` restart validation. Additional inspection established the scalar single-row `publication_work` dependency and bare-artifact readers in reconciliation and detached recovery. An initial attempted array intent failed the existing integer CHECK in isolated tests; it was replaced, without altering the CHECK or DDL, by the explicitly authorized typed JSON binding below.

Production allocates stable, distinct reference/audition IDs from principal/idempotency key/exact request and freezes both plus exact audition text in the hashed plan. Trusted supplied audition text is preserved; absent text defaults explicitly to `Synthetic fixture audition.` Original uploaded metadata is resolved through the existing principal-owned asset path. The original request segment and reference transcript remain exact and appear in the candidate's prepared-reference role. Save still selects a committed candidate and retains the prepared reference in its immutable version/configuration. Generation resolves that saved version and only its reference dependencies; it removes preparation-only roles from the new speech plan.

Both synthetic workers use one bounded fixture builder and return the two typed role assets. Fixed owned staging files are `reference.wav` and `audition.wav`; no caller-selected filename or generic output descriptor was introduced. The 48kHz reference fixture has exactly `out_tick - in_tick` frames (including nonzero segment starts), PCM16 mono, with a distinct low-amplitude PCM marker. Audition is the historical 4,800-frame silence fixture. Their hashes differ. Reference fixtures are limited to 480,000 ticks/10 seconds and the three explicit synthetic profile namespaces. This is **synthetic PCM fixture audio, not source decoding, cleaning, resampling, recognized speech, cloning or spoken audition text**. The prepared role retains required original ID/hash/kind provenance association; that association is not acoustic transformation evidence. Audition provenance explicitly remains synthetic. No real profile can use this fixture builder's prepared-reference path.

Core accepts only the closed ordered prepared-reference pair for role-bearing prepare, or the existing single `output.wav` path. It measures each actual staged file's full hash, frames, rate, channels and PCM subtype, validates ownership/no-follow/link/size fences, and verifies complete synthetic fixture bytes before the existing publication oracle. Both outputs must match the frozen roles and exact manifest. Actual Kokoro remains the existing one-output 24kHz generated policy, bounds and provenance; `_verified` retains its existing bytes return for single-output callers. No actual Kokoro execution was performed in this lot.

Design's deliberate generated-reference rule remains unchanged in reviewed contracts and fixtures: same generated reference/audition, actual declared spoken text and design lineage. This lot adds no design runtime route and does not apply prepare's two-file rule to design. Existing historical synthetic plans and single-output intents retain their original representation; no stored plan/voice/fixture was rewritten.

## Durable pair intent and recovery

Existing `publication_work` table/columns, scalar constraints, P1 revision and canonical schema identity remain unchanged. Legacy binding remains the exact bare artifact. For the new pair only, `binding` contains the closed object `{version: prepared-reference-adoption-v1, artifact, manifest, files}`. `files` is exactly two ordered entries containing literal destination ID, unique adoption alias, SHA-256 and byte length. Scalar destination/temporary/hash/length remain literal first-file values and must corroborate the first entry. They are never arrays, overloaded hashes or encoded lengths.

`persistence.publication_intent` and `publication_record` share closed validation across Core, current reconciliation, and detached recovery documents/reference enumeration. They reject unknown versions, extra/missing entries/keys, duplicate IDs/aliases, wrong order, role conflation, hash/length/scalar mismatches, and artifact/manifest/job-attempt/frozen-role mismatch. The pair length rule matches the closed PCM16 WAV fixture. Recovery's existing raw binding digests retain the complete new object; no partial decoder bypass or DDL snapshot upgrade was needed.

The complete validated pair intent, manifest and artifact commit before any file creation. Each file is written completely and fsynced, linked without replacement, then verified through the existing alias/directory barriers. Both files must pass before the single existing metadata transaction inserts either asset, the artifact and success. Second-file failures cannot expose the first asset through published metadata. Filesystem ambiguity retains the original intent and names. A complete linked pair can recover through the existing barrier; an incomplete pair fails without metadata publication. No missing link is reconstructed and no old intent is repaired or cleaned. Existing exact redundant-alias completion remains unchanged. Commit uncertainty reads durable terminal state and preserves idempotent history. No new attempt/deadline/reap or reservation lifecycle was introduced.

Reconciliation follows both explicit destinations/aliases and recognizes the two staging basenames only under an owning frozen prepared-reference plan. Legacy `output.wav` handling remains. The expanded in-memory work entries retain the original artifact ownership binding.

## Executed checks

All checks used `/private/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and owning-checkout imports. Inspected `tests/conftest.py` and `tests/service/conftest.py`; no pytest collection, guard, listener fixture or full suite ran. Direct pytest-style calls used only assertion/MonkeyPatch helpers, not pytest execution.

- New `test_reference_publication.py`: **7 methods PASS**, including loops over before-adoption, each file/write/link boundary, after-adoption and after-commit exceptions/restart; each file write/link/fsync failure; metadata COMMIT error before/after actual commit; no half metadata, idempotent completion/history; legacy literal intent/detached snapshot/restart; closed new intent/current detached document tampering; partial/hash-corrupt first and second media, symlink/foreign provenance, duplicate publication; exact independently frozen transcript/audition text and prepared-reference save.
- Existing `test_production*.py` unittest discovery: **13 methods PASS**. Includes owned upload/prepare/select/save/generate, API inert producer, idempotency, stale candidate, missing/foreign reference, revision conflicts, restart reconciliation and saved-reference reuse. Migrated API assertion explicitly requires the prepared reference rather than audition.
- `test_production_synthetic.py`: **3 methods PASS**, covering legacy proposal compatibility, distinct role assets/exact segment frames and transcript/original, and actual isolated synthetic runner with injected transport writing only the planned pair. No sandbox/runtime process or engine is invoked by that isolated-runner test.
- Reviewed reference-role unittest: **7 methods PASS**, including generated-reference design and subsequent saved-version generation semantics.
- Inspected `/private/tmp/r235-contract-wrapper.py`, then ran it with `PYTHONPATH=.`: **144 direct contract cases PASS**, including actual canonical generator/determinism and producer/oracle boundaries. No canonical bytes changed.
- `/private/tmp/r238-persistence-wrapper.py`: **18 direct existing legacy persistence cases PASS**, expanding selected write/alias/barrier matrices and covering original intent, checkpoint/WAL, write-denied recovery, historical reads and inventory. Explicitly excludes process-death/crash tests. Final run followed final Core changes.
- `git diff --check`: PASS. Source worktree clean after source commit.

Total **192 executed cases/methods**, with additional mutation/fault loops inside methods; not 192 separately enumerated assertions. Earlier fixture-only failures (integer CHECK, stale audition assertion/staging inventory names, incomplete injected profile readiness, test DB property injection and test transcript/config mismatch) were corrected and the affected checks rerun successfully. Contract/source checks do not establish runtime containment or acoustics.

No actual Docker/model/inference/network/provisioning, ML/hardware, HELD operations or actual process-crash invocation. Only small owned temporary fixtures and normal own-fixture cleanup. No held evidence or external operational data was opened or changed.

## Independent-review handoff

Review source `41460ba3ae37736fdf11de7854f7b720f61df974`, especially the explicitly authorized persistence/recovery shared dependency, scalar compatibility, complete ordered pair binding, both inode barriers and metadata commit uncertainty. Parent owns adoption after independent review. No self-GO is claimed.

Actual Pocket adapter work follows separately: consume authorized source bytes and selected ticks, produce a real prepared reference, synthesize the separately frozen audition, retain both measured outputs and exact transcript/lineage, and separately qualify audible clone/restart/cancel behavior. Current synthetic fixtures deliberately provide none of that acoustic evidence. Actual Kokoro, installation and real-engine qualification were not rerun; their existing path was preserved by bounded source changes and legacy one-output regressions.
