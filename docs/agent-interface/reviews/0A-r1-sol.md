# Lot 0A independent adversarial review — round 1

**NO-GO** for the executable contract-pack handoff at `9f1027b95229ed30f12d918b27bba989fbfe5d9d`, against base `a25169bd674b38c232e3476a23ede251b99d20e4`.

Reviewer: separate Sol-low review, 2026-10-16. Source inspected read-only in `/Users/ben/.txcode/worktrees/py-audio-box/t3-0fe72d9f`; HEAD matched candidate and status was clean before and after checks. Scope: the 26 changed files, product/entry point/protocol, canonical shared contracts and Phase 0, lot ticket, and relevant unchanged parser/source surfaces. No implementation corrections, checkout, integration, commit, ledger/ticket edit, private audio, real models, cloud actions or remote e2e execution. Only this report was written in the root repository; temporary diagnostics were written under `/tmp`.

## Required corrections

### F1 — P1: media upload and retrieval have no usable byte transport

Locations: `contracts/v1/operations.json:65-114`; `contracts/v1/schema.json:1078` (`asset_upload_request`); `docs/agent-interface/phase-0/CONTRACTS.md:8`.

`asset_upload` is JSON-only and its closed request contains only `content_sha256`, a numeric `bytes` count, and `source`. There is no body/multipart/chunk/staging-upload operation for the actual file. `asset_get` and `artifact_get` return JSON metadata, and neither resource contains a download link or byte transport descriptor. There is no separate content/range operation in the complete catalogue. Thus a client can neither upload a microphone/local WAV nor retrieve/play the synthesized bytes using these contracts. A hash and byte count cannot reconstruct an uploaded recording.

Reproduction: follow `asset upload` → receipt → job → artifact → asset using the named schemas. Every transmitted field remains JSON metadata; there is no permitted location for input bytes and no defined response carrying output bytes. This is a static contract failure, not a demand for a server implementation in 0A.

Correction/verification: freeze actual bounded upload and authorized download/range contracts, with HTTP/CLI/MCP correspondence and idempotency/content-hash behavior. Add a synthetic end-to-end contract transcript that identifies how the exact fixture bytes enter and leave, including invalid size/hash/range cases. Consumers must not invent these routes independently.

### F2 — P1: resource and worker workflows require missing information

Locations: `contracts/v1/schema.json:210` (`voice`), `:1177-1224` (voice mutation requests), `:1885` (`worker_message`); `contracts/v1/operations.json:209-245`, `:903-913`.

There are several concrete holes in the promised shared vocabulary:

- `voice_save`/`voice_archive` require `expected_revision`, but `voice_get` returns a closed `voice` with no revision and no specified ETag/revision header. A client cannot obtain the precondition it must send.
- Prepare/design return a job receipt, but the job exposes only artifact IDs; artifacts expose asset IDs and generic identity. No candidate resource/result mapping supplies the `candidate_id` required by `voice_save`. There is also no voice-version read operation despite versions being distinct resources required by speech inputs.
- History returns `list`, whose entries are opaque ID strings. There is no revision-get operation or mapping from those IDs to the numeric `source_revision` expected by restore, nor a way to retrieve recorded revision operations for refresh-safe undo.
- An `execute` worker message contains IDs, a fingerprint, identity and deadline but no operation or frozen executable input/plan (speech text/settings or render composition). A hash is not a payload. Workers cannot access the database, and no plan-fetch/transfer protocol is defined. A native Metal worker cannot execute the same typed protocol without inventing another channel.

Reproduction: construct a client using only the catalogue and closed response definitions, then attempt voice-get → archive, design → save, history → restore, or control → worker speech execution. Each chain encounters an unavailable required value or input body.

Correction/verification: define versioned revisions/candidate/results/history and frozen worker input transfer, with meaningful success/failure transcripts. This need not implement the store or worker, but must make required values obtainable and immutable. Re-review the complete chains, including concurrent edits and disconnected/restarted clients.

### F3 — P1: generation equality is accepted as attempt ownership/fencing

Locations: `contracts/validate.py:102-106`, `:123-126`; `tests/contracts/test_contracts.py:71-81`.

`publication` never verifies that the supplied attempt belongs to `job['attempts']`, is the current attempt, or has the expected owner/staging identity. `worker_message` receives only a scalar active generation and asset allowlist, so it cannot verify job ID, attempt ID, frozen input hash, staging ID or sender ownership. Generations are not specified as globally unique across jobs. An unrelated or wrong-owner result at the same generation passes. The existing publication test uses an attempt containing only `generation`, so it cannot exercise ownership at all.

Executed reproduction (all returned normally):

```python
j = load('job.json'); a = load('artifact.json')
rogue = {**j['attempts'][0], 'id': 'rogue',
         'owner_id': 'wrong-owner', 'staging_id': 'wrong-stage'}
c.publication(j, rogue, a, {'impulse': 'verified'})
# worker result: unrelated job_id/attempt_id/input_sha256/staging_id,
# generation=1, empty input assets; c.worker_message(message, 1, []) passes.
```

Correction/verification: make the oracle consume authoritative active attempt/owner context and compare the full job/attempt/input/staging tuple, with an explicit authenticated transport-owner boundary. Define/test monotonically advancing generations and owner-death/reap requirements for replacement, not just scalar equality. Add same-generation wrong-job, wrong-attempt, wrong-owner, wrong-input and wrong-stage counterexamples, plus cancel/result races. Actual process kill/reap remains later E2/E3 work.

### F4 — P2: nested resolution invariants are skipped on standalone artifacts/messages

Locations: `contracts/validate.py:18-29`, `:44-50`, `:123-126`.

Semantic dispatch only handles project, capabilities, resolution and profile. Validating an artifact (or take/worker message) does not enforce its nested requested/effective resolution. For example, changing an artifact's effective model while retaining its requested model passes `validate('artifact', ...)`. This is the exact identity substitution forbidden by the normative contract. The worker helper also accepts such a resolution because it only performs structural validation and generation/assets checks.

Executed reproduction:

```python
b = deepcopy(load('artifact.json'))
b['identity']['effective']['model'] = 'substitution'
c.validate('artifact', b)  # accepted
```

Correction/verification: enforce nested intrinsic invariants at each advertised resource/message validation entry point, including unresolved effective device/precision, prohibited fallback and model/bundle substitution. Contextual media existence/admission is legitimately a later caller responsibility; requested/effective equality is intrinsic and already checkable here. Test the resource entry points consumers actually use, not only `validate('resolution', ...)`.

### F5 — P2: ordinary edits can rewrite immutable takes through the edit oracle

Locations: `contracts/validate.py:128-138`; `docs/agent-interface/phase-0/CONTRACTS.md:24` (immutable take identity).

Existing take preservation runs only for `candidate_arrival`. `block_text` and `blocks_reorder` compare clips/tracks but allow existing take text/settings/voice identity to change under the same ID, as long as the final project is structurally/relationally valid. That defeats frozen history and stale-state truthfulness. In particular an editor could rewrite the take text to match an edited block and suppress its stale marker without new generation.

Executed reproduction:

```python
before = load('production-30.json'); after = deepcopy(before)
after['revision'] += 1
after['takes'][0]['text'] = 'rewritten immutable take'
c.edit_semantics(before, after, 'block_text')  # accepted
```

Correction/verification: preserve existing immutable takes/assets/voice versions across applicable edit operations, while allowing explicit new IDs and dependency-safe lifecycle operations. Test a real before/after block edit that also rewrites its old take text/settings/hash and ensure rejection; verify legitimate new candidates remain allowed. Calling the standalone `immutable` function on two hand-written dictionaries does not verify edit preservation.

### F6 — P2: ripple approves an overflowing clip end

Location: `contracts/validate.py:108-112`.

Ripple checks the new start only, whereas the timeline invariant bounds start plus duration. A valid 4,800-tick clip starting at `9007199254740991 - 4800`, shifted by one tick, passes ripple even though its new end exceeds the JavaScript safe integer ceiling. Consumers relying on this oracle can apply an edit that immediately fails project validation.

Executed reproduction:

```python
p = load('production-30.json')
p['clips'][0]['start'] = 9007199254740991 - 4800
c.validate('project', p)  # valid before edit
s = p['clips'][0]['start']
c.ripple(p, 'speech', s, 9007199254740991, s, 1, ['c0'])  # accepted
```

Correction/verification: validate resulting ends as well as starts (and resulting project invariants), with boundary tests at the exact safe maximum and one tick beyond. Preserve atomic failure semantics.

## Checks actually performed

- Read actual candidate resources, operations, validators, generator, fixtures, tests, compatibility/migration/evidence documents and Dockerfile delta; inspected the exact 26-file change scope. Source HEAD verified against the requested SHA.
- Reused image only after verifying image ID `sha256:5b5cae63a569e685c95c9c3069af813f5de0ba1c59baa7cfd66507e93a434b1c` and comparing SHA-256 for **130 copied files** against candidate source across contracts, lib, apps, tests, benchmarks, scripts and launch/config files: **zero mismatches**. No user mounts.
- Sequential Docker commands used `--rm --network none --cpus 2 --memory 3g`.
- Ran the reported selected regression command, including all contract tests: **85 passed in 20.61s**, exit 0. This includes the **14 focused tests**, including deterministic regeneration inside the disposable container, without rewriting read-only source.
- Ran offline collection with `-m 'not integration and not redis_integration' --collect-only -q`: **320 tests collected in 2.55s**, exit 0. Collection is not full-suite execution.
- Independently reconstructed exact AST argument/parser/default entries (line, kind, arguments and keyword expressions) from current sources: **8 apps / 296 entries**, zero differences against the ledger. Suite also verified recorded source/launcher hashes. The unchanged-legacy mapping is explicit and does not falsely advertise service support.
- Ran `/tmp/pab-0a-r1-counterexamples.py` using the existing temporary validator venv with `PYTHONDONTWRITEBYTECODE=1` and candidate on `PYTHONPATH`. All six diagnostic calls printed `ACCEPTED`: rogue publication, unrelated worker result, ripple end overflow, nested project-create missing media, artifact substitution, immutable take rewrite. The project-create case is recorded as a structural/contextual boundary observation, **not an additional finding**, because CONTRACTS.md explicitly reserves contextual existence checks to consumers. Other reproductions are included above.
- Candidate `git status --short` remained empty after checks. Containers were disposable and removed. No full legacy suite, e2e script, real migration or inference was run.

Exact selected test command:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-contract-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

## Bounded conclusion and evidence limits

The synthetic fixture scale, exact text/IDs, legacy coverage, explicit requested/effective identities, finite provisional profile ceilings and non-destructive migration policy are useful foundations. CPU/CUDA/Metal fixtures are correctly unqualified/not-ready; no hardware, acoustic or performance qualification is inferred. Migration documents correctly retain legacy jobs, require distinct destination ownership/copies, preserve unknown provenance and distinguish data restore from image rollback. Actual containment/races/journaling/restore remain expressly deferred integration work, not proven here.

**NO-GO for 0A as the frozen executable boundary for downstream consumers.** F1–F3 leave consumers inventing essential transport/resource/ownership contracts; F4–F6 demonstrate required invariants escaping the executable checks despite the passing suite. Correct these bounded contract/test defects and obtain independent re-review of every required correction and the resulting exact SHA. Do not infer integration or phase approval from this report.

All review results are **E1 only**: static/schema/offline tests, including tests run in a container. No E2 durable service/process-failure proof, E3 hardware/real-model proof, E4 browser/agent workflow acceptance or E5 human listening/production acceptance was obtained.
