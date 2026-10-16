# Lot 0A independent adversarial re-review — round 2

**NO-GO** for exact candidate `aa9a61a183545df65c96b5ccc5df071aa464c714`, branch `t3/fix-lot-0a-contracts`, inspected read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-032fdaf1`.

Planning base: `e8aa9a92f6e00778d113fa2d6a141fe5fbf16683`. Original rejected candidate `9f1027b95229ed30f12d918b27bba989fbfe5d9d` is represented here by cherry-pick `dd5b869bb82a8ade68c031e41e06b074c7946537`. Reviewer: independent Sol-low, 2026-10-16. No corrections, commits, source writes, app changes, model/cloud/private-audio work, remote e2e, push or deployment. Only this report was written in the parent checkout; its untracked research was left alone.

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical shared contracts/Phase 0, lot 0A, the full prior review and correction response, EVIDENCE-0A and its E1 limits. Inspected the candidate contract resources/catalogue, validators, generator, fixtures, tests, compatibility/migration policies and test-image changes, including the correction delta. No prior objection was waived.

## F1–F6 disposition

| Finding | Result | Concrete evidence |
| --- | --- | --- |
| F1 actual bytes | **Closed at E1.** | `schema.json:1099`, `operations.json` asset_upload/asset_content, `validate.py:197–212`, CONTRACTS:66. Actual strict base64 upload, length/hash checks and authenticated explicit end-exclusive JSON range replies now exist. Synthetic fixture bytes are reconstructed exactly in the passing transcript. |
| F2 missing workflow values/frozen executable inputs | **Partially closed; unresolved.** | Voice revisions, job voice-results/candidate IDs, version-get, numeric history/revision-get and restore inputs are now obtainable. Closed typed inline plans carry requests and explicit synthesis settings. However supported `origin=preset` plans still lack the engine's preset/speaker selector: R2-1 below. Resolved execution identity also remains outside the frozen hash: R2-2. |
| F3 ownership and replacement fencing | **Original finding closed at E1.** | `validate.py:107–129,152–162` now checks authoritative full tuple, authenticated owner, current ordered attempt and prior reap; replacement requires death/reap and next generation with fresh attempt/staging IDs. Original rogue publication/unrelated result reject with `attempt_mismatch`. Existing correction tests cover same-generation tuple changes and cancellation/replacement. Separate new identity/asset-boundary failures are below. |
| F4 recursive intrinsic resolution | **Closed for the original invariant.** | `validate.py:20–21,190–195`; standalone artifact/take/message substitution, unresolved identity and forbidden intrinsic fallback regressions pass. Original effective-model-only substitution rejects `identity_substitution`. This does not close the cross-object binding failure in R2-2. |
| F5 immutable ordinary edits | **Closed at E1.** | `validate.py:215–229` preserves all existing takes/assets/voice versions for every edit call; original same-ID take rewrite rejects `immutable_identity`. Legitimate new candidates and non-interference regressions pass. |
| F6 ripple end overflow | **Closed at E1.** | `validate.py:132–141` checks end and start, validates a copy and returns it. Original maximum-end-plus-one rejects `invalid_range`; exact maximum and atomic failure tests pass. |

Base64 JSON is a pragmatic bounded byte contract for this lot: CLI adapters can encode/decode local files, HTTP has explicit ranges and MCP has a typed representation. It costs about one-third extra wire bytes and can produce large tool payloads; this is not measured token economy or agent-host acceptance. Clients can request small download ranges. A binary HTTP redesign or runtime implementation is not required to close F1. For a nonempty asset, a one-byte initial range supplies `total_bytes`, so the lack of a byte count on asset metadata does not itself leave ordinary download impossible. Catalogue endpoints remain individually typed; no opaque command/argv bus was found.

## Required remaining corrections

### R2-1 — P1: frozen preset voice input still cannot identify the voice to execute (F2 unresolved)

Locations: `contracts/v1/schema.json:159–209` (`voice_version`), `:2589–2759` (`worker_plan`); `contracts/validate.py:164–178`; `contracts/generate.py:23`; unchanged `lib/tts.py:965–977` and `apps/voice-cpu/voice-cpu.py:125–137`.

A voice version permits `origin=preset`, but its closed fields contain only opaque service voice/version IDs, origin, reference asset IDs/text, engine/model/bundle/device/precision, a settings hash and provenance. There is no backend preset/speaker identifier or typed binding to one. The frozen plan adds language/sampling settings, but no such selector. Model/bundle identifies weights, not which voice within those weights. Qwen CustomVoice requires `speaker`; Kokoro requires `preset`. Workers have no DB access, and opaque `voice_id` is not a documented backend speaker name.

The candidate's own `execution()` fixture builds an accepted speech plan using vv0, `origin=preset`, empty references/text and no selector. Its three synthetic preset voices differ only in service IDs/lineage, not executable voice configuration. `validate('worker_plan', plan)` and `plan_hash(plan)` accept this. A downstream worker must invent an ID convention, consult an unspecified mapping, or choose a default voice. A hash cannot recover the missing value.

Bounded correction: freeze a closed origin/engine-appropriate voice configuration, including preset/speaker selection where required, and carry it in the version/plan. Test two distinct presets of the same bundle through version read → frozen execution input, plus rejection of missing/unsupported selectors. The compatibility ledger correctly preserves advanced controls as unchanged legacy; that does not make an admitted preset speech plan executable. No engine run or expansion to every legacy control is requested.

### R2-2 — P1: frozen request policy and published identity can diverge under the same input hash

Locations: `contracts/validate.py:115–123,152–162,186–188`; `contracts/v1/schema.json:1246–1285,2589–2759`; CONTRACTS:32,46,72.

The plan hash binds the request, snapshots and synthesis settings, but not the resolved message identity/profile/fallback. `worker_message` compares only `identity.requested` to request.identity; it never compares request.fallback. Its recursive resolution check evaluates the worker's own fallback declaration. Thus a request explicitly forbidding fallback is accepted with a worker-chosen CPU fallback. `publication` compares the ownership/input tuple and intrinsically validates artifact identity, but has no authoritative frozen resolution to compare. Replacing BOTH requested and effective artifact model with another model is accepted under the original job/input hash. This violates no-silent-substitution and frozen render/execution identity requirements even with honest hash/media verification.

Executed reproductions using the candidate's test helper, inside the verified image:

```python
import sys
sys.path.insert(0, 'tests/contracts')
from test_corrections import execution
from contracts import validate as c

j, m, a = execution()
m['plan']['request']['identity']['device'] = 'cuda'
m['plan']['request']['fallback'] = 'forbid'
m['identity']['requested']['device'] = 'cuda'
m['identity']['effective']['device'] = 'cpu'
m['identity'].update(fallback='allow_cpu', reason='worker decided')
j['input_sha256'] = m['input_sha256'] = c.plan_hash(m['plan'])
c.worker_message(m, j, ['impulse'], transport_owner='worker1')  # ACCEPTED

j, m, a = execution()
a['identity']['requested']['model'] = 'substitution'
a['identity']['effective']['model'] = 'substitution'
c.publication(j, j['attempts'][0], a,
              {'impulse': 'verified'}, transport_owner='worker1')  # ACCEPTED
```

Bounded correction: freeze the control-approved resolved identity/policy with the executable input, bind it to the stored hash/context and compare worker and artifact identities to it. A worker cannot grant fallback forbidden by the request. Test changed requested/effective model together, device/precision/profile changes, forbidden fallback and legitimate declared fallback. This is an E1 relation check, not a demand for real admission/hardware proof.

### R2-3 — P2: plan-required assets bypass the explicit allowed-assets check

Locations: `contracts/validate.py:159–170`; CONTRACTS:72.

`worker_message` authorizes only the separately supplied message.input_asset_ids. It does not derive required inputs from the frozen plan or require that the list includes them. A clone plan referencing impulse can pass with both input_asset_ids and trusted allowed_assets empty. `frozen_plan` checks that the asset metadata exists in the plan, which is not authorization to transfer/read its bytes.

Executed reproduction (same imports as above):

```python
j, m, a = execution()
v = m['plan']['voice_versions'][0]
v['origin'] = 'clone'
v['reference_asset_ids'] = ['impulse']
m['input_asset_ids'] = []
j['input_sha256'] = m['input_sha256'] = c.plan_hash(m['plan'])
c.worker_message(m, j, [], transport_owner='worker1')  # ACCEPTED
```

An adapter following input_asset_ids cannot supply the reference and cannot execute; one following the plan escapes this oracle's trusted allowlist. Actual authorization on each content transfer remains required later, but the executable oracle already has enough information to reject this mismatch now.

Bounded correction: derive the operation's required asset dependency set from frozen inputs and require it to be represented and authorized; define whether extra plan metadata is allowed without granting read authority. Test prepare source, clone reference and render clip dependencies with omitted/unauthorized IDs and valid complete inputs. Do not merely check that caller-listed IDs are a subset.

## Verification actually run

- HEAD matched the requested SHA before/after review; source worktree status remained clean. No applicable AGENTS.md found in the candidate/parent search.
- Verified image ID `sha256:bf5e59d5c3a8703ee6bbb2a4eea1a91e133a2946863e43ada9420bf68c43f865`. Independently SHA256-compared **131 copied files** against candidate across contracts/lib/apps/tests/benchmarks/scripts and launcher/config files: **zero mismatches**. Image reuse was therefore exact-source reuse, not acceptance of its prior test claim.
- Sole heavy test lane used the following final selected command, with no user mounts:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-fix-r1-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

Result: **118 passed in 31.98s**, exit 0. This includes all **47 final contract tests**, synthetic byte roundtrip, deterministic regeneration in the disposable container, final five additions and relevant legacy regressions. No collection count is substituted for execution; no full-suite execution is claimed.

- Independent diagnostic container calls used the same `--rm --network none --cpus 2 --memory 3g` restrictions and no mounts. Original rogue publication, unrelated result, intrinsic artifact substitution, immutable take rewrite and ripple overflow all rejected with the expected semantic errors, using the new signatures/valid protocol envelope rather than counting TypeErrors as fixes. New R2-2/R2-3 reproductions above all printed ACCEPTED.
- Independently reconstructed full AST parser entries (line/kind/argument expressions/keyword expressions), rather than trusting the suite's count-only comparison: **8 apps / 296 controls**, exact match. Source/launcher hash checks also passed in the suite. Legacy defaults/application files were not modified by the lot.
- `git diff --check e8aa9a9 HEAD` passed. Temporary schema diff was under `/tmp`; containers exited/removed. No migration, real inference, public STT, Voicebox or remote e2e was run.

CPU/CUDA/Metal profiles remain truthfully synthetic, unqualified and not-ready. Finite provisional ceilings and topology distinctions are useful contracts, not hardware support or memory measurements. Migration remains non-destructive policy into a distinct root, preserving pending legacy jobs and explicitly separating DB/files restore from image rollback; actual journaling/race/restore proof is deferred.

**Bounded verdict: NO-GO for the exact candidate as the complete frozen executable boundary consumed by 0B/0C.** Most original corrections are substantive and verified. F2 is not fully closed, and the new cross-object identity/asset cases prevent full handoff approval. Fix these relations and missing preset input in the contract owner lane, then review the exact resulting SHA. No broad redesign is required. The orchestrator owns any narrowing/escalation under the correction-loop protocol; this report does not automatically launch another agent.

All conclusions are **E1 static/schema/offline evidence only**. No E2 durable service/process/recovery, E3 hardware/model, E4 runtime browser/agent acceptance or E5 listening/production acceptance is claimed.
