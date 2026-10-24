# H1 production domain r182 — parent seam required

Status: BLOCKED BEFORE IMPLEMENTATION; no source adoption or engine readiness claim.

## Pins and scope

Own task checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-90cd27d0`, branch `t3/h1-voice-generation`. Entry HEAD `7c63716`; fast-forwarded to inspected current root `5df7257dd4ef68d8d2ceadd333f2bfe253515d27` (upload integration plus parent next-step record). No historical checkout was changed. Sole deliverable at this checkpoint is this report, copied verbatim to root; no source commit exists.

Read PRODUCT.md, IMPLEMENTATION-PLAN.md H0/H1/H2/H4, service/core.py, voice_store.py, ingest.py, persistence.py, contracts/validate.py, operation declarations and prepare transcript, synthetic workers, dispatcher admission, and both conftests. No AGENTS.md was found by the bounded parent-directory search.

## Concrete dependency requiring parent decision

The requested trusted uploaded reference -> prepare job -> published candidate -> explicit save -> clone generation journey cannot currently execute using the existing synthetic lifecycle:

- Core.submit_synthetic rejects every operation except speech_generate, restricts settings to preset modes, and rejects all input assets.
- Core._message supplies no input asset IDs. Publication validation also supplies an empty allowed-assets set. Input binding must be threaded consistently, without relaxing owner/attempt/frozen-plan checks.
- worker/synthetic.py rejects voice_prepare and always emits speech_generate with candidates=[].
- worker/isolated_synthetic.py rejects non-speech operations, input asset IDs and non-qwen_preset voices; it likewise emits no candidates. supervised_synthetic delegates to that generator.
- Dispatcher submission remains a closed installed preset fixture. This report does not request HTTP/client or dispatcher activation work.
- contracts.validate.publication already requires exactly one prepare candidate, a published audition asset, matching reusable configuration/settings, reference text, conditioning and exact engine/model/bundle identity. The prepare transcript is contract evidence, not a worker capable of producing that result.

Inventing candidates from caller metadata, treating an upload as a successful prepare job, or writing fake success rows would violate this commission. Adding another runner/store would violate reuse requirements. Worker files are outside the literal lane allowlist.

Requested parent decision: authorize worker/synthetic.py and worker/isolated_synthetic.py edits for fixed synthetic prepare/clone results using the existing lifecycle, alongside the already allowed minimal Core changes; or commission that lifecycle seam separately and return its reviewed interface. No contract change is presently identified as necessary. Real H4 engines and installed dispatcher activation remain separate work.

## Checks and limits

Read-only source inspection and git pin/status checks only. Both pytest conftests were inspected: top-level installs a maintenance guard and service conftest can launch uvicorn. Neither was executed. No unittest was run because implementation is paused at the required parent seam decision; no tests or source have changed. No pytest, guard, full suite, Docker, network, model, ML, hardware, HELD operation, delegation or historical cleanup was performed.

After the seam decision: implement the smallest coherent domain slice, add tests/service/test_production.py with real temporary Core/SQLite/files, run only isolated unittest via /private/tmp/pab-0c-venv/bin/python, commit source separately from the completed report, then independent review. Restart/idempotency/ownership/stale selection and immutable exact identity remain required. Synthetic results must remain explicitly synthetic. No self-adoption.
