# Astra review — contextual dictation clarification

15 October 2026. Bounded planning/source review of
[the clarification](research-contextual-dictation.md) and contextual sections of
[PLAN](PLAN.md), [ACCEPTANCE](ACCEPTANCE.md) and
[the STT synthesis](STT-AND-PORTABILITY.md). Independently read
[TxCode issue #8](https://github.com/Coriou/txcode/issues/8) with read-only `gh`:
open proposal, last updated 11 October. No benchmark, install, model/provider
request, TxCode change or external write was performed.

**GO — first-class contextual speech input at the contract/prototype boundary.**
The clarification correctly makes useful final wording and reduced correction
effort the success criterion. Preserving raw ASR is necessary evidence; it does
not excuse a dictation experience that repeatedly damages project terminology.
The user's Claude/Codex experience defines the desired outcome without proving
those products' internal recognition/context pipeline.

The service can compose supported recognition hints and bounded terminology
resolution under one request. The client supplies compact, versioned work context
and owns microphone/caret/draft handling. This gives the core a useful contextual
capability without requiring broad memory access, another mandatory heavyweight
LLM, a new sibling application or a hosted default. TxCode is a concrete client
integration target; its proposed OpenRouter endpoint/model/accounting details are
neither adopted nor qualified by this review.

The meaning-preservation contract is sufficient for this stage: retain valid
unexpected speech, language switches, negation, quantities and literal code;
resolve only supported candidate matches; leave ambiguity visible. Contextual
resolution is distinct from optional prose rewriting. The final proposal carries
raw/derived identity, changed spans and context/processor provenance, with a
reversible client edit. Microphone release, cancellation, changed-draft protection
and ordinary review/send prevent the result from becoming an unsolicited command
or an overwritten draft.

## Necessary corrections verified

Two gaps found during review were incorporated into PLAN and ACCEPTANCE and
reread before this verdict:

- **Retention:** ephemeral dictation declares separate audio, raw-text,
  resolved-text and context retention, including caches, logs, backups and
  cancellation/expiry deletion. General durable-media provenance no longer
  implies permanent retention of dictation capture or context. Exact deadlines
  and deletion/recovery semantics remain contract fixtures to specify and test.
- **Failure and fallback:** unsupported contextual policy is rejected or an
  explicitly caller-authorized downgrade is exposed. A failed contextual stage
  reports partial/failed state; blind ASR cannot be labelled contextual success,
  auto-inserted or silently rerouted to a cloud provider/different model.

**No unresolved blocker remains for the clarified planning scope.** Acceptance
evaluates the proposed final text on matched held-out FR/EN/mixed utterances,
including wrong/stale/adversarial context. Correction effort, critical-term and
semantic fidelity, false substitutions, latency and compute cost take precedence
over raw WER alone. Report false-substitution denominators and failed/abandoned
utterances so lower editing effort cannot be obtained by hiding failures.

**No runtime or product-readiness GO.** The earlier CPU profiles remain
unqualified; contextual quality and latency have not been measured. Keep the
existing bounded CPU/package diagnosis as the next runtime prerequisite. Schema,
fake-worker and draft-safety fixtures can proceed independently. No further model
tests, native-client implementation, provider selection or sibling creation are
authorized by this review itself.
