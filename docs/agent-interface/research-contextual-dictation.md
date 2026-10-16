# Contextual dictation: product intent and contract correction

2026-10-15. Planning/source review only; no models, benchmarks, installs or external writes. Read [TxCode issue 8](https://github.com/Coriou/txcode/issues/8) through read-only `gh issue view`; it remains an open proposal, updated 2026-10-11. Only this document changed.

## Correct the success criterion

**Useful contextual dictation is a first-class intent. Raw recognition is evidence, not the final UX success criterion.** A transcript that repeatedly damages the user's project terminology fails that intent even when its ordinary-word accuracy is respectable. The user's observed Claude/Codex experience establishes the desired outcome; it does not establish which internal recognizer, context retrieval or correction mechanism produced it.

The earlier STT report was right to preserve raw recognition and prohibit silent editorial rewriting, but “literal recognition plus optional cleanup” under-specifies this need. Separate **recognizer hints**, **bounded contextual resolution**, and **optional rewriting**. Contextual resolution is not the same product feature as making prose smoother.

## What the TxCode precedent actually says

The issue proposes desktop-composer speech insertion at the caret, ordinary review/send, short Opus clips capped at 60 seconds, automatic FR/EN detection, microphone release before ASR, cancellation/discard and a guard against late draft replacement. It proposes small on-screen context candidates and constrained substitutions, preserving language and intent. Its hosted OpenRouter path pins cheap/stronger model IDs, requires explicit escalation and counts cost against a daily ceiling. [Issue 8](https://github.com/Coriou/txcode/issues/8).

This is a client-requirement precedent, not verified endpoint/model availability or authorization to add a paid provider. The shared service remains local and Docker-first. A router's model name, advertised task and returned accounting need live qualification before any separate hosted implementation. The issue's audio-discard policy also differs from archival media transcription; a dictation request must support ephemeral capture.

## Three distinct stages

1. **Recognition:** pass supported, bounded vocabulary/initial context to the selected recognizer. Preserve its actual output and resolved model/context settings. Hints may help but cannot force a word or guarantee correct spelling.
2. **Context resolution:** repair an obvious recognition variant to an explicitly supplied candidate: a project name, identifier, path or command. Return proposed text plus source-span/candidate references and a substitution list. Ambiguity means retain the recognition or flag the span. Never add absent facts, tasks, negation, quantities or a language conversion.
3. **Rewriting:** punctuation style, disfluency removal, restructuring, summarization and translation are separate explicit policies. They are not prerequisites for contextual terminology accuracy and remain off by default.

Do not introduce a new cleanup LLM by default. Start with supported recognizer hints and conservative candidate resolution; evaluate whether they meet the user's correction burden. Where an existing caller agent already has authorized work context, let its host supply a compact context pack or perform the constrained resolution using its existing facility. Do not upload a full thread or make the speech service acquire broad agent memory. A model-assisted resolver, if evaluated, is separately identified, budgeted and constrained; it has no tools or authority to execute transcribed instructions.

## Minimal request and insertion contract

Add `intent=contextual_dictation` alongside `media_transcription`. Its plan includes:

* Audio limit/retention, automatic language policy, recognizer profile and cancellation identity.
* A bounded candidate list with exact spelling, type, provenance, source revision and local priority. Include a small draft window around the caret only when supplied by the caller; larger context is not automatically better.
* `resolution_policy=conservative_candidates`; `rewrite_policy=none`; explicit permitted transformations and unsupported-feature behavior.
* Draft ID/revision and selection/caret anchors, held by the client. These are insertion preconditions, not text for the recognizer to obey.

Keep raw recognition and resolved text separately under the retention policy; raw text need not be the displayed default, and audio need not be archived. Record substitutions and unresolved spans without leaking full context into ordinary logs. Candidate strings and recognized speech are untrusted data: “ignore instructions” remains content.

Release the microphone before processing. On cancellation, discard capture and reject late insertion. On a changed draft/selection, offer the result as a pending suggestion instead of overwriting newer text. Accepted insertion is one undoable composer edit, never an automatic send or agent session. Provide a brief way to inspect/revert substitutions without requiring a confirmation dialog for every correct word.

## Evaluate usefulness and harmful bias

Use matched FR, EN and code-switching clips with: project/person names; confusable names; paths; identifiers; spoken commands; negation; quantities/dates; acronyms; and literal code. Include “ne supprime pas”, “fifteen, not fifty”, repeated words and a correction spoken mid-sentence. Supply correct, irrelevant, stale, wrong and adversarial context, including candidates never spoken.

Compare raw recognition, recognizer hints, hints plus conservative resolution and any explicitly evaluated model-assisted resolver. Hold audio/profile constant; retain outputs and candidate provenance. Primary outcomes are **user corrections per accepted utterance**, terminology accuracy, semantic errors and **false context substitution rate**. Report the denominator: wrong candidate substitutions among substitutions and among eligible utterances. Also measure language preservation, negation/number/code changes, unresolved spans, latency and added token/compute cost. Aggregate WER alone is insufficient. No results are claimed here.

## Precise plan changes

In `PLAN.md` and the STT report, promote contextual dictation from incidental vocabulary support to an explicit quality intent and acceptance journey. Preserve raw/derived separation, but replace the implication that raw ASR alone meets dictation acceptance. Distinguish bounded candidate resolution from optional rewriting; add ephemeral capture/insertion preconditions and false-substitution evaluation. Keep general media transcription evidence-preserving, retain local Docker execution as the default, and leave provider choice/native-client adoption open until qualified.
