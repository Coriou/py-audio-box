# Qwen3 Voice Stack Integration Plan (Agent-Optimized, Container-First)

Status: Proposed implementation plan
Date (validated): 2026-02-20
Owner: `py-audio-box`
Scope: Qwen3-only (no CosyVoice/F5/E2, no external API providers)

Progress snapshot (repo audit): 2026-02-21
- Phase 0: implemented (rollout guard `QWEN3_ENABLE_CUSTOMVOICE`, default on).
- Phase 1: implemented.
- Phase 2: implemented.
- Phase 3: implemented.
- Phase 4: implemented.
- Phase 5: implemented (capability probe + metadata schema tests + smoke matrix automation; full clone/built-in/designed smoke validated in container).

## 1. Mission and Hard Constraints

This plan upgrades the project from a strong clone-only pipeline into a clear, controllable Qwen3 voice platform while preserving current strengths:

- Container-only operation (`./run`, Docker Compose) with zero host installs.
- Existing fast iteration loop via cached prompts.
- Clean CLI and predictable artifacts for humans and AI agents.

Hard constraints:

- Keep Qwen3 family only.
- Do not add external TTS APIs.
- Keep backward compatibility for current `voice-register`, `voice-clone`, and `voice-synth` workflows.
- Keep all new behavior deterministic and scriptable by default.

## 2. Up-to-Date Qwen3 Research (Primary Sources)

### 2.1 Verified model families and intended use

As of 2026-02-20, the official Qwen3-TTS stack exposes three distinct generation paths:

1. Base (`Qwen3-TTS-12Hz-0.6B/1.7B-Base`)
- API: `generate_voice_clone`, `create_voice_clone_prompt`
- Purpose: reference-audio voice cloning
- Notes: can run x-vector-only mode (quality tradeoff)

2. CustomVoice (`Qwen3-TTS-12Hz-0.6B/1.7B-CustomVoice`)
- API: `generate_custom_voice`
- Purpose: built-in premium timbres + instruction-controlled speaking style
- Notes: supports `speaker`, optional `instruct`, and language selection

3. VoiceDesign (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`)
- API: `generate_voice_design`
- Purpose: natural-language voice creation from description
- Notes: can be followed by clone-prompt extraction for reusable character voices

### 2.2 Official CustomVoice built-in speakers

Official supported speaker list shown in Qwen docs/model cards:

- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ryan`
- `Aiden`
- `Ono_Anna`
- `Sohee`

Implication for this repo: built-in timbres are real and directly usable in our local setup. They are not currently integrated into our CLI surface.

### 2.3 Control surface differences (critical)

- Base clone path: control is dominated by reference audio + generation sampling.
- CustomVoice path: control is dominated by `speaker` + `instruct` + text semantics.
- VoiceDesign path: control is dominated by `instruct` during design stage, then stabilized via prompt reuse.

This explains the "black box" feeling in clone mode: clone mode has no native `instruct=` argument at synthesis time.

### 2.4 Performance and quality notes from upstream

- Qwen reports low-latency streaming architecture and broad language coverage in official docs.
- Qwen reports benchmark results on Seed-TTS and InstructTTS-Eval in official evaluation section.
- `qwen-tts` package is new and evolving quickly; latest PyPI listed version in source at validation time is `0.1.1` (released 2026-02-06).

Planning implication: we should codify version pin and compatibility checks because upstream APIs are likely to evolve quickly.

## 3. Current Repository Assessment

### 3.1 What is already strong

- Excellent multi-stage cache architecture.
- Good voice registry abstraction for named voices and tone-labeled prompt variants.
- Interactive reference selection already reduces clone uncertainty.
- Existing `design-voice` command already implements VoiceDesign -> Clone flow.

### 3.2 Core gaps blocking controllability

1. No CustomVoice integration in runtime path
- `lib/tts.py` synthesis path currently calls clone API only.

2. No built-in-speaker UX
- No `list-speakers` command.
- No direct `--speaker` + `--instruct` speak path.
- No way to register built-in timbres as first-class named voices.

3. Clone uncertainty still high
- Candidate selection quality features are limited.
- Voice-synth QA is intelligibility-centric only (no style consistency policy).
- Generation parameter presets are not standardized per voice/tone.

4. Style UX inconsistency
- `lib/styles.yaml` and style helper functions exist, but CLI flow does not consistently use them.
- Some comments/examples imply style flags that are not active in current paths.

### 3.3 Agentic UX gaps

- Missing machine-readable "capabilities" introspection command.
- Missing deterministic "best take" policy and reproducible ranking profile.
- Missing explicit per-voice defaults profile for sampling and language.

## 4. Target End-State (Qwen3-Only)

After this plan, the project should support three first-class voice modes with a single clean user mental model:

1. Clone voice from reference audio.
2. Use built-in Qwen CustomVoice timbre (speaker) with optional instruct control.
3. Design a new voice by instruction, then reuse it as a named voice.

### 4.1 Canonical UX principles

- One obvious command for each intent.
- Named voices remain primary abstraction.
- Direct mode is still available for quick experimentation.
- Every run emits reproducible metadata with mode, model, and parameters.
- Every feature works through container commands only.

### 4.2 Proposed command surface (additive, backward-compatible)

Keep existing commands, add:

- `./run voice-synth list-speakers [--model <customvoice-model>]`
- `./run voice-synth speak --speaker Ryan --text "..." [--instruct "..."] [--model <customvoice-model>]`
- `./run voice-synth register-builtin --voice-name announcer --speaker Ryan [--instruct-default "..."] [--model <customvoice-model>]`

And extend existing:

- `voice-synth speak --voice <slug>` resolves to either:
  - clone prompt (current behavior), or
  - registered built-in profile (new behavior), or
  - designed clone prompt (current behavior).

## 5. Architecture Changes

### 5.1 Synthesis engine abstraction

Add a small engine router in `lib/tts.py`:

- `synthesise_clone(...)` -> `generate_voice_clone`
- `synthesise_custom_voice(...)` -> `generate_custom_voice`
- `synthesise_design(...)` remains in design workflow

Keep old `synthesise(...)` wrapper for backward compatibility and route internally by mode.

### 5.2 Voice registry schema extension (additive)

Current `voice.json` remains valid. Add optional fields:

- `engine`: `"clone_prompt" | "custom_voice" | "designed_clone"`
- `custom_voice`: object
  - `model`
  - `speaker`
  - `instruct_default`
  - `language_default`
- `generation_defaults`: object
  - `temperature`
  - `top_p`
  - `repetition_penalty`
  - `max_new_tokens_policy`

Rules:

- If `engine` absent, infer existing clone behavior.
- For clone voices, existing `prompts` + `tones` behavior is unchanged.

### 5.3 Metadata contract upgrades

Extend run metadata (`takes.meta.json`, clone meta) with:

- `engine_mode`
- `source_type`
- `speaker` (if custom voice)
- `instruct` and `instruct_source` (explicit/default/none)
- `generation_profile` (stable/balanced/expressive/custom)
- `selection_policy` and `selection_metrics`

This is required for agent traceability and repeatability.

## 6. Implementation Roadmap (Rock-Solid Phasing)

## Phase 0: Baseline, Freeze, and Guardrails

Goal: lock behavior before feature expansion.

Tasks:

1. Add a snapshot test script that captures current CLI behavior.
2. Add API compatibility smoke checks for qwen-tts model methods at runtime.
3. Create feature flags for new modes (`QWEN3_ENABLE_CUSTOMVOICE=1` default on after rollout).

Files:

- `apps/voice-synth/voice-synth.py`
- `lib/tts.py`
- `README.md`
- `apps/*/README.md`

Acceptance:

- Existing clone/design commands output identical audio/meta for fixed seed.
- No host-side setup changes.

## Phase 1: CustomVoice Read/Use Path

Goal: make built-in Qwen timbres immediately usable.

Tasks:

1. Add `list-speakers` command in `voice-synth`.
2. Add direct CustomVoice synthesis path:
- `speak --speaker ... --instruct ...`
- enforce mutual exclusivity: `--voice` xor `--speaker`
3. Add routing in synthesis layer to call `generate_custom_voice` when speaker mode is selected.
4. Update metadata and output directory naming for speaker-mode runs.

Files:

- `apps/voice-synth/voice-synth.py`
- `lib/tts.py`
- `apps/voice-synth/README.md`
- `README.md`

Acceptance:

- `list-speakers` returns non-empty list for CustomVoice model.
- `speak --speaker Ryan` generates valid audio in container.
- Existing `speak --voice <clone>` behavior unchanged.

## Phase 2: Register Built-in Voices as Named Voices

Goal: unify built-in and cloned voices under one "named voice" UX.

Tasks:

1. Add `register-builtin` command to persist CustomVoice profile to registry.
2. Extend `resolve_voice` to resolve `engine=custom_voice` entries.
3. Support tone variants for built-in voices by storing multiple instruct presets.
4. Add `voice-synth list-voices` display fields for built-in profiles.

Files:

- `apps/voice-synth/voice-synth.py`
- `lib/voices.py`
- `apps/voice-synth/README.md`
- `README.md`

Acceptance:

- `speak --voice <builtin-slug>` works without speaker flag.
- Export/import preserves built-in profile metadata.
- Backward compatibility with existing exported zips confirmed.

## Phase 3: Clone Predictability and Control Hardening

Goal: reduce uncertainty in cloned output without new external models.

Tasks:

1. Introduce generation profiles:
- `stable` (low stochasticity)
- `balanced` (default)
- `expressive` (higher variance)

2. Add deterministic best-take mode:
- `--variants N --select-best`
- ranking by weighted intelligibility + pacing sanity + duration fit

3. Improve reference candidate scoring with non-ML acoustic heuristics:
- clipping detection
- RMS floor/noise heuristic
- speech ratio continuity

4. Persist selected profile per voice (optional defaults in registry).

Files:

- `apps/voice-clone/voice-clone.py`
- `apps/voice-synth/voice-synth.py`
- `lib/audio.py`
- `lib/voices.py`
- `apps/voice-clone/README.md`
- `apps/voice-synth/README.md`

Acceptance:

- Same seed + same profile => stable repeatability.
- Best-take metadata includes ranking breakdown.
- Regression tests pass on clone quality gate logic.

## Phase 4: Style/Instruction UX Cleanup

Goal: remove ambiguity and dead paths.

Tasks:

1. Decide one of two paths and document clearly:
- Path A (recommended): keep style YAML only for instruction templates in CustomVoice/VoiceDesign.
- Path B: remove style YAML entirely from clone/speak docs.

2. Align comments, examples, and CLI flags so no stale `--style` references remain.

3. Add explicit warning when user attempts instruction-like text hacks in clone mode.

Files:

- `lib/styles.yaml`
- `lib/tts.py`
- `apps/voice-clone/voice-clone.py`
- `apps/voice-synth/voice-synth.py`
- `README.md`
- `apps/*/README.md`

Acceptance:

- Documentation and CLI help are consistent.
- No examples advertise unsupported flags.

## Phase 5: Agentic and Operational Hardening

Goal: make this robust for autonomous usage in CI/agents.

Tasks:

1. Add machine-readable command outputs (`--json` where practical for list commands).
2. Add a capability probe command:
- available models
- available CustomVoice speakers
- CUDA/CPU mode
- qwen-tts package version

3. Add CI-level container smoke matrix:
- clone voice
- built-in voice
- designed voice

4. Add deterministic golden tests for metadata schema.

Files:

- `apps/voice-synth/voice-synth.py`
- `apps/voice-clone/voice-clone.py`
- `Makefile`
- `README.md`
- test directory (new)

Acceptance:

- Agent can discover capabilities without scraping text output.
- CI can fail fast on upstream qwen-tts API drift.

## 7. Detailed Documentation Update Plan

Documentation must be updated in the same PR as behavior changes.

Required files:

- `README.md`
- `SETUP.local.md`
- `apps/voice-synth/README.md`
- `apps/voice-clone/README.md`
- `apps/voice-register/README.md`
- `apps/voice-split/README.md` (only where cross-links are needed)

Required content updates:

1. New command cheat sheet for built-in voices.
2. Clear model-mode decision table:
- Clone vs CustomVoice vs VoiceDesign.
3. Explicit explanation of controllability limits in clone mode.
4. Reproducibility best practices:
- seed, profiles, variants, best-take policy.
5. Troubleshooting by mode.

Definition of done for docs:

- Every command in docs exists in CLI help.
- Every CLI option in help appears in docs when user-facing.
- No stale `--style` references unless actually implemented.

## 8. Test and Validation Strategy (Container-Only)

All commands below must run through `./run` and/or `make` targets.

### 8.1 Functional smoke matrix

1. Clone smoke
- `./run voice-clone self-test`

2. Built-in direct smoke (new)
- `./run voice-synth list-speakers`
- `./run voice-synth speak --speaker Ryan --text "Hello from built-in voice"`

3. Built-in named smoke (new)
- `./run voice-synth register-builtin --voice-name eng-news --speaker Ryan`
- `./run voice-synth speak --voice eng-news --text "Top story tonight"`

4. Designed voice smoke
- existing `design-voice` flow + `speak --voice <slug>`

### 8.2 Regression matrix

- Existing named clone voices continue to resolve and synthesize.
- Legacy prompt IDs still resolve.
- Export/import remains backward compatible.
- GPU variant still works (`TOOLBOX_VARIANT=gpu`) with same commands.

### 8.3 Determinism matrix

For fixed seed and profile:

- metadata equality on generation config
- runtime variance acceptable but bounded by quality thresholds

## 9. Risks and Mitigations

1. Upstream qwen-tts API churn
- Mitigation: startup capability checks + tight version pin + smoke tests.

2. CPU latency for larger models
- Mitigation: keep 0.6B default; make 1.7B opt-in; preserve timeout and chunking.

3. UX complexity growth
- Mitigation: additive commands, strict mutual exclusivity, and unified docs table.

4. Registry schema drift
- Mitigation: additive fields only, schema versioning in metadata, migration function.

## 10. Delivery Milestones and Gates

Milestone M1: CustomVoice direct synthesis
- Gate: `speak --speaker` and `list-speakers` working in container.

Milestone M2: Built-in named voice registration
- Gate: register -> speak -> export/import end-to-end passes.

Milestone M3: Predictability hardening
- Gate: deterministic profile tests + best-take scoring metadata.

Milestone M4: Docs parity
- Gate: docs/CLI parity checklist complete.

## 11. Agent Execution Checklist (Machine-Friendly)

Use this list in order. Do not skip gates.

1. Implement Phase 1 code paths and tests.
2. Run smoke matrix for Phase 1.
3. Update docs for Phase 1 in same PR.
4. Implement Phase 2 registry and resolve logic.
5. Run export/import compatibility tests.
6. Update docs for Phase 2 in same PR.
7. Implement Phase 3 predictability features.
8. Run determinism and regression matrix.
9. Complete Phase 4 style/instruction cleanup.
10. Complete Phase 5 agentic output and capability probe.
11. Final full smoke on CPU and GPU compose variants.

## 12. Recommendations (Decision Summary)

Recommended path for this repository:

1. Prioritize CustomVoice integration first.
- This directly solves "built-in timbre" and control requirements.

2. Keep clone mode, but do not overpromise instruction control there.
- Improve predictability with deterministic profiles and best-take ranking.

3. Standardize on named voice abstraction across all modes.
- Clone, built-in, and designed voices should all be invokable with `--voice <slug>`.

4. Treat docs as part of runtime behavior.
- If command/help/docs diverge, quality is considered broken.

5. Keep container-first discipline.
- No host install assumptions introduced at any phase.

## 13. Primary Sources (Validated 2026-02-20)

- Qwen3-TTS official repository and examples:
  - https://github.com/QwenLM/Qwen3-TTS
- Qwen3 CustomVoice model card:
  - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- Qwen3 Base model card:
  - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- Qwen3 VoiceDesign model card:
  - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- qwen-tts package release history:
  - https://pypi.org/project/qwen-tts/
