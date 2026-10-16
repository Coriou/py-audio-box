# Voicebox export-function compatibility probe

Executed 2026-10-15 in the existing local test image `sha256:42008ccbab95ca5e7afbe52dd2335f041c0fc87e8fd7620c844b88b44bbbdece`, network disabled, 1 CPU/512 MiB, read-only root and bounded temporary filesystem. No model or private audio was used. Upstream snapshot: `jamiepine/voicebox@8af7efe62fab8d33e2a5dfbedabaa45f68a5184f`.

The harness AST-extracts the actual `export_story_audio` function, keeping its body unchanged, and supplies mocked database/path/audio-loader inputs with synthetic PCM. It uses real NumPy mixing and WAV encoding through soundfile. This is not an installed-app, database integration, API, browser or listening test. `result.json` records the full source-file SHA256 and observed outputs.

Four fixtures established:

- A missing source alongside valid media is skipped; a one-second output is still returned.
- Two overlapping 0.75-amplitude sources produce a peak-normalized result of 1.0, rather than an explicit render-preset/headroom decision.
- A zero clip-volume value yields 0.75 output, because the inspected implementation treats zero as its default gain.
- A nonzero half-gain value correctly yields 0.375 output.

These are concrete contract/adaptation requirements, not an assertion that the entire product is unsuitable. The volume issue is small in isolation; project revision, lifecycle and canonical-state integration remain the larger adoption questions. The first harness invocation lacked AST location metadata and failed before executing the function; `ast.fix_missing_locations` corrected the harness, without modifying upstream code.

Reproduce by mounting the pinned `backend/services` directory read-only at `/upstream`, this directory at `/results`, and executing `/results/probe.py` in a compatible NumPy/soundfile Python environment with no networking. The local image tag alone is not a reproducible identity; use the recorded digest or reproduce its dependency environment.
