# Third-party notices

The project's AGPL-3.0-only election covers project source, subject to existing
third-party notices. It does not relicense dependencies, base images, vendor code,
model weights, user recordings or generated audio. Preserve their notices and
check the terms for the particular artifacts you redistribute.

Existing Studio evaluation notices remain verbatim in the source checkout:
`studio/evaluation/LICENSE-wavesurfer.txt` (BSD-3-Clause) and
`studio/evaluation/LICENSE-waveform-playlist.txt` (MIT). These evaluation assets
are not included in the headless/native source installer archives.

Dependency versions and hashes remain in `poetry.lock`, `studio/package-lock.json`,
`service/requirements.lock` and `requirements/`. Their individual licenses still
apply; the project metadata and OCI license label describe project source only.
Redistributing installed dependencies or images also requires retaining their
applicable license texts and notices; this file is not a complete image notice inventory.

The native-only `requirements/wheels/qwen_tts-0.1.1-1native12hz-py3-none-any.whl`
is a modified Apache-2.0 Qwen-TTS dependency, Copyright 2026 The Alibaba Qwen team.
Its original LICENSE is retained; modified source files are marked and its NOTICE
identifies the lazy 25 Hz import change. Reproduction, exact hashes and upgrade
impact are in `docs/agent-interface/NATIVE-DEPENDENCY.md`. This dependency's
license does not relicense application source or model weights.

Pinned model declarations in `requirements/tts/` distinguish Kokoro weights
(Apache-2.0) and runtime (MIT), Qwen (Apache-2.0), and Pocket French weights
(CC-BY-4.0 with gated operator access). Preserve their revisions, attribution,
access restrictions and receipts. These are repository declarations, not a new
upstream terms audit. Model access does not grant rights in uploaded voices.
