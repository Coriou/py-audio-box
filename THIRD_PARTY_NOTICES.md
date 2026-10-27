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

Pinned model declarations in `requirements/tts/` distinguish Kokoro weights
(Apache-2.0) and runtime (MIT), Qwen (Apache-2.0), and Pocket French weights
(CC-BY-4.0 with gated operator access). Preserve their revisions, attribution,
access restrictions and receipts. These are repository declarations, not a new
upstream terms audit. Model access does not grant rights in uploaded voices.
