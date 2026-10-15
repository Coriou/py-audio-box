# Cloud Qwen failures: prompt identity, registration and worker exit

These corrections cover local contracts and failure handling. The cloud logs and
GPU/container from the reported incident were not available in this checkout;
the actual cause of that worker death remains unconfirmed. No cloud audio,
private references, cloud registry or running service was modified.

## Prompt compatibility

Clone prompts belong to the checkpoint **and resolved revision** that produced
them. A 0.6B Base prompt cannot be reused for 1.7B Base by changing the target
model argument. `synthesise_clone` now rejects mismatched or missing provenance
before generation, with `clone_prompt_mismatch` or `clone_prompt_identity_missing`.
It does not reinterpret tensors, convert their dimensions or silently rebuild
using a different model/reference.

New prompts use `tts.ClonePrompt`, containing the original payload, checkpoint
and revision. The supervised model's `create_voice_clone_prompt` and the cache
builder attach this provenance. Local aliases and HF snapshot paths are
canonicalized; a lazy backend resolves its immutable revision before comparison.
The public synthesis helper unwraps the payload only after validation.
Missing/blank revisions are refused even when both sides name the same model.
An ordinary mutable local directory is also refused: a supplied revision label
cannot prove that its weights have not changed. Use the resolved Hugging Face
snapshot directory (or an alias to it), or load the repository so the worker
resolves its snapshot. Replacing files inside an immutable snapshot is outside
this trusted-cache contract.

Existing scripts that load **their own trusted local** pickle and pass it to
`synthesise_clone` can use newly built prompts directly. A legacy raw prompt
requires `prompt_metadata=` from a sidecar validated by the existing trusted-cache
checks, including pickle/reference hashes and checkpoint identity. Merely reading
an arbitrary sidecar is not validation. Rebuilding from the authorized reference
is the simplest migration. These provenance checks do not make pickle safe for
untrusted imports; portable imports remain WAV/JSON only.

Rebuild a named clone using the exact intended checkpoint; adapt paths to the
existing cloud mounts:

```sh
python apps/voice-clone/voice-clone.py synth \
  --ref-audio /work/authorized-reference.wav \
  --ref-text 'The exact words spoken in the reference.' \
  --voice-name cloud-speaker \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --force \
  --text-file /work/full-script.txt --cache /cache --out /work/rebuilt
```

For a 1.7B clone, use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` during both prompt building
and subsequent synthesis. Use the dtype supported by that runtime. Existing
named entries/references remain recoverable; successful rebuilding publishes a
complete prompt without deleting other tones.

For a named voice with both model variants already cached, `voice-synth speak
--voice cloud-speaker --model Qwen/Qwen3-TTS-12Hz-0.6B-Base` selects an existing
compatible 0.6B prompt instead of blindly taking the newest 1.7B prompt. Selection
checks the resolved revision, prompt bytes and that prompt's own reference. It
does not convert or rebuild prompts. If no compatible variant exists, build it
once for the target model. Direct pickle paths and legacy prompt IDs stay exact.
Without `--model`, named voices select a compatible prompt for the existing
default Base model; building a larger-model prompt does not change that default.

`--tone` additionally requires the same registered tone. New registrations record
the tone in each prompt sidecar so variants survive a later model switch. For old
sidecars without tone metadata, only the current explicit tone index establishes
that association; an unidentifiable older variant may need rebuilding once.

On Linux hosts where Docker requires sudo, the Bash launcher accepts
`TOOLBOX_DOCKER_SUDO=1 ./run voice-synth …`; combine it with
`TOOLBOX_VARIANT=gpu` when needed. Elevation is explicit and never an automatic
retry of a failed generation.

If the cloud host can download models only through `docker run --network host`,
use `TOOLBOX_NETWORK=host` with the Bash launcher as well:

```sh
TOOLBOX_DOCKER_SUDO=1 TOOLBOX_NETWORK=host TOOLBOX_VARIANT=gpu ./run voice-synth speak --voice cloud-speaker --text 'Bonjour'
```

Omit the sudo/GPU options if unnecessary. This explicitly enables the host network
for that invocation; it does not pre-download a model or diagnose the bridge's
timeout. The normal Compose network remains the default, with no automatic retry.
See the root README for Linux/Docker Desktop prerequisites. Cloud connectivity
still needs to be confirmed on that host.

## Named voice publication

`voice-clone synth` no longer creates a named entry before reference preparation,
model loading and prompt construction succeed. Missing/empty prompt metadata
fails the operation instead of producing a success warning. Reference, prompt,
tone index and explicitly saved generation defaults are staged under the registry
writer lock, then promoted together. An exception rolls back to the previous
record; failed new registration leaves no new named entry. VoiceDesign uses the
same finalization path. Readers do not acquire the writer lock: the two-rename
promotion can briefly make an existing voice absent, and a system interruption
can leave a hidden recovery backup. This is not a durability or zero-gap promise.

`prepare-ref` intentionally creates reference-only state; it does not build a
synthesis prompt. Its successful completion is not `speak` readiness. Complete
that stage with `voice-clone synth`. Existing incomplete entries are not silently
deleted or repaired by selecting a model/reference on the user's behalf.

## Worker death and memory evidence

EOF, broken pipe, failed spawn and premature worker exit now become
`QwenWorkerError`. Its bounded diagnostics include operation, startup/send/response
stage, checkpoint basename, requested dtype, exit code/signal and available Linux
cgroup-v2 memory limit/`oom_kill` counter delta. Workers are stopped and reaped;
a subsequent request starts fresh. Constructor/start failures close both pipe ends.

`SIGKILL` alone does **not** establish OOM. A rising cgroup counter establishes
an OOM kill in the shared group, without proving that this particular worker was
the victim. Missing cgroup information remains missing. The CLI reports expected
worker/provenance failures with exit status 1 and a concise stderr message.
`voice-synth speak --json-result FILE` additionally selects structured stderr on
failure; it does not claim to have produced a successful result file.

On the reported cloud runtime, bfloat16 was reported to work while float32 failed.
The tool keeps dtype selection explicit (`auto` uses the existing hardware policy)
and does not silently retry with lower precision. Continue to use the working
configuration while checking the new signal/cgroup diagnostics. Float32 with a
roughly 9 GB limit is not automatically refused based on an unverified memory estimate.

## One context for the whole script

`voice-synth speak` already defaults to one supplied text per generation take.
The explicit `--no-chunk` option makes this choice reviewable in cloud scripts.
Avoid wrapping that command in a per-sentence loop when assessing tone continuity.
`--chunk` remains available for bounded pieces; `--chunk` and `--no-chunk` conflict.
Original text and chunk count remain in take metadata.

Example on a runtime where bfloat16 is supported:

```sh
python apps/voice-synth/voice-synth.py speak \
  --speaker Ryan --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --dtype bfloat16 --language French \
  --text-file /work/full-script.txt --no-chunk --profile stable --seed 7 \
  --cache /cache --out-exact /work/cloud-whole-text \
  --json-result /work/cloud-whole-text/result.json
```

A longer single call can need more memory/time/tokens. It reduces application-level
context resets but does not guarantee a fixed tone, pronunciation or vocal identity.
Compare it by listening with the same script, model, speaker/reference and settings.

## Validation

Regression coverage uses synthetic prompts and real spawned processes without
model weights. It covers checkpoint/revision conflicts and unknown provenance,
legacy cache binding, source alias resolution, failure-before-generation,
registration rollback/new-entry absence/concurrent tones, real process exits and
SIGKILL, transport races, shared-cgroup OOM evidence, restart, whole-text CLI calls,
error exit status and cleanup of model owners. Physical 1.7B GPU inference and the
reported cloud memory limit still require a cloud rerun.
