# Qwen local: existing voices, troubleshooting and retest

Use the existing `./run` toolbox to repair and retest Ryan, Bruce and John. A new
“Qwen local” service installation is not required for that workflow. Docker and
the existing CPU or CUDA toolbox image provide Python and the Qwen runtime.
This guide describes commands present at PR #1 baseline
`cfc3654d2ac9f38f30ccf3ae5a072d10155ba376`; acceptance must record the exact
commit containing the final integrated repairs. It is an operator checklist,
not a record of a successful runtime test.

## Choose the voice and matching model

| Voice | Existing route | Model choice |
| --- | --- | --- |
| Ryan | `voice-synth speak --speaker Ryan` | CustomVoice; the default is 0.6B-CustomVoice |
| Bruce | `voice-synth speak --voice bruce` | Base; existing prompts are 0.6B only |
| John | `voice-synth speak --voice john` | Base; existing prompts are 0.6B only |

`bruce` and `john` below are examples of registry slugs: use the exact slug or
unambiguous legacy prompt ID returned by `list-voices`. Display names are not
necessarily slugs. Preserve the current registry and prompt files before any
repair. Do not delete the cache to diagnose a lookup failure.

From the checkout you intend to test, using the existing image and cache:

```sh
./run voice-synth list-voices --json
./run voice-synth speak --speaker Ryan --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --language English --seed 7 --text "Hello, this is the local voice test."
./run voice-synth speak --voice bruce --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --language English --seed 7 --text "Hello, this is the local voice test."
./run voice-synth speak --voice john --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --language English --seed 7 --text "Hello, this is the local voice test."
```

These are real synthesis commands: they can download weights and write audio to
`./work/`. Run them only in the authorized Linux acceptance session. Keep the
existing CPU/GPU choice consistent: prefix each command with
`TOOLBOX_VARIANT=gpu` for the existing CUDA setup, and with
`TOOLBOX_DOCKER_SUDO=1` if Docker requires sudo. Do not rebuild or pull merely to
test a source-only fix: Compose mounts the checkout at `/app`. If dependencies
actually changed, record that separately and use the matching authorized image.

Ryan does not use Bruce/John's clone pickles. 0.6B-CustomVoice supports preset
speech but rejects text direction. For `--instruct`, select
`--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` explicitly; that is a different
checkpoint and may require a separate download. Base does not support presets
or text direction. A 0.6B-Base clone prompt cannot be used with 1.7B-Base,
CustomVoice, or a different revision. Creating a 1.7B clone requires rebuilding
from the authorized reference with that target Base checkpoint. Keep 0.6B
explicit for the current Bruce/John repair.

## Tokens, caches and networking

The repository declares the Qwen snapshots public: no gated-model acceptance
or HF access token is required for those declared snapshots. Do not confuse this
with the private bearer credential used to call the headless service. Pocket's
French model is gated and has its own HF access prerequisites. Neither Pocket
installation nor a service credential fixes legacy Qwen cache lookup.

The toolbox binds `./cache` to `/cache` and `./work` to `/work`; the image sets
`HF_HOME=/cache/huggingface`. However, at the baseline the supervised Qwen child
overrides HF/cache paths into a temporary private directory. Persistent app
prompt data therefore does not prove persistent model-weight reuse. The cache
repair must be verified on its final commit without weakening worker isolation,
ownership or reap/cleanup. Do not work around it by bypassing the supervised
worker or by copying arbitrary pickles into a new runtime.

Distinguish three observations: cached reference preparation, cached clone
prompt construction, and cached model weights. Every `./run` invocation starts
a fresh container/process, so even successful disk reuse still loads the model
and generates audio. A second call may check remote metadata; a metadata request
is different from transferring the model weights again. Record actual network
bytes and cache paths rather than inferring reuse from elapsed time or a progress
bar. A newly selected checkpoint can legitimately need a first download.

On Linux, if the same download fails on the Docker bridge but succeeds with host
networking, this existing launcher option is available:

```sh
TOOLBOX_NETWORK=host ./run voice-synth speak --speaker Ryan --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --language English --seed 7 --text "Hello, this is the local voice test."
```

Add the same GPU/sudo prefixes when applicable. Host networking is a diagnostic
comparison and possible operator workaround for network transport. It does not
repair prompt metadata, make 0.6B prompts compatible with 1.7B, or prove model
cache reuse. A timeout alone does not identify DNS, proxy or firewall as the
cause. Keep network mode fixed when comparing cache behavior.

## Missing or incompatible Bruce/John prompts

`list-voices` inventories the current cache; listing a name is not proof that its
prompt passes compatibility checks. Current synthesis checks checkpoint, resolved
revision, schema, mode, pickle digest and the prompt's own trusted reference WAV
and digest. Old metadata or a missing per-prompt `.ref.wav` can cause refusal
even when a `.pkl` exists. Inspect JSON metadata without deserializing untrusted
pickles. Do not fabricate revision/hash fields or relabel a 0.6B prompt as 1.7B.

Before recovery, retain the original `voice.json`, prompt/sidecar files and
reference audio. Establish the exact source checkpoint/revision, prompt mode,
reference/transcript and voice/tone association from trusted evidence. Recovery
also needs the authorized original WAV or another verified reference; a pickle
alone does not establish missing provenance. If evidence is incomplete, rebuild
from the trusted reference rather than claiming metadata-only recovery succeeded.
Dedicated lookup/repair command additions are outside this baseline guide;
their final syntax must be documented after integration.

The existing rebuild route, when a named voice still has its trusted reference:

```sh
./run voice-clone synth --voice bruce --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --text "Hello, this is the local voice test."
```

Repeat for John's actual slug only when needed. Rebuilds write prompts and registry
data; preserve originals first. For an explicitly chosen segment of an authorized
recording in `./work/`, the existing flags use seconds:

```sh
./run voice-clone synth --ref-audio /work/bruce-source.wav --voice-name bruce-rebuilt --ref-start 12.0 --ref-end 20.0 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --text "Hello, this is the local voice test."
```

Replace file, name and bounds with verified inputs. This creates/updates a
separate named voice; it does not recover old metadata. Reference selection,
transcription and quality gates still apply. Do not add `--force-bad-ref` or
`--force` to a normal reuse test.

## Linux acceptance record

1. Record the original main checkout's exact SHA and the final tested PR SHA
   (`git rev-parse HEAD`), plus `git status --short` for each. “main” or “PR #1”
   alone is not reproducible. Keep unrelated worktrees unchanged. Use the same
   authorized script, text, seed, engine/device, dtype, network mode and inputs
   for comparisons; baseline flags may differ, so retain each exact command.
2. Record Docker context/host, CPU/GPU, Compose overlay and resolved mounts,
   toolbox image ID/digest (not just its tag), dependency identity and model
   checkpoint/resolved revision. Preserve `cache/voices`, legacy prompt data
   and actual HF snapshot identity. Record cache paths, initial sizes and whether
   each checkpoint was already present. Use isolated preserved copies if a
   comparison could rewrite the same cache; do not switch two revisions against
   a live mutable registry or erase a warm cache to manufacture a cold run.
3. Run each of Ryan, Bruce and John once, then repeat the identical invocation
   in a fresh `./run` container. Retain exit code, full logs, duration, model
   downloads/network bytes, cache bytes before/after, output WAV and
   `takes.meta.json`. Record first versus second invocation separately for each
   checkpoint. Explain metadata-only traffic versus repeated weight transfers;
   zero cache growth alone does not prove zero downloads.
4. Check both clone voices select their compatible 0.6B prompt, without silent
   cross-model fallback. If metadata repair is needed, record its prerequisites,
   preserved originals, exact applied command and before/after inventory. Run the
   synthesis pair again only after that separately recorded repair. A missing
   reference/provenance remains a blocker, not a successful recovery.
5. After the normal container/process exit, repeat from a fresh shell with the
   same mounted cache/image. Verify voice inventory, prompt/model reuse and valid
   audio persist. Retain worker exit/reap evidence; do not bypass cleanup to keep
   weights resident. Listen to all three outputs for speaker identity, intelligible
   complete text and truncation. Report failures at their actual stage.

No heavyweight acceptance was performed for this document. Source or focused
fixture tests cannot qualify Linux runtime, audio quality or service readiness.

## Headless service remains a separate, unqualified route

The service implements Base 0.6B EN/FR cloning and a `docker-qwen`
`install`/`up`/`status`/`stop` family. It is not Ryan's CustomVoice toolbox route.
Its managed model is pinned, inference is offline, and provisioning/building are
separate explicit actions. Service voice assets and identity differ from the
legacy registry: no supported command magically imports a legacy `.pkl` prompt.
A service journey must upload the reference, prepare/audition, save and generate
through its existing API/CLI/MCP contract. Its CPU path uses transcript-free
x-vector cloning, so legacy full/ICL prompt behavior is not implied.

The [Qwen CPU status](agent-interface/QWEN-CPU.md#current-status--2026-10-27)
retains unresolved installed activation timeout and fresh capacity/admission
requirements. Real inference, cancellation/reap, restart/history, public-client
journey and listening acceptance still require authorized execution on the exact
integrated source and image. Legacy success would not close those blockers or
promote release readiness.
