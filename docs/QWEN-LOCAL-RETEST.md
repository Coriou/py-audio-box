# Qwen local: existing voices, troubleshooting and retest

Use the existing `./run` toolbox to repair and retest Ryan, Bruce and John. A new
“Qwen local” service installation is not required for that workflow. Docker and
the existing CPU or CUDA toolbox image provide Python and the Qwen runtime.
This guide covers the integrated PR #1 cache repair and registry lookup/repair
commands, plus selection-specific reference preparation caches. Acceptance must
record the exact tested commit containing the final integrated repairs. It is an
operator checklist, not a record of a successful runtime test.

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
`HF_HOME=/cache/huggingface`. The repaired legacy Qwen child
retains configured persistent HF retrieval paths (`HF_HOME`, hub cache overrides
and `TRANSFORMERS_CACHE`); its home, temporary and compiler state remain private.
Managed service backends retain private/offline model handling. Persistent app
prompt data alone does not prove persistent model-weight reuse. Verify the repair
on its final commit without weakening worker isolation, ownership or reap/cleanup.
Do not work around it by bypassing the supervised worker or by copying arbitrary pickles into a new runtime.

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
Registry lookup accepts case-insensitive slugs, aliases, unique display names
and unambiguous slug prefixes; `voice-synth speak` also accepts unique legacy
prompt IDs. Ambiguous matches fail. Prefer the canonical slug from the
inventory. An alias adds another lookup name without renaming files; it writes
`voice.json` immediately (there is no alias dry-run):

```sh
./run voice-clone add-alias --voice bruce --alias bruce-local
./run voice-synth list-voices --json
```

For an existing orphan directory whose `voice.json` is missing, use an explicitly
trusted original metadata backup placed in `./work/`. Preserve a separate complete
copy of the original directory, prompts, sidecars, references and backup before
any write. Preview first; only after checking the result, apply explicitly:

```sh
./run voice-clone repair-registry --voice bruce --metadata /work/bruce-voice.backup.json
./run voice-clone repair-registry --voice bruce --metadata /work/bruce-voice.backup.json --apply
./run voice-synth list-voices --json
```

The default repair is a read-only dry-run. Apply atomically restores only missing
`voice.json`; it refuses an existing registry file and does not rewrite prompts
or deserialize pickles. The backup must identify the slug and all prompt/tone
indexes. Retained prompt sidecars need pinned checkpoint/revision/schema,
compatible mode, pickle digest and trusted per-prompt reference bytes/digest.
Hash checks detect mismatches; they do not establish provenance or make an
untrusted pickle safe. Repair cannot invent absent evidence or repair incompatible
prompt payloads. If this evidence or a trusted complete backup is absent, use the
trusted-WAV rebuild route below with a new voice name instead. Repeat for John's
actual slug only when applicable.

When a registered voice still resolves and retains a trusted reference, the
existing rebuild route is:

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
separate named voice; it does not recover old metadata. Preparation caches now
separate manual bounds from automatic selection and include Whisper model, language and transcript/embedding options. Changing bounds
selects separate segment/transcript artifacts even for the same recording; old
source-only preparation artifacts remain preserved but are not reused as proof
of the new selection. Identical preparation options and bounds reuse their cache.
Bounds must be paired, finite, ordered, nonnegative and within the normalized
recording. Reference selection, transcription and quality gates still apply.
Do not add `--force-bad-ref` or `--force` to a normal reuse test.

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
   zero cache growth alone does not prove zero downloads. For warm-cache acceptance,
   require no model-weight downloads in either invocation when that exact checkpoint
   is already present; any required new download falls outside that acceptance pair.
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

## Headless two-model service: local preview and acceptance

The current service setup accepts sealed Base 0.6B and CustomVoice 0.6B
bundles for the same authenticated principal. Base cloning uses the exact
reference WAV and transcript with ICL conditioning; Ryan uses the installed
CustomVoice preset. Its `docker-qwen` family provides `install`, `up`,
`status` and `stop`. The managed bundles are pinned and inference is offline.
Service voice assets and identity differ from the legacy registry: no
supported command imports a legacy `.pkl` prompt. The named-voice client
uploads a reference, prepares and auditions it, saves a version and generates
speech. Ryan can be spoken through the same client after preset registration.

The earlier [Qwen CPU status](agent-interface/QWEN-CPU.md#current-status--2026-10-27)
records an installed activation timeout on an older candidate. The
`967ef93a67f31d5cf629b318e63e8644a0340d36` source activation fix has
not been qualified by a fresh installed run. Capacity/admission, actual
inference, cancellation/reap, restart/history, public-client journey and
listening acceptance require authorized execution on the exact integrated
source and image. Legacy success would not promote release readiness.

### Fresh headless-service journey for an authorized Linux host

Use this sequence only after the operator has identified a qualifying host,
existing sealed Base and CustomVoice bundles, loaded images, a trusted mono
PCM16 reference WAV, and its exact transcript. The local `up` lane is an
**unqualified preview**, with no production qualification receipt. It does not
establish the quality or license of a particular reference recording. Record
the recording's source and permission separately. Do not use a legacy prompt
pickle as the service reference.

1. Record `git rev-parse HEAD`, `git status --short`, host architecture, Docker
   version/context, cgroup mode and limits, image IDs, dependency identity,
   bundle manifests/revisions, disk headroom and reference WAV digest. Use the
   final integrated commit, not `967ef93a67f31d5cf629b318e63e8644a0340d36`
   as a proxy for later edits. Keep the actual commands and terminal output.
2. Select an absolute Python 3.11 executable on the host. Verify that ordinary
   and `sudo` execution resolve to the same executable, then run the local
   setup with absolute existing inputs and a **new** handle. Replace the
   illustrative paths and image references below. This launcher is the actual
   `speech-box` entrypoint; local setup needs `sudo` for its dedicated cgroup.

   ```sh
   PYTHON_311='<absolute-python-3.11-executable-on-host>'
   test "${PYTHON_311#/}" != "$PYTHON_311" && test -f "$PYTHON_311" && test -x "$PYTHON_311" || exit 1
   python_real="$("$PYTHON_311" -c 'import os,sys; assert sys.version_info[:2] == (3,11); print(os.path.realpath(sys.executable))')" || exit 1
   test "$(sudo "$PYTHON_311" -c 'import os,sys; assert sys.version_info[:2] == (3,11); print(os.path.realpath(sys.executable))')" = "$python_real" || exit 1
   sudo "$PYTHON_311" scripts/headless-install.py up \
     --clone-bundle /absolute/sealed/base --ryan-bundle /absolute/sealed/customvoice \
     --control-image LOCAL_CONTROL_IMAGE --worker-image LOCAL_WORKER_IMAGE \
     --helper-image LOCAL_HELPER_IMAGE --reference-wav /absolute/trusted/reference.wav \
     --reference-text 'The exact words in the reference recording.' \
     --handle /absolute/private/new-handle.json
   ```

   The [local startup instructions](agent-interface/QWEN-RESIDENT-SERVICE.md#local-startup-before-qualification)
   describe the required cgroup and bundle preflight. Stop on the first
   failure or timeout; preserve its handle and receipts for diagnosis. Do not
   automatically retry setup or replace a retained failure with a new handle.
3. As the ordinary client account, use the returned `client_command` arguments,
   including `--installation` when the root is custom, but run them with the
   verified `"$PYTHON_311"` in place of the printed `python3` executable.
   The following are the actual `clients.cli` verbs; replace paths and text
   with the authorized recording and client root.

   ```sh
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root status
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root enroll --name bruce --wav /absolute/trusted/bruce.wav --reference-text 'Exact words in Bruce recording.' --audition-text 'A short audition.'
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root speak --voice bruce --text 'Hello from Bruce.' --output /absolute/new-bruce.wav
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root speak --voice Ryan --text 'Hello from Ryan.' --output /absolute/new-ryan.wav
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root voices list
   "$PYTHON_311" -m clients.cli --installation /absolute/private/client-root jobs list
   ```

   Do not print or copy `control.key`. The client verifies downloaded bytes
   against the service SHA-256 and refuses an existing output path. Retain the
   returned voice version, job, artifact and asset IDs, result JSON, and output
   digest. A timed-out or uncertain submission needs inspection using those
   confirmed handles and the [resume rules](SERVICE-VOICE-CLIENT.md), not a
   blind repeat. Record whether the service actually generated audio; status
   alone does not prove it.
4. Validate the resulting file as a nonempty PCM WAV and listen to it for
   intelligible, complete speech and the intended voice. For example, run this
   against the new output (replace the path):

   ```sh
   "$PYTHON_311" -c 'import sys,wave; w=wave.open(sys.argv[1],"rb"); print({"channels":w.getnchannels(),"sample_width":w.getsampwidth(),"rate":w.getframerate(),"frames":w.getnframes()}); assert w.getnchannels()==1 and w.getsampwidth()==2 and w.getnframes()>0; w.close()' /absolute/new-bruce.wav
   ```

5. Use `sudo "$PYTHON_311" scripts/headless-install.py docker-qwen stop --handle /absolute/private/new-handle.json`
   and preserve terminal cleanup/reap evidence. After confirmed stop, use
   `sudo "$PYTHON_311" scripts/headless-install.py docker-qwen up --handle /absolute/private/new-handle.json`
   once, then client `status`, `voices list`, and `jobs list` with the same
   client root and `"$PYTHON_311"`.
   Verify the saved voice, bundle identity and prior job remain, then speak to
   a **new** output path and validate/listen again. Finally run
   `sudo "$PYTHON_311" scripts/headless-install.py docker-qwen stop --handle /absolute/private/new-handle.json`
   and retain the second terminal cleanup/reap evidence. Stop on the first
   failure; do not infer persistence from a source fixture or a still-running
   process.

Report each stage as PASS, FAIL or NOT RUN with the exact SHA, image IDs,
installed dependency identity, timestamps and evidence location. A local
preview success remains unqualified. Production qualification requires its
separate measured setup and receipt, as well as authorized real-audio review.
