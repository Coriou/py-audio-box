For the distinct ordinary Mac/Linux Docker-client lane, see
[DOCKER-VOLUMES.md](DOCKER-VOLUMES.md), with family-specific worked results in
[CPU-QUICKSTART.md](CPU-QUICKSTART.md) and [POCKET-QUICKSTART.md](POCKET-QUICKSTART.md).
The strict host route below retains its
original Linux-root, host-path and host-inspection requirements.

# Packaged headless installation

For daemon-side CPU policy generation and the explicit Intel Mac Linux-VM
boundary, use [CPU-DOCKER.md](CPU-DOCKER.md).

## CPU candidate and current evidence

This is a headless CPU candidate: use authenticated HTTP, CLI or MCP, with no
browser UI. For an ordinary Docker client (including a suitable GitHub/cloud
runner), start with [DOCKER-VOLUMES.md](DOCKER-VOLUMES.md). The protected Linux-host
route below is a separate installation lane; its host-root requirements do not
apply automatically to the ordinary Docker-client route.

For a build/provision/measure/install worked example, use
[CPU-QUICKSTART.md](CPU-QUICKSTART.md). The pinned French Kokoro engine and PCM24
converter have passed historical actual contained execution with their recorded runtime image.
The `actual-install14` installed HTTP speech/export/download/stop/restart journey
and actual CLI/MCP checks have passed on Linux/amd64 under OrbStack; see the
[bounded installed report](reviews/CPU-KOKORO-INSTALLED-r407-parent.md). This is not
a CUDA/Metal or complete-release claim, and an engine profile alone does not
establish an installed journey.

Pocket cloning is a separate installation; see [POCKET-QUICKSTART.md](POCKET-QUICKSTART.md).
The ordinary Docker-client route is
`docker-pocket`; the protected-host route is also documented below. Actual
contained Pocket reference synthesis and the ordinary Docker-client installed
upload/prepare/save/generate/download/restart journey passed in
`pocket-public-install05` at source `7c56f72`. Additional checks passed two distinct
reference workflows, running-job cancellation through cancelled/reaped/idle ACK,
a second same-container restart retaining voices/jobs, and new generation from
both saved versions after restart with SHA-256 checked downloads. Identical-key
replay returned the same receipt; changed payloads with that key returned 409.
See the [bounded CPU release report](reviews/CPU-RELEASE-r407-parent.md).
The fresh public05 nine-job journey also passed its final clean stop. Its retained
history used 872 of 2048 inspection rows; this is finite measured acceptance,
not a guarantee of unlimited retained jobs or history.

Qwen CPU has a separate finite `docker-qwen` source family; see its
[current status and historical evidence](QWEN-CPU.md#current-status--2026-10-27).
Current measurement/startup is disk-blocked; historical actual05 measurement/import/init
passed, but activation timed out. No qualified Qwen worked recipe is claimed.

Listening for the installed runs above remains **UNMEASURED**, the protected-host lane **UNRUN**, and the CPU
regressions/source archive checks have passed. CUDA and Apple Metal
remain **UNQUALIFIED**; neither is a gate for this CPU candidate. Published
headless image availability is not established: build and enroll exact local
images from the reviewed source.

This source unit supplies the finite `speech-box init`, `complete`, `up`,
`status`, and `stop` command. The ordinary-client families are `docker`,
`docker-export`, `docker-pocket`, and `docker-qwen`, each with finite
`install/up/status/stop` commands; Qwen remains operationally unqualified. Qualification of this protected Linux-host lane
remains separate from the passed ordinary Docker-client Kokoro journey. Do not interpret an
injected test, dependency resolution, source archive or Dockerfile as that evidence.

First target: x86_64 Linux, trusted Python 3.11, local rootful Docker at
`unix:///var/run/docker.sock`, ordinary local volumes, cgroup v2, no remapping,
private PID/cgroup namespaces. Docker Desktop, rootless/remote daemons and
unproved mount layouts refuse. The host administrator owns Docker and protected
launch authority; no container receives the Docker socket, host PID namespace,
source checkout, protected receipts in control, or host authority directory.

## Distribution and dependencies

Build the deterministic distribution using stdlib only:

```
python3 scripts/headless-install.py archive --output /absolute/new/speech-box.zip
```

Extract it into a trusted, non-writable installation directory such as
`/opt/speech-box`. It contains its transitive service, worker, contracts and lib
source/data, pinned dependency locks and closed release build context. The root
`speech-box` file is the command (invoke with the trusted interpreter; ZIP tools
may not restore executable mode). No repository or repository PYTHONPATH is
needed. `--help` and archive creation need only stdlib. This is a source
host distribution; it does not bundle a Python interpreter or compiled wheels.

Provision an isolated trusted **Python 3.11** environment separately using the
included `service/requirements.lock`, `pip --require-hashes`, and an explicitly
supplied matching offline wheelhouse (`--no-index --find-links`). Invoke the root
command with that environment's interpreter. Runtime dependency installation is
never attempted by `init/up/stop/status/complete`. The legacy Poetry project is
unchanged; `[tool.speech-box-headless]` describes this separate archive format.
The lock was resolved with existing versions constrained for CPython 3.11,
x86_64 manylinux_2_28 against official PyPI metadata; no compatibility inference
is made from the Python 3.14 test environment. Installing these dependencies or building an image alone does not qualify the
installed runtime; use the end-to-end checklist below.

`service/Dockerfile.release` has `control`, `worker`, `worker-export`, `worker-pocket` and
`worker-qwen` targets using the existing
pinned Python 3.11 CPU base. The Kokoro worker installs the unchanged `requirements/kokoro.lock`
and control/service lock together. Only explicitly allowed source/data/locks enter
the build context. Image construction is a separately authorized release operation.
Build/load control and the selected family's worker before installation; the
control image is reused for finite helpers. Qwen source inclusion does not
qualify its installed runtime. Enroll their exact local
`sha256:` image IDs. `up` performs no build, download, pull or model provisioning.
`compose.release.yml` documents the fixed policy; use the finite command for the
transaction, rather than Compose up/down, which cannot perform registration or
preserve the launch protocol.

## Explicit provisioning and init

Supply the real enrolled configuration: one CPU Kokoro profile (or the explicit Pocket alternative below), finite resource
budget, explicit principal/context templates and exact verified bundle manifest,
checkpoint and revision. No fixture profile or synthetic qualification is supplied
by this package. Provision the sealed model tree at the host policy's explicit
model source (10001:10001; directories 0555, files 0444). Supply canonical config
JSON at the explicit checked input source (root:10001, 0440), and credentials at
the explicit credential source (root:10001, 0640), in the existing
`installed_credentials` format. Credential contents never enter arguments,
provenance, Docker environment, command output or launcher logs.

Provide four distinct, empty, root:root 0700 local named-volume roots with no
consumers. Supply scope JSON containing the exact daemon/context and volume
name/driver/creation identities. Supply the complete explicit `host_policy`
record defined in `service/install_policy.py`: exact volume and input paths and
stat identities, public config/model hashes, finite CPU/tmpfs budgets and
loopback control port. The credential record carries identity only, never its
hash or contents. Existing reviewed host inspection corroborates these claims,
including disjointness/mountinfo; metadata assertions are not host evidence.

```
/opt/speech-box/venv/bin/python /opt/speech-box/speech-box init \
  --configuration /protected/install.json --scope /protected/scope.json \
  --host-policy /protected/host-policy.json
```

Init exclusively creates
`/var/lib/py-audio-box/p1-host-authority/installed/<scope-digest>/host-policy.json`
and `launch.lock` (root:root 0600, root:root 0700 directory), retaining partial
creation on failure. Supplied inputs are checked, never silently copied,
repaired, reminted or adopted. All subsequent operations hold the checked
nonblocking launch lock and reread exact protected policy/identities.
Initialization uses the reviewed create-only split-owner transaction and
permanently dropped Core child, not an initializer container or duplicated SQL.

For an interrupted deterministic initialization, use `complete` with the exact
`--initialization-digest` from the retained initialization record and the same
configuration/scope. It reuses the existing protected authority; it does not
recreate partial authority or infer a missing digest. Ambiguous Core creation,
changed bytes/identities and already enrolled installations refuse.

## Finite operation and retained failure

Use the same `--configuration` and `--scope` for `up`, `status` and `stop`.
First up retains exclusive creation intent and each exact new container ID,
using `docker container create --pull=never`. A crash during creation retains
an obligation; the next invocation refuses instead of recreating lost IDs.
Clean restart starts the same retained containers. Missing/recreated historical
containers, dirty shutdown, changed inputs/images and unproved consumers refuse.

Up starts final control in its bounded waiting factory phase, then the final
PID1 worker. It waits for healthy authenticated hello through the reviewed
control-identity child, performs protected registration plus same-ID notification,
and checks the actual incarnation again after bounded owner/listener startup.
The fixed exec readiness reader returns only nonsecret registration identity;
the result is checked against host observations. No control DB owner/listener
exists before the activation gate. Existing Dispatcher reconciliation, resource,
integrity, orphan and cancellation/reap guards remain authoritative.

Stop requests TERM on exact control, waits for normal exit (including owner drain
and close), verifies clean terminal/ack/reap history through existing readers,
then TERM/waits exact worker and rechecks stopped topology and history under
ordinary locks. It never uses Docker stop's KILL escalation. A timeout, failed
close, dirty receipt, unknown exit or failed identity bracket retains containers,
volumes and authority. No force, reset, remove, prune, maintenance or lane reuse
exists in this command. Status is a checked instantaneous observation, not a
future health guarantee. Active work can make stop refuse within its finite
budget; that is retained uncertainty, not proof of death or released capacity.

Finite journal/UID/history budgets remain. Recovery/destructive maintenance is
not part of this installer. For this protected Linux-host lane, actual permissions, setgid inheritance, peer/mount/
process identities, installed authenticated journeys, CPU Kokoro audible speech,
retained voice/artifact reuse, cancellation/reap and clean/failed restart must
still be qualified on the separately authorized Linux target.


## First configured Kokoro voice

A fresh installation has an empty voice catalogue. Startup does not register voices.
With an active `read` + `production-write` credential, read `/skeleton` (CLI `status`,
MCP `speech_skeleton_status`) for `configured_presets`, exact model/bundle identity
and activation status. This installation supports the configured Kokoro preset for
generation; clone preparation is unavailable. Registration does not qualify hardware.

Explicitly POST `/v1/operations/voice_preset_register` with `Idempotency-Key` and
`{"preset":"ff_siwis","voice_id":"my-voice","expected_revision":0}` (replace the
selector with the advertised configured selector). CLI:
`python -m clients.cli voice preset-register --request '{"preset":"ff_siwis","voice_id":"my-voice","expected_revision":0}' --idempotency-key first-preset`.
MCP: `speech_voice_preset_register` with those fields plus `idempotency_key`.
The response contains the immutable `version.id`; use that ID for `speech_generate`.
List with `voices list`, inspect with `voice version get ID`, then use the existing
job → artifact → asset content commands to retrieve and verify the audio.
Identical replay of the same key/request returns the same result. A changed request
with that key conflicts; a fresh key cannot overwrite the immutable version.
After a clean completed shutdown and checked same-volume restart, list and reuse
the retained version without registering again. Arbitrary imported versions do not
inherit the configured preset's execution authority. The ordinary Docker-client
Kokoro journey has passed speech and clean restart checks. Listening acceptance,
cancellation/reap, failed-restart handling and the protected-host lane remain
separate checks.

## Explicit alternative: Pocket CPU clones

The same archive also supports **one** `pocket-cpu` profile under ownership policy
`docker-pocket-split-v2` and enrollment scope `owned-pocket-v2`.
The legacy `docker-pocket-split-v1` / `owned-pocket-v1` policy remains unchanged;
do not relabel or migrate retained v1 state. Select this instead
of Kokoro when creating a fresh installation. An existing enrolled Kokoro
configuration remains immutable; changing its policy, bundle or profile is not an
upgrade path. For the protected-host route, rootful Linux, Python 3.11, local
volumes and every existing host, registration, activation and retained-history
check still apply.

### Ordinary Docker-client Pocket commands

The archive also exposes the separate finite `docker-pocket` command. It selects
`installed-pocket-daemon-volumes-v2` explicitly, with configuration version 1,
ownership policy `docker-pocket-split-v2`, scope `owned-pocket-v2`, and the existing
publication/store format 1. The ordinary Docker-client installed Pocket checks
listed above have passed; the protected-host lane remains **UNRUN**.
The command does not qualify a Docker platform, model, voice or image.

Prepare the same closed request envelope documented in
[DOCKER-VOLUMES.md](DOCKER-VOLUMES.md), supplying the Pocket configuration, exact
Pocket bundle files, private credential file and already loaded immutable helper,
control and Pocket worker image IDs. Use a fresh handle and fresh named volumes:

```sh
./speech-box docker-pocket install --context YOUR_LOCAL_CONTEXT --request pocket-request.json --handle pocket-installation.json
./speech-box docker-pocket up --handle pocket-installation.json
./speech-box docker-pocket status --handle pocket-installation.json
./speech-box docker-pocket stop --handle pocket-installation.json
./speech-box docker-pocket up --handle pocket-installation.json
```

Installation imports the supplied bytes and initializes the fresh enrollment.
Later commands retain the same installation/container IDs; clean stop and retained
history checks precede restart. A partial operation retains its receipts and
refuses automatic repair. This route uses trusted Docker inspection and real
container-local observations; it does not require protected-host root authority
on the operator machine. Every daemon/security requirement of the ordinary
Docker-client route still applies. The worker receives only a read-only named
volume subpath `core/assets` at `/references/assets`; unsupported subpath behavior
refuses. No build, pull, acquisition, migration or readiness assertion occurs in
these commands. The authenticated upload/prepare/save/generate API, CLI and MCP
workflow below is shared by both Pocket routes.

Use the `control` image target and the separate `worker-pocket` target in
`service/Dockerfile.release`, then enroll their actual immutable local image IDs.
The Pocket target installs `requirements/pocket-installed.lock` with hash checking;
it contains the existing Pocket lock plus service-only dependencies. Pocket pins
win overlapping distribution names (in particular worker FastAPI 0.143.1 versus
control 0.143.2). Every stanza/hash comes verbatim from the existing locks. The public target has been built and completed actual contained Pocket CPU
execution; installed public acceptance is tracked separately in the Pocket
quickstart. Pocket runtime declarations are `pocket-tts=3.3.0`,
`torch=2.10.0+cpu`, `tokenizers=0.23.3`; reviewed bundle runtime metadata must match.
No package hash, model asset or image ID is generated by the installer.

Use the existing explicit managed provisioning boundary with an independently
reviewed Pocket expected manifest: immutable 40-hex upstream revision, cloning
weights, portable local config, tokenizer files, exact bytes/SHA256/origins and
license/access metadata. The installed runtime manifest is the closed output of
`bundle_manifest`; descriptor-only `config` and `allowed_origins` remain provisioning
policy. Seal the provisioned model tree as 10001:10001, directories 0555/files 0444.
The host inventories the entire tree and streams Pocket weight hashes with finite
byte/time limits. No download or provisioning occurs during startup.

Configuration retains the existing closed fields `version`, `ownership_policy`,
`profiles`, `budget`, `contexts`, `bundles`, `images`, `mounts`. Supply one Pocket CPU
profile with both prepare and clone-generation compatibility, and one trusted
bundle/profile/budget template per enrolled principal. The protocol-shaped template
is validated but its original reference is not opened at startup. The reference-free
factory resolves each selected owned upload or saved immutable voice on Core's
owner thread. Public requests cannot supply filesystem paths, descriptors, managed
roots or qualification. No context replacement is needed between preparing and
using multiple saved voices.

The Pocket worker receives exactly the control volume's `core/assets` subdirectory
at `/references/assets`, read-only; it receives neither the control root nor its DB.
`compose.release.pocket.yml` records this alternative topology; the finite launcher
constructs and inspects the exact derived bind source, destination and read-only
flag. The checked host also verifies the literal private Core/assets directories.
Managed media remains UID/GID 10001 and mode 0600 inside a mode 0700 assets directory.
A bounded, nonresident UID10001 helper opens only the selected no-follow asset and
transfers an O_RDONLY descriptor. The supervisor requires its successful reap,
independently verifies descriptor ownership/mode/link count, complete bytes/hash/PCM
and bundle binding, and passes only that descriptor plus the execution socket to
the UID20000+ child. The child verifies the descriptor again and makes its existing
private conditioning snapshot. No global chmod, extra group, read-bypass capability,
public reference path or database mount is used. Actual kernel/UID/SCM_RIGHTS and
mount enforcement are separate qualification obligations.

From an empty catalogue, upload each WAV through the authenticated asset-upload
API/CLI, then call `voice_prepare` with its asset ID, exact frame-aligned ticks,
empty `reference_text`, explicit `audition_text` and the configured Pocket identity.
CLI supports `voice prepare --request … --audition-text … --idempotency-key …`;
MCP exposes the same closed request and idempotency field. Read `voice_results` and
retrieve the audition asset; `voice_save` retains the prepared reference, not the
audition. Save two candidates to different voices, then alternate `speech_generate`
using their immutable version IDs. After clean completed shutdown, use `up` with
the same retained volumes/IDs and reuse both versions. Do not seed Core or swap a
context to perform this journey. Kokoro `voice_preset_register` is unavailable for
this profile.

Discovery reports Pocket preparation/save/generation and the configured profile.
`activation_ready` describes checked owner activation; `ready_for_speech` additionally
requires the declared qualified/measured/ready Pocket profile. Installation,
provisioning, enrollment and registration do not mint qualification. The ordinary
Docker-client lane passed the installed speech, cancellation/reap and same-volume
restart checks recorded above. The protected-host lane remains **UNRUN**;
perceptual clone/reference fidelity remains **UNMEASURED**, and dedicated
cold/warm resource characterization remains separate. An interrupted nonempty
conditioning directory remains UNKNOWN and retained;
there is no recursive removal, general recovery or relaxed restart guard.


## Pinned source onboarding

This review uses audited commit `9fff9c1faf9c6ca0f66f95d3d6065e82aea732cd`;
it is not an approved release. A release owner must later name the approved
candidate and its artifact identities. On a suitable Linux/amd64 Docker runner:

```sh
git clone https://github.com/Coriou/py-audio-box.git
cd py-audio-box
git checkout --detach 9fff9c1faf9c6ca0f66f95d3d6065e82aea732cd
python3.11 -m venv .venv-client
.venv-client/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
```

Choose one lane and obtain its model access before execution:
[Kokoro presets/export](CPU-QUICKSTART.md) or [Pocket French cloning](POCKET-QUICKSTART.md).
Their worked commands build/provision/measure/install and retrieve the first
verified artifact. Supply an explicit existing local Unix Docker context and a
fresh output path beneath a persistent private evidence parent outside `/tmp`.
Keep credentials, model access and reference audio private. Apply fresh resource
admission and the chosen recipe's exact limits before any heavy operation.
For Qwen, read the [blocked status and remaining evidence](QWEN-CPU.md#current-status--2026-10-27)
instead of treating source availability as a ready recipe.

The legacy Poetry non-package toolbox and its `voice-tools` pulls are separate
from this headless client/archive. No published headless image is established;
there is no headless registry pull shortcut. The standalone `speech-box` archive
path above remains separate: Pocket's worked script needs a Git checkout even
though it is included in that archive. Archive creation is source evidence only.

After the chosen recipe, use its printed URL and privately configured credential:

```sh
.venv-client/bin/python -m clients.cli status
.venv-client/bin/python -m clients.cli discover
.venv-client/bin/python -m clients.cli operation-detail speech_generate
.venv-client/bin/python -m clients.cli asset content ID --output /absolute/private/new-output.wav
```

Set `SPEECH_URL` and `SPEECH_CREDENTIAL` privately in the client host; replace `ID`
with the returned asset ID and use a new output file. CLI `--url` is global and
precedes the subcommand. `--request` accepts a JSON object, not a filename;
idempotent writes require `--idempotency-key`. For MCP SDK setup, follow the
selected quickstart's separate environment, not the client lock alone.

## GitHub/cloud agent CPU acceptance checklist

Use a disposable, suitable Linux Docker runner with enough disk for pinned
images/models and a persistent evidence directory outside `/tmp`, for example
`$HOME/.local/state/py-audio-box/release-r407`. Keep model/license access and
credentials in the runner's secret facility. Never include credentials in
arguments, logs, reports or uploaded artifacts. A runner without Docker or the
required daemon policy cannot qualify this installation.

Run heavy builds and actual synthesis serially. Bound source tests to 2 CPUs and 1 GiB memory. Actual engine/install checks use
the explicit limits in their selected recipe (Kokoro 2 GiB, Pocket up to 4 GiB
per container). Qwen uses its selected configuration and fresh parent-owned
admission, not Pocket's 4 GiB/180-second recipe bounds. Retain explicit timeouts
and finite output limits. A
resource refusal or timeout is a failed/incomplete check, not permission to
silently increase limits. Do not run concurrent model engines on this host.

1. Record the source commit, platform, exact image IDs, dependency locks, model
   manifest/revision/hash and selected installer policy. Build/load the images
   explicitly; follow the chosen installation guide without fixture credentials
   or fabricated qualification fields.
2. Initialize fresh volumes, run `up`, then read authenticated `status`, `discover`
   and operation schemas. Record activation and execution availability separately.
3. Register the advertised Kokoro preset through `voice preset-register`; retain
   the request, idempotency key and returned immutable version ID. For the separate
   Pocket or separately qualified Qwen lane, upload a reference, prepare, retrieve the audition and save the
   voice through public operations instead.
4. Submit `speech generate` using the returned version and the advertised schema.
   Follow `job events` / bounded `job wait` until terminal; a healthy worker or
   accepted job alone is insufficient. Retrieve the artifact, asset metadata and
   `asset content --output` to an exclusive file. Verify bytes/hash, decodable WAV,
   nonempty frames and duration. Record listening separately from machine checks.
5. In the Kokoro/export-v2 lane, submit `audio export` using its public schema, wait for completion and retrieve
   the exported asset. Replay each identical request/idempotency key and verify
   the retained result; a changed request with that key must conflict.
6. Exercise cancellation and verify terminal/reap evidence. After completed work,
   perform clean `stop` and checked same-volume `up`; reuse the retained voice and
   read the prior job/artifact before a new generation. Preserve refusal evidence
   and partial state rather than deleting it to obtain a pass.

Use `python -m clients.cli --url <loopback-url> …` with `SPEECH_CREDENTIAL` supplied
privately. `operation-detail speech_generate` and `operation-detail audio_export`
provide current closed request schemas. Equivalent authenticated HTTP/MCP calls
are valid; neither a web UI nor direct Core/database writes belong in acceptance.
The final report must distinguish source checks, actual engine execution, actual
installed public journeys and human listening, and mark every unrun step UNRUN.

For MCP reference uploads, optionally configure `SPEECH_UPLOAD_ROOT` as an absolute
operator-authorized directory in the agent host. It grants filesystem read scope
and exposes `speech_asset_upload_file` with a relative `path` (no `..`, no symlink
following). Otherwise upload through CLI `asset upload FILE` or authenticated HTTP
and pass the returned asset ID. MCP direct content is limited to 3072 decoded
bytes per call; retrieve full verified audio with
`python -m clients.cli asset content ID --output FILE` outside model context.

### Evidence snapshot — 2026-10-27

Audited HEAD `9fff9c1faf9c6ca0f66f95d3d6065e82aea732cd`; reported Unit CI
`40488015700`: 4573 tests / 1600 subtests, 11 shards. These unit results do not
qualify any installed lane. Historical rows retain their own source/image IDs.

| Lane | Engine measurement | Import/init | Up/activation | Public API/CLI/MCP | Cancellation/reap | Restart/history | Listening |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kokoro export-v2, actual-install14 | Historical pass | Historical pass | Historical pass | Historical pass | Separate acceptance needed | Historical bounded pass | Installed run unmeasured |
| Pocket v2, public05 (`7c56f72`) | Historical pass | Historical pass | Historical pass | HTTP pass; CLI/MCP contracts shared, no independent full cloning pass claimed | Historical pass | Historical bounded pass | Installed run unmeasured; backend pilot fidelity failures |
| Qwen CPU | Current unrun/disk-blocked; actual05 ~135 s historical pass | actual05 historical pass | Historical activation timeout unresolved; corrected startup unrun | Unqualified | Unqualified | Unqualified | Prior numbers/liaisons rejected; no approval |

Fresh current Qwen admission, startup, public cloning/download, cancellation/reap,
restart/history and human listening remain parent-owned gates. Retain exact source,
image, lock and model identities, failures and terminal outcomes. Older Apple
dependency CI passed with MPS skipped; Apple synthesis/Metal remain unqualified.
CUDA is deferred/unqualified. Public release qualification is not ready. The owner
must also state the project source license and retain selected model notices and
attribution before distribution; model descriptors do not choose a project license.
