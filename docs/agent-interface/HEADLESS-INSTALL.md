For the distinct ordinary Mac/Linux Docker-client lane, see
[DOCKER-VOLUMES.md](DOCKER-VOLUMES.md). The strict host route below retains its
original Linux-root, host-path and host-inspection requirements.

# Packaged headless installation

For daemon-side CPU policy generation and the explicit Intel Mac Linux-VM
boundary, use [CPU-DOCKER.md](CPU-DOCKER.md).

This source unit supplies the finite `speech-box init`, `complete`, `up`,
`status`, and `stop` command. Independent source review and actual installed
Linux/CPU speech qualification remain pending. Do not interpret an
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
is made from the Python 3.14 test environment. Actual wheel installation, Python
3.11 execution, image build and installed runtime remain qualification work.

`service/Dockerfile.release` has control, Kokoro CPU worker and alternative Pocket CPU worker targets using the existing
pinned Python 3.11 CPU base. The Kokoro worker installs the unchanged `requirements/kokoro.lock`
and control/service lock together. Only explicitly allowed source/data/locks enter
the build context. Image construction is a separately authorized release operation.
Build/load the two images before installation and enroll their exact local
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
not part of this installer. Actual permissions, setgid inheritance, peer/mount/
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
`{"preset":"af_heart","voice_id":"my-voice","expected_revision":0}` (replace the
selector with the advertised configured selector). CLI:
`python -m clients.cli voice preset-register --request '{"preset":"af_heart","voice_id":"my-voice","expected_revision":0}' --idempotency-key first-preset`.
MCP: `speech_voice_preset_register` with those fields plus `idempotency_key`.
The response contains the immutable `version.id`; use that ID for `speech_generate`.
List with `voices list`, inspect with `voice version get ID`, then use the existing
job → artifact → asset content commands to retrieve and verify the audio.
Identical replay of the same key/request returns the same result. A changed request
with that key conflicts; a fresh key cannot overwrite the immutable version.
After a clean completed shutdown and checked same-volume restart, list and reuse
the retained version without registering again. Arbitrary imported versions do not
inherit the configured preset's execution authority. Actual offline speech, listening,
cancellation/reap and installed restart qualification remain separate host checks.

## Explicit alternative: Pocket CPU clones

The same archive also supports **one** `pocket-cpu` profile under ownership policy
`docker-pocket-split-v1` and enrollment scope `owned-pocket-v1`. Select this instead
of Kokoro when creating a fresh installation. An existing enrolled Kokoro
configuration remains immutable; changing its policy, bundle or profile is not an
upgrade path. Rootful Linux, Python 3.11, local volumes and every existing host,
registration, activation and retained-history check still apply.

Use the `control` image target and the separate `worker-pocket` target in
`service/Dockerfile.release`, then enroll their actual immutable local image IDs.
The Pocket target installs `requirements/pocket-installed.lock` with hash checking;
it contains the existing Pocket lock plus service-only dependencies. Pocket pins
win overlapping distribution names (in particular worker FastAPI 0.143.1 versus
control 0.143.2). Every stanza/hash comes verbatim from the existing locks. This
source composition has not been installed or resolved again and does not establish
runtime compatibility. Existing Pocket runtime declarations are `pocket-tts=3.3.0`,
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
provisioning, enrollment and registration do not mint qualification. Actual offline
speech, audible clone/reference fidelity, cold/warm resource budgets, cross-UID
permissions, cancellation/reap and same-volume restart remain UNRUN for this source
lot. An interrupted nonempty conditioning directory remains UNKNOWN and retained;
there is no recursive removal, general recovery or relaxed restart guard.
