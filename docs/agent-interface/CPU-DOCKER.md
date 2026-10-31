For the new `installed-daemon-volumes-v1` ordinary Docker-client candidate, see
[DOCKER-VOLUMES.md](DOCKER-VOLUMES.md). This document describes the unchanged
strict daemon-host alternative and its separate evidence boundary.

# CPU Docker setup: daemon-side boundary

Source candidate only. Clean installation and audible Kokoro speech on Linux and
Intel Mac remain UNRUN. Release NO-GO, Metal UNQUALIFIED, legacy CLOSED. The 6 GiB
heavy-operation gate is unchanged. No command below was executed for this lane.

Two named layouts share one finite installer:

* `linux-local`: x86_64 Linux administrator shell on the Docker daemon host.
* `intel-mac-linux-vm`: administrator shell **inside an explicitly managed x86_64
  Linux VM** on an Intel Mac, running its own rootful Docker engine. Python,
  package, protected inputs, authority and named volumes all live inside that VM.

Both require the local `/var/run/docker.sock`, cgroup v2, ordinary same-filesystem
mounts, no rootless/userns remapping and real Linux `/proc` process identity.
The VM root administrator and Docker daemon are trusted: either can impersonate
workers and read mounted secrets. The Mac Docker client is not this authority.
Docker Desktop's default LinuxKit daemon is explicitly denied by existing policy;
its host socket forwarding and macOS shared files do not satisfy these checks.
OrbStack's default Docker service is not claimed supported. An operator-accessible
OrbStack Linux machine running a separate ordinary Docker engine is a candidate
only if **all observed metadata and filesystem/proc checks pass inside it**.
No product-name detection grants support; actual VM qualification remains UNRUN.
No privileged initializer, host-root bind, Docker socket or host PID namespace is
added to service containers. Unsupported layouts must use a suitable Linux VM or
Linux host; do not remove guards or chmod shared Mac files to imitate Linux.

## Explicit prerequisites

Use the distribution and offline Python 3.11 dependency procedure in
[HEADLESS-INSTALL.md](HEADLESS-INSTALL.md). The source test interpreter is separately
Python 3.14.7 and does not qualify that installed interpreter. Supply already loaded
immutable control/worker image IDs from `service/Dockerfile.release` targets
`control` and `worker`; the base image digest and `service/requirements.lock` plus
`requirements/kokoro.lock` are the exact package prerequisites. No image is built
or pulled by setup. Do not use Compose up/down; `service/compose.release.yml` is
only the matching topology reference.

Supply one reviewed Kokoro CPU configuration in the existing closed
`install_policy.configuration` format: exact principal, profile snapshot, frozen
context, finite budgets, bundle ID, manifest SHA256, checkpoint, immutable revision,
model/runtime descriptor and image IDs. No fixture identity or qualification is
minted. `requirements/tts-assets.json` and the reviewed provisioning manifest define
required Kokoro asset bytes/origins; the sealed tree must contain exactly the
configured `/models/bundles/<bundle-id>` manifest/assets accepted by `_model_tree`.
No model is selected, substituted or downloaded. Pocket is an explicit alternative
installation described in HEADLESS-INSTALL; this setup command refuses it and
never substitutes it for Kokoro or Qwen voices.

On the daemon/VM filesystem, pre-provision four distinct empty local named-volume
roots root:root 0700. Provision sealed model directories/files 10001:10001
0555/0444; config root:10001 0440; existing `installed_credentials` JSON
root:10001 0640. All seven sources must be disjoint, no symlinks, on the same
ordinary filesystem without alias/nested mounts. VM shared-folder mounts refuse.
The credential is only mounted into control, never worker or setup output.
Initialization performs existing explicit ownership transitions and one Core
creation using its dropped-identity child. No second database is introduced.

## One finite command surface

Write an explicit protected request JSON with exactly these fields:

```
{
  "layout": "intel-mac-linux-vm",
  "daemon_root": "/var/lib/docker",
  "configuration": {"...": "exact existing installed configuration"},
  "scope": {
    "daemon": "exact Docker daemon ID", "context": "default",
    "volumes": {
      "control": {"name":"speech-control","driver":"local","created":"exact CreatedAt"},
      "staging": {"name":"speech-staging","driver":"local","created":"exact CreatedAt"},
      "receipts": {"name":"speech-receipts","driver":"local","created":"exact CreatedAt"},
      "ipc": {"name":"speech-ipc","driver":"local","created":"exact CreatedAt"}
    }
  },
  "inputs": {"models":"/srv/speech/models","config":"/srv/speech/install.json","credential":"/srv/speech/control.key"},
  "cpu": {"control":1000000000,"worker":2000000000},
  "tmpfs": {"control":{"bytes":65536,"inodes":64},"worker":{"bytes":65536,"inodes":64}},
  "control_port": 8000
}
```

The configuration placeholder is deliberately not executable: supply the reviewed
configuration also present verbatim in the config input. Worker CPU must cover
its descriptor thread count; use appropriate finite budgets, not sample values
as performance advice. `daemon_root` must equal observed DockerRootDir and contain
every selected volume source. Volume identities are explicit, not name discovery.
Inside the Linux host/VM, as root with the trusted packaged Python:

```
/opt/speech-box/venv/bin/python /opt/speech-box/speech-box setup --request /protected/setup.json --output /protected/new-bundle.json
/opt/speech-box/venv/bin/python /opt/speech-box/speech-box init --setup-bundle /protected/new-bundle.json
/opt/speech-box/venv/bin/python /opt/speech-box/speech-box up --setup-bundle /protected/new-bundle.json
/opt/speech-box/venv/bin/python /opt/speech-box/speech-box status --setup-bundle /protected/new-bundle.json
```

Setup is inspect-only apart from exclusive output creation; it does not claim
runtime readiness. Existing output refuses. Init/up hold existing protected
root authority and revalidate every frozen identity. Wrong daemon/root, remote
socket, non-Linux/arm daemon, rootless/Desktop, unsafe owners, credential modes,
model bytes, mount layout, image or consumers refuse before a configuration result.
`setup_*` refusals identify the prerequisite; `installed_*`/`unsupported_*` identify
the existing exact policy fence. Correct inputs/layout explicitly and choose a new
output path. Do not repair retained partial init/up state automatically: retain it
for review. The CLI never reports success after a failed authority exit.

## Public first speech and reopen

Use authenticated CLI `status` and `discover` (MCP `speech_status` and
`speech_discover`) to read activation and configured presets. Explicitly register
that preset using `voice preset-register --request <JSON> --idempotency-key first`.
Generate with its returned immutable version ID through `speech generate`; read
`job events ID`, `job wait ID --wait-seconds 5`, then artifact, asset metadata and
bounded asset content through the same public CLI/API/MCP. See HEADLESS-INSTALL for
the exact registration body; public operation schemas define the generation body.
Keep the request and idempotency key. After completion and clean `stop`/`up` with
the same bundle, read the same version/job/artifact, replay the same request/key
without another launch, and generate using a new key. Control-only reopen is covered
by the ASGI fake-runtime fixture; real process and audible journeys remain separate.

The worker retains receipts/staging/IPC with its existing root supervisor and
UID20000+ children; control stays UID10001 with its own Core and credential, and
read-only staging/IPC. Native freshness/renewal, publication/reap fences and finite
capacity are unchanged. No CUDA/Metal installation or universal routing is claimed.
