# Two-model Qwen CPU service source path

This optional installed policy selects one Base 0.6B clone template and one
CustomVoice 0.6B Ryan template for the **same** authenticated principal. The
existing single-Base installed configuration remains valid. The two-model
configuration starts the `qwen-resident-cpu-v1` worker automatically. One model
owner stays warm and serializes calls; a model switch unloads before loading.

## Prepare exact inputs

Use the reviewed descriptors in `requirements/tts/qwen-0.6b-base.expected.json`
and `requirements/tts/qwen-0.6b-customvoice-cpu.expected.json`. The bounded
`qwen-snapshot` command repairs verified bytes in **private staging** only.
It does not publish a shared Docker model or activate a service. The installed
`docker-qwen install` transaction imports both pinned bundles into a fresh
private model volume and verifies the complete sealed tree before activation.
The two bundles total about 5.02 billion source bytes; allow space for staging,
the installed volume, and temporary import bytes. The shared single-model
`--shared-models` selection cannot carry this two-model configuration.

The direct setup command accepts paths to the two previously verified sealed
bundles, an explicit local Docker context, three locally loaded image digests,
an absolute private root, a principal and port, a dedicated custody path, a
local mono PCM16 reference WAV with its exact transcript, and measured resource
inputs. It verifies both complete bundle trees and derives the frozen contexts,
profile, credentials, complete file manifest, and installer request. The seed
WAV is template metadata; enroll a reference through the authenticated API
before generating a clone. The measurement receipt and resource values must
come from real operator qualification. Setup does not measure model memory or
claim that installation proves inference.

Example request shape (replace every placeholder with actual local evidence):

```json
{
  "principal": "alice",
  "bundles": {"clone": "/absolute/sealed/base", "Ryan": "/absolute/sealed/customvoice"},
  "docker_context": "EXPLICIT_LOCAL_DOCKER_CONTEXT_NAME",
  "images": {"control": "sha256:<64 lowercase hex>", "worker": "sha256:<64 lowercase hex>", "helper": "sha256:<64 lowercase hex>"},
  "root": "/absolute/private/qwen-service",
  "port": 8765,
  "custody": "/sys/fs/cgroup/speech-box-qwen/<32 lowercase hex>",
  "reference": {"wav": "/absolute/local/reference.wav", "text": "The exact words in the reference recording."},
  "measurement": {"receipt": "actual-qualification-receipt", "checked_at": "actual ISO-8601 measurement time", "memory_bytes": 0, "disk_bytes": 0, "deadline_seconds": 0},
  "ram_bytes": 0, "pids": 0, "threads": 0,
  "cpu": {"control": 0, "worker": 0},
  "tmpfs": {"control": {"bytes": 0, "inodes": 0}, "worker": {"bytes": 0, "inodes": 0}}
}
```

All zero values above are placeholders and are rejected. CPU values are
nanocores. The measured model memory belongs in `measurement.memory_bytes`;
`ram_bytes` is the container budget and must cover it. The setup root and
handle must be new. It invokes the existing `docker-qwen install` then `up`
and retains their receipts. A failed install or launch requires review of the
retained handle; running setup again with a new handle is not recovery.

```sh
python3 speech-box qwen-service-setup --request operator-input.json --handle fresh-qwen.json
python3 speech-box docker-qwen status --handle fresh-qwen.json
python3 speech-box docker-qwen stop --handle fresh-qwen.json
python3 speech-box docker-qwen up --handle fresh-qwen.json
```

The generated private `install-request.json` is the existing `docker-qwen`
request. The generated `client.json` supplies local endpoint and credential
discovery; the Docker importer checks the mounted credential against the
configuration. The second `up` above is a restart after a confirmed `stop`,
using the retained handle. Raw `qwen-service-config`, `qwen-local-init`, and
`docker-qwen` entrypoints remain available for explicit lower-level workflows.
The two-model install request requires `custody`, an exact
`/sys/fs/cgroup/speech-box-qwen/<32 lowercase hex>` path. The Linux host
operator must precreate that dedicated cgroup v2 subtree as root:root mode
`0700` and delegate membership moves and `cgroup.kill` to the worker's fixed
root UID. Docker
bind-mounts only that subtree into the networkless worker at
`/run/attempt-cgroups` and uses it as the worker's cgroup parent. The exact
bind and parent are checked in container inspection. A host that cannot
provide this narrow delegation refuses the resident lane. All Docker commands
require a local rootful Linux amd64 daemon with cgroup v2 and the cgroupfs
driver, pinned loaded
images, exact fresh volumes, and sufficient RAM/disk. They do not download
models or install dependencies.

The resident attempt runs under its unique unprivileged UID in the worker's
private PID namespace and host cgroup namespace. The global cgroup mount must
be read-only; only the dedicated subtree bind can be written. The worker is
PID 1. A pipe gate prevents execution
until the child has moved into a dedicated cgroup; all descendants inherit it,
including detached `setsid` descendants. Cancellation uses `cgroup.kill`, and
terminal receipt requires an empty `cgroup.events` population, UID cross-check,
and reaped leader/descendants. The container keeps its existing read-only root,
network isolation, dropped capabilities except the fixed worker transition set,
and no Docker socket. An unsupported or ambiguous cgroup, `/proc`, PID, or UID
observation quarantines the worker instead of promoting a result.

Source and inert-engine checks establish configuration and routing only. Real
Linux installed activation, model loading, CPU inference, transcript fidelity,
Ryan identity, cancellation/reap, restart history, and listening remain
separate acceptance evidence. Check retained `status` and runtime qualification
before using production speech.
