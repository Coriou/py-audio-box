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
bundles, an explicit local Docker context, three explicitly named loaded local
images (tags or pinned IDs), a dedicated custody path, a local mono PCM16
reference WAV with its exact transcript, and an actual qualification receipt
with resource measurements. It resolves each named image through read-only
Docker inspection to a pinned ID. It verifies both complete bundle trees and
derives the frozen contexts, profile, credentials, complete file manifest,
and installer request. The seed
WAV is template metadata; enroll a reference through the authenticated API
before generating a clone. The measurement receipt and resource values must
come from real operator qualification. Setup does not measure model memory or
claim that installation proves inference.

Put your actual qualification evidence in a private regular file, for example
`/protected/qwen-measurement.json`:

```json
{
  "receipt": "actual-qualification-receipt",
  "checked_at": "2026-10-29T10:00:00+00:00",
  "memory_bytes": 2147483648,
  "disk_bytes": 10737418240,
  "deadline_seconds": 600
}
```

The numbers above illustrate the file format; use measurements from your own
qualification, not these values. `--ram-bytes` is the chosen container budget
and must cover measured model memory. `--threads` drives the worker CPU and PID
budgets; the control CPU and both tmpfs budgets use the existing conservative
setup values. The setup root and
handle must be new. It invokes the existing `docker-qwen install` then `up`
and retains their receipts. A failed install or launch requires review of the
retained handle; running setup again with a new handle is not recovery.

```sh
python3 speech-box qwen-service-setup \
  --clone-bundle /absolute/sealed/base --ryan-bundle /absolute/sealed/customvoice \
  --context LOCAL_CONTEXT --control-image LOCAL_CONTROL_IMAGE \
  --worker-image LOCAL_WORKER_IMAGE --helper-image LOCAL_HELPER_IMAGE \
  --custody /sys/fs/cgroup/speech-box-qwen/0123456789abcdef0123456789abcdef \
  --reference-wav /absolute/local/reference.wav \
  --reference-text 'The exact words in the reference recording.' \
  --measurement /protected/qwen-measurement.json --ram-bytes 12884901888 \
  --threads 2 --handle /protected/fresh-qwen.json
python3 speech-box docker-qwen status --handle /protected/fresh-qwen.json
python3 -m clients.cli enroll --name bruce --wav /absolute/local/bruce.wav \
  --reference-text 'Exact words Bruce recorded.' --audition-text 'A short audition.'
python3 -m clients.cli speak --voice bruce --text 'Hello from Bruce.' --output /absolute/new-bruce.wav
python3 -m clients.cli speak --voice Ryan --text 'Hello from Ryan.' --output /absolute/new-ryan.wav
python3 -m clients.cli enroll --name john --wav /absolute/local/john.wav \
  --reference-text 'Exact words John recorded.' --audition-text 'A short audition.'
python3 -m clients.cli speak --voice john --text 'Hello from John.' --output /absolute/new-john.wav
```

The default setup root is `~/.local/state/speech-box/qwen-service` and the
default loopback port is 8000. With its `client.json` present, CLI and MCP
discover that root automatically when no endpoint, credential or installation
override is set; no token copy or environment setup is needed. For a custom
root pass `--root /absolute/private/root` to setup, then use the returned
`client_command` (which includes `--installation /absolute/private/root`) for
CLI calls, or set `SPEECH_INSTALLATION` to that root for MCP. The setup result
also gives the private generated `install-request.json` path for inspection.
`--prepare-only` writes that validated plan without installing or starting.
`--request operator-input.json` remains available for advanced closed input.

The generated private `install-request.json` is the existing `docker-qwen`
request. The generated `client.json` supplies local endpoint and credential
discovery; the Docker importer checks the mounted credential against the
configuration. A later `up` is a restart after a confirmed `stop`, using the
retained handle. Raw `qwen-service-config`, `qwen-local-init`, and
`docker-qwen` entrypoints remain available for explicit lower-level workflows.
The two-model install request requires `custody`, an exact
`/sys/fs/cgroup/speech-box-qwen/<32 lowercase hex>` path. The Linux host
operator must precreate that dedicated cgroup v2 subtree as root:root mode
`0700` and configure `memory.max` no greater than the installation RAM budget,
`memory.swap.max` as `0`, `pids.max` no greater than the installation PID budget,
and a finite `cpu.max` no greater than the worker container's CPU quota.
These limits belong on the dedicated parent: the worker container, resident
model owner, and each attempt are its children. The host must delegate
membership moves and `cgroup.kill` to the worker's fixed root UID. Docker
bind-mounts only that subtree into the networkless worker at
`/run/attempt-cgroups` and uses it as the worker's cgroup parent. The exact
bind and parent are checked in container inspection. A host that cannot
provide this narrow delegation and aggregate limits cannot start the resident lane
before workload launch. The supervisor verifies the live parent identity and
limits at startup and before each child group creation, then verifies each
child's cgroup membership before releasing its execution pipe gate. All Docker commands
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
