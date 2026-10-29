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

Create a JSON request with exactly `principal`, `clone`, `Ryan`, `profile`,
`budget`, and `images`. `clone` and `Ryan` are complete frozen execution
contexts with the same measured `qwen-cpu` profile; their pinned descriptors,
public model identities, CPU tuple, and compatibility entries must pass the
closed validators. The clone context may use an exact reference transcript
with ICL conditioning. The profile must advertise `voice_prepare/prepare`,
`speech_generate/qwen_clone`, and `speech_generate/qwen_preset`. The builder
normalizes bundle IDs to the manifest digests and refuses any other model.

```sh
python3 speech-box qwen-service-config --request two-model-contexts.json --output install.json
python3 speech-box qwen-local-init --root /absolute/private/qwen-service --port 8765 --principal alice
python3 speech-box docker-qwen install --context EXPLICIT_CONTEXT --request install-request.json --handle fresh-qwen.json
python3 speech-box docker-qwen up --handle fresh-qwen.json
python3 speech-box docker-qwen status --handle fresh-qwen.json
python3 speech-box docker-qwen stop --handle fresh-qwen.json
python3 speech-box docker-qwen up --handle fresh-qwen.json
```

The `install-request.json` is the existing `docker-qwen` request: configuration,
helper image, an explicit file manifest for both model bundles plus config and
credential, CPU budgets, tmpfs budgets, and control port. The credential
document minted by `qwen-local-init` must be that installed input. Use the
same principal as the service configuration and the same port as the Docker
control port. Its private
`client.json` supplies local endpoint and credential discovery to the client;
the Docker importer verifies the mounted credential and its principal against
the configuration. `up` launches the authenticated control and worker. The
second `up` above is a restart after a confirmed `stop`, using the retained
handle. The two-model install request also requires `custody`, an exact
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
