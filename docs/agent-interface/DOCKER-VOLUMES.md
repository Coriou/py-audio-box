# Ordinary Docker-client installation candidate

`installed-daemon-volumes-v1` is a **new-installation SOURCE candidate** for
Kokoro CPU. Independent review and actual qualification remain required. Linux,
THIS Intel Mac Docker Desktop, and THIS Intel Mac OrbStack are separate UNRUN
qualification rows. No commands here were operationally executed for this lane.
Release NO-GO; Metal UNQUALIFIED; legacy CLOSED. Accepted native freshness and
renewal remain unchanged. The 6 GiB heavy-operation gate is unchanged.

The operator machine needs the trusted source distribution, Python stdlib and the
Docker CLI. Linux service dependencies run in the preprovided pinned image. No
Mac root, VM shell/SSH, Linux wheelhouse on the Mac, DockerRootDir access, host
bind folders, host stat records, or manual ownership changes are required.
The previous [strict host route](HEADLESS-INSTALL.md) remains a distinct option.
No conversion, adoption, repair or migration of existing installations is provided.

## Prerequisites and trust

Separately supply the reviewed configuration, complete exact model bundle bytes,
credential file, and already loaded immutable Linux image IDs. The control image
from `service/Dockerfile.release` also runs finite helper entrypoints under an
explicit root user. Image construction/load/acquisition are separate operations;
install/up never build, pull, download or choose a replacement model. Configuration
uses the existing closed Kokoro CPU schema, frozen contexts, manifests, profiles,
resource budgets and exact control/worker image IDs. Installation creates no
performance or audio qualification.

Explicitly select one local Unix Docker context: `desktop-linux`, `orbstack`, or
an operator-configured local Linux context. The resolved Unix endpoint and daemon
ID are frozen and rechecked; ambient DOCKER_HOST/DOCKER_CONTEXT changes do not
redirect an operation. Product brand grants no qualification. Require a rootful,
no-remapping Linux/amd64 daemon with cgroup v2, private namespaces and ordinary
local named volumes. Remote TCP/TLS/SSH, rootless/userns, custom drivers/options,
NFS/CIFS/bind-backed volumes, shared Mac folders and arbitrary preexisting volumes
refuse. Apple Silicon emulation and native Metal are not qualified by this lane.

The operator, pinned client/helper/supervisor, Docker administrator, VM and kernel
are trusted. A daemon administrator can read secrets, replace volumes or impersonate
containers. Trusted Docker inspection plus authenticated lifecycle and local PID1
facts are not cryptographic container attestation against that administrator.
Docker's diagnostic State.Pid is never dereferenced or compared with a peer PID.

## Prepare a reviewed request

Write one local request with exactly these keys (replace illustrative placeholders
with the reviewed concrete values; this is a schema example, not runnable data):

```json
{
  "configuration": {"...": "existing closed Kokoro CPU installed configuration"},
  "helper_image": "sha256:<64 lowercase hexadecimal digits>",
  "cpu": {"control": 1000000000, "worker": 2000000000},
  "tmpfs": {
    "control": {"bytes": 65536, "inodes": 64},
    "worker": {"bytes": 65536, "inodes": 64}
  },
  "control_port": 8000,
  "inputs": [
    {"role":"models", "path":"bundles/<bundle-id>/manifest.json", "source":"/local/preprovided/manifest.json", "bytes":123, "sha256":"<exact hash>"},
    {"role":"models", "path":"bundles/<bundle-id>/<manifest-enumerated-asset>", "source":"/local/preprovided/asset", "bytes":123, "sha256":"<exact hash>"},
    {"role":"config", "path":"install.json", "source":"/local/preprovided/install.json", "bytes":123, "sha256":"<exact hash>"},
    {"role":"credential", "path":"control.key", "source":"/local/private/control.key", "bytes":123}
  ]
}
```

Enumerate **every** model file exactly once, including the existing canonical
manifest. Its enrolled checkpoint/revision/manifest/content hashes must match.
The config input is canonical JSON matching `configuration` byte for byte.
No generic archive extraction is accepted: files are streamed over private attached
stdin in bounded chunks, with at most 4096 entries, 512 MiB per file and 2 GiB total.
Paths are relative, without empty, dot, parent or duplicate components. Local
sources are no-follow single-link regular files. The helper independently checks
content and the existing sealed-model-tree logic inside Linux. The credential
entry deliberately has **no hash field**. It must contain the existing credential
schema, for example (use your separately generated secret, never this sample):

```json
{"version":1,"credentials":[{"token":"<32-to-4096 allowed secret characters>","principal":"alice","scopes":["read","production-write"]}]}
```

Tokens never enter argv, environment, handle, provenance, public hashes or helper
stdout. Keep the original local credential protected. Use finite CPU/RAM/PID/tmpfs
budgets appropriate to the reviewed descriptor; the example is not performance
advice. Worker CPU must cover its enrolled thread count. Configuration remains
one explicit Kokoro profile; no Pocket/Qwen substitution is introduced.

## Few-command journey

From the extracted trusted distribution, with an existing local context:

```sh
python3 speech-box docker install --context desktop-linux --request reviewed-install.json --handle new-install.json
python3 speech-box docker up --handle new-install.json
python3 speech-box docker status --handle new-install.json
python3 speech-box docker stop --handle new-install.json
```

Select `orbstack` or the explicit Linux Unix context instead when appropriate.
Existing handle/output paths refuse. Keep the handle **and all sibling receipts**:
initial intent precedes volume creation; each created volume and helper/container
ID is retained before start. The `.binding` sibling holds nonsecret immutable
configuration/daemon/volume selection, not substitute launch authority. Docker
Mountpoints in it are opaque daemon metadata, never paths the Mac must stat.
A normal up after clean stop reuses the same exact runtime IDs. Status is bounded
observation, not future health. A missing/recreated ID, changed endpoint/daemon,
mount, image, input view, boot or unproved history refuses and retains state.

Eight distinct unpredictable installation-scoped local named volumes hold control,
receipts, staging, IPC, authority, models, configuration and credential. Volume
copy-up is disabled. Authority remains root:root 0700 with exclusive, fsynced 0600
records. It is separate from UID10001 control, and neither runtime mounts it.
Input import and Core initialization use separate finite helpers, so the Core
child's helper never mounts models or credentials. Helpers are networkless,
read-only-rootfs, resource-bounded and get only their phase's volumes. Install/init
capabilities are CHOWN/FOWNER/DAC_OVERRIDE/SETUID/SETGID/KILL; later helpers retain
DAC_OVERRIDE/SETUID/SETGID/KILL. DAC_OVERRIDE is necessary for finite
locked reads of UID10001-private Core files; CHOWN/FOWNER are dropped after
initialization. Runtime workers retain only SETUID/SETGID/KILL. No container gets Docker socket, host PID/root, devices,
privileged or SYS_ADMIN. The fixed-view child permanently drops saved UID/GID to
10001, clears supplementary groups, closes other descriptors and verifies zero
permitted/effective/inheritable/ambient capabilities under no-new-privileges.
It calls the existing `_split_core`; no second Core schema or scheduler exists.

The helper holds its actual volume launch exclusion through client Docker actions,
registration, notification and final brackets. Frames bind nonce, phase and exact
sequence. Control starts waiting without active Core/public listener; worker starts
as final PID1; UID/GID+nonce/HMAC lifecycle, exact Docker inspections and container
local observations precede the dropped registration writer and same-ID notification.
Control rechecks its enrollment/current incarnation before and after acquiring Core.
Only a final successful helper exit allows client success.

Stop sends TERM to control, waits for clean owner exit, checks existing journal,
ACK/reap/publication history, then TERM/waits worker and checks stopped history.
It never uses Docker stop's KILL escalation. Interrupted imports, creation,
registration, notification or final checks leave pending intent and refuse subsequent
activation. Unknown consumers, live/failed historical helpers, ambiguous Core and
unreaped/partial publication history refuse. There is no automatic rollback,
rm/prune, reset, repair or cleanup command. Retain all evidence for review. At most
128 helper transactions are admitted; exhaustion refuses explicitly. Normal exited
helpers are retained and exactly inspected, rather than silently ignored.

`service/compose.daemon-volumes.yml` is the matching runtime topology reference.
Compose up/down is unsupported: it omits this activation transaction. Helper
volumes/capabilities are generated from the same closed policy module.

## Authenticated first voice and generation

The public HTTP API is canonical. Set your existing bearer secret through your
protected client configuration, then GET `/skeleton` and POST
`/v1/operations/discover` with `{}`. A fresh catalogue is empty; use the configured
preset selector returned by discovery. Register it explicitly:

```http
POST /v1/operations/voice_preset_register
Authorization: Bearer <secret>
Idempotency-Key: first-preset
Content-Type: application/json

{"preset":"af_heart","voice_id":"my-voice","expected_revision":0}
```

Use the returned immutable `version.id` for generation. The closed generation request is:

```json
{"voice_version_id":"<returned version.id>","text":"Hello from my installed voice.","identity":{"engine":"kokoro","model":"<advertised model>","bundle_sha256":"<advertised bundle hash>","device":"cpu","precision":"<advertised precision>"},"fallback":"forbid","variants":1,"reuse":"never"}
```

Use the exact compatible identity advertised by the configured profile and saved
version. Inspect `/skeleton/operations/speech_generate` before
submitting. Do not put trusted model paths/configuration into a public request.
POST `/v1/operations/speech_generate` with a fresh `Idempotency-Key`, then use
`job_get`, `job_events`, `job_wait`, `artifact_get`, `asset_get`, and `asset_content`.
Every read is an authenticated POST to `/v1/operations/<operation>` with the
following closed body (these reads do not take an idempotency key):

| Operation | Body |
| --- | --- |
| `job_get` | `{"job_id":"<job-id>"}` |
| `job_events` | `{"job_id":"<job-id>","cursor":"","limit":20}` |
| `job_wait` | `{"job_id":"<job-id>","cursor":"","wait_seconds":5}` |
| `artifact_get` | `{"artifact_id":"<job.artifact_ids[0]>"}` |
| `asset_get` | `{"asset_id":"<artifact.asset_ids[0]>"}` |
| `asset_content` | `{"asset_id":"<asset-id>","start":0,"end_exclusive":1}` |

The content response supplies `total_bytes`; fetch within its bounded chunk
limit and decode `content_base64`, then compare the assembled bytes' SHA256.
Events/wait return canonical cursors; use the returned cursor for continuation.
For the thin clients, the corresponding commands are:

```sh
python3 -m clients.cli voice preset-register --request '{"preset":"af_heart","voice_id":"my-voice","expected_revision":0}' --idempotency-key first-preset
python3 -m clients.cli speech generate --request '<closed generation JSON>' --idempotency-key first-generation
python3 -m clients.cli job events <job-id>
python3 -m clients.cli job wait <job-id> --wait-seconds 5
```

The public CLI needs its HTTP client dependencies separately; only the `speech-box
docker` operator surface is stdlib-only. HTTP clients may use curl with a protected
header/config file instead of placing a credential in argv. Reuse the same request
and key after control reopen to receive the same job without another launch; a new
key requests a new generation using the same immutable version. Verify retrieved
asset bytes against returned SHA256. SOURCE fixtures use synthetic media/runtime
observations; audible Kokoro, actual installed dependencies/Python 3.11, kernel UID
and mount enforcement, PID1 descendant reap, desktop loopback forwarding, and
cold/warm resource behavior remain separate actual qualification gates. The public SOURCE fixture models published speech, control reopen, clean
worker stop/up, old-key replay and new-key generation on the same retained IDs.
Its injected terminal/ACK journal is not real post-speech worker qualification.
