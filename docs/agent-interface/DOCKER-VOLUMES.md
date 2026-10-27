# Ordinary Docker-client installation candidate

This page describes the generic/v1 schema and its historical SOURCE evidence.
For historical actual ordinary-client results, use the engine-specific
[Kokoro export-v2 recipe](CPU-QUICKSTART.md) (`actual-install14`) and
[Pocket v2 recipe](POCKET-QUICKSTART.md) (`pocket-public-install05`, source `7c56f72`).
Those results do not qualify v1 or every family/platform.
[Qwen CPU](QWEN-CPU.md#current-status--2026-10-27) has a source family, with
current startup disk-blocked and historical activation timeout unresolved.
Use fresh resource/disk admission for the selected recipe/operator plan; the
historical 6 GiB gate is not a universal sufficiency threshold.

At the v1 checkpoint, `installed-daemon-volumes-v1` was a **new-installation SOURCE candidate** for
Kokoro CPU. Independent review and actual qualification remain required. Linux,
THIS Intel Mac Docker Desktop, and THIS Intel Mac OrbStack are separate UNRUN
qualification rows at that checkpoint. No commands here were operationally executed for that v1 lane.
Release NO-GO; Metal UNQUALIFIED; legacy CLOSED. Accepted native freshness and
renewal remain unchanged.

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

Child completion becoming unknown is an exceptional retention state. On child
output overflow or timeout, the helper stops advancing the transaction, keeps
its root/authority/launch exclusions and incomplete intent, drains stdout without
retaining further output, and waits for actual exit/reap. Exactly one child stays
owned; there is no replacement, force/KILL, rollback, success or intent clearing.
This wait has **unbounded duration with bounded resources** and may require manual
intervention. A caller timeout is not helper completion: retain the exact helper
ID and all receipts; competing operations must refuse. Do not remove or replace
retained resources. Normal protocol and collection deadlines still bound success.

Docker CLI stdout retains at most 1 MiB and helper-child stdout at most 64 KiB,
plus one read chunk of at most 64 KiB and fixed pipe/selector/request overhead;
there is no disk spool. Readiness waits use 100-ms selector/wait intervals rather
than polling without delay. Terminal protocol success additionally requires exact
stdout EOF within 10 seconds and the remaining operation deadline; buffered or
subsequent trailing bytes, including a second frame, refuse.

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

{"preset":"ff_siwis","voice_id":"my-voice","expected_revision":0}
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
python3 -m clients.cli voice preset-register --request '{"preset":"ff_siwis","voice_id":"my-voice","expected_revision":0}' --idempotency-key first-preset
python3 -m clients.cli speech generate --request '<closed generation JSON>' --idempotency-key first-generation
python3 -m clients.cli job events <job-id>
python3 -m clients.cli job wait <job-id> --wait-seconds 5
```

The public CLI needs its HTTP client dependencies separately; only the `speech-box
docker` operator surface is stdlib-only. HTTP clients may use curl with a protected
header/config file instead of placing a credential in argv. Reuse the same request
and key after control reopen to receive the same job without another launch; a new
key requests a new generation using the same immutable version. Verify retrieved
asset bytes against returned SHA256. The following qualification caveats describe
this page's v1 SOURCE fixtures; later family-specific results are linked above. SOURCE fixtures use synthetic media/runtime
observations; audible Kokoro, actual installed dependencies/Python 3.11, kernel UID
and mount enforcement, PID1 descendant reap, desktop loopback forwarding, and
cold/warm resource behavior remain separate actual qualification gates. The public SOURCE fixture models published speech, control reopen, clean
worker stop/up, old-key replay and new-key generation on the same retained IDs.
Its injected terminal/ACK journal is not real post-speech worker qualification.

## Opt-in shared Qwen models (source contract)

The reviewed A+B+C implementation adds `docker-models` and fresh
`docker-qwen install --shared-models SELECTION.json`. This is source evidence
only: actual Docker/runtime/capacity/audio and release qualification remain open.
The v1 recipe above and legacy Qwen eight-private-volume installs are unchanged.
Generic, Pocket, export, strict-host and native routes do not accept shared daemon
records as authority. Old binaries may refuse the new closed binding; no migration
or downgrade is promised.

The store is daemon-local. `pab-model-store-v1` is a helper-only catalog, never a
runtime mount. Each published object is one physical local volume named
`pab-model-sha256-<K>-<32hex-attempt>`. It contains only the sealed tree
`/models/bundles/<K>/manifest.json` and every descriptor file plus its exact parent
directory closure. Files are UID/GID 10001:10001 mode 0444; directories 0555.
K is SHA256 of the exact canonical runtime manifest bytes, including its newline,
not the acquisition descriptor hash or inode-sensitive tree observation digest.
The production Qwen K is
`d2d42c4a726518f747bb094f5b5180746a4a965020f9abc5ee739ee3d1c76fa8`.

### Prepare explicit offline requests

Run the following from the extracted trusted headless distribution. Supply an
absolute `BUNDLE` directory containing the already reviewed descriptor payload;
this script reads metadata only and performs no acquisition. `IMAGE` must be an
already-loaded immutable helper image ID, `CONTEXT` an explicit admitted local
Unix context. Choose fresh output paths for every invocation.

```sh
export BUNDLE=/absolute/path/to/reviewed-qwen-bundle
export CONTEXT=orbstack
export IMAGE='sha256:REPLACE_WITH_EXISTING_64_LOWERCASE_HEX_IMAGE_ID'
python3 - <<'PY_SHARED_REQUEST'
import json, os
from pathlib import Path
from service.shared_models import descriptor
expected = json.loads(Path('requirements/tts/qwen-0.6b-base.expected.json').read_text())
key, raw, files, directories = descriptor(expected)
bundle = Path(os.environ['BUNDLE'])
assert bundle.is_absolute()
def save(name, value):
    with open(name, 'x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
save('model-import.json', {'descriptor': expected, 'sources': {
    entry['path']: str(bundle / entry['path']) for entry in expected['files']}})
save('model-check.json', {'descriptor': expected, 'sources': {}})
print(key)
PY_SHARED_REQUEST
python3 speech-box docker-models init --context "$CONTEXT" --image "$IMAGE" --output store-init
python3 speech-box docker-models import --context "$CONTEXT" --image "$IMAGE" --request model-import.json --output store-import > publication.json
python3 speech-box docker-models verify --context "$CONTEXT" --image "$IMAGE" --request model-check.json --output store-check > verified-publication.json
```

The shell IMAGE value above is a placeholder: replace it before executing.
The request has exactly `descriptor` (the complete reviewed acquisition descriptor)
and `sources` (descriptor-relative file paths to absolute no-follow single-link
regular local files). Do not include `manifest.json` in sources: the importer
constructs its exact canonical bytes. A fresh import requires every payload source.
Verify, recovery and reuse can use empty sources. Init accepts no request; all
commands require context, image and a fresh private output directory. There is no
implicit download/import in installation, up or native provisioning.

Each successful import/verify/recover-import prints the same immutable publication
object with exactly `schema, daemon, catalog, key, attempt, candidate, observation,
writer, verifier`. Schema is `shared-model-store-v1`; key is K, attempt is 32
lowercase hex. Catalog and candidate are exact reduced Docker records with keys
`Name, Driver, Options, Scope, Mountpoint, CreatedAt, Labels`; retain their returned
values, including incarnation timestamps and opaque daemon Mountpoints. Catalog
labels are exactly `{"speech-box.model-store":"shared-model-store-v1"}`;
candidate adds `speech-box.manifest: K` and `speech-box.attempt: attempt`.
Driver/scope are `local`, options null or empty. Daemon has exactly `endpoint, id, os, architecture, security, cgroup`, as in
the existing closed Docker binding record. Observation is exactly `{root: [device, inode], tree_sha256}`.
Each writer/verifier proof is exactly `{id, nonce, image, inspection_sha256, started,
finished}` (64hex ID/digest, 32hex nonce, immutable image ID, terminal timestamps).
Init returns the catalog schema record `{schema, daemon, catalog}`. These public
records contain no credentials and do not substitute for protected catalog authority.

Output directories contain exclusive `intent.json` with exactly
`schema, operation, context, image, key, attempt, expires`, and monotonically
numbered `NNNNNN-<kind>.json` receipts. Kinds are `volume-intent` (`name, labels`),
`volume-created` (reduced volume), `helper-intent` (`role, nonce, command_sha256,
candidate`), `helper-created` (`id, role, nonce`), `start-intent` (`id, role, nonce`),
`terminal` (`id, role, inspection_sha256`) and `complete` (command result).
Retain the whole directory, exact container IDs and protected catalog records,
including failed attempts; stdout alone cannot recover unknown create outcomes.

### Select the publication for a fresh installation

```sh
python3 - <<'PY_SHARED_SELECTION'
import json
from pathlib import Path
from service.shared_models import digest
published = json.loads(Path('publication.json').read_text())
expected = json.loads(Path('model-check.json').read_text())['descriptor']
selection = dict(descriptor=expected, catalog=published['catalog'],
                 models=published['candidate'], publication_sha256=digest(published))
with open('shared-selection.json', 'x') as stream:
    json.dump(selection, stream, sort_keys=True, separators=(',', ':'))
PY_SHARED_SELECTION
python3 speech-box docker-qwen install --context "$CONTEXT" --request shared-install.json --handle fresh-qwen.json --shared-models shared-selection.json
python3 speech-box docker-qwen up --handle fresh-qwen.json
python3 speech-box docker-qwen status --handle fresh-qwen.json
python3 speech-box docker-qwen stop --handle fresh-qwen.json
```

The selection has exactly the four keys shown. `publication_sha256` hashes compact,
sorted, UTF-8 JSON without a newline (`service.shared_models.digest`), distinct
from K's manifest serialization. It is authenticated under the protected key lock.
Author and review `shared-install.json` separately: its exact keys remain
`configuration, helper_image, inputs, cpu, tmpfs, control_port`, as in the family
request. Inputs must be only config/install.json (with exact bytes/SHA256) and
credential/control.key (bytes, no hash); model entries refuse. Config bytes are
canonical JSON equal to configuration. For this NEW shared configuration only,
every Qwen execution bundle_id and bundle mapping must consistently use K and
`/models/bundles/<K>`. Do not rewrite frozen contexts, existing Core or handles.
The production manifest pin, CPU execution and resource/readiness gates still apply.

Install reserves a permanent catalog reference before allocating seven fresh
private volumes: control, receipts, staging, IPC, authority, config, credential.
The models role is the selected published object, RO even in the install helper.
The final binding adds exactly `model_storage` with `kind` =
`shared-model-bundle-v1`, `manifest_sha256` = K, `bundle_id` = K, exact `catalog`,
`publication_sha256`, and `reference_id` = installation ID. Bound configuration,
attachment intent/exact IDs, then enrollment are protected separate records.
Runtime control/worker never mount the catalog. Subsequent up/status/stop select
only the retained handle/binding; no selection, request or context override.
Keep all handle siblings and `.shared*` receipt directories.

### Recovery, retention and availability

After reviewing the retained exact attempt and all actor outcomes, an operator can
request only that store import's recovery (set ATTEMPT to its exact retained
32-hex attempt, never a newly invented value):

```sh
python3 speech-box docker-models recover-import --context "$CONTEXT" --image "$IMAGE" --request model-check.json --attempt "$ATTEMPT" --output store-recovery > recovered-publication.json
```

Recover-import requires the exact retained attempt and descriptor, terminal writer
proof and a complete fresh RO verification. It cannot resume a failed installation,
enroll it, clear its incomplete intent or repair bytes. Unknown writer/create
outcomes, active/restarted actors and malformed/conflicting publication refuse.
A lost success after publication reuses the identical object after full verification;
a new import may allocate a new candidate only after old writer custody is proven.
No operation removes the old candidate or overwrites a publication.

Reservations, incomplete attachments, failed candidates, stopped helpers and enrolled
references are permanent holds. Stop, process exit and elapsed time release nothing.
There is no automatic GC, refcount decrement, lease expiry, cleanup or retry/reset.
This change frees no historical disk space. Missing/busy/corrupt catalog or ambiguous
consumer authority can conservatively refuse coordinated lifecycle operations,
including stop; running engines do not mount the catalog. Never reconstruct trust
from labels alone or bypass refusal by deleting evidence.

Reuse saves transfer/allocation, not SHA256 I/O or model RAM. Import/reuse/attach
and install/init/up/status verification read the full sealed payload (about 2.5 GB
for the pinned bundle), exact inventory and stable identities; stop does not add a
full rehash to TERM custody. Tamper is a shared failure domain and refuses rather
than repairs. Root-protected catalog and RO mounts rely on the trusted Docker
administrator/VM/kernel: retained terminal RW importers are not immutable against
a daemon owner restarting or replacing them. Native host paths remain separate.
The nonrenewing 600-second Owner deadline is a work budget, not a hard reap/return
guarantee; permanent close/wait/reap faults can retain custody beyond it.
No storage receipt or source test grants runtime or release readiness.
