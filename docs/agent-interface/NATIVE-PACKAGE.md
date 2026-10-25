# Native Qwen package SOURCE portion (r297)

This separate `native-qwen-v1` archive is a worker source package, not a complete
Lot B installation. Linux Kokoro/Pocket enrollment, root/PID1/UID rules, images,
locks and history readers are unchanged. The bounded private first-start owner factory is described below; there is no
complete native installed activation, Docker launcher, enrollment transaction, readiness promotion or fresh catalogue
bootstrap in this portion. Do not use Linux registration for native history.
Actual Metal speech remains UNAVAILABLE / UNQUALIFIED.

Build without acquisition from the checkout using an explicitly selected Python:

```
/absolute/python -I -B -c "import sys;sys.path.insert(0,'/absolute/source');from service.native_package import main;main()" archive --output /absolute/new-native.zip
```

Extract into an operator-owned immutable source directory. The archive contains
only the explicit `native-dist-manifest.json` allowlist and `native-worker.py`.
It contains no Python binary, dependencies, wheels, weights, credential or local
configuration. It contains no Linux dependency locks. Invoke the extracted entry
with the selected absolute managed Darwin arm64 executable (no PATH/shebang):

```
/absolute/managed/python -I -B /absolute/native-worker/native-worker.py check --config /absolute/private/supervisor.json --runtime-manifest /absolute/private/runtime.json
```

`serve` repeats preflight and enters the existing NativeSupervisor and serve function in the
same interpreter using the checked configuration snapshot. It is an explicit private listener operation and is not needed
for source checks. No command installs, resolves, downloads or repairs anything.
The supervisor exec child inherits its existing minimal offline environment and
single private socket; config/TLS credentials are not inherited.

Both documents must be canonical JSON, owner-only 0600, single-link regular
files, in owner-only 0700 directories. Every path is literal absolute and walked
without following symlinks. CA, certificate and key also require 0600/0700 roots.
Use a dedicated service account and exclude hostile same-UID writers. Rotation
requires a stopped and reconciled deployment; never edit trust under a live owner.

The supervisor document is the existing closed NativeSupervisor CLI shape:
`root,deployment_id,configuration_sha256,peer_pin,runtime_manifest_sha256,
registrations,python,child_entry,host,port,proven_host_network,cafile,certfile,keyfile`.
`python` must equal the current managed `sys.executable`; symlink executables are
refused. `child_entry` is the exact extracted `worker/native_child.py`.
`peer_pin` is the exact Docker client certificate DER SHA256, not a CA pin.
TLS remains 1.3/CERT_REQUIRED with the existing exact per-frame peer pin.

Select a numeric reachable host interface and port explicitly. Wildcards and
multicast refuse. Loopback needs explicit proven host-network topology; it is
not a Docker Desktop/OrbStack reachability assumption. Restrict the host firewall
to the selected Docker bridge source/interface and this one port. Resolve any
host gateway name outside the RPC path and configure NativeRuntime with that
explicit numeric address and the exact server certificate DER pin. These source
checks do not establish routing or firewall enforcement. When Docker control
packaging is completed, mount client credentials only into control, never a
Linux worker or native inference child. No native server key belongs in Docker.

The runtime manifest is closed with these fields (no supplied example is a real
platform/dependency lock):

- `version: 1`, `variant: "native-qwen-v1"`, `qualification: "unverified"`.
- `python`: exact `executable,version,build,implementation,machine,system,sha256`.
  `version` is full `sys.version`, `build` is the two-element python_build list,
  system/machine must be Darwin/arm64; sha256 hashes the managed executable.
- `distributions`: normalized name → exact installed version, the entire set
  including packaging/tooling. Local `+cpu` versions refuse. Requires-Dist edges
  and platform markers are checked without importing ML. URL/extra dependencies
  are unsupported and refuse. Missing packaging itself also refuses.
- `artifacts`: nonempty list of `{path,sha256,distributions}` mapping every
  distribution exactly once to an operator-provided offline artifact. Paths
  must be owner-controlled immutable files. Hashes authenticate selected bytes;
  this does not prove an artifact was installed or its ABI is compatible.
- `source_archive`: `{path,sha256}` for the exact retained native source archive.
  Every named extracted source byte must equal the archive; extra archive
  members refuse. Keep the archive outside the extracted tree.
- `models`: list of `{bundle_id,path,manifest_sha256,checkpoint,revision,license,access}`.
  License/access are explicit operator provenance metadata, not legal validation.
  Model bytes/capabilities and exact execution mapping are verified by the
  existing Qwen bundle verifier. No model is installed. All declared models must
  be used by the bounded existing supervisor registration templates.

The runtime manifest digest is SHA256 of its canonical bytes. It must match
supervisor config and every template's frozen native metadata along with
configuration/deployment identity. Registrations keep the existing
`snapshot,execution,bundles,inventory` format and F6 template authorization.
They are not created by public requests. Boot/lane enrollment and completed
history still need their separate native policy in the remainder of Lot B.

A successful check means only `package=prepared`,
`dependencies=metadata-verified`, `abi=unrun`, `qualification=unqualified`,
`speech=unavailable`. Missing or mismatched inputs refuse; no usable status is
written. No reviewed Darwin Python or wheelhouse is supplied. Real ARM ABI/import,
model/MPS, TLS/topology, process/FD/resource and lifecycle qualification remain
unrun. CPU/CUDA release rows and clone lifetime limitations remain separate.

`service.native_package.enrolled_runtime(configuration_file)` is the private
Docker control consumer for a separately provisioned native configuration. Its
closed fields are `version:1,variant:"native-qwen-v1",native,host,port,peer_sha256,
cafile,certfile,keyfile,spool,staging`. `native` is the existing exact five-field
`deployment_id,configuration_sha256,lane,boot,runtime_manifest_sha256` tuple.
It validates private config/client trust files, creates the actual pinned TLS
NativeRuntime, requires a healthy hello matching every configured identity
including boot, and refuses uncertain retained imports. It returns that runtime
for an explicitly configured owner; it does not enroll a Dispatcher, refresh
boot from hello, waive history, write an enrollment record or promote readiness.
Control credentials contain only client key/cert/CA; server key remains on host.
Completed-history activation and the installed entrypoint still require the
remaining native policy. Calling this factory performs real TLS I/O unless its
existing connector seam is injected; the r297 tests inject it.

Each dependency artifact must now be one wheel for one declared distribution.
The wheel filename name/version and supported interpreter/platform tags must
match; its bounded METADATA and WHEEL headers must agree. Linux/x86-only tags
refuse on Darwin arm64. This does not authenticate the installed distribution's
file provenance or prove native extensions load. Source distributions and runtime
aggregate archives are unsupported; absent compatible wheels leave unavailable.
Inert tests use explicitly nonqualifying metadata-only ZIPs, never native wheels.

r299 metadata relation: installed metadata and each selected wheel must both close
against the exact declared distribution set. Their canonical active edge sets
(normalized dependency name and canonical version specifier) must agree exactly;
equivalent but differently expressed constraints can conservatively refuse.
Valid inactive markers are ignored, including optional-extra edges; malformed
markers refuse. Active URLs and extras refuse. Wheel Requires-Python remains
checked. This relation proves neither installed binary provenance/ABI nor source
authenticity.

Before ZIP constructors, classic single-disk ZIP preflight caps source archives
at 64 MiB/512 entries and wheels at 4 GiB/32768 entries, with a 4 MiB central
directory ceiling. ZIP64 and trailing data are unsupported. Metadata is capped
at 65536 bytes; source members and launcher must declare exact expected sizes
before bounded stream reads (expected length plus one). Exact membership and
source-byte comparison remain required. These are package processing limits,
under the same immutable operator-managed input premise, not an extraction or
hostile-writer sandbox.

The r301 singleton policy requires exactly one case-insensitive Name and Version
in installed and selected-wheel METADATA, and at most one Requires-Python and
Metadata-Version. Any duplicate, including an identical or folded duplicate,
refuses before graph acceptance. A present Requires-Python must admit the current
interpreter. WHEEL Wheel-Version and Root-Is-Purelib are optional in this bounded
preflight, but may occur only once; present values must be 1.0 and true/false,
respectively. Repeated Tag and Requires-Dist remain intentional.

After the byte-bounded ZIP constructor, actual central entries must equal the
preflight EOCD count and satisfy the source/wheel entry ceiling before any member
read. This corrects bounded count accounting; it adds no unbounded-allocation or
installed-runtime qualification claim.

r303 adds a finite private durable SOURCE enrollment preparation API in
`service.native_package`; this is still not installed activation. An operator
calls `enroll_native(record_file, configuration_file=..., templates_file=...,
owner_root=...)` explicitly while the deployment is stopped. No public endpoint,
CLI default, environment lookup or automatic enrollment is added. The record
parent and owner root must already be private owner-controlled 0700 directories.
The new record is exclusive 0600, fsynced with its parent, never overwritten or
repaired. A partial record blocks reuse. Retain it for operator reconciliation.

The canonical 0600 templates document has exactly `version:1`,
`variant:"native-qwen-v1"`, `profiles`, `budget`, `inventory`, `registrations`.
Inventory uses the existing hardware wire representation. Budget uses the
existing finite Admission shape. Registrations are 1..64 closed rows of
`principal,profile,voice_version,context`; their three-part keys must be unique.
Actual `NativeQwenContexts` validates frozen plans, native identity, model/voice
metadata, profile admission and inventory freshness. These are operator-selected
source registrations, not supported fresh-catalogue voice creation or proof of
Metal readiness. Supervisor registration and sealed model verification remain
independent requirements.

The record binds the literal control/templates paths and canonical document
hashes, individual client trust byte hashes, exact five-field native identity,
and owner root path/device/inode/UID. Native runtime construction still validates
control shape, trust roots, pinned TLS and exact healthy hello including configured
boot; uncertain retained imports refuse. Enrollment brackets hello with repeated
configuration/trust/template/owner reads. The existing immutable operator-input,
no hostile same-UID writer premise applies; this is not hostile-writer isolation.

`native_startup_inputs` requires that same durable record and exact explicit
inputs, repeats the bracket and returns existing runtime/context objects plus
profiles/budget. Missing, malformed, changed or unhealthy state refuses; boot is
never learned or updated from hello. This function is preparation for future
startup wiring: **do not pass its result directly to an installed Dispatcher as activation
authority**; the r307 private NativeActivation consumer adds first-start fences. It acquires no Core lock and reads no completed history. Native
owner-thread pre/post acquisition checks, completed-history and publication/import
uncertainty validation, installed lifespan/entrypoint wiring and topology/mount
validation remain closed. Existing Linux enrollment/history policy is unchanged.
A clean new native boot requires separately designed stopped/reconciled renewal;
this write-once slice deliberately refuses automatic rotation.


r307 adds the private `service.entrypoint.create_native_owner(record_file=...,
configuration_file=..., templates_file=..., owner_root=..., staging=...)` SOURCE
factory. It consumes the explicit durable enrollment on the existing Dispatcher
control-writer thread, constructs NativeRuntime there, and repeats owner/trust/
record/hello binding after Core lifetime exclusion. It takes no public runtime,
connector or trusted configuration object. The caller owns ordinary Dispatcher
close/drain; a failed close retains its owner obligation and is not retried.

This is **first start only**: the enrolled owner directory must already exist and
be completely empty. Every initialized Core, even a clean empty database from a
previous activation, refuses. Any spool or staging entry refuses, including a
completed import. Current native hello must have exactly the configured boot,
no quarantine and zero launches. There is no boot learning, renewal, history
waiver, repair or deletion. A successful source owner construction does not
promote configured profile readiness or qualify Metal.

This factory does not yet mount the authenticated public installed ASGI app or
provide a native installed command/lifespan/topology policy. Existing Linux
factory and history readers remain separate and unchanged. Remaining completion
gate: strict native completed-history schema/receipt/Authority/publication and
durable import-intent linkage, authenticated installed credential/lifespan wiring,
topology/mount validation and supported fresh owned voice onboarding. Until that
native history gate is reviewed, **all retained Core roots remain refused**.
Actual Metal is UNAVAILABLE / UNQUALIFIED; no reviewed arm64 wheelhouse is supplied.

r309 supersedes r307's first-start-only limitation for the private owner factory.
The native policy runs on the Core owner thread under lifetime exclusion, before
import recovery, reconciliation writes or scheduling. It accepts a clean zero-job
owner or fewer than 64 succeeded, single-attempt, non-clone native jobs with the
same immutable configured five-field identity and boot. Exact native canonical
schema/SQL constraints, closed documents/frozen Authority, retained reservations,
confirmed durable outcomes/ACKs, actual publication bytes and publication intent,
matching retained supervisor terminal receipt/proposal/output, and exact private
import-intent plus import-complete are required. Unknown namespaces, diagnostics,
unfinished commands/work, missing or changed trust and incomplete/legacy imports
refuse without repair, cleanup, ACK or boot learning. NativeRuntime remains on the
control writer; TLS receives only detached Authority, never Core or SQLite.

The startup consumer may construct a runtime with completed imports pending proof;
this alone leaves its admission closed. Only the native validator discharges that
in-memory closure after all proofs. The normal enrollment/startup preparation
consumer still refuses pending imports. Linux installed history and its explicit
native-history refusal remain unchanged. The launch count must equal the retained
completed native job count; unknown host launches refuse. A host restart with a
changed boot still needs separately designed stopped/reconciled renewal.

Failed/cancelled/retried/clone histories, queued work, crash recovery, maintenance
or Linux-enrolled roots, legacy imports without intent/completion, and histories
outside the finite reader budgets remain unsupported. No configured profile is
promoted. Authenticated installed public lifespan/credentials, topology/mounts,
fresh voice onboarding and hardware qualification remain separate. No arm64
wheelhouse is supplied; actual Metal remains UNAVAILABLE / UNQUALIFIED. CPU/CUDA
release rows and the known legacy lib/tts.py digest RED remain unwaived.

## Explicit native public control launch (r317 SOURCE candidate)

The separate native installed control command now mounts the canonical authenticated
HTTP app through the same retained lifespan ownership used by Linux. Existing
Linux installer commands, routes and ownership rules are unchanged. This command
neither provisions nor enrolls. First prepare the explicit write-once native
record/config/templates with the private enrollment API described above, using
independently provisioned client trust and immutable registered contexts. Run
under the dedicated owner account with no hostile same-UID writers.

Supply an owner-only 0600 credential document in a 0700 owner directory. It uses
the existing closed `version:1,credentials:[{token,principal,scopes}]` schema:
1..32 credentials, token 32..4096 ASCII letters/digits/`._:-`, principal already
present in native templates, scopes `read` and optionally `production-write`.
No credential value belongs in a command argument or log. No default credential,
environment lookup, self-enrollment or readiness promotion is provided.

After separately provisioning and starting the native supervisor with its exact
configured boot/trust, an operator can explicitly start control from the extracted
source directory with a separately provisioned compatible control interpreter:

```
/absolute/control/python -I -B /absolute/native-worker/scripts/native-installed.py serve \
  --record-file /absolute/private/enrollment.json \
  --configuration-file /absolute/private/control.json \
  --templates-file /absolute/private/templates.json \
  --owner-root /absolute/private/owner --staging /absolute/private/staging \
  --credential-file /absolute/private/credentials.json --host 127.0.0.1 --port 8780
```

Only explicit `127.0.0.1` or `::1` HTTP binds are supported. This is the control
host's loopback; no Docker/host route, mount or TLS topology is inferred or proven.
Do not expose this plain HTTP listener via forwarding to untrusted networks.
TERM/INT uses ordinary owner drain; failed/unknown close retains its strong owner
obligation and reports failure. Cancelled asynchronous startup retains eventual
acquisition and closes it once; it does not abandon an owner or retry failed close.
The native runtime is constructed on the existing Dispatcher owner thread.
Same immutable enrollment/config/templates/boot/trust and the finite reviewed
native completed-history policy still apply. No automatic boot/trust renewal.

The existing clients use that same API: set `SPEECH_URL` explicitly to
`http://127.0.0.1:8780` and supply `SPEECH_CREDENTIAL` securely in the client
process environment. Existing `python -m clients.cli discover`, `voices list`,
`job get ID`, and `python -m clients.mcp` continue to use canonical requests,
authentication, scopes and errors. Native discovery returns configured native
profile metadata without promotion; `/skeleton` separates activation from speech
and reports `ready_for_speech:false`, UNAVAILABLE / UNQUALIFIED.

This is source wiring, not a complete installed journey. Fresh public native
owned-voice onboarding, topology/mount validation and stopped/reconciled renewal
remain subsequent slices. No reviewed arm64 wheelhouse, compatible installed ABI,
actual hardware/model/MPS, denied-peer/topology, audio, cancellation/crash/resource
or completed installed restart qualification is supplied. CPU/CUDA qualification
and legacy lib/tts digest RED remain separate and unchanged/unwaived.
