# Packaged headless installation

This source unit supplies the finite `speech-box init`, `complete`, `up`,
`status`, and `stop` command. Independent source review and actual installed
Linux/CPU Kokoro speech qualification remain pending. Do not interpret an
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

`service/Dockerfile.release` has control and CPU worker targets using the existing
pinned Python 3.11 CPU base. Worker installs the unchanged `requirements/kokoro.lock`
and control/service lock together. Only explicitly allowed source/data/locks enter
the build context. Image construction is a separately authorized release operation.
Build/load the two images before installation and enroll their exact local
`sha256:` image IDs. `up` performs no build, download, pull or model provisioning.
`compose.release.yml` documents the fixed policy; use the finite command for the
transaction, rather than Compose up/down, which cannot perform registration or
preserve the launch protocol.

## Explicit provisioning and init

Supply the real enrolled configuration: one CPU Kokoro profile, finite resource
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
