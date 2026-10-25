# r392 — checked export owner activation and public journey candidate

2026-10-25. **Implementation candidate; independent acceptance pending. No self-adoption.**
The remaining owner and helper runtime gates are executable. A fresh INERT
installed owner produces and reopens its own export history, without copied local
history rows. Full r378 SOURCE acceptance remains pending fresh Sol adversarial
review and final Astra review, both parent-owned. Release **NO-GO**; legacy **CLOSED**.

## Authority and exact pins

Own checkout:
`/Users/ben/.txcode/worktrees/py-audio-box/t3-a145b721`.
Verified clean at required base
`1bec52ea8e1fedd8411099c0afc5001ff1c3031a`, verified ancestry and fast-forwarded
ONLY this checkout to `580f0570c133113e26a84fe8ca7ff6c3ca39965a`.
Read-only source `/Users/ben/.txcode/worktrees/py-audio-box/t3-f3e21aa0`
was at that pin, and was rechecked clean and unchanged before committing.

Source candidate: **`48da311dfce93f3a5a6ad09dd1448ee251e3153c`**, direct child of
580f057. This report is a subsequent documentation commit. No root/original
writes, adoption merge, push, delegation, review commissioning or operational
cleanup. No applicable AGENTS.md found in checkout or checked ancestor paths.
The lightweight memory registry search returned no relevant entry.

Read the full required absolute ROOT reviews under
`/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`, and candidate
r380/r382/r384/r386/r388/r390 reports. Input SHA-256:

- r378: `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`
- r391: `351d44bff1c393bd1c02a5a8c966c6012fe49af43a6ac4ae5dd26efd17862eaf`
- r387: `1da16379dc57b41cbbd1153fa69137f26f7160505c917cdb4a159e99c9fc99a4`
- r383: `4617c07c962b8a9a5ff06aa7f36392d9262f45d837d3f25ece23a74c22554651`
- r381: `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`

## Executable integration

- The distinct trusted `daemon_export_main` selects FAMILY/LAUNCH from code.
  PID1 publishes new-family local control facts, including assets, then uses the
  existing installed server and retained owner lifespan. Old daemon selection
  remains the v1 branch; no marker grants family authority.
- Private notification publication, WaitingControl decoding and the fixed status
  exec reader have explicit family selection. The bounded canonical parser
  requires the extended hello/seal shape, strict integer format 2, closed keys
  and digest types. Parsing alone grants no authority: pre-Core and post-owner
  reads repeat full enrolled configuration, registration, observation, inspected
  topology/images/start/assets, live lane/boot/seal and control incarnation checks.
- InstalledActivation detaches validated config, verifies exact internally
  combined Kokoro/export admission profiles and budget, refuses supplied
  AudioExports, and prepares the checked enrollment expectation. Dispatcher
  carries that exact expectation into Core before its writer initialization.
  After Core acquisition, the existing private reader repeats under Core's
  lifetime exclusion; only then is AudioExports constructed from checked config.
  Reconciliation and authenticated hello still gate admission/capability.
- The ASGI factory consumes closed new-family config and corroborates export
  principals against actual production-write credentials. Kokoro mapping/context
  verification reuses the unchanged legacy-field projection. SupervisorRuntime
  receives explicit FAMILY plus detached new config. No caller converter map,
  qualified receipt, default pin or readiness upgrade is created.
- The authenticated API's installed configuration branch now recognizes the
  exact checked activation family; the old v1 parser stays closed. Without this
  change the newly acquired owner would still fail when mounting its public app.
- Submit, retry and availability replace blanket installed refusal with the
  exact checked-family/publication-v2 condition. New work requires healthy
  activation/current readiness and the configured eligible measured export
  profile. Duplicate lookup still precedes those new-work gates. Old v1 and
  native installed export remain unavailable.
- Online owner reconciliation selects FAMILY/SPLIT from Core's checked
  expectation and passes exact enrolled public facts into the reviewed scanner.
  It uses the existing separate 160 MiB media-name / 16 MiB scalar budgets and
  canonical v2 checks, without inferring family from SQL markers.
- New-family publication explicitly fchmods its private output copy to 0600
  before writing. The direct factory fixture exposed that relying on PID1's
  umask alone produced 0644 copies and correctly failed checked reopen.
  Old/local publication code follows its existing branch.
- Helper up/status/stop/ready now run through the existing coordinator. Same-ID
  stopped restart selects receipts through the reviewed new-family stopped
  selector, rather than a loose pair/history comparison. Eight volumes, exclusive
  handles/intents, unknown ownership retention, TERM/drain and all existing
  cleanup responsibility remain in place. No force kill, repair, rollback,
  migration, service, scheduler or extra reservation pool was added.
- Updated AUDIO-EXPORT.md describes explicit private selection, the checked
  journey and missing operational prerequisites.

## Fresh INERT journey and injection boundaries

`tests/service/test_export_owner_activation.py` starts with Session's real tiny
fresh initializer and dropped-writer functions under injected UID/Docker facts.
The ordinary helper coordinator creates/starts fake inspected containers and
performs real registration pre/post/notify brackets. The sealed worker fixture
uses actual export_startup on inert immutable executable/closure bytes and actual
RuntimeSupervisor hello construction. No executable runs.

The owner uses actual notification publication/WaitingControl, prepare,
precheck, checked Core acquisition, postcheck, reconciliation and authenticated
hello. The factory case additionally exercises the existing ASGI lifespan,
credentials and checked API mounting. Only fixture model mount resolution/context
construction is injected there; actual installed model paths are not available
on this host. The direct journey uses real Kokoro resolver methods with detached
configured fixture templates.

Public upload is four seconds of 24 kHz mono PCM. Actual owner/Admission/Dispatch
SQL produces a 48 kHz stereo PCM24 proposal of **1,152,044 bytes**. The injected
runtime independently calls the actual sealed worker export verifier with a
fake managed-reference acquisition returning a real O_RDONLY private asset FD.
Actual Core PCM/hash/provenance/publication checks accept the proposal. Public
job/artifact/output metadata, complete download in bounded chunks, an arbitrary
byte range, and SHA-256 are checked. HTTP, CLI and MCP replay the same key/job;
CLI's normal download verifies the full output. Source bytes remain unchanged.

The fake runtime records terminal/ACK journal facts from the actual dispatched
Authority and injected receipt, rather than copying history from a local owner.
After checked close, the actual reviewed retained reader accepts this newly
produced history. Checked reopen starts with live admission already lost:
same key returns the same job, no prepare call occurs, reservations are identical,
and a new key refuses. Configuration/enrollment are not rewritten to simulate
readiness loss.

A separate public preset-register -> Kokoro-context speech-generate journey
publishes a small injected generated PCM asset, then exports it through the same
owned resolver and sealed descriptor verification. Speech stays below its
unchanged 1 MiB ceiling. This is generated-asset integration, not Kokoro inference.

Focused installed cases cover late receipt/timeout, malformed/stale output,
new retry refusal without readiness, unknown descendant observation with no
artifact/ACK and executing reservation retained, then cancellation followed by
explicit injected reap evidence before ACK. Unqualified configuration omits
export capability. Caller maps, changed seal/boot, notification replacement and
malformed/missing extended hello refuse. Existing selected worker tests exercise
actual managed_reference.acquire with fake Popen/pipes/unknown waits, retained
fake process and closed FD, separately from the owner journey.

**Evidence boundary:** fake Docker, UID, helper FD transfer, PCM proposals and
terminal/reap/ACK receipts are explicit injections. They are never qualification.
The fake worker constructor bypasses PID1/socket/thread setup; real seal and
source-verification methods run. Helper unknown-wait evidence and owner unknown
descendant/ACK evidence are separate fixtures, not one end-to-end public
helper-uncertainty scenario. A single public helper-unknown -> retained owner-close
fixture remains an evidence gap for independent review; no safe automatic
unquarantine/cleanup was invented to make it pass. No production integration
gate is intentionally left as an unavailable stub.

## Exact finite execution and failures

Read both conftests before imports. No broad pytest collection or listener
fixture. Every application/test execution used the exact interpreter:

`/private/tmp/pab-0c-venv/bin/python -I -B -`

Stdin runners install the following audit block before application/test imports,
assert CPython **3.14.7** in the final selections, set cwd to this exact checkout,
and insert only its root, tests/service and tests paths:

```python
import sys
forbidden=[]
def audit(event,args):
    if event in ('subprocess.Popen','os.system','socket.bind','socket.connect'):
        forbidden.append(event)
        raise RuntimeError(event)
sys.addaudithook(audit)
```

Receipts remain in this task's tool output; no disk runner/log was created.

1. Affected selection: **37 methods, 36 PASS / 1 FAIL, 68.752 s**, forbidden
   attempts `[]`. Eleven owner tests, nine ExportDaemonTests, twelve
   ExportSealTests, one v1 full helper lifecycle, one fresh checked initialization,
   acknowledged-phase mutation matrix, v1 pre/post lifetime-lock activation,
   and cancellation during the existing retained close-thread lifecycle.
   The sole failure was a stale registration-only fixture assertion expecting
   status ready despite deliberately not retaining containers.json. Actual
   coordinator lifecycle tests passed.
2. Corrected that assertion to initialized. After extending the notification
   structural parser, ran only owner seal/notification mutation, full large
   owner journey, and the corrected registration test:
   **3 PASS, 10.001 s**, forbidden attempts `[]`.
3. After adding fake runtime journal recording plus reviewed retained-reader
   acceptance to the actual owner-produced journey, ran the complete final
   ExportOwnerTests once: **11 PASS, 32.105 s**, forbidden attempts `[]`.

This is not a claimed all-green single 37-method run. The targeted follow-ups
cover the changed parser/fixture and final integrated journey; unchanged tests
were not broadly repeated. Final git diff --check passed.

Development runs exposed fixture inheritance accidentally acquiring an
unrelated local owner (and double-close timeout), missing family selection in
owner reconciliation, insufficient fixture disk quota, missing injected staging
ownership, a full-content request exceeding the public 64 KiB chunk contract,
and the real 0644 publication/reopen issue corrected above. Another combined
failure test correctly encountered the latched invalid-media resource fence;
it now uses independent fresh fixtures rather than clearing that fence.
These failed receipts are retained in tool output and are not relabelled as
passes. All completed development runners with counters reported forbidden
attempts `[]`; no forbidden event was observed in the other audit-blocked runs.

## r378 gap matrix and parent gates

| r378 exit condition | Candidate state / remaining gate |
| --- | --- |
| 1. Fresh exact v2, same family, legacy refusal | Existing reviewed initializer retained; fresh real SQLite/checked Core reopen exercised. No migration or old-family expansion. Independent final review pending. |
| 2. Insert-only writer / immutable brackets | Real fake-Docker helper registration and existing seal/image/profile/principal/boot/start/assets mutation matrix exercised. No writer protocol change. Independent final review pending. |
| 3. Checked owner, sealed mappings, truthful capability | Implemented, including factory/API/notification/reconciliation and unqualified/caller-map refusal. No operational pin or measured receipt supplied. |
| 4. Public large export, generated asset, download/reopen/replay | Fresh installed owner-generated history; HTTP/CLI/MCP, 1,152,044 bytes, full/range/hash/CLI download, generated Kokoro asset, same-key no new dispatch/reservation with live readiness lost. INERT only. |
| 5. Failure/cancel/uncertainty/canonical/ACK fences | Installed timeout/bad/stale output, retry refusal and unknown-descendant/cancel/ACK cases pass. Existing canonical/acknowledged-phase matrix and worker helper-unknown tests pass. Single public helper-unknown/retained-close fixture remains the precise evidence gap described above. Actual containment UNRUN. |
| Independent source acceptance | Fresh Sol adversarial review, then final Astra complete-family review are parent-owned. Not commissioned or self-accepted here. |
| Operational acceptance | Actual converter pins/complete closure, build/install and measured profiles; Linux/amd64, Intel Mac Docker Desktop and OrbStack all UNRUN. CUDA/Metal mandatory, unqualified/UNRUN. |

No condition was declared technically impossible. The remaining finite fixture
gap is retained explicitly rather than replaced with copied history, a fake
success receipt or weakened cleanup. Parent review must decide whether that
separate-seam evidence is sufficient or commission the retained-close fixture.

No actual Docker, network/listener, application child, FFmpeg, ML/hardware,
HELD/STT, installation/migration/build/acquisition/download, real operational
cleanup, heavy 6 GiB waiver or actual resource qualification. Seven real-child
reference-publication tests remain **BLOCKED/UNRUN**. No fresh disk/RAM claim.
All actual Linux/Desktop/OrbStack qualification **UNRUN**; CUDA and Metal
**mandatory, unqualified/UNRUN**. Release **NO-GO**; legacy **CLOSED**.
