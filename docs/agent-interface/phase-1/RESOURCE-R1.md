# R1 resource admission candidate

Baseline f6bd3f3042dedcc6a2cac535d50a85c9a0076f95, contract 1.3.0,
assigned worktree /Users/ben/.txcode/worktrees/py-audio-box/t3-3115080c,
branch t3/resource-admission. W2 integrated synthetic GO 601c558 is inherited;
public and persistent activation remain closed. Independent Sol-low review next.

## Frozen internal interface and state semantics (before implementation)

ResourceStore uses only Core's serialized SQLite writer and transaction. Admission
reserves a durable key/job/category, immutable demand/configuration and disk charge.
The categories inference/decode/render/provision share this allocator; only fixed
synthetic speech has an executor. No registry, render or provision feature is added.
Queued reserves queue and disk; executing additionally reserves lane, resident
configuration, RAM/VRAM and PID capacity; retained releases execution only after a
bound receipt with confirmed reap (or cancellation before any allocation). Disk
charges never decrease, including terminal jobs, stages and publication intents.
Replacement adds a fresh disk charge and checks the original frozen profile.
Idempotent retrieval precedes mutable readiness. Job/settings/reservation insertion
and attempt/execution acquisition each have one SQLite transaction. Unknown retains.
Startup reconstructs from the journal before new admission, with missing or
unjournaled ownership fail-closed. No cleanup/recovery or power-loss claim.

Operator budget is a finite allocation to this deployment, excluding capacity
reserved for legacy/external workloads. It is not discovered host capacity or an
exclusive-host claim. Default diagnostic budget: one lane/config, 192 MiB worker
RAM, zero VRAM, 48 PIDs, 32 queued, 64 MiB managed bytes and 16 MiB filesystem free
reserve. Synthetic worker charges are containment settings, not model RAM-fit.
Qualified stronger profiles may independently increase finite ceilings through the
same allocator; the installed one-worker dispatcher refuses unsupported profiles.

Control checks actual managed/staging byte growth while polling and again before
publication, plus WAV bytes/decoded duration. Fixed synthetic worker writes only
9,644 output bytes, no inputs, one variant; supervisor enforces deadline while
control is absent, container enforces memory/PIDs. Future runtime consumers MUST
incrementally enforce upload/decode/output bounds before allocating/writing,
respect frozen text/variant/deadline ceilings and own hard memory/PID containment;
metadata reservations alone do not authorize those unimplemented consumers.

## Implemented boundary

- `service/admission.py`: current trusted profile validation via the unchanged
  1.3.0 oracle, separately from stored context/hash; closed installed-runtime
  capability, request/plan/output checks and managed-filesystem monitoring.
- `service/resource_store.py`: one SQLite journal for all four categories; finite
  independent RAM/VRAM/PID/slot/resident/queue/disk ceilings and FIFO acquisition.
  Its low-level caller supplies a trusted demand, not a client profile. Only
  Admission derives an executable demand in this lot. Generic category tests do
  not qualify a backend, accelerator or future runtime consumer.
- Core hooks commit job/events/settings/reservation together, and unstarted
  cancellation/release together. DispatchStore commits execution acquisition with
  attempt/Authority intent. The dispatcher releases with durable terminal outcome
  only after validating the exact supervisor runtime/binding/deadline/reap receipt.
  Lost ACK never repeats execution or releases twice. Unknown keeps its executing
  reservation. Replacement retains the old disk charge and adds one attempt charge.
- Duplicate job retrieval precedes startup/quarantine/profile checks. The allocator
  also retrieves an identical existing reservation before mutable capacity checks.
  No automatic fallback, download, profile re-resolution or retry is introduced.
- Startup reads durable reservations before accepting commands, reconciles queued
  cancellation, observes existing dispatch ownership without re-executing, and
  blocks roots containing old jobs without reservations. There is no invented
  migration budget or recovery of an unknown owner.
- Root/staging scans include retained stages, DB/WAL and publication intents/assets.
  Observed high-water bytes PLUS all durable disk charges count against quota;
  deleted/missing individual stage bytes do not reclaim either charge. A missing
  accounting root, unreadable tree, symlink/special file or disk pressure refuses
  admission. Local managed-filesystem statvfs is used for free reserve; no Docker,
  host inventory or test observation is supplied to running service code.
- Resource violations latch durably, request bounded termination when possible,
  continue observation, and prohibit publication. Signalling does not release the
  reservation; receipt/reap is still required. No operator unlatch/recovery exists
  in R1. SQLite admission/cancel failures roll back atomically; allocator-loop persistence
  failures quarantine the writer and retain reservations.

The immutable job context contains the text/variant/media/deadline ceilings; its
insertion is atomic with the demand. This executor accepts no input assets,
exactly one variant, a bounded 48,000-byte context and fixed 4,800-frame/9,644-byte
output. It checks output growth during work and actual WAV header duration/format
before publication; Core independently verifies actual bytes/hash/media. The
worker-owned watchdog remains authoritative while control is absent. Reservation
memory/PIDs are the worker container's 192 MiB/48 hard containment, not the fixture
Qwen model's measured requirements. Worker and runtime policy files are unchanged.

The budget is the expensive worker pool allocated to this deployment. Operators
must ALSO allocate the separate control container's fixed 384 MiB/64 PID cap and
other deployment overhead before assigning the pool; capacity for legacy/external
work is excluded. One Core root/owner is one deployment allocator. Multiple
independent roots are independent deployments and must not each be assigned the
same physical capacity. No process-wide or host-wide exclusive ownership is claimed.
The packaged singleton worker remains one lane, independently of the generic
allocator's stronger finite budget tests. Alternate profiles (including qualified
GPU/Metal rows) are explicitly refused by this fixed executor. Future adapters must
supply trusted profile-derived demands and independently qualified hard containment.

## Packaging and preservation

Dedicated `Dockerfile.admission`, its ignore file and `compose.admission.yml` keep
W2's pinned dependency-overlay and separate network-none worker/control topology.
The sole topology delta is staging root mode 1777 rather than 1733: control must
list names to account bytes; sticky ownership still prevents cross-owner rename.
Attempt permissions and worker secrets/receipt/DB exclusions are unchanged. The
worker image remains the W2 image. No runtime source extension, dependency, public
contract/API/client/studio/registry/ledger/review or legacy change was needed.

The test runner uses the existing nonblocking heavy-lane lock, random owned named
resources, no user-data mounts, and confirms stopped state before removing owned
resources. It retains prior inventory using count/hash receipts, without recording
unrelated container/volume IDs. No prune, model/private audio/cloud/STT/Voicebox,
remote E2E, push/PR/deploy, extra checkout or subagent. The prohibited
`scripts/run-e2e-test.sh` was never invoked.

## Validation history and limitations

Initial host `python3 -m pytest tests/service/test_dispatcher.py -q -p no:cacheprovider`
failed collection because that interpreter lacks jsonschema. No dependency was
installed; subsequent host checks used the existing `/tmp/pab-0c-venv/bin/python`.
The first new unit run exposed four overly specific expected capacity error names
and one real missing during-work output-size check (it was inside publication-only
validation). The assertion expectations and actual monitor were corrected. Focused
final checks passed 54 tests. An initial packaged pass passed 143 E1 tests and 35 E2
steps; it preceded the final atomic cancellation/FIFO and extra tests, so its image
is not the final candidate and its receipt is retained only in `/tmp`.

Final validation uses the complete affected `tests/service` + `tests/release` suite
once against frozen final service/tests/packaging, then 35 installed steps. Steps
include crash/recovery pairs, not 35 independent fault families. Existing W2 cases
are rerun through the R1 topology; lower-level W2c installed families are inherited,
not silently counted as freshly repeated. The evidence file records exact image
IDs, full source manifests and packaging hashes, commands, results, topology,
retained prior-inventory digests and zero-owned-resource checks.

These are E1 deterministic policy/SQLite/concurrency/fake-runtime checks and E2
installed fixed-synthetic dependency-overlay evidence. They do not measure ML RAM
fit, throughput, CPU/GPU/Metal hardware qualification, acoustics, fresh install or
power-loss durability. Generic stronger-budget tests are E1 reservation tests only.

P1 remains mandatory: safe cleanup/reference/grace policy, orphan/adopted-byte and
missing-terminal-media reconciliation, directory durability barriers, partial
writes/file+directory fsync failures, SQLite intent/commit/checkpoint/recovery
failure matrix, receipt/owner/operator recovery and installed persistent-volume
acceptance. Resource faults and old unaccounted roots intentionally need that
future operator policy; R1 supplies no destructive workaround. D1–D5, real adapters,
residency and independently measured profiles remain later gates. Public,
unattended persistent and full Phase 1A activation stay closed. This is an
implementation candidate awaiting independent Sol-low review, with no self-GO.

The first attempted final full-suite run caught an over-broad edit of the optional
cancellation hook: it also inserted `_admission` into historical `reconcile`, where
that name is undefined. Three historical Core regressions failed (152 passed).
The unintended reconciliation line was removed; historical behavior is preserved.
`r1-evidence/failure-1.json` retains this failed candidate receipt. The final full
packaged pass was restarted after this correction; no failed run is called passing.

Commands used for validation (all from the assigned worktree):

```sh
python3 -m pytest tests/service/test_dispatcher.py -q -p no:cacheprovider
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_dispatcher.py -q -p no:cacheprovider
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_admission.py -q -p no:cacheprovider
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_admission.py tests/service/test_dispatcher.py -q -p no:cacheprovider
R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh
git diff --check
```

The packaged runner's exact build, image/source hash, isolated pytest, Compose and
fault-injection commands/results are in `r1-evidence/run.json`. Its complete
`python -m pytest tests/service tests/release -q -p no:cacheprovider` runs inside
network-none, read-only-root, 1 GiB/128-PID bounded Docker with only a 64 MiB /tmp.

## Final candidate evidence

Final command `R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh`
exited 0: **155 E1 tests passed in 42.34s; all 35 installed E2 steps passed**.
Final `git diff --check` passed. Post-run comparison of all 96 control, 18 worker
and four packaging manifest entries to worktree bytes found **zero mismatches**.
The owned project has zero containers/volumes left; all 15 prior containers and
140 prior volumes were preserved. No source changes followed this packaged pass.

- Dependency overlay: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b` (unchanged W2 worker).
- Control: `sha256:0a41e2eb3165c0f3d0a8726fa6913dc43a5d1c6a3a15797f36d04e49dc465e9e`.
- Final run receipt SHA256: `3b5f795a5222e7929fab8396728f9a8f66f247e475384b9a11b14dcd5b69e264`.
- `r1-evidence/final-audit.json` records source checks, exact worktree/branch,
  image identities and preserved-inventory digests. Final commit SHA is supplied
  in the implementation handoff (a report cannot contain its own commit hash).

Owned changed paths: `service/{admission.py,resource_store.py,dispatcher.py,
dispatch_store.py,core.py,Dockerfile.admission,Dockerfile.admission.dockerignore,
compose.admission.yml}`, `tests/service/test_admission.py`,
`tests/release/test_admission.py`, `scripts/test-resource-r1.sh`, this report and
`r1-evidence/{run.json,failure-1.json,final-audit.json}`. Everything else is preserved.
Next action is independent Sol-low review of the exact committed candidate;
no acceptance or integration decision is made here.
