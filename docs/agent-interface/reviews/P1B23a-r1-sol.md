# P1B2.3a independent adversarial review — round 1

2026-10-17 · gpt-6.1-sol / low. **NO-GO for bounded B2.3a source/E1 acceptance** at exact candidate `05afbaec0c3d61f60bb2f430d9108da30f285d0c`. Two reproducible topology gaps remain: an aliased subdirectory consumer can be omitted before stop, and the trusted reader accepts foreign mounts below the four authority roles. Apply remains unconditionally unavailable; no actual kernel/apply bypass or installed-host qualification is claimed.

## Identity, scope and review basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-c1fea5bd` read-only. Entry HEAD matches exactly; candidate status is empty. Actual baseline is `38b622142844b7d5d9cc011295eba5a1420bae96`. Baseline-to-HEAD contains exactly `scripts/p1-operator.py`, `service/recovery.py`, `tests/service/test_recovery.py`, and the appended `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Frozen `eb6e0c69205d49e39123ef3597cb569a747052d9` to HEAD changes only the handoff; source/test comparison is empty. Baseline diff check passes.

Independently compared the actual baseline recovery bytes with the current prefix: **131,362 bytes, byte-for-byte unchanged**, SHA-256 `e4206e1a719dfa969fe73962cae8e67f98e6c1d7064dd0193c22e5cfa5ed6e64`. No kernel, preparation, apply wiring, packaging, dependency, contract, ledger or existing-review mutation is present. No applicable ancestor/scoped AGENTS.md was found. Root has preexisting untracked research, which was preserved.

Read FULL P1B2-KERNEL-ASTRA, including the selected host boundary, preparation limitations, exact B2.3a brief/exits and remaining gates; FULL P1B2-BOUNDARY-ASTRA architecture; FULL candidate RECOVERY-P1B2 original/frozen contracts, all tranches/corrections and complete B2.3a response including the caller-clock defect correction; P1B22d-r5, orchestration protocol and R1 fault-authority context. Inspected actual producer, transport reader, validators, existing registration/hello transport, lifecycle ordering and the complete source/test delta. A lightweight memory registry search supplied no B2-specific authority or acceptance evidence.

## F1 — high — aliased volume subdirectories evade all-consumer coverage

Locations: `scripts/p1-operator.py:126-135` (`_mount_touches`), `166-185` (`host_inventory`), and `371-421` (`freeze_host`).

`_mount_touches` recognizes an exact volume name, lexical source ancestry after `Path.resolve()`, or device/inode equality with a **volume root**. Linux bind mounts are not symlinks: `resolve()` does not translate a bind alias to its backing filesystem path. A descendant has its own inode. Therefore a bind source through an alias of an ancestor can expose a deployment-volume subdirectory while satisfying none of these comparisons.

Concrete topology: deployment staging root `/fake/docker/volumes/staging/_data`, device 7 / inode N; host `/alias` is a bind alias of `/fake/docker/volumes`; an unknown container mounts `/alias/staging/_data/retained-subdir`, device 7 / inode 999. Its Docker mount has type `bind`, empty name, and an unrelated lexical path. This is ordinary bind storage access; the unknown container needs no added capability, privilege or shared PID namespace. The deployment volumes can remain local and option-free. An accidental/preexisting host alias is sufficient; no malicious administrator or mid-maintenance restart is required.

Independent E1 probe supplied that Linux stat/inspect boundary, keeping the rest of the candidate fixture and production freeze unchanged. Both LIVE and STOPPED unknown-container variants were enumerated and inspected but silently excluded from `consumers`. Freeze then executed control stop/wait and worker stop/wait and **published a capsule** covering only the two registered owners. In the live variant the unknown consumer remained running with PID > 0. Logical SQL stayed unchanged and all three mocked container resources remained. Probe exit 0, asserting this incorrect acceptance.

Consequence: the producer's all-consumer/no-alias proof is incomplete, and it performs stops where the brief requires refusal before any stop. A capsule can assert complete stopped coverage while another container still accesses the deployment storage. Apply closure limits today's reachability but does not satisfy B2.3a's producer exit.

Minimal fix: resolve mount sources into a verified host mount/filesystem topology (including ancestor bind aliases and descendant paths), or conservatively refuse sources whose disjointness cannot be proved. Root-inode equality plus lexical `realpath` is insufficient. Bound and repeat the necessary OS observations. Do not fix this by adding an allow-label/hash or broadening trusted consumers. Add unknown LIVE/STOPPED aliased-descendant negatives requiring zero stop/wait and no capsule; retain exact-name, direct-root alias and complete unchanged-owner positives.

## F2 — medium — trusted reader validates role roots but accepts nested foreign storage

Locations: `service/recovery.py:2232-2242`, `2255-2263` (`read_host_capsule`).

The reader rejects nested mounts under the proof directory, but for `/control`, `/staging`, `/receipts` and `/ipc` it checks only one mount at the role root and that root's device/inode/source. It never rejects mounts **beneath** those roots. A mount at `/control/core` or `/receipts/journal.json` changes the authority files subsequently reached through the role while leaving the base directory identity intact. Repeated equality of the same mount list does not make that topology safe.

Independent E1 probe first established the unchanged positive fake RO transport. It then added one stable foreign writable mount at a time below each of the four roles, with device `99:99` and root `/foreign`. Every call returned the original capsule. More specific `/control/core`, `/receipts/journal.json`, `/staging/foreign` and `/ipc/lifecycle.sock` cases also returned the capsule. Capsule bytes and logical SQL remained unchanged. Probe exit 0, asserting incorrect acceptance.

Consequence: this verifier does not establish that the actual authority paths remain on the capsule-bound deployment volumes. This is an unsupported-launch/topology refusal defect, not a claim to defeat a malicious host administrator. C1 must independently pin and verify the launcher tuple, but that pending obligation does not replace this lot's actual-kernel-topology reader checks. No installed launcher or kernel mutation was exercised.

Minimal fix: reject nested mounts under every fixed role unless an explicitly reviewed exact submount contract exists; this lot has none. Validate this on every repeated topology observation and preserve the fixed proof mount/role identity checks. Add coherent one-extra-mount negatives at both directory and managed-file destinations, with unchanged base identities; require reader refusal and evidence preservation. No packaging expansion is needed.

## Reproduction

Run from the exact candidate with the existing interpreter and bytecode disabled. This is an in-memory E1 probe using the candidate's coherent fixture; all Docker/proc/mount/UID behavior remains explicitly fake. It creates only disposable local synthetic roots. The assertions below deliberately demonstrate the incorrect acceptance, rather than count it as a passing safety test.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'
import importlib.util, tempfile, copy
from pathlib import Path
from types import SimpleNamespace
from pytest import MonkeyPatch
s = importlib.util.spec_from_file_location('review_cases', 'tests/service/test_recovery.py')
t = importlib.util.module_from_spec(s); s.loader.exec_module(t)
for live in [True, False]:
    with tempfile.TemporaryDirectory(prefix='b23-review-alias-') as d, MonkeyPatch.context() as mp:
        p = Path(d).resolve()
        h = t.b23_host.__wrapped__(t.installation.__wrapped__(p), p, mp)
        m = h['module']; before = t.logical(h['roots'])
        source = '/alias/staging/_data/retained-subdir'
        u = copy.deepcopy(h['consumers'][1])
        u.update(id='e'*64, running=live, pid=199 if live else 0,
                 finished='0001-01-01T00:00:00Z' if live else '2026-10-17T02:00:00Z')
        u['mounts'] = [dict(type='bind', name='', source=source, destination='/data', rw=True)]
        h['consumers'].append(u)
        real_stat = m.os.stat
        def fake_stat(path, *a, **kw):
            if str(path) == source: return SimpleNamespace(st_dev=7, st_ino=999)
            return real_stat(path, *a, **kw)
        mp.setattr(m.os, 'stat', fake_stat)
        c = m.freeze_host(h['observer'], h['state'], h['request'])
        assert t.logical(h['roots']) == before
        assert u['running'] == live
        assert u['id'] not in [v['id'] for v in c['topology']['consumers']]
        assert len([a for a in h['calls'] if a[1] in ('stop', 'wait')]) == 4
        print('LIVE' if live else 'STOPPED', 'alias omitted; capsule published')
with tempfile.TemporaryDirectory(prefix='b23-review-mount-') as d, MonkeyPatch.context() as mp:
    p = Path(d).resolve()
    h = t.b23_host.__wrapped__(t.installation.__wrapped__(p), p, mp)
    c = h['module'].freeze_host(h['observer'], h['state'], h['request'])
    directory, mounts = t._b23_fake_mount_transport(mp, h, c)
    before = (directory/'capsule.json').read_bytes(); sql = t.logical(h['roots'])
    assert t.recovery.read_host_capsule() == c
    for role in ('control', 'receipts', 'staging', 'ipc'):
        mounts.append(dict(id='99', device='99:99', root='/foreign',
                           target='/'+role+'/foreign', options=['rw']))
        assert t.recovery.read_host_capsule() == c
        assert (directory/'capsule.json').read_bytes() == before
        assert t.logical(h['roots']) == sql
        mounts.pop(); print(role, 'nested foreign mount accepted')
    mp.setattr(t.recovery.time, 'time', lambda: c['expires']+1)
    try: t.recovery.read_host_capsule()
    except ValueError as e: assert str(e) == 'host_capsule_expired'
    else: raise AssertionError('expired capsule accepted')
    try: t.recovery.read_host_capsule(now=c['issued'])
    except TypeError: pass
    else: raise AssertionError('caller clock accepted')
    print('OS expiry and no-caller-clock controls passed')
PY
```

The first exploratory invocation failed during fixture setup because a macOS temporary path was not canonicalized before comparing the fixture's IPC path. Reissued with `Path(d).resolve()` as above. That setup failure is not defect evidence. The subsequent original and strengthened probes completed successfully; no candidate file was edited. No production force/test authorization option was used.

## Preserved source behavior and clock correction

The four prior kernel continuations remain unreachable from all public/programmatic/CLI apply entrypoints. `recovery.apply` still raises `P1B2_apply_unavailable` unconditionally. The host helpers do not call them; activation remains closed. The appended boundary does not rewrite the frozen kernel proof or plan digests.

For recognized topology, actual source retains complete retained-registration union, all-container enumeration including stopped containers, full incarnation/image checks, fixed context/operations, shell disabled, bounded client streams/process/global deadline checks and sanitized subprocess diagnostics. Control stop and successful wait precede worker stop; nonzero container exit status is permitted. Final stopped/PID-zero/meaningful-finish checks and repeated daemon/volume/registration observations precede capsule publication. Failure has no remove/prune/restart/finally container cleanup. F1 concerns the incomplete storage-alias recognition feeding that sequence.

Registration brackets authenticated hello with repeated local process/namespace/cgroup/IPC observations and rejects unsupported remote/Desktop/privileged/shared topology. This remains source/E1 evidence, not actual host association or installed proof. Capsule validators are content-only, closed and bounded; hashes bind content and do not authenticate origin. The fixed reader has no environment/label/E1/force fallback, and absent trusted mount refuses.

Independently verified the pre-handoff clock correction through an otherwise valid fake RO transport: OS time after expiry raises `host_capsule_expired`; supplying `now=issued` raises TypeError. Content-only retained-original validation remains distinct from the trusted reader. The implementer's earlier 116-pass result, reproduced expired-clock RED, interrupted focused exit -15, and statement that full had not started are attributed historical receipts, not fresh reviewer executions. The corrected frozen 117/810/962-pass claims are checked by the fresh suites below.

## Fresh verification

Existing `/tmp/pab-0c-venv/bin/python` only, no installs. Bytecode/cache disabled. Focused and full suites run sequentially against unchanged exact candidate source/tests. Independent stdin probes use separate disposable local roots and no actual Docker/proc/network/model lane.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Focused result: **810 passed in 542.31s (9:02), exit 0**. This freshly covers all 117 added B2.3a cases and inherited recovery/packaged lifecycle/Admission/Dispatcher cases. Full service/release started afterward on the same unchanged candidate.

Full result: **962 passed, 18 skipped in 587.49s (9:47), exit 0**. Sequential after the focused suite, on the same unchanged exact candidate. All 18 skips remain unqualified. Both fresh populations match the corrected frozen candidate's claimed counts; no candidate test failed. The independent adversarial probes establish the two missing negatives despite these passing suites. No redundant broad campaign ran.

Final HEAD remains exactly `05afbaec0c3d61f60bb2f430d9108da30f285d0c`; candidate status is empty. `git diff --check 38b6221 HEAD` passes. Frozen-to-final remains handoff-only, and the protected-path comparison excluding the exact four authorized paths is empty. Root has no tracked diff; the newly written review is untracked alongside preserved preexisting research. No source/test edit occurred during or between the sequential suites.

## Evidence limits and write boundary

**E1 mocked host evidence is not actual-host authentication.** No actual Docker operation, network access, other-user proc inspection, audio/model/paid runner, run-e2e-test, install, push/deploy, prune or user cleanup ran. The alias and mount counterexamples are coherent modeled OS boundaries plus source reasoning; they are not real Linux bind-mount acceptance receipts. No adversarial claim against a malicious host administrator is made.

C1's installed image/entrypoint, host-admin identity and mapped UID 0, protected proof mount, new-installation ownership model and exact maintenance-consumer exception remain pending. No macOS/container UID equality or chown of existing data is assumed. B2.3b preparation and its conservative existing-high-water footprint proof, B2.3c wiring/resume/startup binding, B3, C1/C2, parent integration, final global Astra and public/unattended acceptance remain pending. All installed/platform skips remain unqualified. **A1 global Docker-inventory drift remains unexplained and unqualified; no cause or waiver is inferred.** Physical ENOSPC/power-loss/daemon/VM/controller durability, clean-host operation and engine/audio/hardware acceptance remain unqualified.

Only `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23a-r1-sol.md` is written as a repository artifact. No candidate edit, commit, subagent, push or deployment. Preexisting root research and other worktrees are preserved. Findings require correction and fresh independent review; passing inherited suites cannot waive them.
