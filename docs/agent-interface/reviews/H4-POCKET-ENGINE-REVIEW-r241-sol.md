# H4 Pocket CPU adapter independent review — r241 Sol

2026-10-24. **Bounded NO-GO for source adoption: one P2 format-validation finding.** Parent owns adoption after correction and review. This verdict does not establish lifecycle, actual clone, acoustic, installed-runtime or release qualification.

## Pins and scope

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62`, branch `t3/pocket-cpu-adapter`, base `2a374c998135a0b18000ac5187ac5e155522da8c`, source `a023bba32100c24d06aba370e76b94d43206712f`, documentation HEAD `8b0c282df61e97390a674a0c52d03d277ca218c4`. Candidate was clean before and after execution. Base-to-source adds exactly `worker/engines/pocket_execution.py`, `worker/engines/pocket.py`, and `tests/worker/test_engine_pocket.py` (446 lines); source-to-HEAD adds only r240's report. Base-to-HEAD `git diff --check` passed.

Read r240 FULL, r188 including assignment 3, and r238/r239 reference-publication reports. Inspected all added source/test lines, unchanged PocketBackend, bundle verification, AudioResult, Kokoro adapter/tests, provisioning's Pocket config construction, and test conftest. No applicable AGENTS.md was found in checked ancestors or candidate paths. A bounded memory-registry lookup returned no relevant match and supplied no premise. No delegation, commit or candidate/root source edit. Sole lasting write is this report; root's pre-existing untracked STT material was left untouched.

## Finding R241-1 — P2: malformed/non-PCM16 reference WAVs reach the backend

**Location:** candidate `worker/engines/pocket.py:108-113`, especially line 110. `wave.getsampwidth() == 2` does not prove a 16-bit PCM declaration, and `getnframes()` plus returned byte count does not prove complete declared RIFF/data chunks. Python's wave reader rounds bits-per-sample up to bytes and floors data chunk size to frames; it also tolerates EOF before some declared lengths. Consequently these checks admit format-invalid authorized bytes before the lazy backend constructor.

Independent reproduction used the existing four-frame, 24 kHz, mono fixture (52 bytes, eight PCM bytes). Each case changes only one header field and updates the descriptor's full reference SHA-256 to bind those exact immutable bytes. No race, foreign reference, mapping substitution, mutable managed storage or hostile host is involved:

| Header mutation | Expected rejection reason | Observed |
| --- | --- | --- |
| Offset 34, uint16 = 12 | Declares 12-bit PCM rather than exact PCM16 | Backend constructed; exact malformed snapshot forwarded; successful three-frame output |
| Offset 32, uint16 = 4 | Block alignment inconsistent with mono PCM16 (must be 2) | Same |
| Offset 40, uint32 = 9 | Declares nine data bytes but only eight exist; incomplete sample/chunk | Same |
| Offset 4, uint32 = 4096 | RIFF container declares bytes absent from the file | Same |

This violates the commissioned pre-ML full format/complete-payload verification and r240's stated complete declared PCM payload guarantee. A correct hash authenticates bytes but does not establish their format. These malformed files can reach the real backend's model loading and conditioning decoder, whose interpretation/refusal is not this adapter's verified exact PCM16 boundary. The probe used an injected backend only; no claim is made about an actual decoder's behavior.

**Required correction:** validate bounded RIFF/chunk extents and alignment, exact supported PCM format fields (including 16-bit precision and consistent block alignment/byte rate), and the full declared data payload before backend construction. A bounded chunk walker can preserve supported ancillary chunks without requiring every valid WAV to have a 44-byte header. If extensible WAV is supported, validate its PCM subtype/valid precision explicitly. Add pre-constructor rejection cases for the four mutations and suitable valid WAV coverage. This is a local adapter/test correction, not a request for a new hostile-host framework or integration rewrite.

Reproducible command, run in the candidate:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B - <<'PY'
import sys, dataclasses, hashlib, struct
sys.path.insert(0, '/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62')
sys.path.insert(0, '/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62/tests/worker')
from test_engine_pocket import PocketTests
for label, offset, fmt, value in [
    ('truncated_odd_data', 40, '<I', 9),
    ('truncated_RIFF', 4, '<I', 4096),
    ('PCM_12_bits', 34, '<H', 12),
    ('wrong_block_align', 32, '<H', 4),
]:
    t = PocketTests(); t.setUp()
    try:
        raw = bytearray(t.reference)
        struct.pack_into(fmt, raw, offset, value)
        raw = bytes(raw); t.ref.write_bytes(raw)
        spec = dataclasses.replace(t.spec, reference_sha256=hashlib.sha256(raw).hexdigest())
        result = t.run_engine(spec)
        print(label, 'ACCEPTED', 'backend_constructed=', bool(t.calls),
              'snapshot_exact=', t.calls[1][1] == raw, 'output_frames=', result['frames'])
    finally:
        t.doCleanups()
PY
```

All four printed `ACCEPTED backend_constructed= True snapshot_exact= True output_frames= 3`; exit 0.

## Other reviewed boundaries

- Descriptor closes wire keys, fixes CPU/engine/output format and binds exact bundle/raw manifest/model/revision/language, reference identity and finite integer limits. Integer bounds reject bool. Unsupported preset/instruction values, including empty strings, refuse; seed/transcript/path are absent from the interface. Result identity is detached for normal supported scalar values and includes the exact UTF-8 text digest.
- Every listed bundle asset is bounded and hashed before backend construction, with safe relative names/extensions, unique inventory, config presence, no-follow traversal, final owner/mode/type/link checks and metadata stability. The legacy backend re-verifies and reopens paths. Those reopens are not independently race-proof, but stable immutable integration-owned bundle storage is an explicit premise of this lot; no blocking TOCTOU finding is asserted by inventing an adversarial same-UID writer outside that premise. The adapter does not activate legacy manifests missing an explicit exact revision.
- Reference bytes are fully hash/length bound and copied to an exclusive mode0600 snapshot in an owned mode0700 attempt namespace. Original reference mutation after capture cannot alter the forwarded snapshot. Namespace/path stability remains caller-owned; same-UID replacement of this private namespace is outside the stated premise. Real PocketBackend takes its own snapshot there and preserves legacy direct-call defaults/cache/seed behavior.
- Returned AudioResult is checked again despite mutability: exact integral rate/mono shape, finite real normalized samples, positive frame count, frame/byte ceilings and measured PCM16 output hash/size. Backend close runs after synthesis/output success or failure. Close or snapshot unlink failure prevents delivery of a successful result; simultaneous failures can mask the earlier exception, but no successful result ambiguity was found. Constructor failure cannot close a nonreturned backend instance; adapter snapshot unwind still applies. Partial construction resources and forced-reap cleanup remain lifecycle-owner responsibilities, as documented. No unrelated cleanup authority is added.
- Existing tests meaningfully exercise refusal before factory use, source-reference mutation, caller namespace/sentinel retention, output mutation, call/close/constructor failure and lazy imports. They do not cover the malformed WAV cases above or injected snapshot-unlink failure. The real-class compatibility test stubs torch, Pocket runtime and YAML; it validates forwarding and snapshot/cache behavior, not real YAML interpretation, upstream compatibility or inference.

## Independent execution and limits

Interpreter `/private/tmp/pab-0c-venv/bin/python`: **Python 3.14.7**. Used `-I -B`, explicit absolute candidate/test import paths and `PYTHONDONTWRITEBYTECODE=1` (also inherited by the tests' inert import-check subprocesses). Inspected the root conftest; no worker conftest exists. No pytest collection, maintenance guard or listener fixture ran.

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62"); sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62/tests/worker"); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ("test_engine_pocket", "test_engine_kokoro")); r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())'
```

**18 methods PASS: 11 Pocket + 7 Kokoro.** The four additional inert malformed-header probes reproduced R241-1. No lifecycle tests were rerun; r240's earlier injected lifecycle PASS and Python 3.11 grammar result remain carried report claims, not independent evidence here. Python 3.11 execution was not performed.

Only small owned temporary source fixtures and their ordinary teardown ran below the stated 6 GiB actual-admission gate. No actual ML/model, Docker, download/network/listener, process crash, operational cleanup or HELD-resource access. No service/core/persistence/recovery/contracts/installer/provisioning changes were made. After the local WAV boundary is corrected, parent still owns context/mapping construction, lifecycle and post-reap cleanup integration, real preparation/audition, actual clone listening/restart/cancel qualification, adoption and release decisions.
