# H4 complete corrected Pocket CPU adapter — independent r243 Sol review

2026-10-24. **Bounded SOURCE GO: R241-1 is resolved; no new material finding in the complete corrected adapter.** This is independent source/inert-test review, not adoption, lifecycle, actual clone quality, installation or release approval. Parent owns those decisions and gates.

## Exact pins and scope

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e`, branch `t3/validate-pocket-wav-riff`, clean at documentation HEAD **`645b7fd20747a59cf4aa367e5ad89e403554228c`**. Parent is `b57a85b813cd50ed975e22ad8d06b2aab2141646`; original source cherry-pick `a023bba32100c24d06aba370e76b94d43206712f` became `5d0475b6627e95ef389ec7da72010099840e278f`, original documentation `8b0c282df61e97390a674a0c52d03d277ca218c4` became `e304e48c9692fa06f268da1ffe51a8bfae04fc89`; correction source is **`805d4d3c1578f1e33d5a4d4fbfee4445784cb22a`**.

Inspected the full parent-to-HEAD three-file source/test addition: `worker/engines/pocket.py` (190 lines), `worker/engines/pocket_execution.py` (56), `tests/worker/test_engine_pocket.py` (316), as well as the correction diff separately. Other additions are only r240/r242 reports. Read full r240, r241, r242 and r188 including assignment 3. Read actual unchanged `lib/tts_backends/pocket_tts.py`, `assets.py`, `contract.py`, and Kokoro adapter/tests. Inspected test conftest: no worker conftest; focused unittest imports bypass pytest collection, maintenance and listener fixtures. No applicable AGENTS.md found in checked ancestors, candidate tree or report destination ancestry. Bounded memory registry search returned no relevant hit and supplied no review premise.

Historical `/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62` remains clean at `8b0c282df61e97390a674a0c52d03d277ca218c4`. No source edits, commits or delegation. Sole lasting write is this report in the root checkout.

## Findings and complete-boundary assessment

**No blocking or other material finding. R241-1 closed.** `pocket.py:58-92` validates complete authenticated RIFF bytes: exact container extent, bounded 4096-chunk walk, every header/payload/padding extent, one fmt preceding one data, exact tag-1 mono PCM16/rate/byte-rate/alignment, and data size exactly twice the descriptor frame count. Sixteen-byte fmt and eighteen-byte fmt with zero extension are accepted; extensible/float/other precision declarations refuse. Opaque ancillary chunks before/after data and nonzero padding remain supported. It does not assume a 44-byte header. Every iteration advances at least eight bytes; bounded input and chunk count prevent an unbounded walker. Validation at `pocket.py:143-146` precedes namespace snapshot, lazy backend import (`154-156`) and construction (`161`). All four original exact-hash-bound malformed fixtures now refuse with no factory call.

The complete descriptor and adapter retain the commissioned boundary:

- `pocket_execution.py:7-56` freezes a closed CPU descriptor, binds raw manifest digest/checkpoint/revision/language, authorized reference identity/hash/bytes/frames/rate, exact threads and finite text/output/memory limits. Integer limits reject bool; wire keys are exact. No public path, transcript, seed or arbitrary model control. `pocket.py:95-104` rejects unsupported preset/instruction even when empty and uses integration-owned ID mappings.
- `pocket.py:22-55,105-142` performs component no-follow traversal, final owner/mode/type/single-link checks, bounded raw manifest verification and exact selected identity, safe unique asset inventory with config present, full chunked hashes and per-file/aggregate bounds before backend construction. Only safetensors/YAML/JSON are admitted. The unchanged backend's local verified-bundle/config-path checks remain in effect. Stable immutable owned mappings/storage are explicit premises; legacy path reopens do not establish protection from an adversarial same-UID writer outside that premise. No new TOCTOU blocker is asserted.
- `pocket.py:143-163` captures exact fully hashed reference bytes into an exclusive mode0600 WAV in the existing owned mode0700 caller namespace. The real backend takes its own reference snapshot there. Snapshot source mutation cannot substitute captured conditioning bytes. Backend receives exact selected root/language/threads/namespace and exact authored text/reference, with no adapter language/model fallback or acquisition path. Legacy defaults, seed and state-cache behavior are unchanged.
- `pocket.py:164-190` revalidates mutable AudioResult rate/channels/shape/real finite normalized samples, enforces output frame/byte bounds and serializes measured mono PCM16 WAV with exact digest, text hash and detached descriptor identity. Backend close runs on output/call failure and success; close or adapter-snapshot unlink failure blocks successful delivery. Constructor partial resources and forced-reap/post-reap namespace handling remain caller/lifecycle responsibilities. This source lot neither deletes the caller namespace nor claims allocator/OS enforcement of memory_bytes.

The supported WAV subset is explicit and appropriate for this boundary; rejection of extensible WAV is intentional, not a claim that every WAV variant is supported. Ancillary payload semantics are opaque. Real upstream decoding, config/weight semantics and acoustic compatibility require later actual qualification. The compatibility test executes the real legacy class with inert Torch/Pocket/YAML stubs; it does not establish real YAML parsing, loading or inference.

## Independent execution

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B --version`: **Python 3.14.7**. Candidate cwd for all checks was the exact reviewed checkout. Used explicit absolute import paths and inherited `PYTHONDONTWRITEBYTECODE=1`, including for the tests' lazy-import subprocesses. No broad discovery or lifecycle suite.

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e"); sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e/tests/worker"); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ("test_engine_pocket", "test_engine_kokoro")); r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())'
git diff --check b57a85b HEAD
```

**22 methods PASS: 15 Pocket + 7 Kokoro**, exit 0; diff whitespace PASS. Includes four original regressions, valid ancillary/fmt-extension cases, malformed RIFF matrix, unlink failure, exact snapshots/identity, binding/file refusals, finite output limits and legacy inert compatibility.

Additional independent small inert probe command (exit 0):

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B - <<'PY'
import sys, struct, ast
from pathlib import Path
root=Path('/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e')
sys.path[:0]=[str(root),str(root/'tests/worker')]
from test_engine_pocket import PocketTests
from worker.transport import Refused
from worker.engines.pocket import _reference_pcm16
t=PocketTests(); t.setUp()
try:
    for offset,fmt,value in [(34,'<H',12),(32,'<H',4),(40,'<I',9),(4,'<I',4096)]:
        raw=bytearray(t.reference);struct.pack_into(fmt,raw,offset,value)
        try:t.run_engine(t.reference_spec(bytes(raw)))
        except Refused as exc:assert str(exc)=='reference_format'
        else:raise AssertionError(offset)
        assert not t.calls
    print('four exact hash-bound R241 mutations: REFUSED before constructor')
    for n in range(1,len(t.reference)):
        raw=bytearray(t.reference[:n])
        if n>=8:struct.pack_into('<I',raw,4,n-8)
        try:_reference_pcm16(bytes(raw),24000,4)
        except Refused:pass
        else:raise AssertionError(n)
    print('51 truncation prefixes, repaired container size where possible: REFUSED')
    fmt=t.chunk(b'fmt ',t.reference[20:36]); data=t.chunk(b'data',b'\0\0'*4)
    raw=t.riff(t.chunk(b'JUNK',b'x')[:-1]+b'\xff',fmt,data,t.chunk(b'JUNK',b'')*4093)
    result=t.run_engine(t.reference_spec(raw));assert result['frames']==3 and t.calls[-2][1]==raw
    print('4096 chunks including nonzero odd padding: ACCEPTED with exact snapshot')
    t.calls.clear();raw=t.riff(fmt,data,t.chunk(b'JUNK',b'')*4095)
    try:t.run_engine(t.reference_spec(raw))
    except Refused:assert not t.calls
    else:raise AssertionError('4097 accepted')
    print('4097 chunks: REFUSED before constructor')
finally:t.doCleanups()
for p in ('worker/engines/pocket.py','worker/engines/pocket_execution.py','tests/worker/test_engine_pocket.py'):
    ast.parse((root/p).read_text(),filename=p,feature_version=(3,11))
print('three-file Python 3.11 grammar: PASS (not 3.11 execution)')
PY
```

All five printed outcomes matched. The 51 prefix probes exercise the pure parser; factory assertions apply to the four original mutations and full-execute chunk-boundary probes. Grammar check is **Python 3.11 syntax compatibility only**, not execution under Python 3.11.

## Limits and parent handoff

Disk observation was 3,903,792 KiB available on `/private/tmp`; only explicitly authorized small temporary inert fixtures ran below the 6 GiB actual gate. No ML/model execution, Docker, network/download, listener, process crash, operational cleanup or HELD-resource access. No service/core/persistence/recovery/contracts/installer/provisioning integration or new validation framework. Candidate and historical checkouts stayed unchanged.

Parent owns adoption, execution-context/mapping authority, resource admission and OS enforcement, lifecycle/cancel/descendant/reap integration and separately authorized post-reap cleanup, prepared-reference/audition workflow and restart reuse, actual CPU clone intelligibility/similarity/listening, installation and release. This GO closes the corrected adapter's bounded source review only.
