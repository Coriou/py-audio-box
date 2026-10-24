# H4 Pocket WAV validation correction — r242 Sol

2026-10-24. R241-1 source correction submitted for parent independent fresh review. No self-adoption, lifecycle qualification, actual inference or release claim.

## Exact pins and ownership

Fresh checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e`, branch `t3/validate-pocket-wav-riff`, entered clean at parent `b57a85b` (the r241 report commit). Before correction, cherry-picked exact candidate source `a023bba32100c24d06aba370e76b94d43206712f` then exact report `8b0c282df61e97390a674a0c52d03d277ca218c4`, producing `5d0475b` then `e304e48c9692fa06f268da1ffe51a8bfae04fc89`. Correction source commit: **`805d4d3c1578f1e33d5a4d4fbfee4445784cb22a`**. This report is a separate subsequent commit.

Historical candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62` was only inspected: clean at exact `8b0c282df61e97390a674a0c52d03d277ca218c4`. No root or historical candidate writes. Correction changes only `worker/engines/pocket.py` and `tests/worker/test_engine_pocket.py`; no shared helper or legacy backend change.

Read full r241 and r240, r188 including assignment 3, descriptor and adapter/test source, root test conftest and existing ingest RIFF validator. No applicable AGENTS.md found in checked ancestors or checkout. No worker conftest exists. Bounded memory-registry query returned no relevant match and supplied no premise.

## Correction and coverage

The adapter now checks exact RIFF/WAVE container length, every chunk header/payload/padding extent, and a maximum of 4096 chunks against the already fully authenticated bounded reference bytes before snapshot creation or backend construction. It requires exactly one fmt before exactly one data chunk. Supported format is classic tag-1 mono PCM, exact 16-bit precision, descriptor rate, byte rate equal to rate times two, block alignment two, and complete even data length equal to descriptor frames times two. Classic 16-byte fmt and 18-byte fmt with zero extension length are supported. Extensible, float and other declarations refuse explicitly. Valid opaque ancillary chunks, including odd payloads with padding and chunks after data, remain supported; no fixed 44-byte header assumption.

Inspected `service/ingest.py:_riff_pcm`: its extent/padding/count pattern is suitable, but it depends on service oracle policy and accepts wider precision/channel formats. A narrow local worker parser avoids importing service or changing ingest behavior within this two-file correction ownership.

Added four exact r241 header mutations with each malformed reference freshly hash-bound and factory calls asserted empty. Added valid ancillary/zero PCM-extension fixtures, truncated header/chunk/payload, missing odd padding, odd data, duplicate fmt/data, reversed/missing chunks, trailing bytes, chunk-count limit, float/extensible format, bad byte rate and invalid extension tests. Added snapshot-unlink failure coverage using the real stdlib closer with only its unlink operation injected: failure propagates after backend close, exact snapshot and caller sentinel/namespace remain. Ordinary owned temporary-fixture teardown runs afterward. No forced-reap or operational-cleanup inference follows.

## Executed checks

Requested filename `/private/tmp/pab-0c-venv/bin/python3.14.7` does not exist. Used existing `/private/tmp/pab-0c-venv/bin/python`, verified by `-I -B --version` as **Python 3.14.7**. No installation or environment modification.

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e"); sys.path.insert(0,"/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e/tests/worker"); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ("test_engine_pocket", "test_engine_kokoro")); r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())'

/private/tmp/pab-0c-venv/bin/python -I -B -c 'import ast,pathlib; root=pathlib.Path("/Users/ben/.txcode/worktrees/py-audio-box/t3-b213ac3e"); paths=("worker/engines/pocket.py","worker/engines/pocket_execution.py","tests/worker/test_engine_pocket.py"); [ast.parse((root/p).read_text(), filename=p, feature_version=(3,11)) for p in paths]; print("Python 3.11 grammar PASS:", *paths)'

git diff --check
```

Final **22 methods PASS: 15 Pocket + 7 Kokoro**. Python 3.11 grammar PASS for the three listed files; no Python 3.11 execution claim. Diff whitespace check PASS. Initial focused run had one test-injection failure: patching tempfile's module unlink did not replace the closer's definition-time default. Corrected the test to inject the closer's unlink argument; no production cleanup change. Same focused suite then passed. No pytest collection, lifecycle suite or broad discovery.

## Limits and handoff

Only small inert source fixtures ran under the user's explicit below-6-GiB test authorization. No delegation, Docker, network/download, actual model, listener, process crash, operational cleanup or HELD access. Existing compatibility test uses inert ML/YAML stubs. Frozen descriptor/mapping authority, exact reference snapshot, unchanged PocketBackend, caller namespace ownership, finite output measurement and detached identity retain their prior scope; this correction supplies no integration or actual-inference evidence.

Parent owns independent fresh review, adoption, lifecycle/context mapping and post-reap integration, and any separately authorized actual CPU clone/acoustic/install/release qualification.
