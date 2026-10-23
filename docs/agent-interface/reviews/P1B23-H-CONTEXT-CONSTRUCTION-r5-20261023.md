# P1B23 H context construction r5 — actual receipt

**STOP. The one authorized construction-only preflight failed. No H acceptance, integration or continuation authority. Independent actual-receipt review and parent decision are next.**

Read in full root `P1B23-H-CONTEXT-CONTROL-SOURCE-r5-sol.md` and prior `P1B23-H-LITERAL-SOURCE-r4-sol.md`. Candidate ABS `/Users/ben/.txcode/worktrees/py-audio-box/h-control-context-20261023` remained read-only and clean at documentation HEAD `c30f058b6d355ea887ba38cdab406b7c3684c87b`; executable pin `4fa9f460841d7c169037e6491b89a483d02bf110`. Their diff is documentation only. All830 tracked files were hashed/copied from the frozen checkout, verified inside Linux before imports and after the failed run, and reverified on host before/after/cleanup. Script SHA256 `6d8c77fbd1fe81e4a202f67d606ee033cefa49d2099e255d6fc7e4c9eca2935e`. No applicable AGENTS.md found. Memory registry search had no relevant hit; no memory fact used. Sole repository write is this report; no commit. Root preexisting STT evidence preserved.

## Single invocation and five required checks

Exactly once, from `/h-candidate`, `timeout 60s python -u docs/agent-interface/evidence/h-exit-context-construction.py`. Outer orchestration deadline60s also applied. Exit1 after 0.677s, no timeout. Raw stdout contains exactly three flushed JSON records, followed by private stderr traceback at script line86, `assert positive_report['recognized'] is True`.

1. Actual selection emitted one matching CALL offset2576. Actual-runner diagnostic recognized True; all named predicates True. Retained adjacency: LOAD_GLOBAL os@2542; LOAD_ATTR _exit@2552; PUSH_NULL@2572; LOAD_SMALL_INT74@2574; CALL1@2576. Its selected/normalized offsets agree.
2. Isolated-positive metadata emitted BEFORE assertions, recognized False. Selected/normalized offset2546. Retained adjacency: POP_TOP@2512; LOAD_GLOBAL os@2514; LOAD_ATTR _exit@2524; LOAD_SMALL_INT74@2544; CALL1@2546. No PUSH_NULL occurs in this reported five-instruction site. Identity, code/instruction bounds, index, literal opcode/exact-int/value74, CALL/exact-int/arity1 are True; owner, attribute and null positional predicates are False. The first rejecting predicate in recognizer evaluation order is owner (the first slot is POP_TOP). This is exact current construction-site evidence; it does not establish a historical failure cause.
3. All-actual-CALL singleton scan is after the failed positive assertion and was NOT reached. The one selected actual site being recognized does not prove this scan.
4. Nine fixed rows were NOT reached; zero control-complete records. No primary-row, foreign-code or non-CALL execution credit.
5. No construction-complete record; successful exit missing. Source/retention proofs completed, but the required overall positive receipt is absent. STOP, no retry/fix/opcode broadening.

Selection record code digest `c12c554100cb1936c292896891e5b4bddcc1f9891a9182cf55f9bb8b4ac45d1a`; actual describe record digest `572ef9205b8d8c22d45c5f712028795688a960068d193057956640a051ee7c14`; isolated digest `d2aaa2794b4de3e3fa8de88e0f493e69a9bcca0d5c235089201c422f80c60702`. The first two reported digest strings differ despite referring to the actual code in this invocation; retained verbatim, cause not investigated under this authorization. Do not treat these serialized digests as a substitute for object identity or source-manifest proof.

## Environment, retention and cleanup

Existing base `sha256:3c4eba47125ade77ef22c86767838d78895bc46e56bc5168e4baf2b7b23a03c7` inspected; new owned local tag `pab-h-context-construction-r5-owned-base:20261023` created and verified before FROM. Source-COPY-only build `--pull=false --network=none`; no install or pull. Derived `pab-h-context-construction-r5-source:20261023` = `sha256:ec839a572d897d0dd4732ee49124c5a043411c310a145d859fa7387fe0f11fa7`. CPython3.14.8 and all42 exact installed versions from `tests/release/requirements.lock` verified before candidate imports; identity raw receipt retained. Installed version equality does not freshly authenticate wheel archives.

Atomic mkdir lane `/tmp/pab-worker-w1-docker-lane.lock` with exact driver PID/random token/pin/evidence owner. No foreign lock removed. Host free space >=6GiB at every host check (also above4GiB admission); final 9970610176 bytes. CPU2, RAM2GiB, PID256, networknone, read-only root, UID/GID10001, capdropALL, no-new-privileges, /tmp tmpfs1GiB nosuid/nodev. Inspect verified no binds, published ports, devices or mounts. No socket/secrets/services/audio supplied. PID1 finite1800s sleep with owned graceful termination. No H node, product function invocation, fork/wait test, SQLite connect, pytest execution/collection, host candidate import, other node, retry or fix.

BEFORE test, live sentinel token plus binary NUL/FF/80/CR/LF was created mode0600 UID10001 and verified through direct binary `docker exec ... tar --hard-dereference` output. AFTER failure and BEFORE stop, entire live `/tmp` exported by `docker exec ... tar --hard-dereference -C /tmp -cf - .` directly to a binary file. 2 members, 1 regular file, 37 bytes; ALL regular-file SHA256 matched independently collected LIVE hashes. Safe names/type/size/UID bounds checked, symlinks metadata only; no extraction/DB opening/docker cp. No export failure or repair.

Exact owner label checked before graceful stop; stopped ExitCode0, OOMKilledfalse, Runningfalse. Exact owned docker rm WITHOUT-v, exact name absence, exact lane owner equality and release verified. No live hold. Host source/limits preserved. Cleanup DONE means retention/cleanup only, not construction success.

Private0700 evidence `/tmp/pab-h-context-construction-r5-20261023` retains source tar/manifest, driver, image/container identities, exact commands/status/timing, raw stdout/stderr and flushed JSON, binary sentinel/full archive, live hash map/inventory, audit, cleanup and evidence-hashes.json. Key hashes:

- construction.stdout `3158faa63c6f18c6f473c5c3dcdc15b325dbdf8a4bb7bbf53f100180cd34829f`
- construction.stderr `fa83f968547a42c83892eff7bcc564e7df0ed4ab6c8344681167d861087a3cd9`
- archive.stdout `72e8b8d3ed807f665ee1c21fc65922f95edeb7cf9cdbae52f74bd82b263a4bd5`
- live-hashes.stdout `fc4128cbde646b4921ed7e598eddc156de9af6b90fde95408c1fe0de9656fd15`
- run.py `a5aa0c16b09157366e6a461a2c14812a832a16b859d0370118b43cfdb18b4a80`

Prior construction failure remains without an exact retained cause; this run does not reinterpret it. No live caller f_lasti, unwinding, SAME exception/full route, Hf, complete Ha;Hs;Hf;, original H matrix, other-family or integration credit. All prior H/runtime/release obligations remain pending. Independent actual-receipt review is required next; no self-acceptance.
