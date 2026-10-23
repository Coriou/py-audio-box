# P1B23 SQLite cap — bounded offline evidence analysis

2026-10-23. **Cap remains HOLD.** Retained bytes establish a checksum-valid current WAL generation with a committed 1,615-page address space (6,615,040 bytes), supported by complete main-file/WAL page coverage while every archived SQLite file is individually <=4 MiB. This resolves the narrow header/overlay question positively. It does not establish full SQLite semantic integrity, the precise runtime reason no SQLite exception escaped, or authority to alter the test or resource policy.

## Scope and identity

Read the entire `P1B23-CHILDWAIT-GUARD-r2-sol.md` and ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-207e07d8/tests/service/test_recovery.py:3406–3450`. Candidate HEAD verified `f8822965c2e6ba970d36937e11225fb0fd0dc2b9`, porcelain empty; source/test frozen pin `3f010b3ae469b68f4ca0716c4019bb5f3e490b42` remains the prior review's provenance, not a new execution qualification. Report checkout HEAD is separately `3c626d352134eceb389e27ca316c2072b19f833a`. Preexisting root STT untracked artifacts preserved. No applicable ancestor AGENTS.md found for report destination.

Prior independent 61 guard / 42 focused passes are attributed to that review, not rerun here. Retained `frozen-17-entry.stdout` confirms 5 passed / 1 failed, helper exit 1 at `assert failed`, not a signal. The four direct OS operation checks passed before SQLite. All 100 INSERT/COMMIT pairs returned without a caught sqlite3.Error and their intervening per-file stat assertions passed. The explicit rollback/checkpoint/close, final limit verification, and outer unchanged-parent-limit assertion after the failing returncode were not reached. Interpreter teardown remains a separate possible source of archived state changes.

## Integrity and finite inputs

Before interpretation, independently validated all **125** entries of `/tmp/pab-guard-linux-20261023/evidence-hashes.json`: zero mismatches, **216,565,692 bytes** hashed per full validation pass. Manifest relative paths reject absolute names and `..`; final parser also rejects symlink inputs and limits each listed file to 80,000,000 bytes. Manifest validation was performed initially and again by parser runs; no hashes were replaced. The manifest is a retained consistency reference, not an independent signature.

Selected uncompressed archive: `frozen-17-entry-tmp.tar`, **74,352,640 bytes**, SHA256 `e10ecb451fd0993f669dcd5cf97db11acbe3e467221b62c551523d64a3498acb`. Iterated 4,181 tar members (hard stop 10,000), read only the three exact regular members below, rejected duplicate selected names, traversal/absolute paths and unexpected sizes. No extraction occurred. Prefix for all names: `./frozen-17-entry/test_lotk_actual_linux_child_f0/`.

| Member | Bytes | SHA256 |
|---|---:|---|
| database.sqlite | 4,194,304 | `4665efb6c66987f591c985ab8b2a30511bd3290c9efe98281fcdd46079d28c9f` |
| database.sqlite-wal | 4,190,072 | `81439c0f6fdaa130e8a819f03e65c9387eb7fcf0ef086944086a69d9d74f4090` |
| database.sqlite-shm | 32,768 | `a178db2afa99a5533660a7938f43698ecb547325845028b5db06fc1383c2cefd` |

Member reads total **8,417,144 bytes**, one bounded read per member per parser invocation. Tar metadata scanning is bounded by archive length/member count. Hash streaming uses <=1 MiB reads. Parsing uses ordinary host `python3` stdlib only: pathlib/hashlib/json/tarfile/struct; no SQLite import/open, product import, pytest, workload, fault generation, GC/allocator/fork diagnostic, Docker, installation, service, model/audio, network workload or paid test.

## WAL and committed overlay facts

Format rules verified against the official [SQLite database file format, sections 4.1–4.5](https://sqlite.org/fileformat2.html#walformat). Current-generation acceptance requires matching salts and cumulative checksums; latest committed page images override main-file pages. Header fields are big-endian; checksum input byte order depends on magic.

- WAL magic **0x377f0682**, format **3007000**, page size **4096**, checkpoint sequence **1**. Checksum input is little-endian, stored checksums big-endian. Header checksum independently matches `(1699629882, 3623586718)`. Salts `(659056527, 3211314617)`.
- **1,017** complete physical frame slots, zero partial trailing bytes. Frames **1–798** consecutively match both salts and cumulative checksums. Every accepted page number is positive/in range. There are **44 commit boundaries**, ending exactly at frame 798, with no uncommitted valid suffix.
- Commit database sizes grow from **922** to **1615** pages. No transaction's maximum page number exceeds its commit database size. Every boundary has complete address coverage using its latest committed overlay plus physically present main pages. The historical main-file contents are not preserved per boundary, so coverage does not prove historical snapshot consistency.
- The first current-generation commit above 4 MiB is frame **145**, dbsize **1034**, **4,235,264** logical bytes. Full per-frame and per-boundary records are retained in `analysis.json`.
- Frame **799** is the first invalid current-generation frame: salts and checksum fail. All **219** remaining physical slots (799–1017) carry salts `(659056526, 1851662757)`, none matching the current header. Slots 800–1017 have checksums consistent with their immediately preceding stored checksum; slot 799 does not match the current predecessor. These are consistent with stale previous-generation slots, not admissible current frames. No old header/prefix is reconstructed, and no old generation is certified. Every physical slot was inspected; stale slots are excluded from commit and page mapping.
- Latest committed overlay: **712** distinct pages, **591** above physical main page 1024, covering every page **1025–1615**. The remaining **903** logical pages use the main file. Thus all 1,615 page addresses have retained bytes. Latest page 1 is frame **781**; its database-page count is **1615**. Both that header and the main header have change counter **2** equal to version-valid-for **2**. Main physical length is only **1024** pages despite the same 1615-page header declaration.

**Established:** the header declaration is supported by a checksum-valid committed WAL overlay and complete page-address coverage; a committed logical address space larger than L spans two individually bounded files here. **Not established:** a complete b-tree/freelist/overflow/schema/SQL integrity verdict or a live successful SQLite reopen. Main pages have no WAL-style checksum, and tar captures are not an atomic runtime snapshot. “Valid committed” here refers to WAL format/checksum/commit validity and byte coverage, not `PRAGMA integrity_check` or durability certification.

## Test premise, budget and causal limits

The original test demands both per-file lengths <=L and **at least one sqlite3.Error within 100 inserts of 65,536 bytes**, with WAL/FULL/mmap_size=0. It does not assert a logical page limit or a combined DB+WAL <=L limit. Its failed `assert failed` is an unmet required-error premise, not evidence of a physical file overrun. Four direct write/pwrite/truncate/ftruncate EFBIG witnesses establish their own finite checks only.

Archived DB+WAL is **8,384,376 bytes**; including SHM is **8,417,144 bytes**. Logical size is **6,615,040**, not either physical sum. WAL stores repeated versions, frame overhead and stale slots. No archived individual file exceeds L. The test's per-iteration observations do not establish continuous peaks, and archived sizes cannot reconstruct all intervening operations.

The existing source model in candidate `RECOVERY-P1B2.md:1080–1110` explicitly budgets **three physical files**, each <=L: `U_all_early = M + 3*L + 1,048,576 + 2*65,536`, checked against retained H, then H plus retained demands against B. The fixed additive allowance is **13,762,560 bytes**. A logical DB spanning DB/WAL does not by itself contradict that physical accounting model or grant an extra logical allowance. Snapshot <=1 MiB, each document <=64 KiB, independent aggregate/entry/free-reserve checks, unchanged H/charges and namespace/freeze assumptions still apply. This test is a separate stress fixture, not an accepted production-budget witness. No policy waiver follows, and the failing test remains failing.

The stale suffix and current header are consistent with WAL reuse; main length/header mismatch is consistent with incomplete transfer to the main file. Neither establishes exact checkpoint timing, errno handling, SQLite/VFS/kernel cause, successful durability, or what interpreter cleanup changed. SHM was hashed and retained, not interpreted as authoritative lock state/mxFrame/nBackfill: its native/transient state cannot prove historical locks, completed syncs, runtime errors or checkpoint order. Archived bytes cannot establish why all 100 calls returned without sqlite3.Error. The production source model also disables automatic checkpoint whereas this test does not explicitly do so; that source difference is relevant context, not a demonstrated cause.

## Reproduction, next gate and cleanup

Retained ordinary static parser `/tmp/pab-sqlite-cap-offline-20261023/parse.py`, SHA256 `ac64517deab72c1bf387bbd126dd23a7551814a3978597080a84fc2566cd44ec`. Output `analysis.json` SHA256 `098fa6b3dae93ce54d55b82501a1ab7cccb6fd57f105b160503cf0fd72bffd8f`; `summary.txt` SHA256 `c373dddac17d4633e6cd5fe0fa16eb28b9f283a322ff80acb1613a4071358dd6`. Exact repeat command (static parsing only):

```sh
python3 /tmp/pab-sqlite-cap-offline-20261023/parse.py > /tmp/pab-sqlite-cap-offline-20261023/summary.txt
shasum -a 256 /tmp/pab-sqlite-cap-offline-20261023/parse.py /tmp/pab-sqlite-cap-offline-20261023/analysis.json /tmp/pab-sqlite-cap-offline-20261023/summary.txt
```

The parser checks every current-generation checksum in order, stops acceptance at the first invalid frame, audits every remaining slot separately, records all commit sizes and missing-page lists, and maps latest committed pages without writing an assembled DB. Its explicit hard bounds are 80,000,000 bytes per manifest file, known 74,352,640-byte tar, 10,000 member headers, exact three member sizes, 1,017 slots, and commit dbsize <=10,000 (actual maximum 1,615). No speculative evidence reconstruction.

**Minimal next step:** parent should decide the intended contract of the existing required-error assertion in light of this physical/logical distinction, with exact shipped SQLite source/VFS/checkpoint error-path review before selecting a runtime diagnostic. If causal resolution remains necessary, separately authorize one unchanged ordinary 100-iteration diagnostic recording per-iteration logical/file sizes, exact execute/commit errors and checkpoint outcomes. This report does not authorize or run it, propose a new limit/journal/iteration/assertion, or approve implementation. Root owns design/implementation and fresh independent review.

Finite analysis completed; no child process/container/service remains from this analysis. Temp parser/JSON/summary intentionally retained, no extracted DB or SQLite handle created; every worktree, evidence archive and STT artifact preserved. Sole repository write is this report; no commits/delegation/source fixes/test retries. Cap HOLD, combined integration and remaining validation gates stay open. No qualification of the full 19 Linux mechanisms, C1/C2, native Darwin cause, installed host, durability, activation or production follows.
