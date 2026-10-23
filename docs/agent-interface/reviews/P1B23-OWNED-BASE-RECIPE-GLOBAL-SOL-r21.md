# P1B23 owned base recipe — independent GLOBAL SOURCE + HOST r21

2026-10-23. **Bounded SOURCE + HOST GO for readiness for ONE separately authorized non-product dummy qualification, total <=600s. No concrete blocking path found.** The shared raw-image-ID FROM defect identified by r20 is repaired in both product preparation and dummy qualification. This review supplies no execution authority and establishes no actual build usability, persistence or product qualification. Independent review of the next actual dummy receipt remains required before product dispatch.

## Scope and exact candidate

Independent Sol low review. Sole write is this report in the parent root; no commit or source edits. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-cc139bcf` is clean at documentation `ce70817437b4b24e0d35711e6450bf680f32e2d4`, executable `a9c604aff77cce87cefd81baa5787fbcc6d41a5e`, based on `7df2dab65c710730dd87c1e36cae310fb82fb211`. The brief's `exec` prefix is a label, not part of the executable SHA. Parent root observed at `624ef38e67e58b1a49a8c0d73f69a1c29f990831`; unrelated untracked STT artifacts left untouched. Applicable ancestor/scoped AGENTS inventory found none. Lightweight memory search supplied no relevant evidence.

Read complete latest HANDOFF/FREEZE sections and QUALIFICATION, parent DUMMY-BUILD-ACTUAL-AUDIT-r20, persistent dummy GLOBAL-SOL-r19, lifecycle GLOBAL-SOL-r18 and controlling STORAGE-PROPORTIONALITY-ASTRA-r16. Recovered relevant truncated outputs with focused reads. Read the entire final launcher, qualification, both HOST harnesses, finite driver and dummy writer, rather than only the changed test assertions. Inspected the complete eight-path base-to-final delta and manifest, source bindings and retained receipts.

Verification used read-only Git/text and stdlib AST/JSON/stat/hash inspection. No candidate/product import, harness rerun, preparation CLI, pytest, native campaign, runtime execution, Docker query/tag/build/volume/container operation, network, extraction, cleanup or delegation. No new unverified concern warranted spending another HOST run. Root lock finalization is outside this review; it was neither inspected operationally nor changed.

Only launcher.py, qualify.py, host-check.py, qualification-host-check.py, source-manifest.json and append-only HANDOFF/FREEZE/QUALIFICATION differ from the base. Executable-to-documentation delta is exactly those three documents. All193 manifest payload mappings and actual bytes match; both original derivative mappings/bytes match. finite-driver.py and dummy-writer.py are byte-identical to base. Entry, observer, identity helper, guard, product/tests/dependencies/locks therefore retain their frozen bytes. Manifest adds only the recipe-contract description; no payload hash churn or self-hash cycle. AST parsing of all six reviewed Python programs and candidate diff --check passed.

Final independently verified SHA256 pins:

- launcher.py: `0ded93a96a7136112c163d546cb269d6fdcbff2b06e8e87fecc5db5e30616c29`
- qualify.py: `9908fd5109a717edfce011b72f037edd62f16aeef581035fdc1510a71fc6af23`
- host-check.py: `a78b2b0ba03dc6bc64bf42be5c1494b4dd94d38bb1aa37bc0c996cbc986f582d`
- qualification-host-check.py: `89e8d3c127625c24701333a84128665d96b1af1ffd3046ceaf2cc89bf082a19f`
- source-manifest.json: `1d97c4f899de7a7cafe2d2137c917d7af6c9a17c037022ee93a478bd19ce219a`
- finite-driver.py: `bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732`
- dummy-writer.py: `858a605b9707e65577937f55f94af04f292b57fc77bfe70669b67b7187c0cf5a`

## Shared recipe and failure closure

`launcher.py:341–426` defines one owner-validated local reference, recipe, prebuild command set and exact recipe/seed validation. FROM is `pab-local-base/<canonical UUID>:<48hex token>`, a valid repository/tag shape. Full BASE `sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb` remains a separate expected local image ID, never a FROM or substituted registry digest. Both product build-proposal and dummy plan disclose the inspect/absence/tag/verify argv, expected ID and retained reference. The latest documentation explicitly discloses the lane, journal, first verification, recipe validation and immediate repeated verification before build. Product launcher proposes this work; it does not dispatch a build or silently tag during preparation.

`qualify.py:151–197` acquires the lane before mutation and executes exact BASE inspect with ID/Linux/amd64/startup admission, fresh exact-reference absence, journaled exact-ID tag, tag inspect/admission, recipe validation, immediate second tag inspect/admission, then sole build with --pull=false/--network=none and existing owner labels/IID binding. No volume acquisition occurs until build/image admission succeeds. A collision stops before tag; lost tag acknowledgement leaves SENT-UNKNOWN without retry, build or removal. Acknowledged tags are inventoried and retained, including subsequent failure. Wrong first tag identity/startup conservatively leaves CONFIRMED and holds the lane; it is never treated as a green build. The trusted lane/daemon boundary is explicit; no atomic compare-and-set claim against a malicious concurrent tag writer is made.

`fatal_build` accepts only COMPLETE/exit1, reaped leader, closed descriptors, no group kill, matching captured stderr byte count and a newline-terminated final explicit BuildKit fatal-solve line with diagnostic. That branch journals TERMINAL/result FAILED and still aborts qualification. Generic exit1, missing newline/truncated terminal, timeout, killed/incomplete capture or lost response remains UNKNOWN; the bounded existing label observation does not establish absence/completion. No retry, downstream volume/product/reader or automatic failed-run lane release follows. Historical actual r19 captures/journal and r20 interpretation remain intact.

The prebuild operations consume the existing admission/overall schedule; build remains <=300s, qualification <=600s with protected supervisor/final reserve. No deadline or storage-policy enlargement is introduced. Native local-tag resolution remains an actual obligation: HOST cannot establish Docker/BuildKit reference behavior.

## Retained HOST evidence and history

Final `/private/tmp/pab-dummy-HOST-970m2p0t/HOST-receipt.json` independently hashes to `ef2972bb751250473fc995c0e8ea28f4a8f2ad7d3cd8c17e5125a24653019a69`: PASS3/3 groups in3.034564911999041s, fixed13 available groups/9s per group/120s total. All seven source hashes match the candidate. All181 success/evidence-hashes entries match retained lengths and hashes. All six recipe-repair proof trees below have directories0700, leaves0600 and no symlinks.

Actual case bodies call both shared product preparation/recipe validation and dummy preparation/full plan validation. They execute corrected product HOST29, reject raw-ID FROM, foreign tag, altered expected ID and build argv, and copy/hash actual193 payloads plus generated derivatives. This repairs the old HOST assertion that incorrectly blessed raw FROM.

The finite exact fake queue exercises eleven failures: collision, wrong tag ID, unsafe startup, lost tag acknowledgement, fatal solve, generic exit1, truncated solve, wait timeout, wrong immediate-reverify ID, unsafe immediate-reverify startup and post-tag recipe tamper. Captured argv/order and operation states match these paths. None dispatches downstream volume/create/start; at most one build occurs. Fatal has COMPLETE/exit1/reaped/closed/not-killed and75 captured stderr bytes; generic exit1 has16 bytes without fatal solve; truncated has74 bytes without final newline; timeout is UNKNOWN/unreaped/group-kill-attempted. Only fatal becomes terminal FAILED. Lost-tag remains SENT-UNKNOWN. Success traverses the changed tag/build path and existing writer/two sequential readers, retention, exact full-ID cleanup and owner/inode release, preserving the tag and volume with acceptance=false.

Fake.spawn requires exact finite argv and cwd/stdio/session/close_fds arguments, with no real fallback. Qualification HOST denies subprocess/Real.spawn/system/killpg/fork. Actual private IO, pipes, capture, parser, journal and publication functions run; process statuses, Docker config and PID1 identity are fixtures. Command expectations partly share production helpers, so native CLI semantics remain unqualified. The harness header's historical12-case wording is superseded by its executable13-case assertion/receipt and latest freeze; selected evidence is three groups, not a claimed full13 execution.

Retained progression, independently rehashed:

- s5v99rnr: `002d555f131d58a5cfa6ffcf2f3af0da5f2a5ebf7ee6a55ccf27fea68b18b349`, RED after context PASS; unconsumed expected commands in the accelerated timeout fixture. Preserve this RED.
- ju_09cdp: `ec88ec619532bd8f4c5e51cf763ef996f0f432bcb9d20ff9b9b7763c9b305fff`, PASS2 after the timeout moved to the real capture wait seam.
- 9unh8s_d: `da20379ccec492ef05e18d8b34bbb631be5263fa7ff3440654c3191b975ac0ea`, PASS3 adding affected success.
- s50ggq8n: `a1d38754933a24e9988e9ec506cf0e6e3425b4802bfc531c41cee0916c9e7f4e`, PASS3 after stricter fatal-line classification/manifest binding.
- uj57v58i: `be3dea0c691b07c8f08a4f8d3c075dd46ed4a972b99a9e3f8b07eddf2a5dc1f7`, PASS3 adding immediate-reverify/tamper coverage.
- 970m2p0t: final above, adding execution of corrected product HOST29.

Receipt source-hash changes support the recorded affected reruns; this is not an unchanged campaign or a retry of actual operations. Earlier dummy vxvsd480 and lifecycle 1hn1pii1/lxq71z25/nhrnd5lv/xgbech_5/5xxexl3z receipt hashes were also rechecked against r19/r18. Their unchanged-seam coverage is reused with its original source/evidence limits, not represented as current full-suite execution.

## Global preservation and remaining actual obligations

L1 remains numeric COPY ownership, private source and seed, metadata-only build-time /tmp ownership/mode and runtime UID10001. L2 remains ordinary persistent local volume, honest name/creation/labels identity, no physical/inode quota claim, and preservation after writer stop/export failure. L3 retains finite PID1, independent alarms, exact stopped/non-OOM admission, sequential RO readers and no competing product reaper. L4 retains token/UUID journal, exact full-ID/config binding, lost-ack HOLD/no mutation retry, retention before owned cleanup and exact lane release. r18 F1 dispatch-relative child/postcheck deadlines and F2 ownership/closure of both pipes before fallible setup remain byte-identical in finite-driver and unchanged outside the recipe additions in launcher. Reserved supervisor accounting, pending/torn publication behavior, binary bounded archive/map equality and failed-retention preservation remain intact.

Keep logical1GiB/64MiB leaves/200000 traversal entries, export2GiB, host4GiB including preparation and supervisor reserve,6GiB admissions/2GiB pressure observations. These do not reserve backing space or prove physical quotas/durability. Image/tag/cache/backend overhead, deleted-open bytes, concurrent disk use, rapid growth, allocation/metadata and blocked kernel/daemon IO remain residual risks, not new invented prerequisites.

Recommend parent proceed with the next single separately authorized dummy from these exact executable/documentation pins: at most one metadata/source build<=300s, one fresh volume, fixed non-product UID10001 dummy<=30s, two sequential same-image RO readers<=60s each, whole<=600s. Stop at first failure; retain tag/image/volume/host evidence and exact uncertain inventory, without repair/retry or product dispatch. This report does not authorize preparation or execution itself, fabricate a concrete plan digest, or reopen the separately accepted r20 root-lock finalization.

The actual receipt must establish native local-tag build usability and exact base/derivative IDs, fresh initialization/source traversal/UID/modes, security/resources/mount/volume binding, detached-wait independence, stopped persistence, terminal9269-byte/non-product binding, binary archive/all-regular-map equality, retained artifacts and final operation/cleanup/release/supervisor state. Independently inspect it before the separately authorized product diagnosis. No additional quota framework or optional microreview is required by this GO.

P47 remains expected RED with independent actual47-field/offset/type/owner/enrollment/raw18944/empty-phase/EAGAIN/guard/call-teardown/unreached-assertion obligations. Historical PID229 remains UNKNOWN. H/S, installed/native/platform/ext4/durability/T2/release/activation and product acceptance do not advance. Candidate/history/private proofs unchanged; sole write this report, no commit and no runtime cleanup obligation acquired.
