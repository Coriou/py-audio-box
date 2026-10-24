# H2 authenticated voice transport — r164 implementation stop

2026-10-24. **STOP: missing explicit owner-thread dispatcher interface. No implementation or release GO.** The commission explicitly requires stopping with a precise interface request if Dispatcher needs a new allowlist entry. That condition holds even for the read-only production factory. No source was changed and no fake integration/test pass is claimed.

## Pins and inspection

Fresh supplied checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-1e48c9df`, branch `t3/1e48c9df`, clean at `e72f08278bc72871a4239a3f1d7d0874734edd7d`; root `/Users/ben/Projects/py-audio-box` had the same HEAD at entry and recheck. No reset or checkout replacement. H0 adoption `aa3dfb5` is an ancestor. Read PRODUCT, IMPLEMENTATION-START, canonical IMPLEMENTATION-PLAN (including H0/H2/parallel ownership), r158/r159/r162, actual API/domain/catalogue/clients, Dispatcher owner loop and command dispatch, Core ownership, voice store reads and relevant contracts. No applicable AGENTS.md was found in the supplied worktree. Memory registry keyword lookup returned no relevant hit; no prior memory fact was used.

## Exact blocker and minimal parent request

`Dispatcher.call` in `service/dispatcher.py` accepts only submit/job/jobs/events/artifact/asset/cancel/status/replace/inventory_page/reconcile_startup/close. Neither a voice operation nor a trusted callback is accepted. Arguments are deep-copied into the bounded queue; `_command` has no voice branch and ends in `getattr(self.core, op)`. Passing a callback is therefore not an existing supported internal seam. Core is constructed and closed inside `_run` on control-writer; its SQLite connection cannot be used by FastAPI worker threads. Injecting `dispatcher.core`, creating a second Core/owner, running a new owner loop, or adding a generic public method bus would bypass the commissioned boundary. Dispatcher/Core/store edits are expressly outside this lane's allowlist.

Parent/dispatcher owner: add these explicit trusted internal command names to `call` and explicit `_command` branches, executed solely on the existing owner loop:

- `voice_list(principal, *, after=0, limit=20)` → `self.core.voices.list(principal, after=after, limit=limit)`.
- `voice_get(principal, voice_id)` → `self.core.voices.get(principal, voice_id)`.
- `voice_version(principal, version_id)` → `self.core.voices.version(principal, version_id)`.
- `voice_resolve(principal, name)` → `self.core.voices.resolve(principal, name)`.

Return the existing detached JSON results through the existing Future/deepcopy path; propagate domain failures for bounded H2 translation. No callback, caller-controlled attribute name, new dispatcher, worker launch, persistent enrollment or activation-fence change is requested. These four commands are sufficient to resume an authenticated read slice; version-history listing and mutations need not block that slice. Principal comes only from H2 credential resolution, never request JSON/query. The factory should borrow an already initialized Dispatcher with explicit caller-owned close lifecycle; it must not implicitly activate persistent service operation. Parent must confirm that borrowing this owner for reads is admitted under the retained persistent lifecycle fences.

H1 compact read methods are not merged here. Existing list caps voice count at 20 but calls get, which materializes every version ID. Existing versions also calls full-history get. Interim H2 must reject oversized serialized pages/details with structured resource_limit rather than silently truncate IDs/authored content; this bounds transport output, not store allocation. H1 should provide compact metadata and cheap existence/version paging for truly bounded history work. Do not advertise a 2,000-ID response as bounded model output.

## Shared contract handoff and deferred work

Existing voices_list/voice_get/voice_version_get declarations provide the read route IDs and CLI/MCP names. Alias resolution and version-history paging lack public operation declarations; parent owns their routes/request/response/cursor contracts. `voice_aliases_request` exists but has no operation declaration. `voice_archive_request` exists, while its operation returns receipt and H0 archive returns voice metadata: a deliberate receipt mapping is required before claiming parity. No arbitrary registration endpoint or redefinition of prepared-candidate voice_save is appropriate.

After the owner seam lands, implement separate production factory and unchanged fixture default, explicit revocable credential configuration with constant-time comparison, authenticated principal/read scope, cheap discovery, bounded structured errors and HTTP/CLI/MCP parity with lazy details. Missing/wrong credentials can use the current forbidden contract; adding a distinct unauthorized code/status is a parent contract decision. No hardcoded secret or production fixture fallback. Alias/archive require additional explicit owner commands and reviewed contract mapping; they remain deferred here.

## Verification, preservation and next owner

Read both pytest conftests: root imports the maintenance guard and service fixtures start HTTP processes. No pytest collection, unittest, dependency install, listener, Core instantiation, model/ML import, Docker, external network, guard/diagnostic/recovery invocation, worker/inference, delegation or activation ran. Tests were not authored against a nonexistent integration. Only source/document inspection and git checks ran; no runtime safety/auth/client acceptance is claimed.

Only this report is written and committed; there is no source commit because the explicit stop precedes source implementation. Report-only git diff/check and root-copy byte equality are the appropriate checks. Only this report is copied to root, without staging or committing parent work. HELD resources/proofs, historical worktrees, STT, lane and recovery guards were not accessed or changed; this is an action-boundary statement, not a live resource inspection.

Next owner: parent supplies the four explicit owner commands (or an equally explicit reviewed typed seam) and shared-contract decisions, then recommissions H2 implementation. Independent Sol review may verify this interface stop; no source implementation is ready for adoption. Production/release remains NO-GO.
