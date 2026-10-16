# Astra final planning review — R2

15 October 2026. Bounded incorporation review against baseline `ae67cd6a572e7105b7d41365aac0be4059d79584` and the current planning packet. Reviewed `PLAN.md`, `ACCEPTANCE.md`, `review-astra-r1.md`, `PRODUCT.md` and the planning index. No fresh broad research was required; no material contradiction called for reopening the three research reports or implementation source. Only this review file was written. No models, private audio, secrets, builds or runtime tests were used.

## Verdict

**GO — READY FOR CONTRACT/PROTOTYPE STAGE.** No unresolved blocker remains at this boundary. The five R1 planning dispositions are incorporated, with corresponding acceptance scenarios and explicit next-stage deliverables.

This is **not** a frozen full implementation specification, authorization for unrestricted implementation, or runtime, security, visual, acoustic or release acceptance. Proposed efficiency budgets and hardware priorities remain unmeasured gates.

## Resolved findings

| R1 disposition | R2 verification |
| --- | --- |
| B1 — one persistence authority | PLAN selects a sole SQLite control owner on local storage, WAL/FULL, no worker DB access and no Redis dispatch/bridge for new jobs. Legacy execution remains separate. Distinct-root dry-run/report/apply migration, generation-fenced control-owner publication, orphan reconciliation and consistent backup/restore are explicit. ACCEPTANCE covers submission through outbox faults, migration preservation and restore. |
| B2 — executable isolation and installation topology | PLAN specifies control, networkless inference worker with private Unix-socket IPC, and a typed networked fetch/provision helper; no Docker socket or source mount is required. Confirmed reap, quarantined unreapable lanes, shared resource admission, model reuse/fairness and staged hardware qualification remain explicit. ACCEPTANCE requires actual Compose isolation, denied canonical access, provisioning failure recovery and resource checks. |
| B3 — full v1 speech studio | PLAN includes imported speech/music/SFX beds, multitrack arrangement, overlap, mute/solo, split/trim/move, gain/fades, accessible timing edits, autosave/history and export. The 48,000-tick contract, immutable derivatives, fixed placement, explicit revision-checked ripple and shorter-take adoption are specified. Strict final rendering, explicit stale/partial policies and preview/render tolerances have corresponding acceptance fixtures. Stage 3 is mandatory for v1. |
| B4 — access and realistic completion | PLAN specifies operator bootstrap, protected browser sessions, attributable scoped agent tokens and baseline stdio MCP over canonical HTTP. Direct remote MCP and host wake behavior require separate qualification. Durable handles/events and callback status remain independent of audio success. ACCEPTANCE adds bootstrap expiry, scopes/revocation, CSRF/Host/Origin, authorized reads, callback policy and the stdio path. |
| B5 — measured efficiency | PLAN and ACCEPTANCE agree on provisional catalogue plus onboarding ≤2,500 tokens, status ≤1 KiB, receipt ≤2 KiB and default lists ≤20 entries. Nine tool families are a prototype starting point, not a permanent count cap. Actual client/schema costs, duplicated content, complete workflows, recovery, model loads and inference starts must be measured. |

## Remaining work and consistency

The next stage explicitly owes versioned schemas, the capability/default matrix, a measured API/MCP catalogue fixture, complete legacy feature mapping, install/storage/network design, a fake-worker durability slice and a synthetic timeline/browser-preview slice. Their absence today is expected work for that stage, not a blocker to entering it. Preserve R1's detailed dispositions when writing those contracts; the shorter PLAN sections do not replace their rationale or acceptance constraints.

PRODUCT records confirmed scope while leaving technical choices as planning decisions. That is consistent with PLAN labelling its defaults as engineering recommendations. The index also explicitly makes PLAN the place where competing independent research recommendations are resolved. No material scope or budget contradiction remains across the reviewed packet.

Product naming, visual preferences, access to certification hardware/agent hosts and final human workflow/listening approval remain later human dependencies. They do not require another product-scope question before contract/prototype work. PLAN's pending-review status and the index may now reference this R2 verdict.
