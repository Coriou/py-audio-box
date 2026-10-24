# H4A independent adversarial Sol review — r174

2026-10-24. **Bounded source NO-GO: one P2 malformed-override defect.** The 14 focused tests pass. This verdict concerns the internal selection slice only; no actual hardware qualification or release approval is asserted.

## Pins and authority

Verified before review in `/Users/ben/.txcode/worktrees/py-audio-box/t3-33672c8a`, branch `t3/hardware-profile-selection`, clean HEAD `73673ef15f036e6ef83e2e4c9f3251bc367faa2e`:

- Base `6e982c828df7c9d4a25974fca5d198916995c44c`.
- Source `d1491b18403bb54bd75eda6ba4be89dcf8eef236`, direct child of base; exactly four added files: `worker/hardware.py`, `service/profiles.py`, `tests/service/test_profiles.py`, `tests/worker/test_hardware.py`.
- Report `73673ef15f036e6ef83e2e4c9f3251bc367faa2e`, direct child of source; only `docs/agent-interface/reviews/H4-HARDWARE-PROFILES-r172-sol.md` added.

Read those files, r172, and canonical `docs/agent-interface/IMPLEMENTATION-PLAN.md` hardware section (53–65), H4 (119–125), and relevant integration/evidence boundaries. No applicable AGENTS.md found. Memory registry search for H4/hardware profiles/r172 had no hit; no memory-derived fact used. Sole authored file is this root report. No source edits, commits, delegation or cleanup.

## Finding requiring correction

**P2 — Reject boolean device indices in explicit request overrides before matching** (`service/profiles.py:104–108`, `184–188`; compare `worker/hardware.py:38–40` and `service/profiles.py:85–86`).

`VoiceWorkRequest.__post_init__` validates request budgets but not `device_override`. Selection then compares the tuple directly. Python treats `False == 0` and `True == 1`, so a malformed explicit override can resolve to a different, valid integer-indexed device and return `ready/selected_override`. This bypasses the exact-int invariant enforced by both Device and ExecutionProfile; it is an internal dataclass consistency defect, not a request for public JSON parsing or qualification-record verification.

Reproduced with existing inert fixtures:

```python
select((profile(),), request(device_override=(F.CPU, False)),
       inventory(), SelectionPolicy(workload='fixture'))
# status='ready', reason='selected_override', profile.profile_id='cpu'
Device(F.CPU, False, True, True, 'fake')
# ValueError: invalid device
```

Validate an override as the declared family/index pair, with a DeviceFamily and exact nonnegative int index, before selection; reject malformed values rather than normalizing bool to an index. Add focused false/true index rejection coverage and retain valid index-zero selection and conflicting/unavailable override refusal. No broader public contract change is needed. Existing override tests at `tests/service/test_profiles.py:80–86` cover missing/conflicting valid values, not malformed indices.

## Reviewed behavior and bounded observations

- Identity matching uses exact engine/checkpoint/revision/conditioning equality (`profiles.py:149–150`), then feature subset, language, requested dtype/attention, installation, verified provisioning, trusted qualification and device metadata, topology, memory and concurrency (`151–171`). No model substitution, provisioning, default profile or engine invocation occurs. Identity/settings binding by the caller remains essential as r172 explicitly states.
- Resources and request budgets reject bool/nonpositive/noninteger/nonfinite values; Performance rejects bool/nonfinite/nonpositive timings. Additional inert checks exercised True, False, NaN, positive/negative infinity, zero and negative one in memory-budget/performance positions. Device/profile index constructors explicitly reject bool. Arbitrary values violating other annotated types are not generally schema-validated: these are trusted internal dataclasses, not a public input-validation boundary. Empty optional dtype/attention currently act as unspecified (`155`); callers should use None under the declared seam. No separate release blocker is inferred from public-schema work deferred to the parent.
- Comparable measurements rank by the chosen cold/warm objective and profile ID (`197–202`); unmeasured selection requires explicit policy. Overrides cannot fall back. Explicit family fallback retains the same identity/compatibility checks and exposes the selected profile plus fallback reason. No dtype capability is inferred from hardware family.
- Preference is applied before measurement filtering (`190–198`). With an otherwise eligible CUDA profile having no performance or a different workload, and a measured CPU profile explicitly allowed as fallback, selection returns `not_ready/measurement_required`, with both eligibility diagnostics empty. Reproduced both cases. This is conservative refusal, not silent substitution. The supplied contract does not clearly say whether an available but unmeasured preferred family must trigger fallback; resolve that policy when binding the caller rather than treating it here as a second proven defect. Override/fallback reasons also supersede the unmeasured reason, so callers must inspect performance/workload when presenting measurement status.
- MPS requires native Darwin arm64/aarch64 (`hardware.py:54–57`); Docker/unknown deployment cannot claim Metal. CUDA topology beyond that is trusted qualified metadata, not independently certified here. Memory is a supplied profile budget, and concurrency a qualified ceiling; no scheduler or aggregate admission is implemented or claimed. Thread counts are preserved, not dynamically enforced.
- Diagnostics have at most 256 profile entries and a fixed finite reason set (`profiles.py:175–182`, `146–172`); caller-owned profile IDs are echoed without byte-length limits. This is bounded record count, not a public wire-size guarantee. Duplicate device/profile identifiers are rejected. Provider exceptions return finite reasons without exception text (`hardware.py:74–84`).
- Import/discovery is stdlib-only and consumes supplied persisted records; no ML loading, model scanning, download or implicit provider probe. The fresh subprocess import guard passes. Explicit fake probes fail closed for absent/negative/invalid/raising providers and incompatible MPS topology. No actual provider was invoked.

## Checks and evidence limits

Executed in the pinned candidate using `/private/tmp/pab-0c-venv/bin/python`, with `PYTHONDONTWRITEBYTECODE=1`:

- `-m unittest tests.service.test_profiles -v`: 10 PASS.
- `-m unittest discover -s tests/worker -p test_hardware.py -v`: 4 PASS, including guarded fresh-process import/discovery.
- In-memory fixture checks: bool override alias reproduced; empty dtype/attention observed; preferred missing/mismatched measurement refusal reproduced; numeric invalid-value rejection checks completed.
- `git diff --check`: PASS. Candidate status remained clean after checks.

No pytest, full suite, native guard, Docker, network, ML import, model/hardware probe, benchmark or HELD-resource operation was run. No temporary fixture files were created. Passing focused fixtures are source/fake-policy evidence only.

## Remaining parent gates

Correct the malformed override and review its focused regression. Parent still owns the already-declared caller integration: immutable authorized voice/settings binding, executable/result identity, public HTTP/CLI/MCP serialization and fallback disclosure, durable qualification/probe provenance and freshness/invalidation, and actual resource admission. Their absence is explicitly outside this slice and is not a newly invented source defect.

Canonical H4 still requires real adapters/provisioning/packaging and actual install/generate/cancel/restart, CPU responsiveness, bounded higher-concurrency/resource measurements and listening on the named CPU, CUDA and native Apple Silicon Metal rows (`IMPLEMENTATION-PLAN.md:53–65,123–125`). Those gates remain open; this review supplies no new hardware evidence or authority to operate resources.
