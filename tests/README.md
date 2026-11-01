# Test collection

Ordinary service and maintenance tests run automatically. Historical B observer
execution probes and frozen-source authority diagnostics in
`service/test_child_wait_guard.py` (`B-S-*`, `B-C-*`, `B-H-*`, `B-A-*`, and
the extracted HOST contract) carry `manual_b_probe` and default to explicit
MANUAL/unrun skips. These historical diagnostics remain UNQUALIFIED against
the current source. Independent synthetic B parser and terminal predicate tests
remain automatic.

To request these probes explicitly, run:

```sh
python -m pytest tests/service/test_child_wait_guard.py --run-manual-b-probes -m manual_b_probe
```

Opting in does not requalify the frozen diagnostic authority. Its exact target
source hash, assertions, and `/tmp/pab-p-one-node-observer` authority must still
match; a changed target remains a refusal or failure, never a waiver or release
qualification. `--noconftest` bypasses the opt-in collection hook and is reserved
for the separately authorized historical control lane described in the tests.
