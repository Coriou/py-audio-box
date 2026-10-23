# Launcher source freeze recipe — NOT execution authority

Run only source/text/Git operations for this preparation. Fresh independent review of this exact launcher is REQUIRED before any ONE execution. No Docker or launcher command was executed here.

The source-manifest pins 193 copied regular files from observer documentation base 8d5aba5fc702719843d055b7d3f693c407deb0e9, including exact observer/entry/guard. Launcher copies only that allowlist, never the checkout wholesale. Future execution uses one `docker exec -w /app <owned-name> python /app/diagnostics/p-one-node/entry.py --one-node`, PYTHONPATH=/app/diagnostics/p-one-node and PYTHONDONTWRITEBYTECODE=1. No separate collection. The Dockerfile is generated as FROM the newly owned inspected local base tag, WORKDIR /app, COPY source/ /app/ only; build uses --pull=false --network=none. No install or pull.

Proof root /tmp/pab-p-one-node-launcher-20261023 must be absent; umask077, root0700, files0600. Source manifest is verified before candidate imports and after test; runtime identity is checked pre/post, including all42 lock versions, Linux image pins, SQLite source/compile options and linked-library digest. Identity stdout must match pre/post. Existing inner deadlines and accelerated test clock remain byte-identical; outer monotonic budget1800s, test subprocess1200s with reserved retention time. All commands have finite timeout bounded by remaining outer budget.

Lane mkdir is atomic at /tmp/pab-worker-w1-docker-lane.lock, owner PID/token/base pin recorded; owner equality required for release, never foreign removal. Host disk reserve6GiB and in-container available memory/disk4GiB admission; tmpfs free900MiB before test. CPU2/RAM2GiB/PID256/networknone/read-only/UID10001/cap-dropALL/no-new-privileges/tmpfs1GiB; inspect rejects binds/devices/ports/mounts. No sockets, secrets, audio, services or host mounts.

Before the test, binary tar sentinel includes NUL/FF/80 and is compared byte-for-byte. After the sole test, direct binary docker exec tar --hard-dereference exports ALL live /tmp before stop; independent LIVE regular-file hashes must equal archive hashes. Safe unique relative member names; symlinks metadata only; no extraction, DB open or docker cp. Export failure leaves LIVE-PROOF-HOLD with owned container/lane; no retry, no destructive cleanup. Successful retention permits exact owned stop/inspect/rm without -v, exact-name absence and owner-bound lane release receipts. Any later cleanup failure preserves remaining resources and evidence for parent handling; never foreign cleanup.

Records are bounded47-integer rows,512bytes/row,8rows/process,16KiB total; required child1 and owner2/3/4 set binds enrolled PID/raw wait. Structural/type/phase mismatch STOP. Even structurally valid records deliberately STOP for independent actual caller/runner offset and separate guard/report binding review; process-local numeric IDs cannot independently name unknown classes. Expected RED remains RED. No automatic cause/acceptance, test retry or tuning.

## Static freeze hashes

- launcher.py: `ea419c21e81966861f79cb6af81c724bad5ebf78618de4c31fd56f554bdf0ee6`
- runtime-identity.py: `81acbdbb70df0a9c2a5db1f9daf86373cb22ea3b92371d7dddd924420a60237c`
- source-manifest.json: `1d8280803813f6e550bbdde1028eb7c315a397c0157f6a4f0fcec2f67841984e`
