# CPU headless release candidate — r407

2026-10-25. Actual CPU functionality below is **GO for the tested ordinary
Docker-client Linux/amd64 lane**. Fresh Pocket public05 passed the complete
nine-job journey and its final clean stop. Repository regressions and the
bounded source-distribution checks below are complete. This is a locally prepared
CPU release candidate, ready for GitHub publication and independent replication. No GitHub push or image publication occurred.
No browser UI is part of this candidate. CUDA and Apple Metal remain UNQUALIFIED;
human listening and perceptual clone fidelity remain UNMEASURED.

## Actual installed evidence

Kokoro's public recipe completed at source `8b2e024`: authenticated HTTP preset
registration, idempotent speech, PCM24 stereo 48 kHz export, bounded hashed
download, clean stop, same-container restart, actual CLI download/status and MCP
stdio initialization/tool/read calls. See
[the retained Kokoro report](CPU-KOKORO-INSTALLED-r407-parent.md).

Pocket's public recipe completed at source `7c56f72`, using pinned French
`pocket-tts-fr-3.3.0`, Pocket 3.3.0, Torch 2.10.0+cpu and tokenizers 0.23.3.
Finished images:

- Control: `sha256:d141a3073b933308f0a4334676f8db57fca3b9fdc6fbc154ddaa6cbce2772917`.
- Worker: `sha256:793f998d30407c1a7b2172fb2ef80563c35cfd979784387303e1a1de4b25ef42`.

The contained probe completed in 21.007526 seconds with
1072259072 bytes peak RSS, under a 3 GiB
address-space ceiling, two threads and the original 180-second execution limit.
The parent corroborated exact image, actual exit zero and no OOM. The executable
child had UID/GID 20000, no supplementary groups or capabilities, and no-new-privs.
The private conditioning namespace was empty after execution and the automatic
cache environment was restored. This probe alone was not treated as installed
qualification.

The fresh public Docker-volume installation then passed:

1. Upload → reference preparation → separate reference/audition download → save
   via revision-zero CAS → generation from the returned immutable voice version
   → SHA-256-verified WAV download. Every mutation replay used its original key.
2. Clean stop and checked restart of the same exact control and worker IDs.
3. Two different eight-second, mono PCM16 48 kHz references, each independently
   uploaded, prepared, saved, generated and downloaded. Their saved version IDs
   and reference IDs differ. Reference hashes:
   `ca1021790719c2099a10cdb7ce9c2e94a8968fa137d95cb4f6997baea5d48e5b` and
   `5db8bad544ec5b188918016c1deaee05ca66e1ccc87889dfce32dba164def526`.
4. Cancellation after observing an actual RUNNING generation; identical-key replay;
   terminal cancelled state with actual reap on every attempt, then idle ACK.
5. A second clean stop/restart after the two-reference and cancellation checks,
   preserving the same container IDs, prior jobs and both saved voices.
6. Fresh post-restart generations with each saved version, unchanged prior job
   reads, identical-key replay, changed-payload/key conflict HTTP 409, and complete
   hashed downloads. No direct Core/database seeding or context swapping occurred.

7. Final clean stop after all nine jobs, with both exact containers exited zero
   and no OOM. Under control exclusion, a private WAL-aware database copy measured
   872 aggregate rows against the 2048-row inspection limit (1176 rows remaining).
   All 679 dispatch commands were returned. This measures this exact journey;
   arbitrary long jobs and unlimited retained history are not qualified.

Private installation `pocket-public-install05` retains handles, credentials,
images, requests, responses, process wait receipts and audio. Credentials and
reference recordings are excluded from the repository and archive. Installation
resource ceilings are 4 GiB independently per control/worker, not an aggregate
memory guarantee. The 1 GiB data quota reserves 8 MiB per retained Pocket job;
Core admission, finite history and retention limits remain active.

## Corrections and retained refusals

Earlier failed trials remain retained and do not gain a successful receipt:

- The small draft installation exhausted retained disk reservations. Public recipes
  now use a 1 GiB data quota; resource-limit errors provide explicit operator
  guidance. Preparation's conservative full-input reservation is 8 MiB.
- A clean Pocket shutdown exposed detached inventory rejection of prepared
  reference/audition files. The correction permits only exact typed preparation
  owners and their exact staging namespaces in the Pocket v2 scope.
- Another trial recorded an uncertain START RPC while verifying the model twice;
  generation later succeeded, but clean-stop history correctly refused. The
  public Pocket v2 START response now has a ten-second budget anchored before
  authentication. Authentication/transmission retain their short bound; the
  original 180-second execution cutoff and historical uncertainty fence remain
  unchanged. The fresh install04 recipe passed without that uncertainty.
- The public04 nine-job final shutdown refused at the aggregate 2048-row
  inspection limit. Commit `7c56f72` changes only successful Pocket ALIVE
  observation polling from 30 to 200 ms; cancellation and resource-stop checks
  retain their immediate paths. No history is deleted and no budget is raised.
  Fresh public05 passed all nine jobs and its final stop with the margin above. This is a
  bounded retained-history service, not an unlimited job/history guarantee.
- Final source also restores explicit pre-P1 historical-local inspection with
  retained producer/media corroboration and typed production/export refusal.
  P1 v1/v2 public-artifact verification remains unchanged. This separate historical
  branch is not exercised by the qualified installed P1 lane.

During the final Pocket checks, no live original SQLite files were opened for
diagnostics. Inspection of failed
trial03 occurred only after both exact containers were verified exited zero,
using control exclusion and a bounded private WAL-aware copy.

## Final checks

The complete service, worker, client, contract and release collection contained
3997 test nodes: 3739 owned definitions and 258 imported duplicates. All **3684
nonmanual owned cases passed**, along with **11 targeted manual controls**.
The remaining 44 manual diagnostics were not run and receive no qualification
credit. These counts are unique test nodes, not a sum of repeated batches or
subtests. There are zero missing nonmanual cases in the final collection ledger.

Tests ran in bounded Linux containers (2 CPUs, 1 GiB memory, read-only source,
no network); large groups used separate processes with at most 100 selected
cases. A memory-limited earlier test process and fixture/environment refusals
were retained, resolved, and followed by explicit successful cases. The
writable determinism check used a private copy with unchanged original hashes;
the offline compactness check used a separately verified tokenizer cache.
No host ML or host pytest was run.

Final test-only corrections preserve real receipt admission, private SQLite
capacity failures, connection ownership and pipe/reap assertions. The B witness
uses the current exception trace-event traceback without relaxing identity or
terminal checks. The final idle source-authority negative control passed with
source hashes unchanged, exact refusal debt, hook restoration and both subsequent
launches refused. Failed earlier attempts remain recorded; they do not gain
terminal evidence. Production source is unchanged since the qualified Pocket
`7c56f72` build. Independent Sol reviews and Astra batch review distinguished
these source/test corrections from actual installed qualification.

The headless source distribution has 154 allowlisted files plus its executable
launcher (155 ZIP entries). Two independent builds are byte-identical; ZIP
integrity, exact allowlist, executable mode and packaged relative documentation
links pass. Isolated extraction passes launcher help and CLI/MCP/Pocket imports
without repository fallback or Torch loading. The final archive and external
`SHA256SUMS` are retained outside the repository; no credentials, model weights,
recordings, evaluation data or browser UI are packaged.

The release is prepared locally only; cloud-runner adoption remains a separate
replication of the documented recipes, never inferred from this machine's pass.
