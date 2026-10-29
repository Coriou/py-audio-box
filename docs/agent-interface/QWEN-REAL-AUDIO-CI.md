# Manual installed Qwen real audio gate

This is a trusted-owner, one-shot gate for an **exact reviewed commit**. The
workflow has no pull-request or push trigger. Its `reviewed_sha` input is a full
40-character commit SHA; checkout and the harness both refuse a different or
dirty source tree. It may be a premerge commit once the workflow is present on
the default branch. The workflow is **not on the default branch yet**, so do not
claim `workflow_dispatch` is currently runnable. GitHub requires a dispatch
workflow on the default branch. A branch name or PR ref is not an exact source
identity.

## Premerge bootstrap: local exact-SHA harness

The trusted owner can run the same harness directly on the dedicated runner
while this workflow is absent from `main`. First review and authorize the exact
commit and media, make that commit available to the runner's local repository,
and create a fresh detached checkout. For example, substitute the full SHA and
private paths below; use a unique empty work directory for this one attempt:

```bash
REVIEWED_SHA='<full-40-character-reviewed-commit-sha>'
REPO='<trusted-local-repository-path>'
CHECKOUT='<new-private-checkout-path>'
WORK='<new-private-empty-work-path>'
REPORT='<private-report-path>'
REFERENCE_WAV='<absolute-authorized-reference-wav>'
REFERENCE_TEXT_FILE='<absolute-exact-transcript-file>'
CACHE_ROOT='<absolute-private-cache-root>'
[[ "$REVIEWED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 1
git -C "$REPO" worktree add --detach "$CHECKOUT" "$REVIEWED_SHA"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REVIEWED_SHA" || exit 1
test -z "$(git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" || exit 1
mkdir -m 700 "$WORK"
sudo -n -E python3 "$CHECKOUT/scripts/ci-qwen-real-audio.py" \
  --reviewed-sha "$REVIEWED_SHA" --work "$WORK" \
  --reference-wav "$REFERENCE_WAV" \
  --reference-text-file "$REFERENCE_TEXT_FILE" \
  --cache-root "$CACHE_ROOT" --report "$REPORT"
```

The owner must check the nonzero exit and JSON report. Keep the checkout,
private work, report and Docker state for review after any failure. There is no
automatic retry or cleanup. This route runs the actual archive and installed
client sequence; it does not create a GitHub Actions status or establish runner
enrollment. After this workflow is merged to the default branch, a trusted owner
may dispatch it manually with the exact reviewed SHA and three private path
inputs. GitHub fetchability of that SHA, runner availability and the source
check still determine whether that attempt starts.

## Runner and inputs

No runner with the dedicated `pab-qwen-docker27-cgroupfs-v2` label is enrolled
in the current inventory. A queued job is **not evidence**. The owner must
provide a dedicated ephemeral Linux amd64 runner with Docker 27, a local rootful
Unix-socket daemon, cgroup v2 with cgroupfs driver, Python 3.11, passwordless
`sudo`, 12 GiB available RAM and 22 GiB free disk. The private
`/sys/fs/cgroup/speech-box-qwen` parent must be root-owned mode 0700 with cpu,
memory and pids subtree controllers enabled. The public `up` command creates
and limits its own dedicated child cgroup. This source change registers no
runner, dispatches no workflow and runs no Docker or model download.

Provide an authorized real mono PCM16 reference WAV and a separate UTF-8 file
containing its exact transcript on the runner. Use their absolute private paths
and a private cache root. Do not supply private Bruce/John recordings or
credentials. The cache is read-only input: an absent snapshot stays absent in
the cache, while `qwen-snapshot` acquires pinned missing bytes into its private
staging root. The extracted archive verifies all bytes against committed Base
and CustomVoice descriptors.

## Evidence and limits

The gate builds Docker images from the extracted headless archive, resolves
image IDs, calls the public unqualified `speech-box up`, then uses the client
as the original runner account. It enrolls the supplied reference, generates
clone and Ryan WAVs twice, tests continued service access after acquisition,
submits and cancels a fresh generation, checks the terminal reaped attempt,
and attempts to stop only its own handle. Every expected WAV must exist and
decode as nonzero PCM16. SHA-256, duration, run timing, pinned bundle manifests,
source SHA, image IDs and bounded command outcomes go into a redacted report.
If both audio and stop fail, the primary audio failure and separate stop failure
remain visible and the gate exits unsuccessfully. The report never contains the
WAV, transcript, credential, client configuration or raw service/Docker output.
Private work, images and failed handles remain on the runner for operator review.
The gate reports `unqualified` throughout; an audio pass cannot create a
production qualification receipt or listening/fidelity proof.

Admission failure stops before model acquisition. A setup, missing-audio or
speech failure cannot be promoted by later steps. The workflow's 180-minute
wall limit and command budgets bound one attempt. Review the report and retained
runner state before any new attempt. Do not use global Docker prune to clean
the runner.
