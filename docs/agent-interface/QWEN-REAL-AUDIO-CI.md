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
PYTHON_311='<absolute-python-3.11-executable-on-runner>'
[[ "$REVIEWED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 1
git -C "$REPO" worktree add --detach "$CHECKOUT" "$REVIEWED_SHA"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REVIEWED_SHA" || exit 1
test -z "$(git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" || exit 1
[[ "$PYTHON_311" = /* && -f "$PYTHON_311" && -x "$PYTHON_311" ]] || exit 1
python_real="$("$PYTHON_311" -c 'import os, sys; assert sys.version_info[:2] == (3, 11); print(os.path.realpath(sys.executable))')" || exit 1
test "$(sudo -n "$PYTHON_311" -c 'import os, sys; assert sys.version_info[:2] == (3, 11); print(os.path.realpath(sys.executable))')" = "$python_real" || exit 1
mkdir -m 700 "$WORK"
sudo -n -E "$PYTHON_311" "$CHECKOUT/scripts/ci-qwen-real-audio.py" \
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
may dispatch it manually with the exact reviewed SHA, three private path
inputs and the absolute Python 3.11 executable path. GitHub fetchability of
that SHA, runner availability and the source check still determine whether that
attempt starts.

## Runner and inputs

No runner with the dedicated `pab-qwen-docker27-cgroupfs-v2` label is enrolled
in the current inventory. A queued job is **not evidence**. The owner must
provide a dedicated ephemeral Linux amd64 runner with Docker 27, a local rootful
Unix-socket daemon, cgroup v2 with cgroupfs driver, Python 3.11, passwordless
`sudo`, 12 GiB available RAM and 22 GiB free disk. Select an absolute Python
3.11 executable path before invoking `sudo`; both procedures verify its version
and that root resolves the same executable before passing it to the harness.
Do not rely on `python3` from either account's `PATH`. The private
`/sys/fs/cgroup/speech-box-qwen` parent must be root-owned mode 0700 with cpu,
memory and pids subtree controllers enabled. The public `up` command creates
and limits its own dedicated child cgroup. This source change registers no
runner, dispatches no workflow and runs no Docker or model download.

Provide an authorized real mono PCM16 reference WAV and a separate UTF-8 file
containing its exact transcript on the runner. These are required for real-voice
acceptance. A synthetic Ryan-generated reference can serve only as a limited
technical smoke input; it cannot establish real-voice or fidelity acceptance.
Use their absolute private paths and a private cache root. Do not supply
private Bruce/John recordings or
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
Each speech repetition submits distinct text and must return a new job ID for
the expected voice version; the report records the job ID, voice version and
text hash for each submission. A successful WAV with a replayed job is a failure.
If both audio and stop fail, the primary audio failure and separate stop failure
remain visible and the gate exits unsuccessfully. The report never contains the
WAV, transcript, credential, client configuration or raw service/Docker output.
The command receipt names a private log ID, phase, elapsed time, exit or timeout,
stream byte counts and omitted byte counts. Private logs live under
`WORK/command-logs` (directory mode 0700, files mode 0600). Each stream retains
its exact first and last 64 KiB, with an omission count for any middle bytes;
the harness drains the full stream without keeping it in memory or writing an
unbounded file. The public receipt may include a code-shaped structured cause
from a helper, never its free-form message. Logs may contain secrets, transcript
or private paths: keep them on the trusted runner, review them privately, and do
not add them to the uploaded report artifact. Only the redacted JSON report is
uploaded by the workflow. Private work, images and failed handles remain on the
runner for operator review.
The gate reports `unqualified` throughout; an audio pass cannot create a
production qualification receipt or listening/fidelity proof.

Admission failure stops before model acquisition. A setup, missing-audio or
speech failure cannot be promoted by later steps. The workflow's 180-minute
wall limit and command budgets bound one attempt. Review the report and retained
runner state before any new attempt. Do not use global Docker prune to clean
the runner.

The first official attempt at `28821e1473a9944e5952f46ad3b9e4b7cc8971c9`
failed admission in 0.04 seconds with `requires_12_gib_available_ram`; its
`steps` and `command_log` were empty. The box exposed about 2.5 GiB available
RAM through virtio ballooning. The policy checks available RAM, not declared
total RAM. This is a retained preflight rejection, with no audio or service
regression result. Code cannot make that host satisfy the 12 GiB precondition.
