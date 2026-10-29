# Manual installed Qwen audio gate

`.github/workflows/qwen-real-audio.yml` is dispatch-only on `main`. No runner
with its dedicated `pab-qwen-docker27-cgroupfs-v2` label exists in this
repository's current runner inventory. A queued run is **not evidence**. An
operator must enroll a dedicated ephemeral Linux amd64 runner with Docker 27,
local rootful Unix-socket daemon, cgroup v2 with cgroupfs driver, Python 3.11,
passwordless `sudo`, 12 GiB available RAM and 22 GiB free disk. The private
`/sys/fs/cgroup/speech-box-qwen` parent must be root-owned mode 0700 with cpu,
memory and pids subtree controllers enabled. The public `up` command creates
and limits its own dedicated child cgroup. No repository runner registration,
workflow dispatch, model download or Docker mutation is part of this source
change.

Provide an authorized real mono PCM16 reference WAV and a separate UTF-8 file
containing its exact transcript on that runner. Use the three dispatch inputs
to give their absolute private paths and a private cache root. Do not supply
private Bruce/John recordings or credentials. The cache may contain pinned HF
snapshot files; the extracted archive's `qwen-snapshot` command verifies all
bytes against the committed Base and CustomVoice descriptors and acquires
missing pinned bytes into private staging. One failed attempt remains retained
for inspection; this workflow does not automatically rerun.

The gate builds Docker images from the extracted headless archive, resolves
image IDs, calls the public unqualified `speech-box up`, then uses the client
as the original runner account. It enrolls the supplied reference, generates
clone and Ryan WAVs twice, tests continued service access after acquisition,
submits and cancels a fresh generation, checks the terminal reaped attempt,
and stops only its own handle. Every WAV must decode as nonzero PCM16; SHA-256,
duration, run timing, pinned bundle manifests, source SHA, and image IDs go into
a redacted report with command outcomes. The report never contains the WAV, transcript, credential,
client configuration or raw service/Docker output. Private work, images and
failed handles remain on the ephemeral runner for operator review. The gate
reports `unqualified` throughout; an audio pass cannot create a production
qualification receipt or listening/fidelity proof.

If the runner cannot meet admission, the gate fails before model acquisition
and reports the reason. If setup or speech fails, its report stays failed and
no later step is promoted. The workflow's 180-minute wall limit and command
budgets bound one attempt. Review the report and retained runner state before
any new dispatch. Do not use global Docker prune to clean the runner.
