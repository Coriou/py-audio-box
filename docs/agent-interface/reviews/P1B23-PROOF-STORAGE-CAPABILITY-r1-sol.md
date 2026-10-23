# P1B23 proof storage — targeted capability audit r1 Sol

2026-10-23. **STORAGE-CAPABILITY STOP before implementation or diagnosis.** No authoritative existing backend satisfying the complete Astra prerequisite was identified in the bounded current-daemon inventory. Do not implement a portable quota abstraction, substitute an ordinary local volume, or weaken Astra. This is a capability-evidence result, not proof that no such backend could exist elsewhere.

## Authority and source boundary

User-authorized Sol-low READONLY research, with this report as the sole root write; no commit. Root checkout `/Users/ben/Projects/py-audio-box`, HEAD `0a747e735dfdb57c9e172fee728349e8e80d1040`. Existing untracked STT evidence was present and left untouched. Checked ancestor instructions through root and scoped `docs` AGENTS inventory; none found. No delegation.

Read FULL root `P1B23-BLOCKED-PIPE-LAUNCHER-LIFECYCLE-ASTRA.md` and `P1B23-BLOCKED-PIPE-LAUNCHER-SOURCE-r1-sol.md`, including L1–L4. Their SHA256 values are respectively `bd0884a333e5e0598a1f934f234a40627f8e21b6e60aa3de1d313704702a4598` and `dca89cff992be204c95bdcf4c9a47c00f15a8ab5c4bc5cd7ce797b3ded678884`. Candidate pins below are source-report provenance, not a fresh candidate checkout audit: documentation `f93b7fef6cc461b756cd9e13ff14e88e243359a4`, executable freeze `e941c7aee7f53348905659904b8299d5500d1250`.

Astra selects one fresh Docker-managed persistent proof volume, aggregate hard ceiling 1 GiB (1,073,741,824 bytes), including the fixed 1 MiB lifecycle reserve and a documented finite backend-metadata reservation; at most 200000 allocated inodes; private UID/GID10001 usability; persistence after writer stop. Admission also requires at least 900 MiB available after lifecycle reservation. Provisioning/inspection must work without privileged setup containers, host binds, new services, plugin installation or daemon reconfiguration. Existing volumes are inventory evidence only and are not candidates for adoption.

L1 remains the copied-source/UID10001 ownership defect. L2 remains proof loss on finite PID1/tmpfs expiry. L3 remains remote-work/deadline/quiescence and protected handback defects. L4 remains acquisition/uncertain-create/ownership accounting defects. This audit repairs or closes none of them. Persistent storage alone would not close the other lifecycle gates.

## Current runtime-time evidence

Read-only Docker calls used the current CLI selection, without context switching or endpoint output. Each subprocess had a 20-second timeout; all completed successfully, with no retry. Inventory timestamps are UTC on 2026-10-23:

- 17:37:33.140875: `docker context show` returned `orbstack`.
- 17:37:33.479583: `docker version --format '{{json .}}'`, filtered before output: client 29.8.2, API1.56, darwin/amd64; server Docker Engine Community 29.8.2, API1.56, linux/amd64, Engine Git commit `8af9fe3`; kernel `7.2.9-orbstack-00391-g2b298d05fbd4`. Components: containerd2.3.6, runc1.5.2, docker-init0.19.0. This is daemon-reported identity, not image/runtime qualification.
- 17:37:33.713018: `docker info --format '{{json .}}'`, allowlisted fields only: operating system OrbStack, architecture x86_64, storage driver overlay2, backing filesystem btrfs, advertised volume drivers `[local]`. The image-layer storage driver/backing filesystem does not establish per-volume quota or inode enforcement.
- 17:37:33.839747: `docker plugin ls --format '{{json .}}'` returned no managed plugins. Combined with the info driver list, no alternate volume driver was advertised by this daemon.
- 17:37:33.884835: `docker volume ls --format '{{json .}}'`, reduced to count and drivers: 141 existing volumes, all `local`.
- 17:37:48.958421: a second `docker volume ls -q` supplied names internally to five sequential `docker volume inspect` batches of at most32. All141 inspected records had Driver=`local`, Scope=`local`, and empty/null Options; aggregate option-key count was empty. No volume names, labels, mountpoints, device paths, status payloads or contents were printed or retained. No backend allocation UUID or enforced quota receipt was established by this filtered inventory. Empty options are not proof of absent hidden filesystem controls; they provide no positive contract for them.

No Docker context endpoint, credentials, environment, proxy configuration or raw info/inspect payload was output or saved. No existing volume contents or host-visible OrbStack storage paths were traversed. No daemon configuration, credentials file or unrelated application configuration was needed or read. Context endpoints and cross-context inventories were deliberately outside this bounded current-daemon audit.

## Authoritative support contract inspected

Official documentation was read directly on 2026-10-23; no broad web search:

- [Docker volume lifecycle](https://docs.docker.com/engine/storage/volumes/), lifecycle and empty-volume population sections: volume data is independent of container lifetime; empty volumes can receive image-path contents. This supports persistence as a general mechanism. It does not certify a particular allocation, UID/mode initialization, byte/inode ceilings or metadata accounting.
- [Docker volume create, driver options](https://docs.docker.com/reference/cli/docker/volume/create/): options are driver-specific; Linux local options follow mount semantics. Its sized/UID tmpfs example is not a persistent hard-quota backend. A generic `size` option or a CLI flag is not evidence of enforcement on ordinary local directories. No creation example was executed.
- [OrbStack volumes and mounts](https://docs.orbstack.dev/docker/file-sharing): documents persistent Linux-side volumes and cloning/export. The inspected page supplies no per-volume hard byte/inode quota or finite backend-overhead reservation contract.
- [OrbStack Docker](https://docs.orbstack.dev/docker/) and [settings](https://docs.orbstack.dev/settings): supporting context/engine documentation; neither inspected page establishes the missing storage contract. The initially guessed `/docker/volumes` URL was inaccessible; the canonical volume page was then followed from the official Docker page. No backend absence claim rests on that failed URL.

Documentation describes product contracts and is not version-pinned enforcement evidence for this allocation. No documented applicable provisioning/inspection contract satisfying all requirements was found in these bounded sources. Btrfs backing storage, Docker labels, free-space observations, sparse-file lengths, application counters, export member limits and default local volumes cannot supply the missing hard allocation guarantees.

## Capability disposition and exact missing prerequisite

Aggregate byte enforcement including reserve/metadata: **NOT ESTABLISHED**. Inode allocation at most200000: **NOT ESTABLISHED**. Finite backend metadata overhead charged within1GiB: **NOT ESTABLISHED**. Private usable UID/GID10001 root and supported initialization without prohibited setup: **NOT ESTABLISHED**. Persistence after container stop: **documented generic volume semantics only**, with no stopped-writer allocation receipt or experiment in this audit. Backend allocation identity and read-only inspectable enforced limits: **NOT ESTABLISHED**. Therefore the conjunction cannot be established without additional authoritative backend evidence; no prohibited probe may be used to invent that evidence.

The smallest concrete environment dependency is an **already available Docker-managed backend on the selected execution daemon with a supported, version-specific allocation and inspection contract** that returns an allocation identity and authoritative enforcement/accounting receipts for all of the following: aggregate allocation <=1GiB including lifecycle and bounded backend overhead; inode ceiling <=200000; private10001:10001 initialization; persistent retained storage after stop; no prohibited setup dependency. No qualifying driver/backend name can responsibly be supplied from this inventory. This is not an installation recommendation or authorization to reconfigure the environment. A backend that needs one of the prohibited setup actions does not satisfy the current task.

Until that dependency is supplied and reviewed, stop before launcher implementation, image construction and diagnosis. Do not add a fake quota adapter or replace enforcement with admission free-space checks. If evidence later establishes the backend contract without a fill/probe, Astra still requires fresh independent source review before any construction execution, separately authorized finite construction qualification/allocation receipts, and separate parent runtime authority. This audit does not grant those authorities.

## Preservation and handback

No volume/container/image creation, start, exec, build, tag, pull, install, stop, removal, quota-fill or runtime probe occurred. No product imports, pytest/collection, database open, cleanup, push, commit or implementation change occurred. No resources, ownership lane or cleanup obligation were acquired. Existing holds/resources and historical evidence remain untouched. Sole write: this report. No diagnosis, runtime approval, feature acceptance, platform or durability claim follows.
