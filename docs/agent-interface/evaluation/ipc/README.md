# Networkless cross-container socket smoke

On 2026-10-15, two containers on this Intel Mac's Docker Linux VM successfully exchanged a synthetic ping/pong through a Unix-domain socket on one task-owned Linux named volume. Both had `--network none`, read-only root, 1 CPU/128 MiB limit, dropped capabilities and no-new-privileges; the client mounted the IPC volume read-only. Both exited successfully. The script removed its task-owned server container and named volume; no host socket or private files were mounted.

Image: `sha256:42008ccbab95ca5e7afbe52dd2335f041c0fc87e8fd7620c844b88b44bbbdece`. Actual output: result.txt. Reproduction: probe.sh. Host/VM inventory: ../host.json.

This verifies only the transport primitive under this Mac Docker setup. It does not test production IPC authentication/authorization, rootless or nonroot UID mappings, Windows/WSL2, host bind mounts, failure/reconnect behavior, protocol compatibility, service readiness or inference isolation as a whole. The test intentionally used no Docker networking; native host named pipes and host filesystem sockets are different deployment paths.

Runtime identity clarification: the observed Linux VM is OrbStack (`7.0.14-orbstack-00380-ga7e0a2dc9535`) accessed through the Docker CLI. Actual local probes do not certify Docker Desktop, Windows/WSL2 or another hypervisor.
