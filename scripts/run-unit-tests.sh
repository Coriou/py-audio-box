#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${PAB_TEST_IMAGE:-py-audio-box-unit-tests}"
docker build -f Dockerfile.tests -t "$image" .
# Ordinary tests remain discoverable, including mocked evaluation/benchmark tests.
# These three release modules are host Docker qualification operators, not units.
# Historical manual B probes retain their separate opt-in authority.
selection=(tests -m 'not integration and not redis_integration and not manual_b_probe'
  --ignore=tests/local --ignore=tests/remote
  --ignore=tests/release/test_dispatcher.py
  --ignore=tests/release/test_packaged_lifecycle.py
  --ignore=tests/release/test_admission.py
  -p no:cacheprovider -q)
# No network or mounts of user work, voices, credentials or model caches.
# 1 GiB scratch capacity also accommodates synthetic disk-reserve fixtures.
docker run --rm --network none --cpus 2 --memory 1g --memory-swap 1g \
  --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  "$image" python -m pytest "${selection[@]}" "$@"
