#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${PAB_TEST_IMAGE:-py-audio-box-unit-tests}"
docker build -f Dockerfile.tests -t "$image" .
# Unit tests have no network and no mounts of user work, voices, or cache.
docker run --rm --network none "$image" python -m pytest tests -m 'not integration and not redis_integration' -q "$@"
