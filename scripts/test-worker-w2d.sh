#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${W2D_PYTHON:-python3}" -m tests.release.test_dispatcher
