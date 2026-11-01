#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${R1_PYTHON:-python3}" -m tests.release.test_admission
