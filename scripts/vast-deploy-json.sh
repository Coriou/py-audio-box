#!/usr/bin/env bash
# scripts/vast-deploy-json.sh — convenience wrapper for machine-readable vast-deploy runs
#
# Usage:
#   ./scripts/vast-deploy-json.sh --summary-json /tmp/summary.json --tasks my-jobs.txt
#   ./scripts/vast-deploy-json.sh --summary-json /tmp/summary.json --fail-fast -- voice-synth speak --voice x --text hi

set -euo pipefail

exec "$(cd "$(dirname "$0")" && pwd)/vast-deploy.sh" "$@"
