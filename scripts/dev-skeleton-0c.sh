#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Supply a dedicated environment installed from service/requirements.lock.
exec "${SPEECH_0C_PYTHON:-python3}" -m uvicorn service.api:app --host 127.0.0.1 --port "${SPEECH_0C_PORT:-8780}" --no-access-log
