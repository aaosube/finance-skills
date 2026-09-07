#!/usr/bin/env bash
set -euo pipefail

: "${CAMOFOX_API_KEY:?CAMOFOX_API_KEY must be set before starting the browser service}"
export CAMOFOX_CRASH_REPORT_ENABLED="${CAMOFOX_CRASH_REPORT_ENABLED:-false}"

exec npx -y @askjo/camofox-browser@1.14.0
