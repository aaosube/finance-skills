#!/usr/bin/env bash
set -euo pipefail

EXPECTED_VERSION="0.4.0"
TASK="${*:-}"

if [ -z "$TASK" ]; then
  echo "usage: $0 \"describe the code change you are about to make\"" >&2
  exit 2
fi

command -v ripwire >/dev/null 2>&1 || {
  echo "ripwire is not installed; see infra/ripwire/README.md" >&2
  exit 2
}

ACTUAL_VERSION="$(ripwire --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]; then
  echo "refusing unreviewed ripwire version: expected $EXPECTED_VERSION, got ${ACTUAL_VERSION:-unknown}" >&2
  exit 3
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "run this helper inside a git repository" >&2
  exit 2
}
cd "$ROOT"

exec ripwire . --for="$TASK"
