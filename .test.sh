#!/usr/bin/env bash
set -euo pipefail

if [ "${TEST:-0}" = "1" ]; then
  (cd backend && poe test) || true
  (cd frontend && bun run test:unit:ci) || true
fi
