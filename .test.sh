#!/usr/bin/env bash
set -euo pipefail

if [ "${TEST:-0}" = "1" ]; then
  (cd backend && poe test)
  (cd frontend && bun run test:unit:ci)
fi
