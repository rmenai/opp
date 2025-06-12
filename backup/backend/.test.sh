#!/usr/bin/env bash
set -euo pipefail

if [ "${TEST:-0}" = "1" ]; then
  (poe test)
fi
