#!/usr/bin/env bash

if [ "${TEST:-0}" = "1" ]; then
  SUPABASE_WORKDIR="supabase/test"
  ENVIRONMENT_MODE="Test"
else
  SUPABASE_WORKDIR="supabase/dev"
  ENVIRONMENT_MODE="Development"
fi

cleanup() {
  echo "Signal received, stopping Supabase ($ENVIRONMENT_MODE environment from $SUPABASE_WORKDIR)..."
  supabase stop --workdir "$SUPABASE_WORKDIR"
  echo "Supabase ($ENVIRONMENT_MODE environment from $SUPABASE_WORKDIR) stopped."
  exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM (sent by process managers like process-compose)
# Also trap EXIT to ensure cleanup runs even if the script exits for other reasons after starting.
trap cleanup SIGINT SIGTERM EXIT

# Start Supabase for the selected environment
echo "Starting Supabase ($ENVIRONMENT_MODE environment) using workdir: $SUPABASE_WORKDIR..."
supabase start --workdir "$SUPABASE_WORKDIR"

echo "Supabase ($ENVIRONMENT_MODE environment) services initiated from $SUPABASE_WORKDIR."
echo "Keeping script alive to manage instance."
echo "Press Ctrl+C in this terminal (if run directly) or send SIGTERM via process-compose to stop."

# Keep the script running in the foreground.
# This loop allows signals to be caught by the trap.
while true; do
  sleep 86400
  wait $!
done
