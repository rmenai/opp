#!/usr/bin/env bash

SUPABASE_WORKDIR="supabase/test"

supabase start --workdir "$SUPABASE_WORKDIR"

SUPABASE_KEY=$(supabase status --workdir "$SUPABASE_WORKDIR" | grep -i 'service_role key:' | awk -F ': ' '{print $2}')

supabase stop --workdir "$SUPABASE_WORKDIR"

if [ "$SUPABASE_KEY" = "" ]; then
  echo "Failed to extract anon key from 'supabase status'"
  exit 1
fi

echo "SUPABASE_KEY=$SUPABASE_KEY" >> ../.env.test
