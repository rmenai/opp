# TODO

- ~~Migrate from poetry to uv~~
- Fix testing

  - Add Supabase testing client
  - Add Hypothesis
  - Add Schemathesis
  - <https://chatgpt.com/share/6826e618-9944-8010-82e2-410210404b64>
  - Add all tests for my current app
  - Test-driven development

- Redo supabase database, but this time with migrations
- Prepare Supabase dev client
- Change .env loading, separate to .env.dev, .env.test
- Fix linting and add more pre-commit hooks

```sh
supabase link --project-ref myapp-test
supabase start            # → ~30 s to spin up containers
supabase db push          # → migrations + seeds
supabase db test          # → SQL tests
pytest                    # → API/WebSocket tests
supabase stop             # → tear down containers
```
