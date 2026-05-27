---
name: verify-quality
description: Run the full lint/typecheck/test pipeline for this project. Use after any code changes. Order is mandatory: lint (auto-fix) → typecheck → tests. 0 errors and 100% passing is the only acceptable result.
---

Run the full quality pipeline in the correct order. Never run tests before lint and typecheck pass clean.

## Backend (Python)

```sh
cd backend
ruff check --fix .   # auto-fix lint, then confirm 0 errors
mypy .               # typecheck — must be 0 errors
pytest               # tests — must be 100% passing
```

Key rules:
- Always run `ruff check --fix .` first; never skip to mypy directly
- After ruff auto-fixes, check that 0 errors remain before proceeding
- Check `pyproject.toml` for available scripts before assuming commands exist

## Frontend (JavaScript/React)

```sh
cd frontend
npm run lint:fix     # runs ESLint + Prettier — NEVER use `npm run lint`
npm test             # must be 100% passing
```

Key rules:
- Always use `lint:fix`, never `lint`
- Check `package.json` for available scripts before assuming commands exist

## Acceptable results

- Lint/typecheck: 0 errors, 0 warnings
- Tests: 100% passing

If either check fails, fix the underlying issue. Do not comment out code, skip tests, or disable rules project-wide to make checks pass.
