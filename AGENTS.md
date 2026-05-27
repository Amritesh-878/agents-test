## Project Context

**Adira Academy Learning Assistant** — a personalized RAG chatbot for students. A complete pipeline processes daily Zoom cloud recording exports (session MP4, mixed M4A, per-student M4As named by student name) to build per-student context: transcribe the full session for class context, transcribe per-student M4As for exact spoken content, parse names from filenames, merge with Zoom attendance data → per-student context objects → ChromaDB RAG + Groq llama-3.3-70b chatbot. Students use it after class for deeper review or to catch up if absent.

Stack: Python 3.11, whisperX, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Groq SDK, Pydantic, pytest, ruff, mypy.

**Alpha phase (as of May 2026):** Pre-release alpha — breaking changes are allowed by default. Revisit before first public release.

---

- **Python:** Use classes for domain logic and services; use plain functions for utilities and helpers. Keep methods focused on single responsibility.

```python
# Good: Service class
class UserService:
    def create(self, email: str) -> User:
        pass

    def update(self, user_id: int, data: dict) -> User:
        pass

# Good: Utility function
def validate_email(email: str) -> bool:
    return '@' in email
```

- **JavaScript/React:** Prefer named function exports for utilities; use arrow functions for callbacks and hooks. Declare complex exported functions with `function` keyword; use concise arrow functions for inline callbacks.

```js
// Good: Exported function
export function computeGameState(board) {
  /* implementation */
}

// Good: Arrow for callbacks/hooks
const onClick = () => setState((prev) => prev + 1);

// Bad: Block-bodied export arrow
export const complex = () => {
  /* large logic */
};
```

---

- **Python:** Avoid using modules purely as namespaces. Prefer plain functions or class methods.

```python
# Bad
class Utils:
    @staticmethod
    def foo(): pass

# Good
def foo():
    pass
```

- **JavaScript:** Avoid class-based design for simple utilities. Use named exports instead.

```js
// Bad
export class Utils {
  static foo() {}
}

// Good
export function foo() {}
```

---

- **Python:** All functions need type hints on parameters and return types. Use `from __future__ import annotations` for forward refs. Use `Optional[T]` for nullable fields.

```python
from __future__ import annotations
from typing import Optional

# Bad
def process(data):
    return None

# Good
def process(data: dict) -> Optional[dict]:
    return {}
```

- **JavaScript:** Prefer explicit imports from modules; avoid inline `import()` unless for code-splitting. Do not use `any` in production code; prefer explicit types or let type inference work.

---

- **Python Tests:** Use proper type hints. Use pytest fixtures instead of bare objects for test setup.

```python
# Bad
from typing import Any
def test_process() -> Any:
    result = None

# Good
import pytest
def test_process(mock_engine: Engine) -> None:
    result = mock_engine.run()
    assert result is not None
```

- **JavaScript Tests:** Use proper types or JSDoc. Avoid `as any`; use fixtures.

```js
// Bad
const mock = {} as any;

// Good
/** @type {Test} */
const mock = { id: 1 }; // test fixture
```

---

- **Python:** Avoid bare `except:` clauses. Always catch specific exceptions. Use type narrowing with `isinstance()`.

```python
# Bad
try:
    process(value)
except:
    pass

# Good
from typing import Union
value: Union[str, int] = get_data()
if isinstance(value, str):
    result = value.upper()
else:
    result = str(value).upper()
```

- **JavaScript:** Prefer `?.` and `??` over casts for null/undefined. Use type guards for narrowing.

```js
// Bad
const a = obj.x ?? 'd';

// Good
const a = obj?.x ?? 'd';
```

---

- Run the `verify-quality` skill after all changes. You must not run tests before lint/typecheck pass without errors, even for test files themselves.

---

- Check the closest `package.json` (frontend) or `pyproject.toml` (backend) for available scripts; do not assume scripts that don't exist.

---

- When seeing 'No such file or directory', confirm with `pwd` and `ls` and navigate correctly.

```sh
pwd
ls -la
```

---

- If auto-fixes occur while you edit, re-run the `verify-quality` skill to confirm the final state before PR.

---

- Do not edit `AGENTS.md` unless the user says "you have permission to edit AGENTS.md" in the current session.

---

- Follow DRY and SOLID principles; avoid code duplication and ensure single responsibility.

---

- When moving/renaming files, prefer `git mv` for moving files to preserve history. `mv` is acceptable if `git mv` is not available.

```sh
git mv src/old.ts src/new.ts
```

---

- Prefer changing file extension to `.txt` to preserve files marked for deletion (so files can be recovered before a final git cleanup).

---

- If files are large, consider splitting into smaller modules to improve readability and reduce lint noise.

---

- **Python:** Keep imports organized: standard library → third-party → local imports. Use absolute imports from package root.

```python
import sys
from typing import Optional

import numpy as np
from pydantic import BaseModel

from game.engine import GameEngine
```

- **JavaScript/React:** Use path aliases from `jsconfig.json` (e.g., `@components`, `@hooks`). Avoid overly deep relative imports.

```js
import GameBoard from '@components/GameBoard';
import { useGameState } from '@hooks/useGameState';
```

---

- **Backend:** Use version pinning in `pyproject.toml`. When adding dependencies, verify type stubs are available (for mypy).

```toml
[tool.poetry.dependencies]
numpy = "^1.24.0"
pydantic = "^2.0"
```

---

- **Frontend:** Add dependencies with `npm add`. Check for TypeScript types or JSDoc support.

```sh
cd frontend && npm add react react-dom
npm install --save-dev @types/react
```

---

- When changing a module used by others, verify integration by running tests and type-checks on dependents.

---

- Use `rg` or `grep` to find usages across the repo quickly.

---

- Prefer function declarations for complex exported functions; allow concise arrow expressions and inline callbacks; avoid block-bodied exported arrows.

```js
// Bad
export const compute = () => {
  /* large */
};

// Good
export function compute() {
  /* large */
}
```

---

- Respect `Note for Agent:` comments in files: read and honor instructions before continuing; remove notes when done. The user may add these notes mid-prompt as type errors or lint errors so they appear when you run checks.

**Python Example:**

```python
from typing import Literal
# Type error: NoteForAgent = Literal['some instruction to follow']
assert_result: NoteForAgent  # "Note for Agent: update validation logic here"
```

**JavaScript Example:**

```js
// Lint error (uncomment to see):
const x = // Note for Agent: refactor this section
  undefined;
```

**General Pattern:**
When you encounter a Note for Agent (via type error, lint error, or code comment), immediately:

1. Read the full instruction
2. Acknowledge understanding
3. Make necessary changes
4. Remove the note when done

---

- Only type cast when necessary; prefer type guards, utility types, or refactor to avoid casts.

**Python Example:**

```python
from typing import Union
# Bad: Using type: ignore
value: Union[str, int] = get_data()
result = value.upper()  # type: ignore

# Good: Use isinstance for type narrowing
if isinstance(value, str):
    result = value.upper()
else:
    result = str(value).upper()
```

**JavaScript Example:**

```js
// Bad
const v = x as unknown as T;

// Good: Use type guards
if (isT(x)) {
    const v = x;
}
```

---

- After linting/typechecking, the only acceptable result is 0 errors and 0 warnings. After running tests, 100% passing is the only acceptable result. This is not up for debate.

---

- You must not comment out tests or code to fix lint/typecheck errors. Always fix the underlying issue.

---

- You must not skip writing tests just because they are complex. Write whatever mocks or helpers are needed to write the test in a way that it mimics real usage as closely as possible.

---

- If absolutely needed, prefer to disable specific rules inline with comments rather than disabling them project-wide.

**Python Example:**

```python
# ruff: noqa: E501 (disable specific rule for this line)
result = some_function_with_a_very_long_name()  # noqa: E501

# mypy: ignore error for this line
value = x  # type: ignore[assignment]
```

---
