from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def resolve_db_url(flag_value: str | None) -> str:
    """Resolve the Postgres DSN, preferring the DATABASE_URL env var over a CLI flag.

    Passing a DSN on the command line leaks the password into process listings,
    shell history, and logs, so a secret-bearing ``--db-url`` flag is honored but
    warned against; ``DATABASE_URL`` (loaded from ``.env``) is the primary source.
    """
    load_dotenv()
    if flag_value and flag_value.strip():
        logger.warning(
            "Passing --db-url on the command line exposes the database password in "
            "process listings and shell history. Prefer setting DATABASE_URL in .env."
        )
        return flag_value.strip()
    return os.getenv("DATABASE_URL", "").strip()
