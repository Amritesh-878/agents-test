from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def resolve_db_url(flag_value: str | None) -> str:
    load_dotenv()
    if flag_value and flag_value.strip():
        logger.warning(
            "Passing --db-url on the command line exposes the database password in "
            "process listings and shell history. Prefer setting DATABASE_URL in .env."
        )
        return flag_value.strip()
    return os.getenv("DATABASE_URL", "").strip()
