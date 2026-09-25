from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

try:
    from ..observability import logger
    from .schema_tool import ALLOWED_TABLES, DEFAULT_DB_PATH, VALID_IDENTIFIER_PATTERN, init_demo_db
except (ImportError, ValueError):
    from observability import logger
    from tools.schema_tool import ALLOWED_TABLES, DEFAULT_DB_PATH, VALID_IDENTIFIER_PATTERN, init_demo_db

MAX_SAMPLE_LIMIT = 20
DEFAULT_SAMPLE_LIMIT = 5


def get_table_sample(
    table_name: str,
    limit: int = DEFAULT_SAMPLE_LIMIT,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Returns a small sample of records from an approved database table.

    Validates table name, enforces allowlist permissions, safely clamps limit,
    and parameterizes query limit to protect against arbitrary execution.
    """
    start_time = time.perf_counter()

    # 1. Input presence validation
    if not isinstance(table_name, str) or not table_name.strip():
        logger.warning("sample_tool_validation_failed reason=empty_table_name")
        return {
            "success": False,
            "error": "Table name must be a non-empty string.",
        }

    clean_name = table_name.strip()

    # 2. Strict identifier validation (defense against SQL injection)
    if not VALID_IDENTIFIER_PATTERN.match(clean_name):
        logger.warning(
            "sample_tool_security_blocked reason=invalid_identifier table=%r",
            clean_name,
        )
        return {
            "success": False,
            "error": "Invalid table name: must be a valid alphanumeric identifier without special characters.",
        }

    normalized_name = clean_name.lower()

    # 3. Allowlist enforcement
    if normalized_name not in ALLOWED_TABLES:
        logger.warning(
            "sample_tool_access_denied table=%r allowed=%r",
            normalized_name,
            sorted(ALLOWED_TABLES),
        )
        return {
            "success": False,
            "error": (
                f"Table '{clean_name}' is not approved for access. "
                f"Approved tables: {', '.join(sorted(ALLOWED_TABLES))}."
            ),
        }

    # 4. Limit validation and clamping
    try:
        sanitized_limit = int(limit)
    except (ValueError, TypeError):
        sanitized_limit = DEFAULT_SAMPLE_LIMIT

    if sanitized_limit < 1:
        sanitized_limit = 1
    elif sanitized_limit > MAX_SAMPLE_LIMIT:
        sanitized_limit = MAX_SAMPLE_LIMIT

    # 5. Resolve database path
    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_db.exists():
        init_demo_db(target_db)

    # 6. Safe parameterized query
    try:
        with sqlite3.connect(f"file:{target_db.resolve()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Verify table actually exists in sqlite_master
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name) = ?",
                (normalized_name,),
            )
            match = cur.fetchone()
            if not match:
                logger.warning("sample_tool_table_not_found table=%s", normalized_name)
                return {
                    "success": False,
                    "error": f"Table '{clean_name}' was not found in database.",
                }

            actual_table_name = match[0]

            # Fetch sample rows using parameterized limit
            cur.execute(f"SELECT * FROM {actual_table_name} LIMIT ?;", (sanitized_limit,))
            rows = [dict(row) for row in cur.fetchall()]

        latency = time.perf_counter() - start_time
        logger.info(
            "sample_tool_completed table=%s limit=%d fetched=%d latency=%.4fs",
            actual_table_name,
            sanitized_limit,
            len(rows),
            latency,
        )

        return {
            "success": True,
            "table_name": actual_table_name,
            "limit": sanitized_limit,
            "row_count": len(rows),
            "rows": rows,
        }

    except Exception as exc:
        latency = time.perf_counter() - start_time
        logger.error(
            "sample_tool_error table=%s error=%s latency=%.4fs",
            clean_name,
            exc,
            latency,
        )
        return {
            "success": False,
            "error": f"Database error fetching sample for '{clean_name}': {str(exc)}",
        }
