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


def get_table_row_count(
    table_name: str,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Returns the total number of rows in an approved database table.

    Validates table name format, enforces allowlist permissions, and queries
    SQLite row counts safely. Never executes unvalidated arbitrary SQL.
    """
    start_time = time.perf_counter()

    # 1. Input presence validation
    if not isinstance(table_name, str) or not table_name.strip():
        logger.warning("row_count_tool_validation_failed reason=empty_table_name")
        return {
            "success": False,
            "error": "Table name must be a non-empty string.",
        }

    clean_name = table_name.strip()

    # 2. Strict identifier validation (defense against SQL injection)
    if not VALID_IDENTIFIER_PATTERN.match(clean_name):
        logger.warning(
            "row_count_tool_security_blocked reason=invalid_identifier table=%r",
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
            "row_count_tool_access_denied table=%r allowed=%r",
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

    # 4. Resolve database path
    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_db.exists():
        init_demo_db(target_db)

    # 5. Safe count query
    try:
        with sqlite3.connect(f"file:{target_db.resolve()}?mode=ro", uri=True) as conn:
            cur = conn.cursor()

            # Verify table actually exists in sqlite_master
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name) = ?",
                (normalized_name,),
            )
            match = cur.fetchone()
            if not match:
                logger.warning("row_count_tool_table_not_found table=%s", normalized_name)
                return {
                    "success": False,
                    "error": f"Table '{clean_name}' was not found in database.",
                }

            actual_table_name = match[0]

            # Count rows
            cur.execute(f"SELECT COUNT(*) FROM {actual_table_name};")
            row_count = cur.fetchone()[0]

        latency = time.perf_counter() - start_time
        logger.info(
            "row_count_tool_completed table=%s row_count=%d latency=%.4fs",
            actual_table_name,
            row_count,
            latency,
        )

        return {
            "success": True,
            "table_name": actual_table_name,
            "row_count": row_count,
        }

    except Exception as exc:
        latency = time.perf_counter() - start_time
        logger.error(
            "row_count_tool_error table=%s error=%s latency=%.4fs",
            clean_name,
            exc,
            latency,
        )
        return {
            "success": False,
            "error": f"Database error querying row count for '{clean_name}': {str(exc)}",
        }
