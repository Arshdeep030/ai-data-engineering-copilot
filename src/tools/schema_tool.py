from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

try:
    from ..observability import logger
except (ImportError, ValueError):
    from observability import logger

ALLOWED_TABLES: frozenset[str] = frozenset({"employees", "orders", "customers"})

# Enforce strict SQL identifier syntax: starts with letter/underscore, followed by alphanumerics/underscores
VALID_IDENTIFIER_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "demo.db"


def init_demo_db(db_path: Optional[str | Path] = None) -> None:
    """Initialize the demo SQLite database with sample tables if not already present."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                employee_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                amount REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                country TEXT NOT NULL
            )
        """)

        cur.execute("SELECT COUNT(*) FROM employees")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO employees (employee_id, name, department, salary) VALUES (?, ?, ?, ?)",
                [
                    (1, "Alice Smith", "Engineering", 125000.0),
                    (2, "Bob Jones", "Data Platform", 115000.0),
                    (3, "Charlie Brown", "Analytics", 105000.0),
                ],
            )

        cur.execute("SELECT COUNT(*) FROM orders")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO orders (order_id, customer_id, order_date, amount) VALUES (?, ?, ?, ?)",
                [
                    (101, 1, "2026-01-15", 250.75),
                    (102, 2, "2026-01-16", 89.50),
                    (103, 1, "2026-02-01", 420.00),
                ],
            )

        cur.execute("SELECT COUNT(*) FROM customers")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO customers (customer_id, name, email, country) VALUES (?, ?, ?, ?)",
                [
                    (1, "Acme Corp", "contact@acme.com", "Canada"),
                    (2, "Global Health", "ops@globalhealth.org", "Canada"),
                    (3, "DataFlow Systems", "admin@dataflow.io", "USA"),
                ],
            )
        conn.commit()


def get_table_schema(
    table_name: str,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Returns the schema of an approved database table.

    Validates table name format, enforces allowlist permissions, and queries
    SQLite table metadata safely. Never executes unvalidated arbitrary SQL.
    """
    start_time = time.perf_counter()

    # 1. Input presence validation
    if not isinstance(table_name, str) or not table_name.strip():
        logger.warning("schema_tool_validation_failed reason=empty_table_name")
        return {
            "success": False,
            "error": "Table name must be a non-empty string.",
        }

    clean_name = table_name.strip()

    # 2. Strict identifier validation (defense against SQL injection)
    if not VALID_IDENTIFIER_PATTERN.match(clean_name):
        logger.warning(
            "schema_tool_security_blocked reason=invalid_identifier table=%r",
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
            "schema_tool_access_denied table=%r allowed=%r",
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

    # 5. Safe metadata query
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
                logger.warning("schema_tool_table_not_found table=%s", normalized_name)
                return {
                    "success": False,
                    "error": f"Table '{clean_name}' was not found in database.",
                }

            actual_table_name = match[0]

            # Fetch columns using PRAGMA
            # Identifier is strictly validated by VALID_IDENTIFIER_PATTERN and verified against ALLOWED_TABLES
            cur.execute(f"PRAGMA table_info({actual_table_name});")
            pragma_rows = cur.fetchall()

            columns: list[dict[str, str]] = []
            for row in pragma_rows:
                # row structure: (cid, name, type, notnull, dflt_value, pk)
                col_name = row[1]
                col_type = row[2].upper() if row[2] else "TEXT"
                columns.append({
                    "name": col_name,
                    "type": col_type,
                })

        latency = time.perf_counter() - start_time
        logger.info(
            "schema_tool_completed table=%s columns_count=%d latency=%.4fs",
            actual_table_name,
            len(columns),
            latency,
        )

        return {
            "success": True,
            "table_name": actual_table_name,
            "columns": columns,
        }

    except Exception as exc:
        latency = time.perf_counter() - start_time
        logger.error(
            "schema_tool_error table=%s error=%s latency=%.4fs",
            clean_name,
            exc,
            latency,
        )
        return {
            "success": False,
            "error": f"Database error inspecting table '{clean_name}': {str(exc)}",
        }
