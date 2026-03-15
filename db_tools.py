"""
SQLite tools for the research agent.

Three plain Python functions — no Claude API here.
The agent wraps these in JSON schema definitions and calls them
when Claude decides to query the database.
"""

import json
import sqlite3

DB_PATH = "research.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def run_query(sql: str) -> str:
    """
    Execute a SQL SELECT statement and return results as a JSON string.

    Returns a JSON array of objects on success, or an error message string
    (also prefixed with 'Error:') so the agent can detect failure and retry.
    """
    # Restrict to read-only queries so the agent can't mutate data
    normalised = sql.strip().upper()
    if not normalised.startswith("SELECT") and not normalised.startswith("WITH"):
        return "Error: only SELECT (and WITH ... SELECT) queries are allowed"

    try:
        conn = _connect()
        cur = conn.execute(sql)
        rows = cur.fetchall()
        conn.close()
    except sqlite3.Error as e:
        return f"Error: {e}"

    if not rows:
        return "Query returned 0 rows."

    # Convert to a list of plain dicts so it serialises cleanly
    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


def list_tables() -> str:
    """
    List all tables in the database with their row counts.
    Use this to understand what data is available before writing a query.
    """
    try:
        conn = _connect()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        if not tables:
            return "Database has no tables."

        lines = []
        for (name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            lines.append(f"  {name}  ({count} rows)")

        conn.close()
        return "Tables in database:\n" + "\n".join(lines)
    except sqlite3.Error as e:
        return f"Error: {e}"


def describe_table(table_name: str) -> str:
    """
    Show the column names, types, and a few sample rows for a table.
    Use this to understand a table's structure before writing a query.
    """
    try:
        conn = _connect()

        # Check the table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if not exists:
            conn.close()
            return f"Error: table '{table_name}' does not exist"

        # Column info
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_lines = [
            f"  {c['name']} {c['type']}{' NOT NULL' if c['notnull'] else ''}"
            for c in cols
        ]

        # Sample rows (first 3)
        rows = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
        sample = json.dumps([dict(r) for r in rows], indent=2)

        conn.close()
        return (
            f"Table: {table_name}\n"
            f"Columns:\n" + "\n".join(col_lines) + "\n\n"
            f"Sample rows (first 3):\n{sample}"
        )
    except sqlite3.Error as e:
        return f"Error: {e}"
