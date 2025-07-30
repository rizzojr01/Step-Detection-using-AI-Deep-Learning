import sqlite3
from typing import Any, Dict

DB_PATH = "config/config.db"

# Ensure the config table exists
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def get_config_value(key: str, default: Any = None) -> Any:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return default


def set_config_value(key: str, value: Any):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()


def get_all_config() -> Dict[str, Any]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM config")
    rows = cur.fetchall()
    conn.close()
    return {k: v for k, v in rows}
