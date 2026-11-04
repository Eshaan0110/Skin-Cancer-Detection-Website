import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.environ.get("HISTORY_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL NOT NULL,
    risk TEXT NOT NULL,
    source TEXT NOT NULL
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_connection()) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def log_prediction(filename, result, source):
    """Persist one classification result. `result` is run_inference()'s return dict."""
    with closing(get_connection()) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, filename, label, description, confidence, risk, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                filename,
                result["label"],
                result["description"],
                result["confidence"],
                result["risk"],
                source,
            ),
        )
        conn.commit()


def _build_where(risk=None, label=None, source=None, query=None):
    clauses, params = [], []
    if risk:
        clauses.append("risk = ?")
        params.append(risk)
    if label:
        clauses.append("label = ?")
        params.append(label)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if query:
        escaped = query.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        clauses.append("filename LIKE ? ESCAPE '!'")
        params.append(f"%{escaped}%")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def get_recent(limit=50, offset=0, risk=None, label=None, source=None, query=None):
    """Return logged predictions newest-first, optionally filtered and paginated.

    A negative `limit` returns every matching row.
    """
    where, params = _build_where(risk, label, source, query)
    with closing(get_connection()) as conn:
        rows = conn.execute(
            f"SELECT * FROM predictions{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]


def count_entries(risk=None, label=None, source=None, query=None):
    """Number of logged predictions matching the filters."""
    where, params = _build_where(risk, label, source, query)
    with closing(get_connection()) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM predictions{where}", params).fetchone()[0]


def get_risk_counts():
    """Return counts of logged predictions grouped by risk level."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT risk, COUNT(*) as count FROM predictions GROUP BY risk"
        ).fetchall()
        counts = {"high": 0, "medium": 0, "low": 0}
        for r in rows:
            counts[r["risk"]] = r["count"]
        return counts


def clear_history():
    with closing(get_connection()) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
