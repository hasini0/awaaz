"""SQLite ticket store. Small, file-based, swappable for Postgres behind the same methods."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    channel TEXT,
    text TEXT NOT NULL,
    lang TEXT,
    department TEXT,
    dept_conf REAL,
    urgency TEXT,
    urg_conf REAL,
    model_urgency TEXT,
    rule_hits TEXT,
    locality_id TEXT,
    locality TEXT,
    lat REAL,
    lon REAL,
    cluster_id TEXT,
    priority REAL,
    status TEXT DEFAULT 'open',
    updated_at TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_dept_status ON complaints(department, status);
CREATE INDEX IF NOT EXISTS idx_cluster ON complaints(cluster_id);
"""


STATUSES = ("open", "in_progress", "resolved")
ACTIVE = ("open", "in_progress")  # not yet resolved: still shown to officers, still accepts duplicates


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.ROOT / config.settings()["db_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        columns = {r[1] for r in self.conn.execute("PRAGMA table_info(complaints)")}
        if "updated_at" not in columns:  # databases created before status tracking
            self.conn.execute("ALTER TABLE complaints ADD COLUMN updated_at TEXT")

    def insert(self, row: dict):
        row = dict(row)
        if isinstance(row.get("embedding"), np.ndarray):
            row["embedding"] = row["embedding"].astype(np.float32).tobytes()
        if isinstance(row.get("rule_hits"), (list, dict)):
            row["rule_hits"] = json.dumps(row["rule_hits"], ensure_ascii=False)
        cols = ",".join(row)
        self._write(f"INSERT INTO complaints ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))

    def candidates(self, department: str, since: datetime) -> list[sqlite3.Row]:
        """Open complaints of one department that could be duplicates of a new one."""
        return self._read(
            "SELECT id, lat, lon, cluster_id, embedding FROM complaints "
            "WHERE department=? AND status!='resolved' AND created_at>=? AND lat IS NOT NULL",
            (department, since.isoformat()))

    def cluster_size(self, cluster_id: str) -> int:
        return self._read("SELECT COUNT(*) FROM complaints WHERE cluster_id=?", (cluster_id,))[0][0]

    def cluster_status(self, cluster_id: str) -> str:
        """Status of an unresolved incident ("in_progress" if officers have started on any of its reports)."""
        rows = self._read("SELECT status FROM complaints WHERE cluster_id=? AND status!='resolved'", (cluster_id,))
        return "in_progress" if any(r[0] == "in_progress" for r in rows) else "open"

    def cluster_urgencies(self, cluster_id: str) -> list[str]:
        return [r[0] for r in self._read("SELECT urgency FROM complaints WHERE cluster_id=?", (cluster_id,))]

    def set_cluster_priority(self, cluster_id: str, priority: float):
        self._write("UPDATE complaints SET priority=? WHERE cluster_id=?", (priority, cluster_id))

    def get(self, complaint_id: str) -> dict | None:
        rows = self._read("SELECT id, created_at, text, lang, department, urgency, locality, cluster_id, priority, "
                          "status, updated_at FROM complaints WHERE id=?", (complaint_id,))
        return dict(rows[0]) if rows else None

    def set_status(self, ids: list[str], status: str, at: datetime | None = None):
        assert status in STATUSES, status
        at = (at or datetime.now()).isoformat(timespec="seconds")
        with self._lock:
            self.conn.executemany("UPDATE complaints SET status=?, updated_at=? WHERE id=?",
                                  [(status, at, i) for i in ids])
            self.conn.commit()

    def set_cluster_status(self, cluster_id: str, status: str):
        """Officers act on incidents: every report merged into the cluster changes status together."""
        assert status in STATUSES, status
        self._write("UPDATE complaints SET status=?, updated_at=? WHERE cluster_id=?",
                    (status, datetime.now().isoformat(timespec="seconds"), cluster_id))

    def df(self, days: int | None = None) -> pd.DataFrame:
        q = "SELECT id, created_at, channel, text, lang, department, dept_conf, urgency, urg_conf, " \
            "model_urgency, rule_hits, locality_id, locality, lat, lon, cluster_id, priority, status, updated_at " \
            "FROM complaints"
        params = ()
        if days:
            q += " WHERE created_at>=?"
            params = ((datetime.now() - timedelta(days=days)).isoformat(),)
        with self._lock:
            d = pd.read_sql_query(q, self.conn, params=params)
        d["created_at"] = pd.to_datetime(d["created_at"])
        return d

    def count(self) -> int:
        return self._read("SELECT COUNT(*) FROM complaints")[0][0]

    def clear(self):
        self._write("DELETE FROM complaints")

    # One connection is shared by every Streamlit session / API worker thread, so reads take the lock too.
    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _write(self, sql: str, params: tuple = ()):
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()
