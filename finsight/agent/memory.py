"""Persistent per-session conversation state (SQLite), so follow-ups like
"and the year before?" work across turns and across API restarts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from finsight.config import DATA_DIR


class SessionStore:
    def __init__(self, path=None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path or DATA_DIR / "sessions.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS turns (session TEXT, ts REAL, data TEXT)")
        self._lock = threading.Lock()

    def history(self, session: str, n: int = 3) -> list[dict]:
        rows = self._db.execute("SELECT data FROM turns WHERE session=? ORDER BY ts DESC LIMIT ?", (session, n))
        return [json.loads(r[0]) for r in rows][::-1]

    def append(self, session: str, turn: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO turns VALUES (?, ?, ?)", (session, time.time(), json.dumps(turn)))
            self._db.commit()
