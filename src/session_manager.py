# src/session_manager.py
"""
SQLite-based session persistence for ChatMD workflow state.

Provides persistent storage for workflow sessions that survive server restarts.

Usage:
    from session_manager import get_session_manager

    manager = get_session_manager()
    session = manager.create("session-123", "simulate copper at 300K")
    session = manager.get("session-123")
    manager.update("session-123", {"stage": "parameters_pending"})
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

import config


class SessionManager:
    """SQLite-based session persistence for workflow state."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or getattr(config, 'SESSION_DB_PATH', 'data/sessions.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    user_request TEXT,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_stage
                ON sessions(stage)
            """)
            conn.commit()

    def create(self, session_id: str, user_request: str) -> Dict[str, Any]:
        """Create a new session."""
        now = datetime.now().isoformat()
        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "stage": "analysis",
            "user_request": user_request,
            "analysis": None,
            "verification": None,
            "parameters": None,
            "script": None,
            "review": None,
            "conversation": [],
            "memories_used": []
        }

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, created_at, updated_at, stage, user_request, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, now, now, "analysis", user_request, json.dumps(session))
            )
            conn.commit()

        return session

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if row:
                return json.loads(row["data"])
            return None

    def update(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update a session with new data."""
        session = self.get(session_id)
        if not session:
            return False

        session.update(updates)
        session["updated_at"] = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """UPDATE sessions
                   SET updated_at = ?, stage = ?, data = ?
                   WHERE session_id = ?""",
                (session["updated_at"], session.get("stage", "unknown"),
                 json.dumps(session), session_id)
            )
            conn.commit()

        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions (summary only, not full data)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, stage, user_request
                   FROM sessions ORDER BY updated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()

            return [dict(row) for row in rows]

    def list_by_stage(self, stage: str, limit: int = 20) -> List[Dict[str, Any]]:
        """List sessions filtered by stage."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, stage, user_request
                   FROM sessions WHERE stage = ? ORDER BY updated_at DESC LIMIT ?""",
                (stage, limit)
            ).fetchall()

            return [dict(row) for row in rows]

    def cleanup_old_sessions(self, days: int = None) -> int:
        """Delete sessions older than specified days. Returns count deleted."""
        expiry_days = days or getattr(config, 'SESSION_EXPIRY_DAYS', 30)
        cutoff = (datetime.now() - timedelta(days=expiry_days)).isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

            stages = conn.execute(
                "SELECT stage, COUNT(*) as count FROM sessions GROUP BY stage"
            ).fetchall()

            oldest = conn.execute(
                "SELECT MIN(created_at) FROM sessions"
            ).fetchone()[0]

            newest = conn.execute(
                "SELECT MAX(updated_at) FROM sessions"
            ).fetchone()[0]

            return {
                "total_sessions": total,
                "by_stage": {row["stage"]: row["count"] for row in stages},
                "oldest_session": oldest,
                "newest_session": newest
            }


# Global instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager():
    """Reset the cached session manager (useful for testing)."""
    global _session_manager
    _session_manager = None
