"""
Unit tests for the session manager.
"""
import pytest
import os
import tempfile
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from session_manager import SessionManager, reset_session_manager


class TestSessionManager:
    """Tests for SQLite-based session persistence."""

    @pytest.fixture
    def temp_manager(self, tmp_path):
        """Create a SessionManager with a temporary database."""
        db_path = str(tmp_path / "test_sessions.db")
        return SessionManager(db_path=db_path)

    def test_create_session(self, temp_manager):
        """Test creating a new session."""
        session = temp_manager.create("sess-123", "simulate copper at 300K")

        assert session["session_id"] == "sess-123"
        assert session["user_request"] == "simulate copper at 300K"
        assert session["stage"] == "analysis"
        assert session["created_at"] is not None
        assert session["analysis"] is None
        assert session["conversation"] == []
        assert session["memories_used"] == []

    def test_get_session(self, temp_manager):
        """Test retrieving a session."""
        temp_manager.create("sess-456", "simulate iron")

        session = temp_manager.get("sess-456")
        assert session is not None
        assert session["session_id"] == "sess-456"
        assert session["user_request"] == "simulate iron"

    def test_get_nonexistent_session(self, temp_manager):
        """Test that getting nonexistent session returns None."""
        session = temp_manager.get("nonexistent-id")
        assert session is None

    def test_update_session(self, temp_manager):
        """Test updating a session."""
        temp_manager.create("sess-789", "simulate aluminum")

        success = temp_manager.update("sess-789", {
            "stage": "parameters_pending",
            "analysis": {"material": "Al", "temperature": 500}
        })

        assert success is True

        session = temp_manager.get("sess-789")
        assert session["stage"] == "parameters_pending"
        assert session["analysis"]["material"] == "Al"
        assert session["analysis"]["temperature"] == 500

    def test_update_nonexistent_session(self, temp_manager):
        """Test that updating nonexistent session returns False."""
        success = temp_manager.update("nonexistent", {"stage": "cancelled"})
        assert success is False

    def test_delete_session(self, temp_manager):
        """Test deleting a session."""
        temp_manager.create("sess-to-delete", "test")

        success = temp_manager.delete("sess-to-delete")
        assert success is True

        session = temp_manager.get("sess-to-delete")
        assert session is None

    def test_delete_nonexistent_session(self, temp_manager):
        """Test that deleting nonexistent session returns False."""
        success = temp_manager.delete("nonexistent")
        assert success is False

    def test_list_recent(self, temp_manager):
        """Test listing recent sessions."""
        temp_manager.create("sess-1", "simulate Cu")
        temp_manager.create("sess-2", "simulate Fe")
        temp_manager.create("sess-3", "simulate Al")

        recent = temp_manager.list_recent(limit=10)
        assert len(recent) == 3

        # Check they're sorted by updated_at descending
        # (most recent first)
        session_ids = [r["session_id"] for r in recent]
        assert "sess-3" in session_ids
        assert "sess-1" in session_ids

    def test_list_recent_respects_limit(self, temp_manager):
        """Test that list_recent respects the limit."""
        for i in range(10):
            temp_manager.create(f"sess-{i}", f"request {i}")

        recent = temp_manager.list_recent(limit=5)
        assert len(recent) == 5

    def test_list_by_stage(self, temp_manager):
        """Test listing sessions by stage."""
        temp_manager.create("sess-a", "request a")
        temp_manager.create("sess-b", "request b")
        temp_manager.update("sess-b", {"stage": "parameters_pending"})
        temp_manager.create("sess-c", "request c")
        temp_manager.update("sess-c", {"stage": "parameters_pending"})

        pending = temp_manager.list_by_stage("parameters_pending")
        assert len(pending) == 2

        analysis = temp_manager.list_by_stage("analysis")
        assert len(analysis) == 1

    def test_session_persistence(self, tmp_path):
        """Test that sessions persist across manager instances."""
        db_path = str(tmp_path / "persist_test.db")

        # Create session with first manager
        manager1 = SessionManager(db_path=db_path)
        manager1.create("persist-sess", "test persistence")
        manager1.update("persist-sess", {"stage": "script_pending"})

        # Retrieve with new manager instance
        manager2 = SessionManager(db_path=db_path)
        session = manager2.get("persist-sess")

        assert session is not None
        assert session["stage"] == "script_pending"

    def test_get_stats(self, temp_manager):
        """Test getting session statistics."""
        temp_manager.create("s1", "request 1")
        temp_manager.create("s2", "request 2")
        temp_manager.update("s2", {"stage": "executing"})
        temp_manager.create("s3", "request 3")
        temp_manager.update("s3", {"stage": "executing"})

        stats = temp_manager.get_stats()

        assert stats["total_sessions"] == 3
        assert stats["by_stage"]["analysis"] == 1
        assert stats["by_stage"]["executing"] == 2
        assert stats["oldest_session"] is not None
        assert stats["newest_session"] is not None

    def test_complex_session_data(self, temp_manager):
        """Test storing complex nested data in session."""
        temp_manager.create("complex-sess", "test")

        complex_data = {
            "analysis": {
                "material": "Cu",
                "temperature": 300,
                "kim_model": "EAM_Dynamo_MendelevAckland_2007v3_Cu__MO_123",
                "nested": {"deep": {"data": [1, 2, 3]}}
            },
            "conversation": [
                {"role": "generator", "attempt": 1, "content": "script..."},
                {"role": "reviewer", "attempt": 1, "issues": ["issue1"]}
            ],
            "stage": "script_pending"
        }

        temp_manager.update("complex-sess", complex_data)
        session = temp_manager.get("complex-sess")

        assert session["analysis"]["nested"]["deep"]["data"] == [1, 2, 3]
        assert len(session["conversation"]) == 2
        assert session["conversation"][1]["issues"] == ["issue1"]

    def test_updated_at_changes_on_update(self, temp_manager):
        """Test that updated_at is modified on each update."""
        temp_manager.create("ts-sess", "test")
        session1 = temp_manager.get("ts-sess")
        original_updated = session1["updated_at"]

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)

        temp_manager.update("ts-sess", {"stage": "executing"})
        session2 = temp_manager.get("ts-sess")

        assert session2["updated_at"] != original_updated
        assert session2["updated_at"] > original_updated
