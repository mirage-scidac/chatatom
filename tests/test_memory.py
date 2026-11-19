# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
Unit tests for the memory abstraction layer.
"""
import pytest
import os
import tempfile
import json
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory import SimpleMemory, MemoryEntry, reset_memory_backend


class TestSimpleMemory:
    """Tests for the SimpleMemory JSON-based backend."""

    @pytest.fixture
    def temp_memory(self, tmp_path):
        """Create a SimpleMemory instance with a temporary file."""
        filepath = str(tmp_path / "test_memory.json")
        return SimpleMemory(filepath=filepath)

    def test_add_memory(self, temp_memory):
        """Test adding a memory entry."""
        memory_id = temp_memory.add(
            content="Cu prefers EAM potentials",
            category="knowledge",
            metadata={"material": "Cu"}
        )

        assert memory_id is not None
        assert len(memory_id) == 8  # UUID first 8 chars

    def test_search_by_keyword(self, temp_memory):
        """Test keyword-based search."""
        temp_memory.add("Cu prefers EAM potentials", "knowledge", {"material": "Cu"})
        temp_memory.add("Al works well with MEAM", "knowledge", {"material": "Al"})
        temp_memory.add("Fe simulation failed due to timestep", "failure", {"material": "Fe"})

        # Search for copper
        results = temp_memory.search("Cu copper EAM")
        assert len(results) >= 1
        assert any("Cu" in r.content for r in results)

    def test_search_with_category_filter(self, temp_memory):
        """Test searching with category filter."""
        temp_memory.add("Successful Cu simulation", "success", {})
        temp_memory.add("Cu simulation failed", "failure", {})

        # Search failures only
        results = temp_memory.search("Cu simulation", category="failure")
        assert len(results) == 1
        assert results[0].category == "failure"

    def test_get_recent(self, temp_memory):
        """Test getting recent memories."""
        temp_memory.add("Memory 1", "success", {})
        temp_memory.add("Memory 2", "success", {})
        temp_memory.add("Memory 3", "failure", {})

        # Get all recent
        recent = temp_memory.get_recent(limit=10)
        assert len(recent) == 3

        # Get recent failures only
        recent_failures = temp_memory.get_recent(category="failure", limit=10)
        assert len(recent_failures) == 1
        assert recent_failures[0].content == "Memory 3"

    def test_clear_all(self, temp_memory):
        """Test clearing all memories."""
        temp_memory.add("Memory 1", "success", {})
        temp_memory.add("Memory 2", "failure", {})

        count = temp_memory.clear()
        assert count == 2

        # Verify empty
        recent = temp_memory.get_recent()
        assert len(recent) == 0

    def test_clear_by_category(self, temp_memory):
        """Test clearing memories by category."""
        temp_memory.add("Memory 1", "success", {})
        temp_memory.add("Memory 2", "failure", {})
        temp_memory.add("Memory 3", "failure", {})

        # Clear failures only
        count = temp_memory.clear(category="failure")
        assert count == 2

        # Verify only success remains
        recent = temp_memory.get_recent()
        assert len(recent) == 1
        assert recent[0].category == "success"

    def test_memory_persistence(self, tmp_path):
        """Test that memories persist across instances."""
        filepath = str(tmp_path / "persist_test.json")

        # Create and add
        mem1 = SimpleMemory(filepath=filepath)
        mem1.add("Persistent memory", "knowledge", {"key": "value"})

        # Create new instance with same file
        mem2 = SimpleMemory(filepath=filepath)
        recent = mem2.get_recent()

        assert len(recent) == 1
        assert recent[0].content == "Persistent memory"

    def test_search_includes_metadata(self, temp_memory):
        """Test that search considers metadata."""
        temp_memory.add(
            "Simulation completed",
            "success",
            {"material": "copper", "ensemble": "NVT"}
        )

        # Search by metadata term
        results = temp_memory.search("copper NVT")
        assert len(results) >= 1

    def test_relevance_scoring(self, temp_memory):
        """Test that results are sorted by relevance."""
        temp_memory.add("copper simulation", "knowledge", {})
        temp_memory.add("copper EAM potential copper", "knowledge", {})  # More matches
        temp_memory.add("iron simulation", "knowledge", {})

        results = temp_memory.search("copper")

        # First result should have more matches
        assert results[0].relevance_score >= results[1].relevance_score if len(results) > 1 else True

    def test_empty_search_returns_empty(self, temp_memory):
        """Test that searching with no matches returns empty list."""
        temp_memory.add("copper simulation", "knowledge", {})

        results = temp_memory.search("nonexistent term xyz")
        assert len(results) == 0


class TestMemoryEntry:
    """Tests for the MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a MemoryEntry."""
        entry = MemoryEntry(
            id="abc123",
            content="Test content",
            category="knowledge",
            metadata={"key": "value"},
            created_at="2026-02-05T10:00:00",
            relevance_score=0.8
        )

        assert entry.id == "abc123"
        assert entry.content == "Test content"
        assert entry.category == "knowledge"
        assert entry.metadata["key"] == "value"
        assert entry.relevance_score == 0.8

    def test_memory_entry_default_score(self):
        """Test that relevance_score defaults to 0.0."""
        entry = MemoryEntry(
            id="abc123",
            content="Test",
            category="test",
            metadata={},
            created_at=""
        )
        assert entry.relevance_score == 0.0
