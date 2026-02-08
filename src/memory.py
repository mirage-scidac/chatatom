# src/memory.py
"""
Memory abstraction layer for ChatMD.

Provides persistent memory storage with support for:
- SimpleMemory: JSON file-based storage with keyword search
- Mem0Memory: Semantic search using Mem0 library (if available)

Usage:
    from memory import get_memory_backend

    memory = get_memory_backend()
    memory.add("Cu prefers EAM potentials", "knowledge", {"material": "Cu"})
    results = memory.search("copper simulation", limit=5)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime

import config


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    category: str  # "success", "failure", "preference", "knowledge"
    metadata: Dict[str, Any]
    created_at: str
    relevance_score: float = 0.0


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""

    @abstractmethod
    def add(self, content: str, category: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory entry. Returns the memory ID."""
        pass

    @abstractmethod
    def search(self, query: str, category: str = None, limit: int = 5) -> List[MemoryEntry]:
        """Search for relevant memories."""
        pass

    @abstractmethod
    def get_recent(self, category: str = None, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories by category."""
        pass

    @abstractmethod
    def clear(self, category: str = None) -> int:
        """Clear memories. Returns count deleted."""
        pass


class SimpleMemory(MemoryBackend):
    """JSON file-based memory with category filtering and keyword search."""

    def __init__(self, filepath: str = None):
        self.filepath = filepath or getattr(config, 'MEMORY_FILE_PATH', 'data/memory.json')
        self._ensure_file()

    def _ensure_file(self):
        """Ensure the memory file and its directory exist."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({"memories": []}, f)

    def _load(self) -> Dict:
        """Load memories from file."""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"memories": []}

    def _save(self, data: Dict):
        """Save memories to file."""
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def add(self, content: str, category: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory entry."""
        import uuid
        data = self._load()
        memory_id = str(uuid.uuid4())[:8]
        entry = {
            "id": memory_id,
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        data["memories"].append(entry)
        self._save(data)
        return memory_id

    def search(self, query: str, category: str = None, limit: int = 5) -> List[MemoryEntry]:
        """Simple keyword-based search (no embeddings)."""
        data = self._load()
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for mem in data["memories"]:
            if category and mem["category"] != category:
                continue

            # Simple relevance: count keyword matches
            content_lower = mem["content"].lower()
            metadata_str = json.dumps(mem.get("metadata", {})).lower()
            combined = content_lower + " " + metadata_str

            # Count matching words
            score = sum(1 for word in query_words if word in combined)

            if score > 0:
                results.append(MemoryEntry(
                    id=mem["id"],
                    content=mem["content"],
                    category=mem["category"],
                    metadata=mem.get("metadata", {}),
                    created_at=mem["created_at"],
                    relevance_score=score
                ))

        # Sort by relevance and return top N
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

    def get_recent(self, category: str = None, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories, optionally filtered by category."""
        data = self._load()
        memories = data["memories"]

        if category:
            memories = [m for m in memories if m["category"] == category]

        # Sort by created_at descending
        memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return [
            MemoryEntry(
                id=m["id"],
                content=m["content"],
                category=m["category"],
                metadata=m.get("metadata", {}),
                created_at=m.get("created_at", "")
            )
            for m in memories[:limit]
        ]

    def clear(self, category: str = None) -> int:
        """Clear memories, optionally by category."""
        data = self._load()
        original_count = len(data["memories"])

        if category:
            data["memories"] = [m for m in data["memories"] if m["category"] != category]
        else:
            data["memories"] = []

        self._save(data)
        return original_count - len(data["memories"])


class Mem0Memory(MemoryBackend):
    """Mem0-based memory with semantic search."""

    def __init__(self, user_id: str = "chatmd"):
        from mem0 import Memory
        self.memory = Memory()
        self.user_id = user_id

    def add(self, content: str, category: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory entry using Mem0."""
        full_metadata = {"category": category, **(metadata or {})}
        result = self.memory.add(
            [{"role": "assistant", "content": content}],
            user_id=self.user_id,
            metadata=full_metadata
        )
        return result.get("id", "unknown")

    def search(self, query: str, category: str = None, limit: int = 5) -> List[MemoryEntry]:
        """Search memories using Mem0's semantic search."""
        results = self.memory.search(query, user_id=self.user_id, limit=limit * 2)

        entries = []
        for r in results:
            mem_category = r.get("metadata", {}).get("category", "unknown")
            if category and mem_category != category:
                continue

            entries.append(MemoryEntry(
                id=r.get("id", ""),
                content=r.get("memory", ""),
                category=mem_category,
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at", ""),
                relevance_score=r.get("score", 0.0)
            ))

        return entries[:limit]

    def get_recent(self, category: str = None, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories from Mem0."""
        all_memories = self.memory.get_all(user_id=self.user_id)

        entries = []
        for m in all_memories:
            mem_category = m.get("metadata", {}).get("category", "unknown")
            if category and mem_category != category:
                continue

            entries.append(MemoryEntry(
                id=m.get("id", ""),
                content=m.get("memory", ""),
                category=mem_category,
                metadata=m.get("metadata", {}),
                created_at=m.get("created_at", "")
            ))

        return entries[:limit]

    def clear(self, category: str = None) -> int:
        """Clear memories from Mem0."""
        if category:
            # Mem0 doesn't support category-based deletion easily
            # Would need to iterate and delete individually
            return 0
        else:
            self.memory.delete_all(user_id=self.user_id)
            return -1  # Unknown count


# Global instance cache
_memory_backend: Optional[MemoryBackend] = None


def get_memory_backend() -> MemoryBackend:
    """
    Factory function to get the appropriate memory backend.

    Uses config.MEMORY_BACKEND to determine which backend to use:
    - "auto": Try Mem0 first, fall back to SimpleMemory
    - "mem0": Use Mem0 (will fail if unavailable)
    - "simple": Use SimpleMemory
    """
    global _memory_backend

    if _memory_backend is not None:
        return _memory_backend

    backend_choice = getattr(config, 'MEMORY_BACKEND', 'auto')

    if backend_choice == 'simple':
        print("[MEMORY] Using SimpleMemory backend")
        _memory_backend = SimpleMemory()
        return _memory_backend

    if backend_choice == 'mem0':
        print("[MEMORY] Using Mem0 backend")
        _memory_backend = Mem0Memory()
        return _memory_backend

    # Auto mode: try Mem0, fall back to SimpleMemory
    try:
        backend = Mem0Memory()
        # Quick test to verify it works
        test_id = backend.add("__test__", "test", {})
        backend.clear("test")
        print("[MEMORY] Using Mem0 backend (auto-detected)")
        _memory_backend = backend
        return _memory_backend
    except Exception as e:
        print(f"[MEMORY] Mem0 unavailable ({e}), falling back to SimpleMemory")
        _memory_backend = SimpleMemory()
        return _memory_backend


def reset_memory_backend():
    """Reset the cached memory backend (useful for testing)."""
    global _memory_backend
    _memory_backend = None
