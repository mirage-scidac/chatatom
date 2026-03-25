# ChatMD Agentify Migration: Revised DSPy Implementation Plan

**Date**: February 5, 2026
**Status**: Ready for Implementation
**Approach**: Complete DSPy Foundation (Skip LangGraph)

---

## Executive Summary

The DSPy implementation is ~70% complete. This plan focuses on wiring up existing components and adding persistent memory/sessions. LangGraph is unnecessary for ChatMD's mostly-linear 3-4 agent workflow.

---

## Current State Assessment

### Already Implemented (Phase 0-1 Complete)

| Component | Status | Location |
|-----------|--------|----------|
| Centralized prompts | ✅ Done | `src/prompts.py` (755 lines) |
| DSPy signatures | ✅ Done | 5 signatures: `AnalyzeRequest`, `VerifySimulationPlan`, `GenerateLAMMPSScript`, `ReviewLAMMPSScript`, `SelectTools` |
| `IterativeLammpsGeneration` | ✅ Done | `src/agents.py:198-311` |
| `PlanVerifier` | ✅ Done | `src/agents.py:324-405` |
| `SmartOrchestrator` | ✅ Done | `src/agents.py:422-478` |
| Checkpoint API endpoints | ✅ Done | 4 endpoints in `app.py:164-485` |
| `KnowledgeBase` class | ✅ Done | `src/prompts.py:112-161` |
| DSPy LM configuration | ✅ Done | `src/agents.py:174-183` |
| In-memory session storage | ✅ Done | `app.py:124-157` |

### Gaps to Address

| Component | Priority | Estimated Lines |
|-----------|----------|-----------------|
| Memory integration (Mem0 or SimpleMemory) | HIGH | ~100 |
| SQLite session persistence | HIGH | ~150 |
| Memory wiring to DSPy modules | MEDIUM | ~50 |
| Frontend checkpoint UI | MEDIUM | ~150 |
| Confidence display | LOW | ~50 |
| Error classification | LOW | ~100 |

---

## Implementation Plan

### Phase 1: Memory System (Priority: HIGH)

#### Step 1.1: Test Mem0 Embeddings

First, test if Mem0 works with Argonne API:

```python
# Test script: tests/test_mem0_integration.py
from mem0 import Memory

def test_mem0_basic():
    """Test if Mem0 works with available embedding model."""
    try:
        m = Memory()
        m.add([{"role": "user", "content": "Cu prefers EAM potentials"}], user_id="test")
        results = m.search("copper simulation", user_id="test")
        return len(results) > 0
    except Exception as e:
        print(f"Mem0 failed: {e}")
        return False
```

#### Step 1.2: Create Memory Abstraction

Create `src/memory.py` with a common interface that supports both Mem0 and SimpleMemory:

```python
# src/memory.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime

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
    """JSON file-based memory with category filtering."""

    def __init__(self, filepath: str = "data/memory.json"):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({"memories": []}, f)

    def _load(self) -> Dict:
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def add(self, content: str, category: str, metadata: Dict[str, Any] = None) -> str:
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

        for mem in data["memories"]:
            if category and mem["category"] != category:
                continue

            # Simple relevance: count keyword matches
            content_lower = mem["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)

            if score > 0:
                results.append(MemoryEntry(
                    id=mem["id"],
                    content=mem["content"],
                    category=mem["category"],
                    metadata=mem["metadata"],
                    created_at=mem["created_at"],
                    relevance_score=score
                ))

        # Sort by relevance and return top N
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

    def get_recent(self, category: str = None, limit: int = 10) -> List[MemoryEntry]:
        data = self._load()
        memories = data["memories"]

        if category:
            memories = [m for m in memories if m["category"] == category]

        # Sort by created_at descending
        memories.sort(key=lambda x: x["created_at"], reverse=True)

        return [
            MemoryEntry(
                id=m["id"],
                content=m["content"],
                category=m["category"],
                metadata=m["metadata"],
                created_at=m["created_at"]
            )
            for m in memories[:limit]
        ]

    def clear(self, category: str = None) -> int:
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
        full_metadata = {"category": category, **(metadata or {})}
        result = self.memory.add(
            [{"role": "assistant", "content": content}],
            user_id=self.user_id,
            metadata=full_metadata
        )
        return result.get("id", "unknown")

    def search(self, query: str, category: str = None, limit: int = 5) -> List[MemoryEntry]:
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
        if category:
            # Mem0 doesn't support category-based deletion easily
            # Would need to iterate and delete individually
            return 0
        else:
            self.memory.delete_all(user_id=self.user_id)
            return -1  # Unknown count


def get_memory_backend() -> MemoryBackend:
    """Factory function to get the appropriate memory backend."""
    try:
        # Try Mem0 first
        backend = Mem0Memory()
        # Quick test
        backend.add("test", "test", {})
        backend.clear("test")
        return backend
    except Exception as e:
        print(f"[MEMORY] Mem0 unavailable ({e}), falling back to SimpleMemory")
        return SimpleMemory()
```

---

### Phase 2: Session Persistence (Priority: HIGH)

Create `src/session_manager.py` for SQLite-based session persistence:

```python
# src/session_manager.py
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

class SessionManager:
    """SQLite-based session persistence for workflow state."""

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
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
        """Update a session."""
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

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, stage, user_request
                   FROM sessions ORDER BY updated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()

            return [dict(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0


# Global instance
_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

---

### Phase 3: Wire Memory to DSPy Modules (Priority: MEDIUM)

Update `src/agents.py` to use memory in the iteration loop:

```python
# Add to agents.py after imports
from memory import get_memory_backend, MemoryEntry

class IterativeLammpsGeneration(dspy.Module):
    """DSPy module with memory-augmented generation."""

    def __init__(self, max_attempts: int = 3):
        super().__init__()
        self.max_attempts = max_attempts
        self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
        self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
        self.memory = get_memory_backend()

    def forward(self, material: str, kim_model: str, ...) -> GenerationResult:
        conversation = []
        current_script = ""
        all_issues = []
        all_suggestions = []

        # Retrieve relevant memories before generation
        relevant_memories = self.memory.search(
            f"{material} {ensemble} simulation",
            category="failure",
            limit=3
        )

        # Build context from memories
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\nPrevious issues to avoid:\n" + "\n".join(
                f"- {m.content}" for m in relevant_memories
            )

        for attempt in range(1, self.max_attempts + 1):
            # Include memory context in visualization_notes
            notes = visualization_notes + memory_context if attempt == 1 else ...

            # ... existing generation logic ...

            # If review fails, store the failure
            if not review_result.is_valid:
                self.memory.add(
                    content=f"Script for {material} failed: {'; '.join(review_result.issues[:2])}",
                    category="failure",
                    metadata={
                        "material": material,
                        "kim_model": kim_model,
                        "attempt": attempt
                    }
                )

        # If successful, store as success pattern
        if result.review_passed:
            self.memory.add(
                content=f"Successful {material} {ensemble} simulation at {temperature}K",
                category="success",
                metadata={
                    "material": material,
                    "kim_model": kim_model,
                    "ensemble": ensemble
                }
            )

        return result
```

---

### Phase 4: Update app.py Session Handling (Priority: MEDIUM)

Replace in-memory session storage with SessionManager:

```python
# app.py changes

from session_manager import get_session_manager

# Remove the old workflow_sessions dict and helper functions

# Update workflow_start endpoint
@app.route('/api/workflow/start', methods=['POST'])
def workflow_start():
    try:
        data = request.get_json()
        user_request = data.get('request')

        if not user_request:
            return jsonify({"error": "Request not provided"}), 400

        # Configure DSPy
        configure_dspy_lm()

        # Create session using SessionManager
        session_manager = get_session_manager()
        session_id = str(uuid.uuid4())
        session = session_manager.create(session_id, user_request)

        # ... rest of existing logic, replacing:
        # - get_session() -> session_manager.get()
        # - update_session() -> session_manager.update()

        # ... existing analysis and verification code ...
```

---

### Phase 5: Add Config Options (Priority: LOW)

Update `src/config.py`:

```python
# Add to config.py

# Memory Configuration
MEMORY_BACKEND = "auto"  # "auto", "mem0", or "simple"
MEMORY_FILE_PATH = "data/memory.json"

# Session Configuration
SESSION_DB_PATH = "data/sessions.db"
SESSION_EXPIRY_DAYS = 30  # Auto-delete sessions older than this
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/memory.py` | CREATE | Memory abstraction with Mem0/SimpleMemory backends |
| `src/session_manager.py` | CREATE | SQLite session persistence |
| `src/agents.py` | MODIFY | Wire memory to IterativeLammpsGeneration |
| `src/app.py` | MODIFY | Replace in-memory sessions with SessionManager |
| `src/config.py` | MODIFY | Add memory/session config options |
| `tests/test_mem0_integration.py` | CREATE | Test Mem0 compatibility |
| `tests/test_session_manager.py` | CREATE | Test session persistence |
| `data/` | CREATE | Directory for memory.json and sessions.db |

---

## Testing Strategy

### Unit Tests

1. **Memory Backend Tests** (`tests/test_memory.py`)
   - Test SimpleMemory add/search/clear
   - Test Mem0Memory fallback behavior
   - Test category filtering

2. **Session Manager Tests** (`tests/test_session_manager.py`)
   - Test create/get/update/delete
   - Test persistence across restarts
   - Test concurrent access

3. **Integration Tests** (`tests/test_dspy_integration.py`)
   - Test IterativeLammpsGeneration with memory
   - Test full workflow with sessions
   - Test memory retrieval affects generation

### Manual Testing

1. Start server, create simulation request
2. Verify session persists after server restart
3. Cause intentional failure, verify memory stores it
4. Run similar simulation, verify memory context is used

---

## What We're NOT Doing (And Why)

### LangGraph Migration
- **Why skip**: ChatMD has 3-4 agents in a mostly-linear flow. The only cycle (Generator ↔ Reviewer) is already implemented in `IterativeLammpsGeneration`. LangGraph adds dependency complexity for marginal benefit.

### Complex Tool Registry
- **Why skip**: Current fixed pipeline works well. Tool selection is only needed if we add many more agent capabilities.

### Full Frontend Overhaul
- **Why skip**: Backend wiring comes first. Frontend checkpoint UI can be added incrementally.

### DSPy Prompt Optimization
- **Why skip**: Current prompts work. Optimization is a polish step after core functionality is stable.

---

## Success Criteria

1. **Sessions persist**: Server restart doesn't lose workflow state
2. **Memory influences generation**: Failed patterns are avoided in subsequent runs
3. **No regressions**: `/api/llm-stream-live` endpoint works correctly
4. **Graceful fallback**: If Mem0 fails, SimpleMemory works transparently

---

## Estimated Effort

| Phase | New Lines | Modified Lines | Files |
|-------|-----------|----------------|-------|
| Phase 1: Memory | ~200 | 0 | 2 |
| Phase 2: Sessions | ~150 | 0 | 1 |
| Phase 3: Wire Memory | ~50 | 30 | 1 |
| Phase 4: Update app.py | ~20 | 40 | 1 |
| Phase 5: Config | ~10 | 0 | 1 |
| Tests | ~150 | 0 | 3 |
| **Total** | **~580** | **~70** | **9** |

---

## Next Steps

1. ~~Run Mem0 embedding test to determine backend choice~~ ✅ Done (using SimpleMemory fallback)
2. ~~Create `src/memory.py` with both backends~~ ✅ Done
3. ~~Create `src/session_manager.py`~~ ✅ Done
4. ~~Wire memory into `IterativeLammpsGeneration`~~ ✅ Done
5. ~~Update `app.py` to use SessionManager~~ ✅ Done
6. ~~Write and run tests~~ ✅ Done
7. Test end-to-end workflow

---

## Phase 6: LLM-Based Intent Detection (Priority: HIGH)

### Problem

The current implementation uses brittle string parsing for intent detection:
- Keywords like "run", "execute", "yes", "ok" are matched via substring
- Natural confirmations like "continue", "fine", "sure", "go ahead" fail
- "run" after script is ready causes a hang (frontend clears context too early)
- Negations like "don't run" incorrectly match "run"

### Solution: Use LLM for Intent Classification

**Principle: Use LLM approaches whenever possible. Minimize string parsing and conditional logic.**

#### Step 6.1: Add Intent Classification DSPy Signature (`src/prompts.py`)

```python
class ClassifyUserIntent(dspy.Signature):
    """Classify the user's intent in the context of a simulation workflow."""

    user_message: str = dspy.InputField(desc="The user's message")
    workflow_stage: str = dspy.InputField(desc="Current stage: 'initial', 'parameters_proposed', 'script_ready'")

    intent: str = dspy.OutputField(desc="One of: 'execute', 'modify', 'cancel', 'question', 'confirm'")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    extracted_modifications: str = dspy.OutputField(desc="If intent is 'modify', what changes are requested. Otherwise empty.")
```

#### Step 6.2: Add Intent Classifier to SmartOrchestrator (`src/agents.py`)

```python
class SmartOrchestrator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeRequest)
        self.tool_selector = dspy.ChainOfThought(SelectTools)
        self.intent_classifier = dspy.ChainOfThought(ClassifyUserIntent)  # NEW

    def classify_intent(self, user_message: str, workflow_stage: str) -> dict:
        """Classify user intent using LLM instead of string matching."""
        result = self.intent_classifier(
            user_message=user_message,
            workflow_stage=workflow_stage
        )
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "modifications": result.extracted_modifications
        }
```

#### Step 6.3: Update `/api/llm-stream-live` in `app.py`

**Remove** string-based intent detection (lines 1062-1076).

**Replace with** LLM-based classification:

```python
if "REVIEW: PASS" in review:
    workflow_stage = "script_ready" if conversation_context else "initial"
    intent_result = orchestrator.classify_intent(user_command, workflow_stage)

    if intent_result["intent"] in ["execute", "confirm"]:
        yield f"data: {json.dumps({'type': 'start_streaming_simulation', ...})}\n\n"
    elif intent_result["intent"] == "modify":
        yield f"data: {json.dumps({'type': 'awaiting_modifications', ...})}\n\n"
    elif intent_result["intent"] == "cancel":
        yield f"data: {json.dumps({'type': 'workflow_cancelled'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'script_ready', 'script': lammps_script})}\n\n"
```

#### Step 6.4: Fix Frontend Context Preservation (`index.html`)

**Problem:** Context is cleared after Phase 2, so "run" command loses context.

**Fix:**
1. Add `pendingScript` state variable to hold script when ready
2. Only clear context AFTER successful execution, not before
3. Include pending script in context when user sends follow-up

```javascript
let pendingScript = null;

// In 'script_ready' handler - DON'T clear context
case 'script_ready':
    pendingScript = data.script;
    // Keep conversationContext intact
    break;

// In 'start_streaming_simulation' handler - NOW clear context
case 'start_streaming_simulation':
    conversationContext = '';
    pendingScript = null;
    break;

// In sendMessage() - include pending script if available
if (pendingScript && !conversationContext) {
    params.append('context', `SCRIPT READY:\n${pendingScript}`);
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/prompts.py` | Add `ClassifyUserIntent` DSPy signature |
| `src/agents.py` | Add `classify_intent()` method to `SmartOrchestrator` |
| `src/app.py` | Replace string matching with LLM intent classification |
| `index.html` | Fix context preservation, add `pendingScript` state |

### Testing

1. Natural confirmations: "sure", "yes please", "go ahead", "this looks good"
2. Negations: "no", "stop", "don't run", "cancel"
3. Modifications: "change to 500K", "make it longer"
4. "run" after script ready (should not hang)

---

## Future Enhancements (TODO)

| Feature | Priority | Description |
|---------|----------|-------------|
| Persistent chat history | MEDIUM | Store frontend chat messages in SQLite/localStorage so they survive page refresh |
| Frontend checkpoint UI | MEDIUM | Modal for reviewing/approving parameters and scripts |
| Confidence display | LOW | Show confidence scores in UI |
| Error classification | LOW | Classify errors by type for adaptive retry strategies |
