# ChatMD Agentify Migration Plan
## Multi-Turn Conversations and Memory Implementation

**Date**: 2025-11-16
**Branch**: `claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n`
**Status**: Planning Phase - No Code Changes Yet

---

## Executive Summary

This document outlines a plan to enhance ChatMD with robust multi-turn conversation capabilities and persistent memory, enabling:
- Continuous conversations across multiple user interactions
- Context retention from previous simulation runs
- Reference to past experiments and parameters
- Learning from user preferences over time

**Target Implementation**: Minimal 3-file architecture (frontend, agents.py, prompts file) plus configuration.

---

## 1. Current Architecture Analysis

### 1.1 Existing Conversation State

**Current Capabilities:**
- ✅ Two-phase conversation (orchestrator Phase 1 → Phase 2)
- ✅ Single-turn context retention via `conversationContext` variable
- ✅ SSE streaming for real-time agent responses
- ✅ WebSocket for live simulation data

**Current Limitations:**
- ❌ No conversation history beyond current request
- ❌ No memory of previous simulation runs
- ❌ Users cannot reference "the last simulation" or "previous Cu run"
- ❌ No learning from user preferences (e.g., favorite KIM models)
- ❌ Context resets with page reload
- ❌ No conversation persistence across sessions

### 1.2 Current File Structure

```
chatmd/
├── app.py (1,118 lines)          # Flask server, routing, workflow
├── agents.py (485 lines)         # 4 agent implementations + prompts
├── config.py (36 lines)          # Configuration
├── index.html (1,100+ lines)     # Frontend UI
├── tools.py (560 lines)          # LAMMPS execution
├── kim_utils.py (729 lines)      # KIM model management
├── ovito_script_generator.py    # Visualization generation
├── ovito_executor.py             # OVITO script execution
├── visualization_rendering.py   # Frame rendering
└── run_manager.py                # Run folder management
```

### 1.3 Current Conversation Flow

```
User Request
    ↓
OrchestratorAgent (Phase 1) → Proposes parameters
    ↓
User Feedback
    ↓
OrchestratorAgent (Phase 2) → Finalizes with context
    ↓
Generator → Reviewer → Visualization → Execution
```

**Key Insight**: We already have a multi-turn pattern within a single request lifecycle, but need to extend this across multiple requests.

---

## 2. Technology Evaluation: DSPy vs Alternatives

### 2.1 Option 1: DSPy (Recommended for Structured Approach)

**Overview**: DSPy is a framework for programming language models with structured signatures and built-in conversation history management.

**Pros:**
- ✅ Native `dspy.History` type for conversation management
- ✅ Built-in adapters handle multi-turn message formatting
- ✅ Structured signatures enforce input/output contracts
- ✅ Optimizable with DSPy's compilation framework
- ✅ Clean integration with existing OpenAI-compatible APIs
- ✅ Memory-enabled ReAct agents with Mem0 integration option
- ✅ Well-documented conversation history patterns

**Cons:**
- ❌ Additional dependency (~10MB package)
- ❌ Learning curve for existing team
- ❌ May be "heavyweight" for minimal implementation
- ❌ Introduces new programming paradigm

**Implementation Complexity**: Medium
**Package Size**: ~10MB
**Code Changes**: Moderate (refactor agents to use DSPy signatures)

### 2.2 Option 2: Simple Buffer-Based Memory (Recommended for Minimal Approach)

**Overview**: Maintain conversation history as list of message dicts, persist to JSON/SQLite.

**Pros:**
- ✅ Zero additional dependencies
- ✅ Simple to understand and maintain
- ✅ Full control over memory management
- ✅ Easy to implement in 3 files
- ✅ Lightweight (~100 lines of code)

**Cons:**
- ❌ Manual message formatting
- ❌ No semantic search (just chronological)
- ❌ Need to implement truncation strategy manually
- ❌ No built-in memory optimization

**Implementation Complexity**: Low
**Package Size**: 0MB
**Code Changes**: Minimal

### 2.3 Option 3: LangChain

**Pros:**
- ✅ Comprehensive ecosystem
- ✅ Multiple memory types (Buffer, Window, Summary, Token-based)
- ✅ Easy persistence with various backends

**Cons:**
- ❌ Heavy dependency (~50MB+ with dependencies)
- ❌ Overkill for our use case
- ❌ Frequent breaking changes between versions
- ❌ Not aligned with "minimal implementation" goal

**Verdict**: ❌ **Not Recommended** - Too heavyweight

### 2.4 Option 4: Vector Database (ChromaDB/LanceDB)

**Overview**: Semantic search over conversation history using embeddings.

**Pros:**
- ✅ Semantic retrieval ("similar conversations")
- ✅ Efficient for large conversation histories
- ✅ Can handle thousands of messages

**Cons:**
- ❌ Requires embeddings API calls (cost/latency)
- ❌ Additional service to manage
- ❌ Overkill for chronological conversation history
- ❌ Not needed for typical 10-50 message conversations

**Verdict**: ❌ **Not Recommended** - Save for future enhancement if needed

---

## 3. Recommended Solution: Hybrid Approach

### 3.1 Core Strategy

**Phase 1 (Minimal Implementation)**: Simple buffer-based memory
**Phase 2 (Optional Enhancement)**: Add DSPy for optimization and advanced patterns

### 3.2 Why Hybrid?

1. **Start Simple**: Buffer-based memory gets 80% of value with 20% of complexity
2. **Easy Migration**: Can upgrade to DSPy later without breaking changes
3. **Minimal Dependencies**: Keep dependency tree clean
4. **3-File Target**: Achievable with buffer approach

---

## 4. Proposed 3-File Architecture

### 4.1 File Breakdown

```
Core Files (Modified):
├── index.html              # Frontend with conversation history UI
├── agents.py               # Agent classes with memory integration
└── prompts.py              # NEW: Extracted prompts with memory templates

Configuration Files (Modified):
├── config.py               # Memory settings (max messages, storage path)

Support Files (Minimal Changes):
├── app.py                  # Add memory load/save endpoints
├── run_manager.py          # Optional: link runs to conversation IDs
```

### 4.2 Can It Be Done in 3 Files?

**Answer**: Yes, with this structure:

1. **`index.html`** (Frontend):
   - Conversation history display
   - Memory persistence to localStorage
   - UI for browsing past conversations
   - Message threading

2. **`agents.py`** (Core Logic):
   - Memory buffer management class
   - Modified agent classes to accept conversation history
   - Methods: `load_history()`, `save_history()`, `add_message()`, `get_context()`

3. **`prompts.py`** (NEW - Prompts + Templates):
   - Extract all system prompts from agents.py
   - Add conversation history templates
   - Memory injection patterns for each agent

**Additional Minimal Changes**:
- `config.py`: 3-5 new config variables
- `app.py`: 2 new endpoints for persistence (optional if using localStorage)

---

## 5. Detailed Design

### 5.1 Memory Data Structure

```python
# ConversationMemory structure
{
    "conversation_id": "uuid-v4",
    "created_at": "2025-11-16T10:30:00Z",
    "last_updated": "2025-11-16T11:45:00Z",
    "messages": [
        {
            "role": "user",
            "content": "Run a Cu simulation at 300K",
            "timestamp": "2025-11-16T10:30:00Z"
        },
        {
            "role": "assistant",
            "agent": "orchestrator",
            "content": "I found 3 Cu potentials...",
            "timestamp": "2025-11-16T10:30:15Z"
        },
        {
            "role": "user",
            "content": "Use the first one",
            "timestamp": "2025-11-16T10:31:00Z"
        },
        # ... more messages
    ],
    "metadata": {
        "total_simulations_run": 2,
        "elements_used": ["Cu", "Al"],
        "last_kim_model": "EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006",
        "preferred_models": {},  # Future: track user preferences
    },
    "simulation_refs": [
        {
            "run_id": "run_20251116_103045",
            "message_index": 5,
            "element": "Cu",
            "temperature": 300,
            "duration_ps": 10
        }
    ]
}
```

### 5.2 Memory Management Class

```python
# agents.py additions

class ConversationMemory:
    """Lightweight conversation history manager"""

    def __init__(self, conversation_id=None, max_messages=100):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_messages = max_messages
        self.messages = []
        self.metadata = {...}
        self.simulation_refs = []

    def add_message(self, role, content, agent=None):
        """Add message to history with timestamp"""

    def get_recent_context(self, n=10):
        """Get last N messages for context"""

    def get_context_for_agent(self, agent_type):
        """Get formatted context for specific agent"""

    def summarize_history(self):
        """Generate summary of conversation for long histories"""

    def link_simulation(self, run_id, params):
        """Link a simulation run to conversation context"""

    def save(self, filepath=None):
        """Persist to JSON"""

    @staticmethod
    def load(filepath_or_id):
        """Load from JSON"""

    def truncate(self, strategy="sliding_window"):
        """Truncate based on strategy (sliding_window, summary, token_limit)"""
```

### 5.3 Agent Integration

**Modified Agent Base Class**:
```python
class Agent:
    def __init__(self, model, system_prompt, temperature=0.7, memory=None):
        # ... existing code
        self.memory = memory  # ConversationMemory instance

    def execute(self, user_input, context=None, conversation_id=None):
        # Build messages with history
        messages = self._build_messages_with_memory(user_input, context)

        # Call LLM
        response = self.client.chat.completions.create(...)

        # Save to memory
        if self.memory:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response_content, agent=self.name)

        return response_content

    def _build_messages_with_memory(self, user_input, context):
        """Construct message array with conversation history"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add relevant conversation history
        if self.memory:
            history = self.memory.get_context_for_agent(self.name)
            messages.extend(history)

        # Add current context (Phase 2 data)
        if context:
            messages.append({"role": "system", "content": f"Previous context: {context}"})

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        return messages
```

### 5.4 System Prompt Enhancements

**New `prompts.py` file**:
```python
# prompts.py - Extract and enhance all system prompts

ORCHESTRATOR_BASE_PROMPT = """
You are an orchestrator for molecular dynamics simulations...
[existing prompt content]
"""

ORCHESTRATOR_WITH_MEMORY_TEMPLATE = """
{base_prompt}

## Conversation History Context
You have access to the following conversation history:
{conversation_summary}

Recent simulations:
{simulation_refs}

When the user references "the last simulation" or "previous run", refer to this context.
"""

def build_orchestrator_prompt(memory=None):
    """Build orchestrator prompt with optional memory injection"""
    if not memory:
        return ORCHESTRATOR_BASE_PROMPT

    summary = memory.summarize_history()
    refs = memory.get_simulation_refs_summary()

    return ORCHESTRATOR_WITH_MEMORY_TEMPLATE.format(
        base_prompt=ORCHESTRATOR_BASE_PROMPT,
        conversation_summary=summary,
        simulation_refs=refs
    )
```

### 5.5 Frontend Changes

**HTML Additions**:
1. **Conversation Selector**: Dropdown to switch between conversations
2. **History Sidebar**: Collapsible panel showing message history
3. **Reference Links**: Click simulation references to view details
4. **New Conversation Button**: Start fresh conversation

**JavaScript Additions**:
```javascript
// Conversation management
class ConversationManager {
    constructor() {
        this.currentConversationId = this.loadOrCreateConversation();
        this.conversations = this.loadAllConversations();
    }

    loadOrCreateConversation() {
        // Check localStorage for active conversation
        // Or create new UUID
    }

    saveMessage(role, content, agent) {
        // Save to localStorage
        // Optional: POST to /api/conversation/save
    }

    loadHistory() {
        // Load from localStorage
        // Or GET from /api/conversation/:id
    }

    renderHistory() {
        // Populate chat UI with history
    }

    switchConversation(conversationId) {
        // Load different conversation
    }

    exportConversation() {
        // Download as JSON
    }
}
```

### 5.6 API Endpoints (Optional - if not using localStorage)

```python
# app.py additions

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Load conversation history"""
    memory = ConversationMemory.load(conversation_id)
    return jsonify(memory.to_dict())

@app.route('/api/conversation/<conversation_id>', methods=['POST'])
def save_conversation(conversation_id):
    """Save conversation history"""
    data = request.json
    memory = ConversationMemory.from_dict(data)
    memory.save()
    return jsonify({"status": "saved"})

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all saved conversations"""
    # Scan storage directory
    return jsonify({"conversations": [...]})
```

---

## 6. Implementation Strategy

### 6.1 Incremental Approach

**Phase 1: Core Memory (Week 1)**
- [ ] Create `prompts.py` with extracted prompts
- [ ] Implement `ConversationMemory` class in `agents.py`
- [ ] Add memory integration to `Agent` base class
- [ ] Test with simple conversation history

**Phase 2: Frontend Integration (Week 1-2)**
- [ ] Add conversation UI components to `index.html`
- [ ] Implement localStorage persistence
- [ ] Add conversation selector
- [ ] Test multi-turn conversations

**Phase 3: Enhanced Context (Week 2)**
- [ ] Link simulations to conversation history
- [ ] Add metadata tracking (preferred models, elements)
- [ ] Implement conversation summarization
- [ ] Add reference resolution ("last simulation")

**Phase 4: Optimization (Week 3)**
- [ ] Implement token-aware truncation
- [ ] Add conversation search/filter
- [ ] Performance testing with long conversations
- [ ] Optional: Backend persistence endpoints

### 6.2 Minimal vs Full Implementation

**Minimal (3 files + config)**:
- localStorage-only persistence
- Simple buffer memory (last N messages)
- No backend API changes
- Basic conversation UI

**Full (Recommended)**:
- Backend persistence (JSON files or SQLite)
- Smart truncation (token-aware)
- Conversation metadata and search
- Simulation linking

---

## 7. Migration Checklist

### 7.1 Pre-Migration

- [x] Document current architecture
- [x] Research memory implementation patterns
- [x] Evaluate DSPy vs alternatives
- [ ] Review existing conversation flow
- [ ] Identify breaking changes
- [ ] Plan backward compatibility

### 7.2 Code Changes

#### A. Create `prompts.py`
- [ ] Extract `ORCHESTRATOR_PROMPT` from `agents.py`
- [ ] Extract `LAMMPS_GENERATOR_PROMPT` from `agents.py`
- [ ] Extract `LAMMPS_REVIEWER_PROMPT` from `agents.py`
- [ ] Extract `OVITO_SCRIPT_PROMPT` from `ovito_script_generator.py`
- [ ] Create prompt builder functions with memory templates
- [ ] Add conversation history injection patterns

#### B. Modify `agents.py`
- [ ] Import prompts from `prompts.py`
- [ ] Implement `ConversationMemory` class
  - [ ] `__init__`: Initialize with conversation ID
  - [ ] `add_message()`: Append message with timestamp
  - [ ] `get_recent_context()`: Retrieve last N messages
  - [ ] `get_context_for_agent()`: Format for specific agent
  - [ ] `save()`: Persist to JSON
  - [ ] `load()`: Load from JSON
  - [ ] `truncate()`: Implement sliding window
  - [ ] `summarize_history()`: Generate summary for long histories
  - [ ] `link_simulation()`: Connect run_id to message
  - [ ] `to_dict()` / `from_dict()`: Serialization

- [ ] Modify `Agent` base class
  - [ ] Add `memory` parameter to `__init__`
  - [ ] Update `execute()` to use `_build_messages_with_memory()`
  - [ ] Implement `_build_messages_with_memory()` method
  - [ ] Save user/assistant messages to memory after each turn

- [ ] Update `OrchestratorAgent`
  - [ ] Use `build_orchestrator_prompt()` from `prompts.py`
  - [ ] Add conversation history to Phase 1/2 context
  - [ ] Handle references to "last simulation"

- [ ] Update `LammpsGeneratorAgent`
  - [ ] Use prompt from `prompts.py`
  - [ ] Add memory awareness (optional)

- [ ] Update `LammpsReviewerAgent`
  - [ ] Use prompt from `prompts.py`

#### C. Modify `index.html`
- [ ] **HTML Structure**:
  - [ ] Add conversation history sidebar (collapsible)
  - [ ] Add conversation selector dropdown
  - [ ] Add "New Conversation" button
  - [ ] Add conversation metadata display (date, message count)
  - [ ] Add export conversation button

- [ ] **JavaScript - ConversationManager Class**:
  - [ ] Implement `ConversationManager` class
  - [ ] `loadOrCreateConversation()`: Load active or create new
  - [ ] `saveMessage()`: Save to localStorage
  - [ ] `loadHistory()`: Retrieve conversation messages
  - [ ] `renderHistory()`: Display messages in UI
  - [ ] `switchConversation()`: Load different conversation
  - [ ] `exportConversation()`: Download as JSON
  - [ ] `deleteConversation()`: Remove conversation

- [ ] **Integration with Existing Code**:
  - [ ] Initialize `ConversationManager` on page load
  - [ ] Render existing conversation history
  - [ ] Save user messages before sending to `/api/llm-stream-live`
  - [ ] Save agent responses after receiving via SSE
  - [ ] Pass `conversation_id` in API requests
  - [ ] Update UI when switching conversations

- [ ] **CSS Styling**:
  - [ ] Style conversation sidebar
  - [ ] Style conversation selector
  - [ ] Add conversation timestamp formatting
  - [ ] Responsive design for mobile

#### D. Modify `config.py`
- [ ] Add `MEMORY_ENABLED = True`
- [ ] Add `MAX_CONVERSATION_MESSAGES = 100`
- [ ] Add `CONVERSATION_STORAGE_PATH = "./conversations"`
- [ ] Add `MEMORY_TRUNCATION_STRATEGY = "sliding_window"` # or "token_limit", "summary"
- [ ] Add `MEMORY_BACKEND = "localStorage"` # or "json", "sqlite"

#### E. Modify `app.py` (Optional Backend Persistence)
- [ ] Add conversation persistence endpoints:
  - [ ] `GET /api/conversation/<id>`: Load conversation
  - [ ] `POST /api/conversation/<id>`: Save conversation
  - [ ] `GET /api/conversations`: List all conversations
  - [ ] `DELETE /api/conversation/<id>`: Delete conversation

- [ ] Update `/api/llm-stream-live`:
  - [ ] Accept `conversation_id` parameter
  - [ ] Load `ConversationMemory` for conversation
  - [ ] Pass memory to agents
  - [ ] Save updated memory after workflow

- [ ] Add conversation storage directory initialization
- [ ] Add cleanup for old conversations (optional)

#### F. Modify `run_manager.py` (Optional)
- [ ] Add `conversation_id` field to run metadata
- [ ] Link simulation runs to conversation context
- [ ] Add method to retrieve runs by conversation ID

### 7.3 Testing

- [ ] **Unit Tests**:
  - [ ] `ConversationMemory.add_message()`
  - [ ] `ConversationMemory.get_recent_context()`
  - [ ] `ConversationMemory.truncate()`
  - [ ] `ConversationMemory.save()` / `load()`
  - [ ] Prompt building with memory injection

- [ ] **Integration Tests**:
  - [ ] Multi-turn conversation flow
  - [ ] Reference to "last simulation"
  - [ ] Conversation persistence (localStorage)
  - [ ] Conversation switching
  - [ ] Long conversation handling (truncation)

- [ ] **End-to-End Tests**:
  - [ ] Create conversation → run simulation → reference in next message
  - [ ] Switch between conversations
  - [ ] Export/import conversation
  - [ ] Page reload preserves conversation

- [ ] **Performance Tests**:
  - [ ] 100+ message conversation
  - [ ] Multiple simultaneous conversations
  - [ ] Memory usage with large histories

### 7.4 Documentation

- [ ] Update README with memory features
- [ ] Document `ConversationMemory` API
- [ ] Add examples of multi-turn conversations
- [ ] Document localStorage schema
- [ ] Add troubleshooting guide for memory issues

### 7.5 Deployment

- [ ] Test migration on clean install
- [ ] Verify backward compatibility
- [ ] Update dependencies (if using DSPy)
- [ ] Deploy to staging
- [ ] User acceptance testing
- [ ] Deploy to production
- [ ] Monitor for memory-related issues

---

## 8. Memory Truncation Strategies

### 8.1 Sliding Window (Simplest)

Keep last N messages, discard older ones.

```python
def truncate_sliding_window(self, max_messages=50):
    if len(self.messages) > max_messages:
        self.messages = self.messages[-max_messages:]
```

**Pros**: Simple, predictable
**Cons**: May lose important context

### 8.2 Token-Aware Truncation

Keep messages until token limit reached.

```python
def truncate_by_tokens(self, max_tokens=4000):
    total_tokens = 0
    keep_messages = []

    for msg in reversed(self.messages):
        msg_tokens = estimate_tokens(msg['content'])
        if total_tokens + msg_tokens > max_tokens:
            break
        keep_messages.insert(0, msg)
        total_tokens += msg_tokens

    self.messages = keep_messages
```

**Pros**: Respects model context window
**Cons**: Requires token counting

### 8.3 Summary-Based (Advanced)

Summarize old messages, keep recent ones.

```python
def truncate_with_summary(self, keep_recent=10):
    if len(self.messages) <= keep_recent:
        return

    old_messages = self.messages[:-keep_recent]
    summary = self._generate_summary(old_messages)

    self.messages = [
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *self.messages[-keep_recent:]
    ]
```

**Pros**: Retains context
**Cons**: Requires LLM call, may lose details

**Recommendation**: Start with **Sliding Window**, upgrade to **Token-Aware** if needed.

---

## 9. Reference Resolution

### 9.1 Handling User References

**Examples**:
- "Run the same simulation but at 500K"
- "Use the last KIM model"
- "Visualize the previous run"

**Implementation**:

```python
def resolve_references(self, user_input):
    """Expand references to previous context"""

    # Pattern: "last simulation", "previous run"
    if "last simulation" in user_input.lower() or "previous run" in user_input.lower():
        last_sim = self.get_last_simulation_ref()
        if last_sim:
            return f"{user_input}\n\n[Context: Previous simulation was {last_sim['element']} at {last_sim['temperature']}K for {last_sim['duration_ps']}ps using {last_sim['kim_model']}]"

    # Pattern: "same parameters"
    if "same" in user_input.lower():
        last_params = self.extract_last_parameters()
        if last_params:
            return f"{user_input}\n\n[Context: Previous parameters: {last_params}]"

    return user_input
```

### 9.2 Simulation Linking

```python
# In run_manager.py
def create_run(self, params, conversation_id=None):
    run_id = f"run_{timestamp}"
    metadata = {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "params": params,
        ...
    }
    # Save metadata

# In ConversationMemory
def link_simulation(self, run_id, params):
    self.simulation_refs.append({
        "run_id": run_id,
        "message_index": len(self.messages),
        "timestamp": now(),
        **params
    })
```

---

## 10. Risks and Mitigations

### 10.1 Context Window Limits

**Risk**: Long conversations exceed model context window (e.g., 128k tokens for GPT-4)
**Mitigation**: Implement token-aware truncation, monitor token usage

### 10.2 Memory Bloat

**Risk**: localStorage fills up with large conversation histories
**Mitigation**:
- Implement auto-cleanup for old conversations
- Add export/archive feature
- Set max conversation age (30 days)

### 10.3 Performance Degradation

**Risk**: Loading large histories slows down UI
**Mitigation**:
- Lazy load conversation history
- Paginate message display
- Use virtual scrolling for large histories

### 10.4 Consistency Issues

**Risk**: localStorage out of sync with backend (if using both)
**Mitigation**:
- Choose one source of truth (localStorage XOR backend)
- Or implement sync mechanism with conflict resolution

### 10.5 Privacy Concerns

**Risk**: Sensitive simulation parameters stored in browser
**Mitigation**:
- Add conversation encryption option
- Implement conversation expiry
- Clear conversation on logout

---

## 11. Future Enhancements (Post-MVP)

### 11.1 Semantic Memory with Vector DB

- Retrieve similar past conversations
- "Find all Cu simulations at high temperature"
- Requires ChromaDB/LanceDB + embeddings

### 11.2 DSPy Migration

- Upgrade to DSPy for structured programming
- Use DSPy's optimizers to improve agent performance
- Implement ReAct pattern with memory

### 11.3 Collaborative Conversations

- Share conversations between users
- Comment on simulation results
- Team memory (shared context)

### 11.4 Conversation Analytics

- Visualize conversation patterns
- Track most-used elements/models
- Identify common user workflows

### 11.5 Smart Summarization

- Auto-summarize long conversations
- Generate simulation reports from conversation
- Export conversation as paper draft

---

## 12. Decision Matrix

| Criterion | Simple Buffer | DSPy | LangChain | Vector DB |
|-----------|---------------|------|-----------|-----------|
| **Complexity** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐⭐⭐ Very High |
| **Dependencies** | ✅ None | ~10MB | ~50MB | ~20MB + service |
| **3-File Target** | ✅ Yes | ⚠️ Possible | ❌ No | ❌ No |
| **Maintenance** | ✅ Easy | ⭐⭐ Medium | ⭐⭐⭐ Complex | ⭐⭐⭐ Complex |
| **Extensibility** | ⚠️ Limited | ✅ High | ✅ High | ✅ Very High |
| **Performance** | ✅ Fast | ✅ Fast | ⚠️ Moderate | ⭐⭐ Slower |
| **Learning Curve** | ✅ Minimal | ⭐⭐ Medium | ⭐⭐⭐ Steep | ⭐⭐⭐ Steep |

**Recommendation**: Start with **Simple Buffer**, migrate to **DSPy** if optimization needed.

---

## 13. Final Recommendation

### 13.1 Minimal Implementation (Phase 1)

**Target**: Working multi-turn conversations in 1 week

1. **Create `prompts.py`**: Extract all prompts (~200 lines)
2. **Enhance `agents.py`**: Add `ConversationMemory` class (~150 lines)
3. **Update `index.html`**: Add conversation UI (~200 lines JS)
4. **Modify `config.py`**: Add 5 config variables
5. **Total New Code**: ~550 lines across 3 files

**Dependencies**: None (0 new packages)

### 13.2 Full Implementation (Phase 2)

**Target**: Production-ready with backend persistence in 2-3 weeks

1. Add backend API endpoints in `app.py`
2. Implement token-aware truncation
3. Add simulation linking in `run_manager.py`
4. Build conversation search/export features
5. Add comprehensive tests

---

## 14. Success Metrics

- [ ] Users can reference "the last simulation"
- [ ] Conversations persist across page reloads
- [ ] Can switch between multiple conversations
- [ ] Agent responses include context from history
- [ ] No performance degradation with 100+ message conversations
- [ ] Token usage stays within model limits
- [ ] Users report improved workflow efficiency

---

## 15. Next Steps

1. **Review this plan** with team/stakeholders
2. **Choose implementation approach** (minimal vs full)
3. **Set up development branch** (already on correct branch)
4. **Create `prompts.py`** skeleton
5. **Implement `ConversationMemory`** class
6. **Build prototype** with simple conversation history
7. **Test and iterate**

---

## Appendix A: Example Conversation Flow with Memory

### Without Memory (Current):
```
User: "Run a Cu simulation at 300K"
Agent: [Proposes parameters]
User: "Use the first model"
Agent: [Runs simulation]
--- Session ends ---

User: "Run the same simulation at 500K"
Agent: "I don't have context of previous simulation. Please specify parameters."
```

### With Memory (Proposed):
```
User: "Run a Cu simulation at 300K"
Agent: [Proposes parameters, saves to memory]
User: "Use the first model"
Agent: [Runs simulation, saves model choice and run_id to memory]
--- Session ends, conversation persists ---

User: "Run the same simulation at 500K"
Agent: "I see you previously ran Cu with EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006 at 300K for 10ps. I'll use the same parameters but at 500K."
Agent: [Runs simulation with context from memory]
```

---

## Appendix B: Code Size Estimates

| File | Current Lines | New Lines | Total Lines | Change % |
|------|---------------|-----------|-------------|----------|
| `prompts.py` | 0 | +200 | 200 | NEW |
| `agents.py` | 485 | +150 | 635 | +31% |
| `index.html` | 1,100 | +200 | 1,300 | +18% |
| `config.py` | 36 | +5 | 41 | +14% |
| `app.py` | 1,118 | +50 | 1,168 | +4% (optional) |

**Total New Code**: ~605 lines
**Complexity**: Low to Medium

---

## Appendix C: Prompt Extraction Checklist

- [ ] `ORCHESTRATOR_PROMPT` (~250 lines) → `prompts.py`
- [ ] `LAMMPS_GENERATOR_PROMPT` (~100 lines) → `prompts.py`
- [ ] `LAMMPS_REVIEWER_PROMPT` (~50 lines) → `prompts.py`
- [ ] `OVITO_SCRIPT_PROMPT` (~80 lines from `ovito_script_generator.py`) → `prompts.py`
- [ ] Add memory injection templates for each prompt
- [ ] Create builder functions: `build_orchestrator_prompt(memory=None)`

---

**Document Status**: ✅ Ready for Review
**Last Updated**: 2025-11-16
**Author**: Claude (AI Assistant)
**Approval Pending**: Team Review
