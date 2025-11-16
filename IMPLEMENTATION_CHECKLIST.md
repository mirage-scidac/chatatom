# ChatMD Agentify Implementation Checklist

**Project**: Multi-Turn Conversations and Memory
**Branch**: `claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n`
**Target**: 3-File Minimal Implementation

---

## Phase 1: Core Memory Implementation (Week 1)

### 1. Create `prompts.py` File
- [ ] Extract `ORCHESTRATOR_PROMPT` from `agents.py` (~250 lines)
- [ ] Extract `LAMMPS_GENERATOR_PROMPT` from `agents.py` (~100 lines)
- [ ] Extract `LAMMPS_REVIEWER_PROMPT` from `agents.py` (~50 lines)
- [ ] Extract `OVITO_SCRIPT_PROMPT` from `ovito_script_generator.py` (~80 lines)
- [ ] Create `build_orchestrator_prompt(memory=None)` function
- [ ] Create `build_generator_prompt(memory=None)` function
- [ ] Create `build_reviewer_prompt(memory=None)` function
- [ ] Add memory injection templates to prompts
- [ ] Test prompt extraction doesn't break existing functionality

### 2. Implement ConversationMemory Class in `agents.py`
- [ ] Create `ConversationMemory` class skeleton
- [ ] Implement `__init__(conversation_id=None, max_messages=100)`
- [ ] Implement `add_message(role, content, agent=None)`
- [ ] Implement `get_recent_context(n=10)`
- [ ] Implement `get_context_for_agent(agent_type)`
- [ ] Implement `save(filepath=None)` - persist to JSON
- [ ] Implement `load(filepath_or_id)` - static method to load from JSON
- [ ] Implement `truncate(strategy="sliding_window")`
- [ ] Implement `summarize_history()` for long conversations
- [ ] Implement `link_simulation(run_id, params)`
- [ ] Implement `to_dict()` and `from_dict()` for serialization
- [ ] Add unit tests for `ConversationMemory` class

### 3. Modify Agent Base Class in `agents.py`
- [ ] Add `memory` parameter to `Agent.__init__()`
- [ ] Import prompts from `prompts.py`
- [ ] Create `_build_messages_with_memory(user_input, context)` method
- [ ] Update `execute()` to use `_build_messages_with_memory()`
- [ ] Update `execute_stream()` to use `_build_messages_with_memory()`
- [ ] Save user messages to memory before LLM call
- [ ] Save assistant responses to memory after LLM call
- [ ] Test agents work with and without memory

### 4. Update Specific Agents
- [ ] `OrchestratorAgent`: Use `build_orchestrator_prompt(memory)` from prompts.py
- [ ] `OrchestratorAgent`: Add handling for "last simulation" references
- [ ] `LammpsGeneratorAgent`: Use prompt from prompts.py
- [ ] `LammpsReviewerAgent`: Use prompt from prompts.py
- [ ] Test each agent with conversation memory

### 5. Update `config.py`
- [ ] Add `MEMORY_ENABLED = True`
- [ ] Add `MAX_CONVERSATION_MESSAGES = 100`
- [ ] Add `CONVERSATION_STORAGE_PATH = "./conversations"`
- [ ] Add `MEMORY_TRUNCATION_STRATEGY = "sliding_window"`
- [ ] Add `MEMORY_BACKEND = "localStorage"` (or "json")

---

## Phase 2: Frontend Integration (Week 1-2)

### 6. HTML Structure Updates in `index.html`
- [ ] Add conversation history sidebar (collapsible panel)
- [ ] Add conversation selector dropdown
- [ ] Add "New Conversation" button
- [ ] Add "Export Conversation" button
- [ ] Add conversation metadata display (date, message count)
- [ ] Add CSS styling for conversation UI
- [ ] Make UI responsive for mobile

### 7. JavaScript ConversationManager in `index.html`
- [ ] Create `ConversationManager` class
- [ ] Implement `constructor()` - initialize with current conversation
- [ ] Implement `loadOrCreateConversation()` - load from localStorage or create new
- [ ] Implement `saveMessage(role, content, agent=null)` - save to localStorage
- [ ] Implement `loadHistory()` - retrieve conversation messages
- [ ] Implement `renderHistory()` - display messages in UI
- [ ] Implement `switchConversation(conversationId)` - load different conversation
- [ ] Implement `exportConversation()` - download as JSON
- [ ] Implement `deleteConversation(conversationId)` - remove conversation
- [ ] Implement `listConversations()` - get all stored conversations

### 8. Integrate ConversationManager with Existing Code
- [ ] Initialize `ConversationManager` on page load
- [ ] Render existing conversation history on load
- [ ] Save user messages before sending to API
- [ ] Save agent responses after receiving via SSE
- [ ] Pass `conversation_id` in `/api/llm-stream-live` requests
- [ ] Update UI when switching conversations
- [ ] Handle page reload - preserve conversation
- [ ] Test localStorage persistence

---

## Phase 3: Backend Integration (Optional - Week 2)

### 9. API Endpoints in `app.py` (Optional)
- [ ] Add `GET /api/conversation/<id>` - load conversation
- [ ] Add `POST /api/conversation/<id>` - save conversation
- [ ] Add `GET /api/conversations` - list all conversations
- [ ] Add `DELETE /api/conversation/<id>` - delete conversation
- [ ] Create conversations storage directory on startup
- [ ] Test endpoints with curl/Postman

### 10. Update Workflow in `app.py`
- [ ] Accept `conversation_id` parameter in `/api/llm-stream-live`
- [ ] Load `ConversationMemory` for conversation_id
- [ ] Pass memory to OrchestratorAgent
- [ ] Pass memory to LammpsGeneratorAgent
- [ ] Pass memory to LammpsReviewerAgent
- [ ] Save updated memory after workflow completes
- [ ] Test multi-turn conversation flow end-to-end

### 11. Link Simulations to Conversations in `run_manager.py` (Optional)
- [ ] Add `conversation_id` field to run metadata
- [ ] Save conversation_id when creating run
- [ ] Add method `get_runs_by_conversation(conversation_id)`
- [ ] Test simulation linking

---

## Phase 4: Testing (Week 2-3)

### 12. Unit Tests
- [ ] Test `ConversationMemory.add_message()`
- [ ] Test `ConversationMemory.get_recent_context()`
- [ ] Test `ConversationMemory.truncate()` with different strategies
- [ ] Test `ConversationMemory.save()` and `load()`
- [ ] Test `ConversationMemory.link_simulation()`
- [ ] Test prompt building with memory injection
- [ ] Test `Agent._build_messages_with_memory()`

### 13. Integration Tests
- [ ] Test multi-turn conversation (3+ messages)
- [ ] Test reference to "last simulation" works correctly
- [ ] Test conversation persistence (save and load)
- [ ] Test conversation switching
- [ ] Test truncation with 100+ message conversation
- [ ] Test localStorage quota handling

### 14. End-to-End Tests
- [ ] Scenario: Create conversation → run simulation → reference in next message
- [ ] Scenario: Switch between multiple conversations
- [ ] Scenario: Export conversation as JSON
- [ ] Scenario: Page reload preserves conversation
- [ ] Scenario: Long conversation (50+ messages) performance
- [ ] Scenario: "Use the same parameters but at 500K" works correctly

### 15. Performance Tests
- [ ] Measure latency with conversation history (0, 10, 50, 100 messages)
- [ ] Test memory usage with large histories
- [ ] Test UI responsiveness with long conversations
- [ ] Monitor localStorage size growth

---

## Phase 5: Documentation & Deployment

### 16. Documentation
- [ ] Update README.md with memory features
- [ ] Document `ConversationMemory` API in docstrings
- [ ] Add examples of multi-turn conversations
- [ ] Document localStorage schema
- [ ] Add troubleshooting guide for memory issues
- [ ] Create user guide for conversation management

### 17. Deployment Preparation
- [ ] Test migration on clean install
- [ ] Verify backward compatibility (no breaking changes)
- [ ] Create migration script if needed
- [ ] Update dependencies in requirements.txt (if using DSPy)
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Create rollback plan

### 18. Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests on staging
- [ ] User acceptance testing
- [ ] Fix any issues found in staging
- [ ] Deploy to production
- [ ] Monitor for errors/issues
- [ ] Gather user feedback

---

## Validation Criteria

**Must Have (MVP)**:
- ✅ Users can have multi-turn conversations
- ✅ Conversations persist across page reloads
- ✅ Can reference "last simulation"
- ✅ Conversation history visible in UI
- ✅ No breaking changes to existing functionality

**Should Have**:
- ✅ Multiple concurrent conversations
- ✅ Conversation switching
- ✅ Export conversation feature
- ✅ Automatic truncation for long conversations

**Nice to Have**:
- ⭐ Backend persistence (in addition to localStorage)
- ⭐ Conversation search
- ⭐ Conversation analytics
- ⭐ Smart summarization

---

## Risk Mitigation Checklist

- [ ] Context window overflow - implement token-aware truncation
- [ ] localStorage quota exceeded - add cleanup mechanism
- [ ] Performance degradation - lazy load history
- [ ] Backward compatibility - test with existing users
- [ ] Privacy concerns - add conversation encryption option
- [ ] Memory leaks - monitor memory usage in long sessions

---

## Code Review Checklist

- [ ] All prompts extracted to `prompts.py`
- [ ] No hardcoded prompts remain in `agents.py`
- [ ] `ConversationMemory` has comprehensive docstrings
- [ ] Error handling for localStorage failures
- [ ] Graceful fallback if memory disabled
- [ ] No breaking changes to existing API
- [ ] Code follows existing style/conventions
- [ ] All tests passing
- [ ] No console errors in frontend
- [ ] No security vulnerabilities introduced

---

## Progress Tracking

**Started**: [DATE]
**Target Completion**: [DATE]

**Phase 1**: [ ] Not Started / [ ] In Progress / [ ] Complete
**Phase 2**: [ ] Not Started / [ ] In Progress / [ ] Complete
**Phase 3**: [ ] Not Started / [ ] In Progress / [ ] Complete
**Phase 4**: [ ] Not Started / [ ] In Progress / [ ] Complete
**Phase 5**: [ ] Not Started / [ ] In Progress / [ ] Complete

**Overall Progress**: 0% (0/89 items completed)

---

## Notes & Issues

### Issues Encountered:
- [ ] Issue #1: [Description]
- [ ] Issue #2: [Description]

### Decisions Made:
- [ ] Decision: Use Simple Buffer Memory (not DSPy) for Phase 1
- [ ] Decision: localStorage as primary storage (backend optional)
- [ ] Decision: Sliding window truncation strategy

### Open Questions:
- [ ] Should we implement backend persistence in Phase 1?
- [ ] What token limit should we use for truncation?
- [ ] Should conversations auto-expire after N days?

---

**Last Updated**: 2025-11-16
**Status**: ✅ Ready to Start Implementation
