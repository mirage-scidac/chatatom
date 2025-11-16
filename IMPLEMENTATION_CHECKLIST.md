# ChatMD Agentify Implementation Checklist v2

**Project**: Agent-to-Agent Multi-Turn Conversations with Iterative Refinement
**Branch**: `claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n`
**Target**: DSPy-based iterative agent collaboration in 3 core files

---

## Overview

This checklist implements **iterative refinement loops** between agents:
- Generator creates script → Reviewer finds issues → Generator fixes → Reviewer approves
- Up to 3 attempts for LAMMPS, 2 for OVITO
- Full conversation history between agents
- Transparent collaboration visible to users

---

## Phase 1: DSPy Setup & Verification (Day 1)

### 1. Install and Configure DSPy
- [ ] Install DSPy: `pip install dspy-ai`
- [ ] Add to requirements.txt: `dspy-ai>=2.5.0`
- [ ] Test import: `python -c "import dspy; print(dspy.__version__)"`
- [ ] Verify no dependency conflicts

### 2. Test DSPy with Existing LLM Infrastructure
- [ ] Create test script: `test_dspy_connection.py`
- [ ] Configure DSPy to use existing API endpoint:
  ```python
  import dspy
  lm = dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  )
  dspy.configure(lm=lm)
  ```
- [ ] Test simple DSPy signature with local LLM
- [ ] Verify OpenAI-compatible API works with DSPy
- [ ] Test streaming support (if needed)

### 3. Create Simple DSPy Example
- [ ] Write hello-world DSPy signature
- [ ] Test ChainOfThought module
- [ ] Test conversation history handling
- [ ] Verify DSPy.Prediction return type
- [ ] Document any quirks or issues found

---

## Phase 2: Create prompts.py with DSPy Signatures (Days 2-3)

### 4. Extract Existing Prompts
- [ ] Read current LAMMPS generator prompt from `agents.py`
- [ ] Read current LAMMPS reviewer prompt from `agents.py`
- [ ] Read OVITO generator prompt from `ovito_script_generator.py`
- [ ] Document prompt structure and key requirements

### 5. Create DSPy Signatures for LAMMPS
- [ ] Create `prompts.py` file
- [ ] Define `GenerateLAMMPSScript` signature:
  - [ ] `parameters: str` input field
  - [ ] `feedback: str` input field for iteration feedback
  - [ ] `conversation: dspy.History` input field
  - [ ] `script: str` output field
  - [ ] `reasoning: str` output field (explain decisions)
  - [ ] Embed full system prompt in docstring
  - [ ] Test signature independently

- [ ] Define `ReviewLAMMPSScript` signature:
  - [ ] `script: str` input field
  - [ ] `parameters: str` input field
  - [ ] `review: str` output field (REVIEW: PASS/FAIL format)
  - [ ] `issues: list[str]` output field
  - [ ] Embed reviewer prompt in docstring
  - [ ] Test signature independently

### 6. Create DSPy Signatures for OVITO
- [ ] Define `GenerateOVITOScript` signature:
  - [ ] `visualization_request: str` input
  - [ ] `lammps_properties: str` input
  - [ ] `feedback: str` input
  - [ ] `conversation: dspy.History` input
  - [ ] `script: str` output
  - [ ] `reasoning: str` output
  - [ ] Embed OVITO prompt in docstring

- [ ] Define `ReviewOVITOScript` signature (NEW):
  - [ ] `script: str` input
  - [ ] `review: str` output
  - [ ] `issues: list[str]` output
  - [ ] Create review criteria (safety, API usage, etc.)

### 7. Test All Signatures
- [ ] Unit test for `GenerateLAMMPSScript`
- [ ] Unit test for `ReviewLAMMPSScript`
- [ ] Unit test for `GenerateOVITOScript`
- [ ] Unit test for `ReviewOVITOScript`
- [ ] Test with real LLM calls
- [ ] Verify output format matches expectations

---

## Phase 3: Implement Iterative Modules in agents.py (Days 3-5)

### 8. Create IterativeLammpsGeneration Module
- [ ] Create `agents.py` (refactored version)
- [ ] Import DSPy and signatures from `prompts.py`
- [ ] Define `IterativeLammpsGeneration` class:
  - [ ] Inherit from `dspy.Module`
  - [ ] `__init__(self, max_attempts=3)`
  - [ ] Initialize `self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)`
  - [ ] Initialize `self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)`

- [ ] Implement `forward(self, parameters: str)` method:
  - [ ] Initialize empty conversation list
  - [ ] Loop for max_attempts:
    - [ ] Call generator with parameters, feedback, conversation
    - [ ] Add generator output to conversation
    - [ ] Call reviewer with script and parameters
    - [ ] Add reviewer output to conversation
    - [ ] Check if "REVIEW: PASS" in review
    - [ ] If PASS: return success prediction
    - [ ] If FAIL: build feedback message with issues
  - [ ] After max attempts: return failure prediction

- [ ] Return `dspy.Prediction` with:
  - [ ] `script`: final script
  - [ ] `review`: final review text
  - [ ] `attempts`: number of attempts used
  - [ ] `conversation`: full agent dialogue
  - [ ] `success`: True/False

### 9. Create IterativeOVITOGeneration Module
- [ ] Define `IterativeOVITOGeneration` class (similar structure)
- [ ] `__init__(self, max_attempts=2)` (fewer attempts for OVITO)
- [ ] Initialize generator and reviewer with OVITO signatures
- [ ] Implement `forward(self, visualization_request, lammps_properties)`
- [ ] Similar iteration loop as LAMMPS
- [ ] Return prediction with same structure

### 10. Unit Tests for Iterative Modules
- [ ] Test `IterativeLammpsGeneration`:
  - [ ] Scenario: First attempt passes (1 iteration)
  - [ ] Scenario: First fails, second passes (2 iterations)
  - [ ] Scenario: All attempts fail (3 iterations, failure)
  - [ ] Verify conversation history format
  - [ ] Verify feedback construction
  - [ ] Mock LLM responses for consistent testing

- [ ] Test `IterativeOVITOGeneration`:
  - [ ] Similar scenarios as LAMMPS
  - [ ] Test with 2 max attempts

### 11. Add Backward Compatibility Wrappers (Optional)
- [ ] Create `LammpsGeneratorAgent` wrapper class
- [ ] Create `LammpsReviewerAgent` wrapper class
- [ ] Ensure existing tests still pass
- [ ] Document migration path from old to new agents

---

## Phase 4: Integrate into app.py (Days 5-6)

### 12. Update app.py Imports and Setup
- [ ] Import DSPy: `import dspy`
- [ ] Import new modules:
  ```python
  from agents import (
      IterativeLammpsGeneration,
      IterativeOVITOGeneration
  )
  ```
- [ ] Initialize DSPy at app startup:
  ```python
  dspy.configure(lm=dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  ))
  ```
- [ ] Create module instances:
  ```python
  lammps_generator = IterativeLammpsGeneration(max_attempts=3)
  ovito_generator = IterativeOVITOGeneration(max_attempts=2)
  ```

### 13. Replace Linear Flow with Iterative Generation
- [ ] Locate current generator → reviewer flow in `/api/chat`
- [ ] Replace with iterative module call:
  ```python
  result = lammps_generator(parameters=analysis)
  ```
- [ ] Handle success case:
  - [ ] Extract script: `script = result.script`
  - [ ] Log attempts: `log.info(f"Script passed after {result.attempts} attempts")`
  - [ ] Store conversation: `conversation.append(result.conversation)`

- [ ] Handle failure case:
  - [ ] Return error with conversation history
  - [ ] Show all attempts and feedback
  - [ ] Include final review

### 14. Integrate OVITO Iterative Generation
- [ ] Replace existing OVITO script generation
- [ ] Call `ovito_generator(visualization_request, lammps_properties)`
- [ ] Handle success/failure similar to LAMMPS
- [ ] Pass successful script to execution

### 15. Update Error Handling
- [ ] Add handling for `result.success == False`
- [ ] Create detailed error message with all attempts
- [ ] Log conversation history for debugging
- [ ] Return appropriate HTTP status codes

### 16. Update SSE Streaming Endpoint
- [ ] Locate `/api/llm-stream-live` endpoint
- [ ] Stream iteration progress in real-time:
  - [ ] "Generator (Attempt 1/3)..."
  - [ ] "Reviewer feedback: ..."
  - [ ] "Generator (Attempt 2/3)..."
  - [ ] "Reviewer: PASS"
- [ ] Test streaming with iterative generation

### 17. Add Logging and Metrics
- [ ] Log each iteration attempt
- [ ] Track success rate by attempt number
- [ ] Log time taken per iteration
- [ ] Add metrics endpoint (optional): `/api/metrics`

---

## Phase 5: Update Frontend (Days 6-7)

### 18. Add UI for Iteration Progress
- [ ] Create new section in chat UI: "Agent Collaboration"
- [ ] Display iteration counter: "Attempt 1/3, 2/3, 3/3"
- [ ] Add progress bar or spinner during generation
- [ ] Show current step: "Generating...", "Reviewing...", "Refining..."

### 19. Display Agent Conversation
- [ ] Create conversation view component
- [ ] For each message in conversation:
  - [ ] Show role: "Generator (Attempt N)" or "Reviewer (Attempt N)"
  - [ ] Show content (script or review)
  - [ ] Show reasoning if available
  - [ ] Color-code: green for PASS, yellow/red for FAIL

- [ ] Make conversation collapsible/expandable
- [ ] Add syntax highlighting for scripts
- [ ] Format issues list as bullets

### 20. Update Chat Message Display
- [ ] Integrate iteration conversation into chat flow
- [ ] Show compact summary by default
- [ ] "Expand" button to see full agent dialogue
- [ ] Link to specific attempt on click

### 21. Add Visual Indicators
- [ ] Success indicator: ✅ "Script approved on attempt 2/3"
- [ ] Failure indicator: ❌ "Failed after 3 attempts"
- [ ] Show issues found in each attempt
- [ ] Highlight fixes made in subsequent attempts

### 22. CSS Styling
- [ ] Style agent messages differently from user messages
- [ ] Add iteration badges (1/3, 2/3, 3/3)
- [ ] Style review status (PASS/FAIL)
- [ ] Add hover effects for expandable sections
- [ ] Ensure responsive design

### 23. JavaScript Integration
- [ ] Parse conversation array from API response
- [ ] Render conversation with `displayIterativeGeneration()`
- [ ] Handle streaming updates for iteration progress
- [ ] Update UI in real-time as attempts happen
- [ ] Test with various iteration scenarios

---

## Phase 6: Configuration Updates (Day 7)

### 24. Update config.py
- [ ] Add `MAX_LAMMPS_GENERATION_ATTEMPTS = 3`
- [ ] Add `MAX_OVITO_GENERATION_ATTEMPTS = 2`
- [ ] Add `DSPY_CACHE_ENABLED = True`
- [ ] Add `DSPY_LOG_LEVEL = "INFO"`
- [ ] Add `ENABLE_ITERATIVE_GENERATION = True` (feature flag)
- [ ] Add `ITERATION_TIMEOUT_SECONDS = 180` (per attempt)
- [ ] Document each config option

### 25. Environment-Specific Config
- [ ] Create `config.dev.py` with lower max_attempts (faster testing)
- [ ] Create `config.prod.py` with production values
- [ ] Add environment variable overrides
- [ ] Document config precedence

---

## Phase 7: Testing (Days 8-9)

### 26. Unit Tests for DSPy Modules
- [ ] `test_iterative_lammps_generation.py`:
  - [ ] Test first attempt success
  - [ ] Test second attempt success after failure
  - [ ] Test third attempt success
  - [ ] Test all attempts fail
  - [ ] Test conversation history structure
  - [ ] Test feedback message construction
  - [ ] Mock LLM responses for deterministic tests

- [ ] `test_iterative_ovito_generation.py`:
  - [ ] Similar test cases as LAMMPS
  - [ ] Test with 2 max attempts

### 27. Integration Tests
- [ ] `test_full_workflow_with_iteration.py`:
  - [ ] User request → Orchestrator → Iterative LAMMPS → Iterative OVITO → Execution
  - [ ] Test with intentionally failing script on attempt 1
  - [ ] Verify script passes on attempt 2
  - [ ] Verify execution succeeds

- [ ] Test error scenarios:
  - [ ] All attempts fail → graceful error
  - [ ] LAMMPS passes but OVITO fails
  - [ ] Network errors during iteration
  - [ ] Timeout during generation

### 28. End-to-End Tests
- [ ] Scenario 1: Missing velocity initialization
  - [ ] Generate script without velocity init
  - [ ] Reviewer catches it on attempt 1
  - [ ] Generator adds it on attempt 2
  - [ ] Reviewer approves

- [ ] Scenario 2: Wrong KIM command order
  - [ ] Generate script with KIM after create_box
  - [ ] Reviewer catches it
  - [ ] Generator fixes order
  - [ ] Passes on attempt 2

- [ ] Scenario 3: First attempt perfect
  - [ ] Generate correct script immediately
  - [ ] Passes on attempt 1
  - [ ] No performance regression

- [ ] Scenario 4: Max attempts exceeded
  - [ ] Generate script with persistent issues
  - [ ] Fails all 3 attempts
  - [ ] Returns meaningful error

### 29. Performance Tests
- [ ] Measure latency per iteration
- [ ] Test with concurrent requests
- [ ] Monitor memory usage during iteration
- [ ] Test DSPy caching effectiveness
- [ ] Benchmark vs old linear approach

### 30. User Acceptance Testing
- [ ] Test with real user workflows
- [ ] Verify UI clarity (can users understand iterations?)
- [ ] Test on different browsers
- [ ] Mobile responsiveness
- [ ] Gather feedback on iteration visibility

---

## Phase 8: Documentation & Deployment (Day 10)

### 31. Update Documentation
- [ ] Update README.md:
  - [ ] Add DSPy dependency
  - [ ] Explain iterative generation
  - [ ] Show example of iteration flow
  - [ ] Update installation instructions

- [ ] Create `AGENT_COLLABORATION.md`:
  - [ ] Explain generator ↔ reviewer pattern
  - [ ] Document max_attempts configuration
  - [ ] Show example conversations
  - [ ] Troubleshooting guide

- [ ] Update API documentation:
  - [ ] New response format with conversation
  - [ ] Iteration fields in JSON
  - [ ] Error codes for failed iterations

### 32. Create Migration Guide
- [ ] Document changes from v1 to v2
- [ ] Explain new conversation format
- [ ] Backward compatibility notes
- [ ] Migration checklist for existing users

### 33. Add Code Examples
- [ ] Example: Using IterativeLammpsGeneration directly
- [ ] Example: Custom iteration strategies
- [ ] Example: Parsing conversation history
- [ ] Example: Adjusting max_attempts per request

### 34. Deployment Preparation
- [ ] Create deployment checklist
- [ ] Update requirements.txt
- [ ] Test on clean install
- [ ] Verify no breaking changes for existing users
- [ ] Create rollback plan

### 35. Deploy to Staging
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Test all iteration scenarios
- [ ] Monitor for errors
- [ ] Performance testing under load

### 36. Production Deployment
- [ ] Deploy to production
- [ ] Monitor error rates
- [ ] Track iteration success rates
- [ ] Monitor API latency
- [ ] Watch for user feedback

---

## Validation Criteria

### Must Have (MVP):
- ✅ Generator and reviewer iterate up to 3 times
- ✅ Failed scripts can be automatically fixed
- ✅ Full conversation history captured
- ✅ UI shows iteration progress
- ✅ No breaking changes to existing functionality
- ✅ Success rate improvement measurable

### Should Have:
- ✅ Real-time streaming of iteration progress
- ✅ OVITO script iteration (2 attempts)
- ✅ Detailed error messages with full conversation
- ✅ Configurable max_attempts per environment

### Nice to Have:
- ⭐ DSPy optimization (compile prompts)
- ⭐ Conversation history persistence
- ⭐ Analytics dashboard for iteration metrics
- ⭐ User feedback on specific attempts

---

## Success Metrics

### Target Goals:
- [ ] 90%+ of scripts pass by attempt 2
- [ ] Average iteration count: ≤ 1.5
- [ ] 70%+ of failed scripts fixed automatically
- [ ] Latency increase: < 2x for scripts needing iteration
- [ ] Zero increase for scripts passing on attempt 1
- [ ] User satisfaction improvement (survey)

### Tracking:
- [ ] Log iteration counts per request
- [ ] Track time per iteration
- [ ] Monitor PASS/FAIL rate by attempt
- [ ] Compare success rates before/after
- [ ] Track most common issues caught by reviewer

---

## Risk Mitigation Checklist

### Technical Risks:
- [ ] Infinite loop protection: Hard max_attempts limit implemented
- [ ] Latency: Stream progress to maintain responsiveness
- [ ] API costs: Monitor and set budget alerts
- [ ] DSPy compatibility: Thoroughly test with OpenAI-compatible API

### Deployment Risks:
- [ ] Backward compatibility: Test with existing workflows
- [ ] Performance regression: Benchmark critical paths
- [ ] Error handling: Comprehensive try/catch blocks
- [ ] Rollback plan: Document quick revert process

### User Experience Risks:
- [ ] Confusion about iterations: Clear UI explanations
- [ ] Longer wait times: Show progress clearly
- [ ] Overwhelming conversation view: Collapsible by default
- [ ] Error fatigue: Meaningful final error message

---

## Code Review Checklist

### Before Merging:
- [ ] All tests passing (unit + integration + E2E)
- [ ] DSPy signatures properly documented
- [ ] No hardcoded values (use config)
- [ ] Error handling comprehensive
- [ ] Logging added at appropriate levels
- [ ] No security vulnerabilities (script injection, etc.)
- [ ] Performance acceptable (< 2x latency increase)
- [ ] Code follows existing style
- [ ] No console errors in frontend
- [ ] Mobile responsive

### Documentation:
- [ ] All new functions have docstrings
- [ ] README updated
- [ ] Migration guide created
- [ ] API documentation current
- [ ] Examples provided

---

## Progress Tracking

**Started**: [DATE]
**Target Completion**: [DATE + 10 days]

### Phase Completion:
- [ ] Phase 1: DSPy Setup (Day 1)
- [ ] Phase 2: Signatures (Days 2-3)
- [ ] Phase 3: Modules (Days 3-5)
- [ ] Phase 4: Integration (Days 5-6)
- [ ] Phase 5: Frontend (Days 6-7)
- [ ] Phase 6: Config (Day 7)
- [ ] Phase 7: Testing (Days 8-9)
- [ ] Phase 8: Deployment (Day 10)

**Overall Progress**: 0/117 items completed

---

## Notes & Decisions

### Key Decisions:
- ✅ Use DSPy for agent collaboration (not manual iteration)
- ✅ 3 attempts for LAMMPS, 2 for OVITO
- ✅ Stream iteration progress in real-time
- ✅ Full conversation history visible to users
- ✅ Backward compatible wrappers for existing code

### Open Questions:
- [ ] Should we allow user to configure max_attempts per request?
- [ ] Should failed attempts be saved for analysis?
- [ ] Should we add A/B testing for iteration vs non-iteration?

### Issues Encountered:
- [ ] Issue #1: [Description]
- [ ] Issue #2: [Description]

---

## Quick Start Checklist (First Day)

If starting immediately, do these first:
1. [ ] Install DSPy: `pip install dspy-ai`
2. [ ] Test DSPy with local LLM
3. [ ] Create `prompts.py` skeleton
4. [ ] Define first signature: `GenerateLAMMPSScript`
5. [ ] Test signature with real LLM call
6. [ ] Create simple iteration loop (no DSPy) as proof of concept

---

**Last Updated**: 2025-11-16 (v2 - Agent Collaboration Focus)
**Status**: ✅ Ready to Start Implementation
**Estimated Timeline**: 10 working days
