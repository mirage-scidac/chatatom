# ChatMD Agentify Implementation Checklist v4 (LEAN)

**Project**: Single-User Agent Collaboration + Simple Memory + Checkpoints
**Branch**: `claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n`
**Target**: Minimal implementation with maximum value

---

## Overview

**What We're Building** (Same powerful features):
- Iterative agent collaboration (Generator ↔ Reviewer)
- Agent memory (learn from feedback)
- User checkpoints (approve before execution)

**What We're Simplifying**:
- Single user only (no auth/isolation)
- JSON file storage (no vector DB)
- Simple category filtering (no semantic search)
- In-memory sessions (no Redis)

**Result**: ~400 lines of code, 7-day implementation, 1 new dependency

---

## Week 1: Core Implementation (Days 1-7)

### Day 1: DSPy Setup

#### 1. Install DSPy
- [ ] Run: `pip install dspy-ai`
- [ ] Add to requirements.txt: `dspy-ai>=2.5.0`
- [ ] Test import: `python -c "import dspy; print(dspy.__version__)"`
- [ ] No dependency conflicts

#### 2. Test DSPy with Local LLM
- [ ] Create test script `test_dspy.py`:
  ```python
  import dspy
  import config

  lm = dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  )
  dspy.configure(lm=lm)

  class SimpleTest(dspy.Signature):
      question: str = dspy.InputField()
      answer: str = dspy.OutputField()

  predictor = dspy.ChainOfThought(SimpleTest)
  result = predictor(question="What is 2+2?")
  print(result.answer)
  ```
- [ ] Verify it works with your LLM
- [ ] Test streaming (if needed)

#### 3. Understand DSPy Basics
- [ ] Read DSPy signature documentation
- [ ] Test ChainOfThought module
- [ ] Test Prediction return type
- [ ] Document any quirks

**End of Day 1**: DSPy working with local LLM ✓

---

### Day 2: Simple Memory System

#### 4. Create SimpleMemory Class
- [ ] Open `agents.py`
- [ ] Add imports:
  ```python
  import json
  import os
  from datetime import datetime
  import uuid
  ```
- [ ] Add `SimpleMemory` class (~50 lines):
  ```python
  class SimpleMemory:
      def __init__(self, filepath="memory.json"):
          # ...

      def _load(self):
          # Load from JSON file

      def _save(self):
          # Save to JSON file

      def add(self, type, category, content, context=None):
          # Add new memory

      def search(self, categories=None, limit=5):
          # Filter by category, sort by reinforcements

      def reinforce(self, memory_id):
          # Increment reinforcement count
  ```
- [ ] See full implementation in v4 migration plan

#### 5. Test Memory System
- [ ] Test `add()`:
  ```python
  mem = SimpleMemory()
  mem.add("preference", "equilibration", "User prefers 2ps equilibration")
  ```
- [ ] Test `search()`:
  ```python
  results = mem.search(categories=["preference"], limit=5)
  assert len(results) > 0
  ```
- [ ] Test `reinforce()`:
  ```python
  mem.reinforce("mem_001")
  # Check that reinforcements increased
  ```
- [ ] Test persistence (save and reload)
- [ ] Verify `memory.json` created correctly

#### 6. Create Initial Memory Template
- [ ] Create `memory.json` with examples:
  ```json
  {
    "memories": [
      {
        "id": "mem_001",
        "type": "preference",
        "category": "kim_model",
        "content": "User prefers EAM_Dynamo_AcklandTichyVitek_1987_Cu for Cu",
        "reinforcements": 1,
        "context": {"element": "Cu"},
        "created": "2025-11-16T00:00:00Z"
      }
    ]
  }
  ```
- [ ] Test loading this template

**End of Day 2**: Memory system working ✓

---

### Day 3: DSPy Signatures & Iteration

#### 7. Create prompts.py
- [ ] Create `prompts.py` file
- [ ] Import DSPy: `import dspy`
- [ ] Extract existing LAMMPS generator prompt from `agents.py`
- [ ] Extract existing LAMMPS reviewer prompt from `agents.py`

#### 8. Define DSPy Signatures
- [ ] Create `GenerateLAMMPSScript` signature:
  ```python
  class GenerateLAMMPSScript(dspy.Signature):
      """Generate LAMMPS script using KIM potentials"""

      parameters: str = dspy.InputField(
          desc="Simulation parameters from orchestrator"
      )

      memories: str = dspy.InputField(
          desc="Past learnings and preferences"
      )

      user_feedback: str = dspy.InputField(
          desc="Specific changes from checkpoint"
      )

      feedback: str = dspy.InputField(
          desc="Reviewer feedback from previous attempts"
      )

      script: str = dspy.OutputField(
          desc="Complete executable LAMMPS script"
      )

      reasoning: str = dspy.OutputField(
          desc="Explanation of key decisions"
      )
  ```
- [ ] Embed full generator prompt in signature docstring
- [ ] Test signature with sample inputs

- [ ] Create `ReviewLAMMPSScript` signature:
  ```python
  class ReviewLAMMPSScript(dspy.Signature):
      """Review LAMMPS script for correctness"""

      script: str = dspy.InputField()
      parameters: str = dspy.InputField()

      review: str = dspy.OutputField(
          desc="REVIEW: PASS or REVIEW: FAIL with issues"
      )

      issues: list[str] = dspy.OutputField(
          desc="List of specific issues"
      )
  ```
- [ ] Embed reviewer prompt in docstring
- [ ] Test signature

#### 9. Create Iterative Module
- [ ] In `agents.py`, add `IterativeLammpsGeneration` class:
  ```python
  class IterativeLammpsGeneration(dspy.Module):
      def __init__(self, memory, max_attempts=3):
          super().__init__()
          self.memory = memory
          self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
          self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
          self.max_attempts = max_attempts

      def forward(self, parameters, user_feedback=""):
          # Get memories
          memories = self.memory.search(
              categories=['preference', 'correction'],
              limit=5
          )
          memory_text = self._format_memories(memories)

          # Iteration loop
          for attempt in range(1, self.max_attempts + 1):
              # Generate
              result = self.generator(
                  parameters=parameters,
                  memories=memory_text,
                  user_feedback=user_feedback,
                  feedback=feedback
              )

              # Review
              review = self.reviewer(
                  script=result.script,
                  parameters=parameters
              )

              # Check pass/fail
              if "REVIEW: PASS" in review.review:
                  # Reinforce memories
                  for mem in memories:
                      self.memory.reinforce(mem['id'])

                  return dspy.Prediction(
                      script=result.script,
                      review=review.review,
                      attempts=attempt,
                      success=True
                  )

              feedback = f"Issues: {', '.join(review.issues)}"

          return dspy.Prediction(success=False, ...)

      def _format_memories(self, memories):
          if not memories:
              return "No previous learnings."

          text = "Learnings from past interactions:\n"
          for m in memories:
              text += f"- {m['content']} (used {m['reinforcements']}x)\n"
          return text
  ```

#### 10. Test Iteration
- [ ] Create test script:
  ```python
  memory = SimpleMemory()
  generator = IterativeLammpsGeneration(memory, max_attempts=3)

  # Initialize DSPy
  dspy.configure(lm=dspy.LM(...))

  # Test with simple parameters
  result = generator(parameters="Cu at 300K for 10ps")

  assert result.success
  assert "REVIEW: PASS" in result.review
  print(f"Passed after {result.attempts} attempts")
  ```
- [ ] Verify iteration works
- [ ] Verify memory reinforcement happens
- [ ] Test with failing script (should iterate)

**End of Day 3**: Iteration + Memory working ✓

---

### Day 4: Checkpoint Backend

#### 11. Add Session Storage to app.py
- [ ] At top of `app.py`, add:
  ```python
  from agents import SimpleMemory, IterativeLammpsGeneration
  import dspy
  import uuid

  # Initialize memory
  memory = SimpleMemory("memory.json")

  # Initialize DSPy
  dspy.configure(lm=dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  ))

  # Initialize generator
  lammps_generator = IterativeLammpsGeneration(
      memory=memory,
      max_attempts=3
  )

  # Session storage
  sessions = {}
  ```

#### 12. Create /api/workflow/start Endpoint
- [ ] Add endpoint:
  ```python
  @app.route('/api/workflow/start', methods=['POST'])
  def start_workflow():
      """Start workflow with parameter proposal"""
      request_text = request.json['request']

      # Get relevant memories
      memories = memory.search(
          categories=['preference', 'kim_model', 'equilibration'],
          limit=5
      )

      # Get orchestrator (existing code)
      orchestrator = OrchestratorAgent(...)

      # Propose parameters (with memory context)
      memory_context = "\n".join([m['content'] for m in memories])
      proposal = orchestrator.execute(
          f"{request_text}\n\nUser preferences: {memory_context}"
      )

      # Create session
      session_id = str(uuid.uuid4())
      sessions[session_id] = {
          "request": request_text,
          "stage": "parameters",
          "proposal": proposal,
          "memories": memories
      }

      return jsonify({
          "session_id": session_id,
          "checkpoint": "parameters",
          "proposal": proposal,
          "memories": [m['content'] for m in memories]
      })
  ```
- [ ] Test endpoint with curl/Postman

#### 13. Create /api/checkpoint/parameters Endpoint
- [ ] Add endpoint:
  ```python
  @app.route('/api/checkpoint/parameters', methods=['POST'])
  def checkpoint_parameters():
      """Handle parameter approval/modification"""
      session_id = request.json['session_id']
      action = request.json['action']  # "approve" or "modify"

      session = sessions[session_id]

      if action == "approve":
          # Store success pattern
          memory.add(
              type="success_pattern",
              category="parameters",
              content=f"User approved: {session['proposal']}"
          )

          # Run generation
          result = lammps_generator(
              parameters=session['proposal'],
              user_feedback=""
          )

          # Update session
          session['stage'] = "script"
          session['script'] = result.script
          session['attempts'] = result.attempts
          session['conversation'] = result.conversation

          return jsonify({
              "checkpoint": "script",
              "script": result.script,
              "attempts": result.attempts,
              "success": result.success
          })

      elif action == "modify":
          feedback = request.json['feedback']

          # Store preference
          memory.add(
              type="preference",
              category="parameters",
              content=feedback
          )

          # Re-propose (simplified: just return modified)
          return jsonify({
              "checkpoint": "parameters",
              "proposal": f"{session['proposal']}\n\nModified: {feedback}"
          })
  ```
- [ ] Test approve action
- [ ] Test modify action

#### 14. Create /api/checkpoint/script Endpoint
- [ ] Add endpoint:
  ```python
  @app.route('/api/checkpoint/script', methods=['POST'])
  def checkpoint_script():
      """Handle script approval/changes"""
      session_id = request.json['session_id']
      action = request.json['action']  # "approve" or "request_changes"

      session = sessions[session_id]

      if action == "approve":
          # User approved - execute!
          script = session['script']

          # Run simulation (existing code)
          from tools import run_lammps_simulation
          result = run_lammps_simulation(script)

          return jsonify({
              "status": "executing",
              "result": result
          })

      elif action == "request_changes":
          feedback = request.json['feedback']

          # Store correction
          memory.add(
              type="correction",
              category="script_generation",
              content=feedback
          )

          # Re-generate with user feedback
          result = lammps_generator(
              parameters=session['proposal'],
              user_feedback=feedback  # User's specific request!
          )

          # Update session
          session['script'] = result.script
          session['attempts'] = result.attempts

          return jsonify({
              "checkpoint": "script",
              "script": result.script,
              "attempts": result.attempts,
              "success": result.success
          })
  ```
- [ ] Test approve action
- [ ] Test request_changes action

#### 15. Test Full Backend Flow
- [ ] Test: start → approve params → approve script
- [ ] Test: start → modify params → approve script
- [ ] Test: start → approve params → request changes → approve script
- [ ] Verify memory gets updated correctly
- [ ] Verify sessions work correctly

**End of Day 4**: Checkpoint backend complete ✓

---

### Day 5: Integration Testing

#### 16. End-to-End Tests
- [ ] Test Scenario 1: First-time user (no memories)
  - [ ] Start workflow
  - [ ] Approve parameters
  - [ ] Script generates correctly
  - [ ] Approve script
  - [ ] Simulation executes
  - [ ] Verify memory created

- [ ] Test Scenario 2: Returning user (with memories)
  - [ ] Add some test memories manually
  - [ ] Start workflow
  - [ ] Verify memories shown in proposal
  - [ ] Approve
  - [ ] Verify memories reinforced

- [ ] Test Scenario 3: User requests changes
  - [ ] Start workflow
  - [ ] Modify parameters
  - [ ] Verify preference stored
  - [ ] Approve script
  - [ ] Request changes to script
  - [ ] Verify correction stored
  - [ ] Approve updated script

#### 17. Memory Learning Tests
- [ ] Test memory accumulation:
  - [ ] Run 3 workflows
  - [ ] Check `memory.json` has memories
  - [ ] Verify reinforcement counts increase

- [ ] Test memory retrieval:
  - [ ] Add 10+ test memories
  - [ ] Search by category
  - [ ] Verify sorted by reinforcements

- [ ] Test memory persistence:
  - [ ] Restart server
  - [ ] Verify memories still there

#### 18. Bug Fixes
- [ ] Fix any issues found in testing
- [ ] Handle edge cases:
  - [ ] Empty memories
  - [ ] Missing session_id
  - [ ] Invalid actions
- [ ] Add error handling

**End of Day 5**: Backend fully tested ✓

---

### Day 6: Checkpoint Frontend UI

#### 19. Create Modal HTML Structure
- [ ] In `index.html`, add modal container:
  ```html
  <div id="checkpoint-modal" style="display:none">
      <!-- Content injected by JavaScript -->
  </div>
  ```

#### 20. Add Modal CSS
- [ ] Add styles:
  ```css
  #checkpoint-modal {
      position: fixed;
      top: 0; left: 0;
      width: 100%; height: 100%;
      background: rgba(0,0,0,0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
  }

  .modal-content {
      background: white;
      padding: 2rem;
      border-radius: 8px;
      max-width: 800px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  }

  .modal-content h2 {
      margin-top: 0;
  }

  .memories-used {
      background: #f5f5f5;
      padding: 1rem;
      border-radius: 4px;
      margin: 1rem 0;
  }

  .memory-tag {
      display: inline-block;
      background: #e3f2fd;
      color: #1976d2;
      padding: 0.5rem 1rem;
      margin: 0.25rem;
      border-radius: 4px;
      font-size: 0.9rem;
  }

  .proposal, .script {
      background: #f9f9f9;
      padding: 1rem;
      border-radius: 4px;
      margin: 1rem 0;
      font-family: monospace;
  }

  .actions {
      display: flex;
      gap: 1rem;
      margin-top: 1.5rem;
  }

  .actions button {
      flex: 1;
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1rem;
  }

  .actions button:first-child {
      background: #4caf50;
      color: white;
  }

  .actions button:hover {
      opacity: 0.9;
  }
  ```

#### 21. Create CheckpointModal Class
- [ ] Add JavaScript:
  ```javascript
  class CheckpointModal {
      constructor(type, data) {
          this.type = type;
          this.data = data;
      }

      show() {
          const modal = document.getElementById('checkpoint-modal');
          const content = this.type === 'parameters'
              ? this.renderParameters()
              : this.renderScript();

          modal.innerHTML = content;
          modal.style.display = 'flex';
      }

      hide() {
          document.getElementById('checkpoint-modal').style.display = 'none';
      }

      renderParameters() {
          const memories = this.data.memories || [];

          return `
              <div class="modal-content">
                  <h2>✋ Review Simulation Parameters</h2>

                  ${memories.length > 0 ? `
                      <div class="memories-used">
                          <strong>Based on your preferences:</strong>
                          ${memories.map(m => `<div class="memory-tag">${m}</div>`).join('')}
                      </div>
                  ` : ''}

                  <div class="proposal">
                      <pre>${this.data.proposal}</pre>
                  </div>

                  <div class="actions">
                      <button onclick="approveParameters('${this.data.session_id}')">
                          ✓ Approve
                      </button>
                      <button onclick="showModifyPanel()">
                          ✎ Modify
                      </button>
                  </div>

                  <div id="modify-panel" style="display:none; margin-top: 1rem;">
                      <textarea id="modification-text"
                                placeholder="What would you like to change?"
                                style="width: 100%; height: 100px; padding: 0.5rem;"></textarea>
                      <button onclick="submitModification('${this.data.session_id}')"
                              style="margin-top: 0.5rem; padding: 0.5rem 1rem;">
                          Submit Changes
                      </button>
                  </div>
              </div>
          `;
      }

      renderScript() {
          return `
              <div class="modal-content">
                  <h2>✋ Review Generated Script</h2>

                  <div class="info">
                      <p>Script approved after ${this.data.attempts} attempt(s)</p>
                  </div>

                  <div class="script">
                      <pre><code>${this.escapeHtml(this.data.script)}</code></pre>
                  </div>

                  <div class="actions">
                      <button onclick="approveScript('${this.data.session_id}')">
                          ▶ Approve & Run
                      </button>
                      <button onclick="showChangesPanel()">
                          ✎ Request Changes
                      </button>
                  </div>

                  <div id="changes-panel" style="display:none; margin-top: 1rem;">
                      <textarea id="changes-text"
                                placeholder="What changes would you like? (e.g., 'Add stress tensor dump')"
                                style="width: 100%; height: 100px; padding: 0.5rem;"></textarea>
                      <button onclick="submitChanges('${this.data.session_id}')"
                              style="margin-top: 0.5rem; padding: 0.5rem 1rem;">
                          Submit Changes
                      </button>
                  </div>
              </div>
          `;
      }

      escapeHtml(text) {
          return text.replace(/[&<>"']/g, (m) => {
              return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[m];
          });
      }
  }
  ```

#### 22. Create Workflow Functions
- [ ] Add workflow management:
  ```javascript
  let currentSessionId = null;

  async function startWorkflow(userRequest) {
      try {
          const response = await fetch('/api/workflow/start', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({request: userRequest})
          });

          const data = await response.json();
          currentSessionId = data.session_id;

          // Show parameter checkpoint
          const modal = new CheckpointModal('parameters', data);
          modal.show();
      } catch (error) {
          console.error('Error starting workflow:', error);
          alert('Error starting workflow');
      }
  }

  async function approveParameters(sessionId) {
      try {
          const response = await fetch('/api/checkpoint/parameters', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  session_id: sessionId,
                  action: 'approve'
              })
          });

          const data = await response.json();

          if (!data.success) {
              alert('Script generation failed after ' + data.attempts + ' attempts');
              return;
          }

          // Show script checkpoint
          const modal = new CheckpointModal('script', data);
          modal.show();
      } catch (error) {
          console.error('Error:', error);
          alert('Error processing parameters');
      }
  }

  function showModifyPanel() {
      document.getElementById('modify-panel').style.display = 'block';
  }

  async function submitModification(sessionId) {
      const feedback = document.getElementById('modification-text').value;

      if (!feedback.trim()) {
          alert('Please enter your modifications');
          return;
      }

      try {
          const response = await fetch('/api/checkpoint/parameters', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  session_id: sessionId,
                  action: 'modify',
                  feedback: feedback
              })
          });

          const data = await response.json();

          // Show updated proposal
          const modal = new CheckpointModal('parameters', data);
          modal.show();
      } catch (error) {
          console.error('Error:', error);
          alert('Error submitting modifications');
      }
  }

  async function approveScript(sessionId) {
      try {
          const response = await fetch('/api/checkpoint/script', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  session_id: sessionId,
                  action: 'approve'
              })
          });

          const data = await response.json();

          // Hide modal
          const modal = document.getElementById('checkpoint-modal');
          modal.style.display = 'none';

          // Show execution message
          alert('Simulation is running!');

          // Continue with existing execution flow...
      } catch (error) {
          console.error('Error:', error);
          alert('Error approving script');
      }
  }

  function showChangesPanel() {
      document.getElementById('changes-panel').style.display = 'block';
  }

  async function submitChanges(sessionId) {
      const feedback = document.getElementById('changes-text').value;

      if (!feedback.trim()) {
          alert('Please describe the changes you want');
          return;
      }

      try {
          const response = await fetch('/api/checkpoint/script', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  session_id: sessionId,
                  action: 'request_changes',
                  feedback: feedback
              })
          });

          const data = await response.json();

          if (!data.success) {
              alert('Failed to generate updated script');
              return;
          }

          // Show updated script
          const modal = new CheckpointModal('script', data);
          modal.show();
      } catch (error) {
          console.error('Error:', error);
          alert('Error requesting changes');
      }
  }
  ```

#### 23. Integrate with Existing UI
- [ ] Update existing chat send button to use new workflow:
  ```javascript
  // Find existing send button handler and update
  sendButton.addEventListener('click', () => {
      const message = chatInput.value.trim();
      if (message) {
          startWorkflow(message);  // Use new checkpoint flow
          chatInput.value = '';
      }
  });
  ```
- [ ] Test integration

#### 24. Test Frontend
- [ ] Test parameter approval UI
- [ ] Test parameter modification UI
- [ ] Test script review UI
- [ ] Test request changes UI
- [ ] Test on different screen sizes
- [ ] Test keyboard shortcuts (Enter to submit, Esc to close)

**End of Day 6**: Frontend complete ✓

---

### Day 7: Polish & Deploy

#### 25. Add Memory Visualization
- [ ] Add "View Learnings" button to UI
- [ ] Show current memories in sidebar:
  ```javascript
  async function showMemories() {
      // Read memory.json via new endpoint or display in UI
      const response = await fetch('/api/memories');
      const memories = await response.json();

      // Display in modal or sidebar
      const html = memories.map(m => `
          <div class="memory-item">
              <strong>${m.type}</strong>: ${m.content}
              <br><small>Used ${m.reinforcements}x</small>
          </div>
      `).join('');

      // Show in UI
  }
  ```
- [ ] Add optional `/api/memories` endpoint in app.py

#### 26. Add Iteration Progress Display
- [ ] Show "Attempt 1/3, 2/3, 3/3" during generation
- [ ] Option 1: Poll status endpoint
- [ ] Option 2: WebSocket updates
- [ ] Option 3: Simple loading message

#### 27. Final Testing
- [ ] Complete workflow test (start to finish)
- [ ] Test with multiple different elements (Cu, Al, Ni)
- [ ] Test memory accumulation over 5+ runs
- [ ] Test error handling (network failures, etc.)
- [ ] Test browser refresh (session lost - expected)

#### 28. Update Configuration
- [ ] Update `config.py`:
  ```python
  # Memory
  MEMORY_FILE = "memory.json"

  # Iteration
  MAX_LAMMPS_ATTEMPTS = 3
  MAX_OVITO_ATTEMPTS = 2

  # Checkpoints
  CHECKPOINT_PARAMETERS_REQUIRED = True
  CHECKPOINT_SCRIPT_REQUIRED = True
  ```

#### 29. Documentation
- [ ] Update README.md:
  - [ ] Add DSPy dependency
  - [ ] Explain checkpoint workflow
  - [ ] Explain memory system
  - [ ] Show example workflow
- [ ] Add inline comments to new code
- [ ] Document memory.json structure

#### 30. Deploy
- [ ] Test on clean environment
- [ ] Verify all dependencies install correctly
- [ ] Test basic workflow
- [ ] Deploy!

**End of Day 7**: COMPLETE! ✓

---

## Success Checklist

### Core Functionality:
- [ ] Agents iterate up to 3 times before giving up
- [ ] Memory stores user feedback correctly
- [ ] Memory retrieves relevant learnings
- [ ] User can approve/modify parameters
- [ ] User can approve/request changes to script
- [ ] Feedback stored in memory after each checkpoint
- [ ] Memory reinforced when patterns work

### Quality:
- [ ] No breaking changes to existing features
- [ ] Error handling for all edge cases
- [ ] UI is responsive and intuitive
- [ ] Memory file doesn't grow unbounded (< 1MB after 100 runs)
- [ ] Average checkpoint latency < 1s

### Learning Effectiveness:
- [ ] Second run shows memories from first run
- [ ] Third run shows stronger memories (higher reinforcements)
- [ ] User preferences actually applied
- [ ] System improves over time (observable)

---

## Quick Reference

### Testing Commands

```bash
# Test DSPy
python -c "import dspy; print('OK')"

# Test memory
python -c "from agents import SimpleMemory; m = SimpleMemory(); print(len(m.memories))"

# Start server
python app.py

# Test workflow
curl -X POST http://localhost:5002/api/workflow/start \
  -H "Content-Type: application/json" \
  -d '{"request": "Run Cu at 300K"}'
```

### File Checklist

- [ ] `prompts.py` created (~200 lines)
- [ ] `agents.py` modified (+250 lines for SimpleMemory + IterativeLammpsGeneration)
- [ ] `app.py` modified (+100 lines for checkpoint endpoints)
- [ ] `index.html` modified (+200 lines for checkpoint UI)
- [ ] `config.py` modified (+10 lines)
- [ ] `memory.json` created (auto-generated)

**Total New Code**: ~400 lines

---

## Progress Tracking

**Started**: [DATE]
**Target**: 7 days

- [ ] Day 1: DSPy Setup - 0/3 sections
- [ ] Day 2: Memory System - 0/3 sections
- [ ] Day 3: Signatures & Iteration - 0/4 sections
- [ ] Day 4: Checkpoint Backend - 0/5 sections
- [ ] Day 5: Integration Testing - 0/3 sections
- [ ] Day 6: Frontend UI - 0/6 sections
- [ ] Day 7: Polish & Deploy - 0/6 sections

**Overall Progress**: 0/30 sections (0/75 tasks)

---

## Troubleshooting

### Common Issues:

1. **DSPy import error**
   - Solution: `pip install dspy-ai`

2. **Memory file not saving**
   - Check file permissions
   - Verify `os.path.exists()` works
   - Check JSON syntax

3. **Checkpoints not showing**
   - Check browser console for errors
   - Verify modal CSS is loaded
   - Check session_id is passed correctly

4. **Iteration not working**
   - Verify DSPy configured correctly
   - Check LLM API is responding
   - Verify "REVIEW: PASS" string matching

---

---

## Day 8: Legacy Code Cleanup

After completing the migration, clean up all legacy code that has been replaced by the new DSPy-based implementation.

### 31. Backup Legacy Code
- [ ] Create backup branch:
  ```bash
  git checkout -b backup/pre-agentify-cleanup
  git push -u origin backup/pre-agentify-cleanup
  git checkout claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n
  ```
- [ ] Document what's being removed in commit message
- [ ] Verify all new functionality is working before cleanup

#### 32. Clean Up agents.py

**Remove Legacy Agent Classes** (agents.py:7-486)

- [ ] **Remove Base Agent class** (lines 7-63):
  - [ ] Remove `__init__` method with OpenAI client
  - [ ] Remove `execute()` method
  - [ ] Remove `execute_stream()` method
  - [ ] These are replaced by DSPy modules

- [ ] **Remove OrchestratorAgent** (lines 65-316):
  - [ ] Remove entire class definition
  - [ ] Remove 249-line embedded system prompt
  - [ ] Prompt is now in prompts.py as DSPy signature
  - [ ] Functionality moved to workflow orchestration in app.py

- [ ] **Remove LammpsGeneratorAgent** (lines 318-425):
  - [ ] Remove entire class definition
  - [ ] Remove embedded system prompt
  - [ ] Replaced by `IterativeLammpsGeneration` DSPy module
  - [ ] Prompt now in prompts.py

- [ ] **Remove LammpsReviewerAgent** (lines 427-486):
  - [ ] Remove entire class definition
  - [ ] Remove embedded system prompt
  - [ ] Integrated into `IterativeLammpsGeneration` module
  - [ ] Prompt now in prompts.py

**What to Keep in agents.py:**
- [ ] New `SimpleMemory` class (or Mem0 integration)
- [ ] New `IterativeLammpsGeneration` DSPy module
- [ ] Any helper functions for memory formatting
- [ ] Imports for DSPy, Mem0, JSON, datetime, uuid

**Expected Result:** agents.py should be ~250 lines (down from 486)

#### 33. Clean Up app.py

**Remove Legacy Endpoints** (app.py has 3 legacy endpoints)

- [ ] **Remove `/api/llm-stream` endpoint** (lines 113-353):
  - [ ] Delete entire function
  - [ ] This endpoint does linear workflow (no iteration)
  - [ ] Replaced by `/api/workflow/start` + checkpoint endpoints

- [ ] **Remove `/api/llm-command` endpoint** (lines 355-503):
  - [ ] Delete entire function
  - [ ] Non-streaming version of legacy workflow
  - [ ] No longer needed with checkpoint system

- [ ] **Remove `/api/llm-stream-live` endpoint** (lines 601-809):
  - [ ] Delete entire function
  - [ ] Duplicate of `/api/llm-stream` with WebSocket support
  - [ ] Simulation execution moved to checkpoint approval

**Remove Legacy Workflow Logic:**
- [ ] Remove Phase 1/Phase 2 conversation context handling
- [ ] Remove `conversation_context` query parameter logic
- [ ] Remove `execution_intent` detection code
- [ ] Remove direct agent instantiation (e.g., `orchestrator = OrchestratorAgent()`)
- [ ] This is all replaced by session-based checkpoint workflow

**What to Keep in app.py:**
- [ ] Flask app setup
- [ ] SocketIO setup
- [ ] Health check functions (check_dependencies, check_llm_api)
- [ ] KIM potentials API endpoints (lines 815-1062)
- [ ] WebSocket handlers for streaming simulation
- [ ] WebSocket handlers for KIM downloads
- [ ] Registry initialization
- [ ] New checkpoint endpoints (`/api/workflow/start`, `/api/checkpoint/parameters`, `/api/checkpoint/script`)
- [ ] New session storage
- [ ] New DSPy + Mem0 initialization

**Expected Result:** app.py should be ~950 lines (down from 1119)

#### 34. Update index.html

**Update Frontend to Use New Endpoints:**

- [ ] **Update sendMessage() function** (lines 1699-1845):
  - [ ] Remove EventSource connection to `/api/llm-stream-live`
  - [ ] Replace with `fetch()` call to `/api/workflow/start`
  - [ ] Remove Phase 1/Phase 2 context handling
  - [ ] Remove `conversationContext` variable
  - [ ] Remove `executionIntent` tracking
  - [ ] Replace with new checkpoint modal workflow

- [ ] **Remove Legacy SSE Handling:**
  - [ ] Remove event listeners for: `agent_start`, `agent_content`, `agent_complete`
  - [ ] Remove `awaiting_confirmation` handler
  - [ ] Remove `needs_more_info` handler
  - [ ] Remove `review_failed` handler
  - [ ] These are replaced by checkpoint modals

- [ ] **Remove Legacy Variables:**
  - [ ] Remove `isStreaming` flag (line 1053)
  - [ ] Remove `conversationContext` (line 1054)
  - [ ] Remove `executionIntent` (line 1055)
  - [ ] Replaced by checkpoint session management

**What to Keep in index.html:**
- [ ] All CSS and styling
- [ ] WebSocket connection for streaming simulation
- [ ] Trajectory viewer
- [ ] Thermodynamic plots
- [ ] KIM potentials tab
- [ ] New checkpoint modal UI (added in Day 6)
- [ ] New workflow functions (added in Day 6)

**Expected Result:** index.html should be ~1900 lines (slightly larger due to checkpoint UI)

#### 35. Update Imports and References

- [ ] In app.py, update imports:
  ```python
  # Remove
  from agents import Agent, OrchestratorAgent, LammpsGeneratorAgent, LammpsReviewerAgent

  # Add (if not already there)
  from agents import SimpleMemory, IterativeLammpsGeneration
  import dspy
  ```

- [ ] Check for any other files that import legacy agents:
  ```bash
  grep -r "OrchestratorAgent\|LammpsGeneratorAgent\|LammpsReviewerAgent" *.py
  ```

- [ ] Update any found references

#### 36. Test After Cleanup

- [ ] **Smoke Test:**
  - [ ] Start server: `python app.py`
  - [ ] Check for import errors
  - [ ] Check health checks pass
  - [ ] Open browser to index.html

- [ ] **Functional Test:**
  - [ ] Start new workflow
  - [ ] Approve parameters checkpoint
  - [ ] Approve script checkpoint
  - [ ] Run simulation
  - [ ] Verify no errors in browser console
  - [ ] Verify no errors in server logs

- [ ] **Endpoint Test:**
  - [ ] Old endpoints should be gone:
    ```bash
    curl http://localhost:5002/api/llm-stream  # Should 404
    curl http://localhost:5002/api/llm-command  # Should 404
    curl http://localhost:5002/api/llm-stream-live  # Should 404
    ```

  - [ ] New endpoints should work:
    ```bash
    curl -X POST http://localhost:5002/api/workflow/start \
      -H "Content-Type: application/json" \
      -d '{"request": "Run Cu simulation"}'
    ```

#### 37. Commit Cleanup

- [ ] Review all changes:
  ```bash
  git status
  git diff
  ```

- [ ] Commit with descriptive message:
  ```bash
  git add agents.py app.py index.html
  git commit -m "$(cat <<'EOF'
  Clean up legacy code after agentify migration

  Removed legacy agent implementations:
  - Base Agent class with OpenAI client (agents.py:7-63)
  - OrchestratorAgent with 249-line embedded prompt (agents.py:65-316)
  - LammpsGeneratorAgent (agents.py:318-425)
  - LammpsReviewerAgent (agents.py:427-486)

  Removed legacy API endpoints:
  - /api/llm-stream (linear workflow, no iteration)
  - /api/llm-command (non-streaming variant)
  - /api/llm-stream-live (streaming variant)

  Updated frontend:
  - Removed Phase 1/Phase 2 conversation handling
  - Removed legacy SSE event listeners
  - Updated to use new checkpoint workflow

  Code reduction:
  - agents.py: 486 → ~250 lines (-236 lines)
  - app.py: 1119 → ~950 lines (-169 lines)
  - Total: -405 lines of legacy code

  All functionality preserved in new DSPy-based implementation.
  EOF
  )"
  ```

- [ ] Push to branch:
  ```bash
  git push -u origin claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n
  ```

#### 38. Verify Code Metrics

- [ ] **Line count verification:**
  ```bash
  wc -l agents.py  # Should be ~250 lines
  wc -l app.py     # Should be ~950 lines
  wc -l prompts.py # Should be ~200 lines
  ```

- [ ] **Total new code added:**
  - [ ] prompts.py: ~200 lines (new file)
  - [ ] agents.py additions: ~250 lines (SimpleMemory + IterativeLammpsGeneration)
  - [ ] app.py additions: ~100 lines (checkpoint endpoints)
  - [ ] index.html additions: ~200 lines (checkpoint UI)
  - [ ] **Total added: ~750 lines**

- [ ] **Total legacy code removed:**
  - [ ] agents.py: -236 lines
  - [ ] app.py: -169 lines
  - [ ] **Total removed: -405 lines**

- [ ] **Net code change: +345 lines** for full agent collaboration + memory + checkpoints!

**End of Day 8**: Codebase cleaned, legacy code removed ✓

---

## Updated Success Checklist

### Core Functionality:
- [ ] Agents iterate up to 3 times before giving up
- [ ] Memory stores user feedback correctly
- [ ] Memory retrieves relevant learnings
- [ ] User can approve/modify parameters
- [ ] User can approve/request changes to script
- [ ] Feedback stored in memory after each checkpoint
- [ ] Memory reinforced when patterns work
- [ ] **Legacy code completely removed**
- [ ] **No broken imports or references**
- [ ] **All tests pass after cleanup**

### Quality:
- [ ] No breaking changes to existing features
- [ ] Error handling for all edge cases
- [ ] UI is responsive and intuitive
- [ ] Memory file doesn't grow unbounded (< 1MB after 100 runs)
- [ ] Average checkpoint latency < 1s
- [ ] **Clean git history with descriptive commits**
- [ ] **Code is well-documented and maintainable**

### Learning Effectiveness:
- [ ] Second run shows memories from first run
- [ ] Third run shows stronger memories (higher reinforcements)
- [ ] User preferences actually applied
- [ ] System improves over time (observable)

---

## Progress Tracking

**Started**: [DATE]
**Target**: 8 days (7 days implementation + 1 day cleanup)

- [ ] Day 1: DSPy Setup - 0/3 sections
- [ ] Day 2: Memory System - 0/3 sections
- [ ] Day 3: Signatures & Iteration - 0/4 sections
- [ ] Day 4: Checkpoint Backend - 0/5 sections
- [ ] Day 5: Integration Testing - 0/3 sections
- [ ] Day 6: Frontend UI - 0/6 sections
- [ ] Day 7: Polish & Deploy - 0/6 sections
- [ ] **Day 8: Legacy Code Cleanup - 0/8 sections**

**Overall Progress**: 0/38 sections (0/90 tasks)

---

**Last Updated**: 2025-11-16 (v4 LEAN with cleanup tasks)
**Status**: ✅ Ready to Implement
**Timeline**: 8 days (7 implementation + 1 cleanup)
**Complexity**: Low
**Perfect For**: Single-user research deployment
