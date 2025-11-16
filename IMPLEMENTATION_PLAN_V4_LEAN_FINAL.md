# ChatMD Agentify Implementation Plan v4 LEAN (FINAL)
## Single-User with Mem0 (Direct Usage)

**Date**: 2025-11-16 (v4 FINAL - Using Mem0 Directly)
**Branch**: `claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n`
**Timeline**: 7 days
**Complexity**: Low

---

## Overview

**What We're Building:**
1. ✅ Iterative agent collaboration (Generator ↔ Reviewer, up to 3 attempts)
2. ✅ Agent memory with Mem0 (learn from user feedback)
3. ✅ User checkpoints (approve parameters & scripts)
4. ✅ Single user (user_id="default")

**Dependencies:**
- `dspy-ai>=2.5.0` - Agent framework
- `mem0ai>=0.1.0` - Memory system (simpler than custom!)

**Total New Code**: ~350 lines across 3 files

---

## File Structure

```
chatmd/
├── prompts.py              # NEW (~150 lines) - DSPy signatures
├── agents.py               # MODIFIED (+100 lines) - Iterative modules with Mem0
├── app.py                  # MODIFIED (+100 lines) - Checkpoint endpoints
├── index.html              # MODIFIED (+200 lines) - Checkpoint UI
├── config.py               # MODIFIED (+10 lines)
└── requirements.txt        # MODIFIED (add dspy-ai, mem0ai)
```

---

## Complete Implementation Checklist

### Day 1: Setup & Dependencies

#### 1.1 Install Dependencies

**Task**: Install DSPy and Mem0
- [ ] Open terminal in project directory
- [ ] Run: `pip install dspy-ai mem0ai`
- [ ] Verify installation:
  ```bash
  python -c "import dspy, mem0; print('DSPy:', dspy.__version__); print('Mem0: OK')"
  ```
- [ ] Should see versions printed without errors

**Task**: Update requirements.txt
- [ ] Open `requirements.txt`
- [ ] Add these lines:
  ```
  dspy-ai>=2.5.0
  mem0ai>=0.1.0
  ```
- [ ] Save file

**Expected Output**: Clean installation, no errors

---

#### 1.2 Test DSPy with Your LLM

**Task**: Create test script `test_dspy.py`
- [ ] Create file `test_dspy.py` in project root
- [ ] Add this code:
  ```python
  import dspy
  import config

  # Initialize DSPy with your LLM
  lm = dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  )
  dspy.configure(lm=lm)

  # Simple test signature
  class SimpleTest(dspy.Signature):
      """Answer a simple question"""
      question: str = dspy.InputField()
      answer: str = dspy.OutputField()

  # Test it
  predictor = dspy.ChainOfThought(SimpleTest)
  result = predictor(question="What is 2+2?")

  print("Question: What is 2+2?")
  print(f"Answer: {result.answer}")
  print(f"Reasoning: {result.rationale}")

  print("\n✅ DSPy is working with your LLM!")
  ```

**Task**: Run test
- [ ] Run: `python test_dspy.py`
- [ ] Verify you get a response from your LLM
- [ ] Should see answer like "4" or "The answer is 4"

**Expected Output**:
```
Question: What is 2+2?
Answer: 4
Reasoning: ...
✅ DSPy is working with your LLM!
```

**If fails**: Check your config.py settings (API_BASE_URL, API_KEY, model name)

---

#### 1.3 Test Mem0

**Task**: Create test script `test_mem0.py`
- [ ] Create file `test_mem0.py`
- [ ] Add this code:
  ```python
  from mem0 import Memory

  # Initialize Mem0 (uses default config)
  memory = Memory()

  print("Testing Mem0...")

  # Add a memory
  messages = [
      {"role": "user", "content": "I prefer 2ps equilibration for Cu simulations"},
      {"role": "assistant", "content": "Noted, I'll remember that you prefer 2ps equilibration for Cu"}
  ]

  result = memory.add(messages, user_id="default")
  print(f"✅ Added memory: {result}")

  # Search memories
  results = memory.search(query="Cu equilibration", user_id="default", limit=3)
  print(f"\n✅ Found {len(results['results'])} memories:")
  for mem in results['results']:
      print(f"  - {mem['memory']}")

  # Get all memories for user
  all_memories = memory.get_all(user_id="default")
  print(f"\n✅ Total memories for user: {len(all_memories['results'])}")

  print("\n✅ Mem0 is working!")
  ```

**Task**: Run test
- [ ] Run: `python test_mem0.py`
- [ ] Should see memory added and retrieved
- [ ] Mem0 creates a `.mem0` directory (this is normal)

**Expected Output**:
```
Testing Mem0...
✅ Added memory: {...}
✅ Found 1 memories:
  - User prefers 2ps equilibration for Cu simulations
✅ Total memories for user: 1
✅ Mem0 is working!
```

**Task**: Understand Mem0 storage
- [ ] Check that `.mem0/` directory was created
- [ ] This is where Mem0 stores memories (vector database)
- [ ] Add `.mem0/` to `.gitignore` if not already there

**End of Day 1**: Dependencies working ✓

---

### Day 2: Create DSPy Signatures (prompts.py)

#### 2.1 Create prompts.py File

**Task**: Create the file
- [ ] Create `prompts.py` in project root
- [ ] Add header:
  ```python
  """
  prompts.py - DSPy Signatures for ChatMD Agents

  This file defines the input/output structure for each agent using DSPy signatures.
  """
  import dspy
  ```

---

#### 2.2 Define GenerateLAMMPSScript Signature

**Task**: Extract existing generator prompt
- [ ] Open `agents.py`
- [ ] Find the `LammpsGeneratorAgent` class
- [ ] Copy the system prompt (the long string in `__init__`)
- [ ] Keep it handy - we'll embed it in the signature

**Task**: Create signature in prompts.py
- [ ] Add to `prompts.py`:
  ```python
  class GenerateLAMMPSScript(dspy.Signature):
      """Generate LAMMPS script using KIM potentials only.

      You are a LAMMPS script generator. Generate complete, executable LAMMPS scripts
      using ONLY KIM potentials (never pair_style or pair_coeff).

      CRITICAL REQUIREMENTS:
      - Use ONLY KIM potentials (kim init, kim interactions)
      - kim init BEFORE create_box
      - kim interactions AFTER create_box
      - Always define mass even with KIM
      - Include velocity initialization matching thermostat temperature
      - Add detailed comments if user prefers them
      - Include CNA computation if structure visualization requested

      [PASTE YOUR FULL GENERATOR PROMPT HERE FROM agents.py]
      """

      parameters: str = dspy.InputField(
          desc="Simulation parameters from orchestrator (element, temp, duration, KIM model, etc.)"
      )

      memories: str = dspy.InputField(
          desc="Relevant memories from past interactions (user preferences, common mistakes to avoid)"
      )

      user_feedback: str = dspy.InputField(
          desc="Specific changes requested by user at checkpoint (e.g., 'add stress dump', 'more comments')"
      )

      feedback: str = dspy.InputField(
          desc="Reviewer feedback from previous generation attempts"
      )

      script: str = dspy.OutputField(
          desc="Complete executable LAMMPS script"
      )

      reasoning: str = dspy.OutputField(
          desc="Brief explanation of key decisions and which memories you applied"
      )
  ```

**Task**: Verify syntax
- [ ] Run: `python -c "from prompts import GenerateLAMMPSScript; print('OK')"`
- [ ] Should print "OK"

---

#### 2.3 Define ReviewLAMMPSScript Signature

**Task**: Extract reviewer prompt
- [ ] Open `agents.py`
- [ ] Find `LammpsReviewerAgent` class
- [ ] Copy the system prompt

**Task**: Create signature
- [ ] Add to `prompts.py`:
  ```python
  class ReviewLAMMPSScript(dspy.Signature):
      """Review LAMMPS script for correctness and KIM compliance.

      You are a LAMMPS script reviewer. Check scripts for correctness and reject
      any that don't follow KIM-only requirements.

      REVIEW CHECKLIST:
      - KIM commands used correctly (init before create_box, interactions after)
      - NO pair_style or pair_coeff commands (automatic rejection)
      - Mass defined for all atom types
      - Velocity initialization matches thermostat temperature
      - Proper timestep and neighbor settings
      - Thermodynamic output configured

      Output format: "REVIEW: PASS" or "REVIEW: FAIL" with specific issues.

      [PASTE YOUR FULL REVIEWER PROMPT HERE FROM agents.py]
      """

      script: str = dspy.InputField(desc="LAMMPS script to review")

      parameters: str = dspy.InputField(desc="Original simulation parameters for validation")

      review: str = dspy.OutputField(
          desc="Review result starting with 'REVIEW: PASS' or 'REVIEW: FAIL', followed by explanation"
      )

      issues: list[str] = dspy.OutputField(
          desc="List of specific issues found (empty list if PASS)"
      )
  ```

**Task**: Verify
- [ ] Run: `python -c "from prompts import ReviewLAMMPSScript; print('OK')"`

---

#### 2.4 Define OVITO Signatures (Optional)

**Task**: Create OVITO generator signature
- [ ] Add to `prompts.py`:
  ```python
  class GenerateOVITOScript(dspy.Signature):
      """Generate OVITO visualization script.

      [Add OVITO generation prompt if you want OVITO iteration]
      """

      visualization_request: str = dspy.InputField()
      lammps_properties: str = dspy.InputField()
      memories: str = dspy.InputField()
      user_feedback: str = dspy.InputField()
      feedback: str = dspy.InputField()

      script: str = dspy.OutputField()
      reasoning: str = dspy.OutputField()


  class ReviewOVITOScript(dspy.Signature):
      """Review OVITO script for safety and correctness."""

      script: str = dspy.InputField()

      review: str = dspy.OutputField(desc="REVIEW: PASS or FAIL")
      issues: list[str] = dspy.OutputField()
  ```

**Note**: OVITO iteration is optional. Can add later if needed.

**Task**: Final verification
- [ ] Run: `python -c "import prompts; print('All signatures loaded OK')"`
- [ ] Should see "All signatures loaded OK"

**End of Day 2**: DSPy signatures defined ✓

---

### Day 3: Create Iterative Modules with Mem0 (agents.py)

#### 3.1 Initialize Mem0 in agents.py

**Task**: Add imports at top of agents.py
- [ ] Open `agents.py`
- [ ] At the top, after existing imports, add:
  ```python
  # === NEW IMPORTS FOR AGENTIFY ===
  import dspy
  from mem0 import Memory
  from prompts import GenerateLAMMPSScript, ReviewLAMMPSScript
  ```

**Task**: Initialize Mem0 (module level)
- [ ] After imports, before any classes, add:
  ```python
  # === MEMORY SYSTEM ===
  # Initialize Mem0 for single-user deployment
  # All interactions use user_id="default"
  memory = Memory()

  # Helper constant
  USER_ID = "default"
  ```

---

#### 3.2 Create IterativeLammpsGeneration Module

**Task**: Add the class to agents.py
- [ ] Add this code to `agents.py`:
  ```python
  class IterativeLammpsGeneration(dspy.Module):
      """
      Iteratively generate and review LAMMPS script until it passes.

      Flow:
      1. Retrieve relevant memories from Mem0
      2. Generator creates script (using parameters + memories + user feedback)
      3. Reviewer checks script
      4. If PASS: store success in memory, return
      5. If FAIL: use feedback for next attempt
      6. Repeat up to max_attempts
      """

      def __init__(self, max_attempts=3):
          super().__init__()
          self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
          self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
          self.max_attempts = max_attempts

      def forward(self, parameters: str, user_feedback: str = ""):
          """
          Generate LAMMPS script with iterative refinement.

          Args:
              parameters: Simulation parameters from orchestrator
              user_feedback: Specific changes from user checkpoint (optional)

          Returns:
              dspy.Prediction with script, review, attempts, success, conversation
          """

          # === STEP 1: Retrieve relevant memories from Mem0 ===
          memory_results = memory.search(
              query=parameters,
              user_id=USER_ID,
              limit=5
          )

          # Format memories for prompt
          if memory_results and memory_results.get('results'):
              memories_text = "Learnings from past interactions:\n"
              for mem in memory_results['results']:
                  memories_text += f"- {mem['memory']}\n"
          else:
              memories_text = "No previous learnings available."

          # === STEP 2: Iterative generation ===
          conversation = []
          feedback = "No previous attempts"
          script = None
          review_result = None

          for attempt in range(1, self.max_attempts + 1):
              # Generate script
              gen_result = self.generator(
                  parameters=parameters,
                  memories=memories_text,
                  user_feedback=user_feedback,
                  feedback=feedback
              )

              script = gen_result.script
              reasoning = gen_result.reasoning

              # Track conversation
              conversation.append({
                  "role": "generator",
                  "attempt": attempt,
                  "content": script,
                  "reasoning": reasoning
              })

              # Review script
              review_result = self.reviewer(
                  script=script,
                  parameters=parameters
              )

              review_text = review_result.review
              issues = review_result.issues if hasattr(review_result, 'issues') else []

              # Track conversation
              conversation.append({
                  "role": "reviewer",
                  "attempt": attempt,
                  "content": review_text,
                  "issues": issues
              })

              # === STEP 3: Check if passed ===
              if "REVIEW: PASS" in review_text:
                  # Success! Store in memory
                  success_messages = [
                      {"role": "system", "content": f"Successful LAMMPS generation: {parameters}"},
                      {"role": "assistant", "content": f"Generated script passed review on attempt {attempt}"}
                  ]
                  memory.add(success_messages, user_id=USER_ID)

                  return dspy.Prediction(
                      script=script,
                      review=review_text,
                      attempts=attempt,
                      conversation=conversation,
                      success=True
                  )

              # === STEP 4: Build feedback for next iteration ===
              if issues:
                  feedback = f"Attempt {attempt} failed. Issues:\n" + "\n".join(f"- {issue}" for issue in issues)
              else:
                  feedback = f"Attempt {attempt} failed. Review: {review_text}"

          # === STEP 5: Failed after max attempts ===
          # Store failure in memory (to learn from mistakes)
          failure_messages = [
              {"role": "system", "content": f"Failed LAMMPS generation after {self.max_attempts} attempts"},
              {"role": "assistant", "content": f"Common issues: {', '.join(issues) if issues else 'See review'}"}
          ]
          memory.add(failure_messages, user_id=USER_ID)

          return dspy.Prediction(
              script=script,
              review=review_text,
              attempts=self.max_attempts,
              conversation=conversation,
              success=False
          )
  ```

**Task**: Verify syntax
- [ ] Run: `python -c "from agents import IterativeLammpsGeneration; print('Module loaded OK')"`
- [ ] Should print "Module loaded OK"

---

#### 3.3 Test the Iterative Module

**Task**: Create test script `test_iteration.py`
- [ ] Create file `test_iteration.py`
- [ ] Add:
  ```python
  import dspy
  import config
  from agents import IterativeLammpsGeneration

  # Initialize DSPy
  lm = dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  )
  dspy.configure(lm=lm)

  # Create module
  lammps_gen = IterativeLammpsGeneration(max_attempts=3)

  # Test with simple parameters
  print("Testing iterative generation...")

  test_params = """
  SIMULATION PARAMETERS:
  Element: Cu
  KIM Model: EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006
  Temperature: 300K
  Duration: 10ps
  Ensemble: NVT
  """

  result = lammps_gen(parameters=test_params, user_feedback="")

  print(f"\n{'='*60}")
  print(f"Success: {result.success}")
  print(f"Attempts: {result.attempts}")
  print(f"Review: {result.review}")
  print(f"{'='*60}")

  if result.success:
      print("\n✅ Script generated successfully!")
      print(f"\nFirst 500 chars of script:\n{result.script[:500]}...")
  else:
      print("\n❌ Script generation failed after max attempts")
      print(f"Last review: {result.review}")

  # Show conversation
  print(f"\n📝 Conversation ({len(result.conversation)} messages):")
  for msg in result.conversation:
      if msg['role'] == 'generator':
          print(f"  Generator (Attempt {msg['attempt']}): [script generated]")
          print(f"    Reasoning: {msg['reasoning'][:100]}...")
      else:
          print(f"  Reviewer (Attempt {msg['attempt']}): {msg['content'][:100]}...")
  ```

**Task**: Run the test
- [ ] Run: `python test_iteration.py`
- [ ] Wait for completion (may take 30-60 seconds)
- [ ] Should see iterative generation working

**Expected Output**:
```
Testing iterative generation...

============================================================
Success: True
Attempts: 1 (or 2 or 3)
Review: REVIEW: PASS - Script is correct and ready...
============================================================

✅ Script generated successfully!

First 500 chars of script:
# LAMMPS Script for Cu MD Simulation
# Generated with KIM potential...

📝 Conversation (2 or 4 or 6 messages):
  Generator (Attempt 1): [script generated]
    Reasoning: Using KIM potential for Cu, setting up...
  Reviewer (Attempt 1): REVIEW: PASS...
```

**Task**: Check memory was stored
- [ ] Run:
  ```bash
  python -c "from mem0 import Memory; m = Memory(); results = m.get_all(user_id='default'); print(f'Total memories: {len(results[\"results\"])}')"
  ```
- [ ] Should see at least 1 memory stored

**If test fails**:
- Check DSPy configuration
- Check your LLM is responding
- Check signatures are loaded correctly
- Look at the error message carefully

**End of Day 3**: Iterative module working ✓

---

### Day 4: Checkpoint Backend (app.py)

#### 4.1 Initialize DSPy and Mem0 in app.py

**Task**: Add imports at top of app.py
- [ ] Open `app.py`
- [ ] After existing imports, add:
  ```python
  # === AGENTIFY IMPORTS ===
  import dspy
  from mem0 import Memory
  from agents import IterativeLammpsGeneration
  import uuid
  from datetime import datetime
  ```

**Task**: Initialize systems (after app creation)
- [ ] Find where `app = Flask(__name__)` is defined
- [ ] After that, add:
  ```python
  # === INITIALIZE AGENTIFY COMPONENTS ===

  # Initialize Mem0
  memory = Memory()
  USER_ID = "default"  # Single user

  # Initialize DSPy
  dspy.configure(lm=dspy.LM(
      model=config.LAMMPS_GENERATOR_MODEL,
      api_base=config.API_BASE_URL,
      api_key=config.API_KEY
  ))

  # Initialize iterative generator
  lammps_generator = IterativeLammpsGeneration(max_attempts=3)

  # Workflow sessions (in-memory dict)
  workflow_sessions = {}
  ```

---

#### 4.2 Create Workflow Start Endpoint

**Task**: Add endpoint to app.py
- [ ] Add this function to `app.py`:
  ```python
  @app.route('/api/workflow/start', methods=['POST'])
  def start_workflow():
      """
      Start new workflow with parameter proposal.

      Request body:
      {
          "request": "Run a Cu simulation at 300K for 10ps"
      }

      Response:
      {
          "session_id": "uuid",
          "checkpoint": "parameters",
          "proposal": "SIMULATION PARAMETERS:...",
          "memories": ["User prefers...", "..."]
      }
      """
      try:
          user_request = request.json.get('request')

          if not user_request:
              return jsonify({"error": "Missing 'request' field"}), 400

          # === STEP 1: Retrieve relevant memories ===
          memory_results = memory.search(
              query=user_request,
              user_id=USER_ID,
              limit=5
          )

          # Extract memory text
          memories_list = []
          memory_context = ""
          if memory_results and memory_results.get('results'):
              for mem in memory_results['results']:
                  mem_text = mem['memory']
                  memories_list.append(mem_text)
                  memory_context += f"- {mem_text}\n"

          # === STEP 2: Get orchestrator (use existing code) ===
          # You already have orchestrator agent - use it
          orchestrator = OrchestratorAgent(
              model=config.ORCHESTRATOR_MODEL,
              system_prompt="[your existing orchestrator prompt]",
              temperature=0.7
          )

          # Add memory context to request
          request_with_context = user_request
          if memory_context:
              request_with_context += f"\n\nUser's known preferences:\n{memory_context}"

          # Propose parameters
          proposal = orchestrator.execute(request_with_context)

          # === STEP 3: Create workflow session ===
          session_id = str(uuid.uuid4())
          workflow_sessions[session_id] = {
              "request": user_request,
              "stage": "parameters",
              "proposal": proposal,
              "memories": memories_list,
              "created": datetime.now().isoformat()
          }

          # === STEP 4: Return checkpoint ===
          return jsonify({
              "session_id": session_id,
              "checkpoint": "parameters",
              "proposal": proposal,
              "memories": memories_list
          })

      except Exception as e:
          print(f"Error in start_workflow: {e}")
          import traceback
          traceback.print_exc()
          return jsonify({"error": str(e)}), 500
  ```

**Task**: Test the endpoint
- [ ] Start your Flask server: `python app.py`
- [ ] In another terminal, test with curl:
  ```bash
  curl -X POST http://localhost:5002/api/workflow/start \
    -H "Content-Type: application/json" \
    -d '{"request": "Run a Cu simulation at 300K"}'
  ```
- [ ] Should get back JSON with session_id, checkpoint, proposal, memories

**Expected response**:
```json
{
  "session_id": "abc-123-...",
  "checkpoint": "parameters",
  "proposal": "SIMULATION PARAMETERS:\nElement: Cu\n...",
  "memories": ["User prefers...", ...]
}
```

---

#### 4.3 Create Parameter Checkpoint Endpoint

**Task**: Add endpoint
- [ ] Add to `app.py`:
  ```python
  @app.route('/api/checkpoint/parameters', methods=['POST'])
  def checkpoint_parameters():
      """
      Handle parameter approval/modification.

      Request body:
      {
          "session_id": "uuid",
          "action": "approve" | "modify",
          "feedback": "optional modification text"
      }

      Response (on approve):
      {
          "checkpoint": "script",
          "script": "# LAMMPS script...",
          "attempts": 2,
          "success": true,
          "conversation": [...]
      }
      """
      try:
          session_id = request.json.get('session_id')
          action = request.json.get('action')  # "approve" or "modify"

          if not session_id or not action:
              return jsonify({"error": "Missing session_id or action"}), 400

          if session_id not in workflow_sessions:
              return jsonify({"error": "Invalid session_id"}), 404

          session = workflow_sessions[session_id]

          # === ACTION: APPROVE ===
          if action == "approve":
              # Store approval in memory
              approval_messages = [
                  {"role": "user", "content": f"Approved parameters: {session['proposal']}"},
                  {"role": "assistant", "content": "Parameters approved, proceeding to generation"}
              ]
              memory.add(approval_messages, user_id=USER_ID)

              # Run iterative generation
              result = lammps_generator(
                  parameters=session['proposal'],
                  user_feedback=""  # No user feedback yet
              )

              # Update session
              session['stage'] = "script"
              session['script'] = result.script
              session['attempts'] = result.attempts
              session['conversation'] = result.conversation
              session['success'] = result.success

              return jsonify({
                  "checkpoint": "script",
                  "script": result.script,
                  "attempts": result.attempts,
                  "success": result.success,
                  "conversation": result.conversation
              })

          # === ACTION: MODIFY ===
          elif action == "modify":
              feedback = request.json.get('feedback', '')

              if not feedback:
                  return jsonify({"error": "Missing feedback for modification"}), 400

              # Store preference in memory
              pref_messages = [
                  {"role": "user", "content": f"User requested modification: {feedback}"},
                  {"role": "assistant", "content": "Noted, I'll remember this preference"}
              ]
              memory.add(pref_messages, user_id=USER_ID)

              # For simplicity, just return modified status
              # (In full implementation, could re-run orchestrator with feedback)
              return jsonify({
                  "checkpoint": "parameters",
                  "status": "modified",
                  "message": "Preference stored. Please revise parameters manually or approve."
              })

          else:
              return jsonify({"error": f"Unknown action: {action}"}), 400

      except Exception as e:
          print(f"Error in checkpoint_parameters: {e}")
          import traceback
          traceback.print_exc()
          return jsonify({"error": str(e)}), 500
  ```

**Task**: Test the endpoint
- [ ] Restart Flask server
- [ ] Test approve action:
  ```bash
  # First start a workflow to get a session_id
  SESSION_ID=$(curl -s -X POST http://localhost:5002/api/workflow/start \
    -H "Content-Type: application/json" \
    -d '{"request": "Run Cu at 300K"}' | jq -r '.session_id')

  echo "Session ID: $SESSION_ID"

  # Then approve parameters
  curl -X POST http://localhost:5002/api/checkpoint/parameters \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$SESSION_ID\", \"action\": \"approve\"}"
  ```
- [ ] Should get back script after generation (may take 30-60 seconds)

**Expected response**:
```json
{
  "checkpoint": "script",
  "script": "# LAMMPS script...",
  "attempts": 1,
  "success": true,
  "conversation": [...]
}
```

---

#### 4.4 Create Script Checkpoint Endpoint

**Task**: Add endpoint
- [ ] Add to `app.py`:
  ```python
  @app.route('/api/checkpoint/script', methods=['POST'])
  def checkpoint_script():
      """
      Handle script approval/change requests.

      Request body:
      {
          "session_id": "uuid",
          "action": "approve" | "request_changes",
          "feedback": "optional change request"
      }

      Response (on approve):
      {
          "status": "executing",
          "message": "Simulation started"
      }
      """
      try:
          session_id = request.json.get('session_id')
          action = request.json.get('action')

          if not session_id or not action:
              return jsonify({"error": "Missing session_id or action"}), 400

          if session_id not in workflow_sessions:
              return jsonify({"error": "Invalid session_id"}), 404

          session = workflow_sessions[session_id]

          # === ACTION: APPROVE ===
          if action == "approve":
              script = session.get('script')

              if not script:
                  return jsonify({"error": "No script in session"}), 400

              # Store approval in memory
              approval_messages = [
                  {"role": "user", "content": "Approved LAMMPS script for execution"},
                  {"role": "assistant", "content": "Script approved, starting simulation"}
              ]
              memory.add(approval_messages, user_id=USER_ID)

              # Execute simulation (use your existing function)
              try:
                  from tools import run_lammps_simulation

                  # Run the simulation
                  output_files, thermo_data, frame_images = run_lammps_simulation(script)

                  return jsonify({
                      "status": "complete",
                      "message": "Simulation completed successfully",
                      "output_files": list(output_files.keys()) if output_files else [],
                      "thermo_points": len(thermo_data) if thermo_data else 0,
                      "frames": len(frame_images) if frame_images else 0
                  })

              except Exception as exec_error:
                  return jsonify({
                      "status": "failed",
                      "error": str(exec_error)
                  }), 500

          # === ACTION: REQUEST CHANGES ===
          elif action == "request_changes":
              feedback = request.json.get('feedback', '')

              if not feedback:
                  return jsonify({"error": "Missing feedback for changes"}), 400

              # Store correction in memory
              correction_messages = [
                  {"role": "user", "content": f"Requested script changes: {feedback}"},
                  {"role": "assistant", "content": "I'll regenerate the script with your requested changes"}
              ]
              memory.add(correction_messages, user_id=USER_ID)

              # Re-generate script with user feedback
              result = lammps_generator(
                  parameters=session['proposal'],
                  user_feedback=feedback  # User's specific request!
              )

              # Update session
              session['script'] = result.script
              session['attempts'] = result.attempts
              session['conversation'] = result.conversation

              return jsonify({
                  "checkpoint": "script",
                  "script": result.script,
                  "attempts": result.attempts,
                  "success": result.success,
                  "message": "Script regenerated with your feedback"
              })

          else:
              return jsonify({"error": f"Unknown action: {action}"}), 400

      except Exception as e:
          print(f"Error in checkpoint_script: {e}")
          import traceback
          traceback.print_exc()
          return jsonify({"error": str(e)}), 500
  ```

**Task**: Test the endpoint
- [ ] Test request_changes:
  ```bash
  # Assuming you have a session with a script
  curl -X POST http://localhost:5002/api/checkpoint/script \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$SESSION_ID\", \"action\": \"request_changes\", \"feedback\": \"Add more detailed comments\"}"
  ```
- [ ] Should get updated script with comments

**Task**: Test full workflow via API
- [ ] Start workflow → approve params → approve script
- [ ] Verify simulation executes
- [ ] Check memories were stored

**End of Day 4**: Checkpoint backend complete ✓

---

### Day 5: Integration Testing

#### 5.1 End-to-End API Test

**Task**: Create comprehensive test script `test_full_workflow.py`
- [ ] Create file `test_full_workflow.py`
- [ ] Add:
  ```python
  import requests
  import json
  import time

  BASE_URL = "http://localhost:5002"

  def test_full_workflow():
      print("="*60)
      print("TESTING FULL WORKFLOW")
      print("="*60)

      # Step 1: Start workflow
      print("\n1. Starting workflow...")
      response = requests.post(f"{BASE_URL}/api/workflow/start", json={
          "request": "Run a Cu simulation at 300K for 10ps"
      })

      assert response.status_code == 200, f"Failed: {response.text}"
      data = response.json()
      session_id = data['session_id']

      print(f"   ✅ Session ID: {session_id}")
      print(f"   Memories used: {len(data.get('memories', []))}")

      # Step 2: Approve parameters
      print("\n2. Approving parameters...")
      print("   (This may take 30-60 seconds for script generation...)")

      response = requests.post(f"{BASE_URL}/api/checkpoint/parameters", json={
          "session_id": session_id,
          "action": "approve"
      })

      assert response.status_code == 200, f"Failed: {response.text}"
      data = response.json()

      print(f"   ✅ Script generated after {data['attempts']} attempt(s)")
      print(f"   Success: {data['success']}")

      if not data['success']:
          print(f"   ❌ Generation failed!")
          print(f"   Last review: {data['conversation'][-1]['content'][:200]}...")
          return False

      # Step 3: Approve script
      print("\n3. Approving script for execution...")
      print("   (This may take several minutes for simulation...)")

      response = requests.post(f"{BASE_URL}/api/checkpoint/script", json={
          "session_id": session_id,
          "action": "approve"
      })

      # Note: Simulation might take a while
      if response.status_code == 200:
          data = response.json()
          print(f"   ✅ Status: {data['status']}")

          if data['status'] == 'complete':
              print(f"   Thermodynamic points: {data.get('thermo_points', 0)}")
              print(f"   Frames: {data.get('frames', 0)}")

      else:
          print(f"   ⚠️  Execution issue: {response.text}")

      print("\n" + "="*60)
      print("WORKFLOW TEST COMPLETE")
      print("="*60)

      return True

  if __name__ == "__main__":
      # Make sure server is running
      try:
          requests.get(f"{BASE_URL}/health")
      except:
          print("❌ Server is not running! Start it with: python app.py")
          exit(1)

      test_full_workflow()
  ```

**Task**: Run the test
- [ ] Make sure Flask server is running in another terminal
- [ ] Run: `python test_full_workflow.py`
- [ ] Wait for completion (may take 2-3 minutes)
- [ ] Verify all steps pass

**Expected output**:
```
============================================================
TESTING FULL WORKFLOW
============================================================

1. Starting workflow...
   ✅ Session ID: abc-123-...
   Memories used: 2

2. Approving parameters...
   (This may take 30-60 seconds for script generation...)
   ✅ Script generated after 1 attempt(s)
   Success: True

3. Approving script for execution...
   (This may take several minutes for simulation...)
   ✅ Status: complete
   Thermodynamic points: 100
   Frames: 20

============================================================
WORKFLOW TEST COMPLETE
============================================================
```

---

#### 5.2 Test Memory Learning

**Task**: Create memory learning test `test_memory_learning.py`
- [ ] Create file:
  ```python
  import requests
  import json
  from mem0 import Memory

  BASE_URL = "http://localhost:5002"
  memory = Memory()
  USER_ID = "default"

  def test_memory_learning():
      print("="*60)
      print("TESTING MEMORY LEARNING")
      print("="*60)

      # Check initial memories
      print("\n1. Checking initial memories...")
      initial = memory.get_all(user_id=USER_ID)
      initial_count = len(initial.get('results', []))
      print(f"   Initial memories: {initial_count}")

      # Run workflow with modification
      print("\n2. Running workflow with parameter modification...")

      # Start
      response = requests.post(f"{BASE_URL}/api/workflow/start", json={
          "request": "Run Al simulation at 400K"
      })
      session_id = response.json()['session_id']

      # Modify parameters
      response = requests.post(f"{BASE_URL}/api/checkpoint/parameters", json={
          "session_id": session_id,
          "action": "modify",
          "feedback": "I prefer 5ps equilibration before production runs"
      })
      print(f"   ✅ Modification stored")

      # Check memory increased
      after_mod = memory.get_all(user_id=USER_ID)
      after_mod_count = len(after_mod.get('results', []))
      print(f"   Memories after modification: {after_mod_count}")
      assert after_mod_count > initial_count, "Memory should have increased!"

      # Run another workflow - should retrieve the memory
      print("\n3. Running second workflow (should use memory)...")
      response = requests.post(f"{BASE_URL}/api/workflow/start", json={
          "request": "Run another Al simulation"
      })

      data = response.json()
      memories_used = data.get('memories', [])

      print(f"   Memories retrieved: {len(memories_used)}")
      for i, mem in enumerate(memories_used[:3], 1):
          print(f"   {i}. {mem[:80]}...")

      # Check if our preference was retrieved
      found_pref = any("5ps equilibration" in mem for mem in memories_used)
      if found_pref:
          print(f"   ✅ Our preference was retrieved!")
      else:
          print(f"   ⚠️  Preference not in top {len(memories_used)} memories")

      print("\n" + "="*60)
      print("MEMORY LEARNING TEST COMPLETE")
      print("="*60)

  if __name__ == "__main__":
      test_memory_learning()
  ```

**Task**: Run the test
- [ ] Run: `python test_memory_learning.py`
- [ ] Verify memories are stored and retrieved

---

#### 5.3 Test Request Changes Flow

**Task**: Create test `test_request_changes.py`
- [ ] Create file:
  ```python
  import requests

  BASE_URL = "http://localhost:5002"

  def test_request_changes():
      print("Testing request changes flow...")

      # Start workflow
      response = requests.post(f"{BASE_URL}/api/workflow/start", json={
          "request": "Run Cu at 300K"
      })
      session_id = response.json()['session_id']

      # Approve parameters to get script
      response = requests.post(f"{BASE_URL}/api/checkpoint/parameters", json={
          "session_id": session_id,
          "action": "approve"
      })

      original_script = response.json()['script']
      print(f"Original script length: {len(original_script)} chars")

      # Request changes
      print("\nRequesting changes: 'Add detailed comments'")
      response = requests.post(f"{BASE_URL}/api/checkpoint/script", json={
          "session_id": session_id,
          "action": "request_changes",
          "feedback": "Add detailed comments explaining each command"
      })

      updated_script = response.json()['script']
      print(f"Updated script length: {len(updated_script)} chars")

      # Updated script should be longer (more comments)
      if len(updated_script) > len(original_script):
          print("✅ Script was updated (likely has more comments)")
      else:
          print("⚠️  Script length unchanged (might not have added comments)")

      print("\n✅ Request changes flow works!")

  if __name__ == "__main__":
      test_request_changes()
  ```

**Task**: Run the test
- [ ] Run: `python test_request_changes.py`

**End of Day 5**: Integration testing complete ✓

---

### Day 6: Frontend Checkpoint UI (index.html)

#### 6.1 Add Modal Container

**Task**: Add modal HTML
- [ ] Open `index.html`
- [ ] Find the closing `</body>` tag
- [ ] Before it, add:
  ```html
  <!-- Checkpoint Modal -->
  <div id="checkpoint-modal" style="display:none;">
      <!-- Content injected by JavaScript -->
  </div>
  ```

---

#### 6.2 Add Modal CSS

**Task**: Add styles
- [ ] In the `<style>` section of `index.html`, add:
  ```css
  /* Checkpoint Modal Styles */
  #checkpoint-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
  }

  .modal-content {
      background: white;
      padding: 2rem;
      border-radius: 8px;
      max-width: 900px;
      max-height: 85vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .modal-content h2 {
      margin-top: 0;
      color: #333;
      border-bottom: 2px solid #4caf50;
      padding-bottom: 0.5rem;
  }

  .memories-used {
      background: #f5f5f5;
      border-left: 4px solid #2196f3;
      padding: 1rem;
      margin: 1rem 0;
      border-radius: 4px;
  }

  .memories-used strong {
      color: #1976d2;
      display: block;
      margin-bottom: 0.5rem;
  }

  .memory-tag {
      display: inline-block;
      background: #e3f2fd;
      color: #1976d2;
      padding: 0.4rem 0.8rem;
      margin: 0.25rem;
      border-radius: 4px;
      font-size: 0.85rem;
      border: 1px solid #90caf9;
  }

  .proposal, .script-display {
      background: #f9f9f9;
      border: 1px solid #ddd;
      padding: 1rem;
      border-radius: 4px;
      margin: 1rem 0;
      font-family: 'Courier New', monospace;
      font-size: 0.9rem;
      max-height: 400px;
      overflow-y: auto;
  }

  .modal-actions {
      display: flex;
      gap: 1rem;
      margin-top: 1.5rem;
  }

  .modal-actions button {
      flex: 1;
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 500;
      transition: all 0.2s;
  }

  .modal-actions button:first-child {
      background: #4caf50;
      color: white;
  }

  .modal-actions button:first-child:hover {
      background: #45a049;
  }

  .modal-actions button:not(:first-child) {
      background: #f0f0f0;
      color: #333;
  }

  .modal-actions button:not(:first-child):hover {
      background: #e0e0e0;
  }

  .feedback-panel {
      margin-top: 1rem;
      padding: 1rem;
      background: #fff3cd;
      border: 1px solid #ffc107;
      border-radius: 4px;
  }

  .feedback-panel textarea {
      width: 100%;
      height: 100px;
      padding: 0.5rem;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-family: sans-serif;
      resize: vertical;
  }

  .feedback-panel button {
      margin-top: 0.5rem;
      padding: 0.5rem 1.5rem;
      background: #ff9800;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
  }

  .feedback-panel button:hover {
      background: #f57c00;
  }

  .info-box {
      background: #e8f5e9;
      border-left: 4px solid #4caf50;
      padding: 1rem;
      margin: 1rem 0;
      border-radius: 4px;
  }
  ```

---

#### 6.3 Create CheckpointModal Class

**Task**: Add JavaScript class
- [ ] In the `<script>` section of `index.html`, add:
  ```javascript
  // ============================================
  // CHECKPOINT MODAL CLASS
  // ============================================

  class CheckpointModal {
      constructor(type, data) {
          this.type = type;  // 'parameters' or 'script'
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
          const proposal = this.data.proposal || 'No proposal available';
          const sessionId = this.data.session_id;

          return `
              <div class="modal-content">
                  <h2>✋ Review Simulation Parameters</h2>

                  ${memories.length > 0 ? `
                      <div class="memories-used">
                          <strong>💡 Based on your past preferences:</strong>
                          <div>
                              ${memories.map(m => `<span class="memory-tag">${this.escapeHtml(m)}</span>`).join('')}
                          </div>
                      </div>
                  ` : ''}

                  <div class="proposal">
                      <pre>${this.escapeHtml(proposal)}</pre>
                  </div>

                  <div class="modal-actions">
                      <button onclick="window.checkpointHandler.approveParameters('${sessionId}')">
                          ✓ Approve
                      </button>
                      <button onclick="window.checkpointHandler.showModifyPanel()">
                          ✎ Modify
                      </button>
                  </div>

                  <div id="modify-panel" class="feedback-panel" style="display:none;">
                      <label><strong>What would you like to change?</strong></label>
                      <textarea id="modification-text" placeholder="e.g., 'Use 5ps equilibration instead', 'Increase temperature to 500K'"></textarea>
                      <button onclick="window.checkpointHandler.submitModification('${sessionId}')">
                          Submit Changes
                      </button>
                  </div>
              </div>
          `;
      }

      renderScript() {
          const script = this.data.script || 'No script available';
          const attempts = this.data.attempts || 0;
          const success = this.data.success;
          const sessionId = this.data.session_id || '';

          return `
              <div class="modal-content">
                  <h2>✋ Review Generated Script</h2>

                  <div class="info-box">
                      ${success
                          ? `✅ Script generated successfully after ${attempts} attempt(s)`
                          : `❌ Script generation failed after ${attempts} attempts`
                      }
                  </div>

                  <div class="script-display">
                      <pre><code>${this.escapeHtml(script)}</code></pre>
                  </div>

                  ${success ? `
                      <div class="modal-actions">
                          <button onclick="window.checkpointHandler.approveScript('${sessionId}')">
                              ▶ Approve & Run
                          </button>
                          <button onclick="window.checkpointHandler.showChangesPanel()">
                              ✎ Request Changes
                          </button>
                      </div>

                      <div id="changes-panel" class="feedback-panel" style="display:none;">
                          <label><strong>What changes would you like?</strong></label>
                          <textarea id="changes-text" placeholder="e.g., 'Add stress tensor dump', 'Add more detailed comments'"></textarea>
                          <button onclick="window.checkpointHandler.submitChanges('${sessionId}')">
                              Submit Changes
                          </button>
                      </div>
                  ` : `
                      <div class="modal-actions">
                          <button onclick="window.checkpointHandler.closeModal()">
                              Close
                          </button>
                      </div>
                  `}
              </div>
          `;
      }

      escapeHtml(text) {
          const div = document.createElement('div');
          div.textContent = text;
          return div.innerHTML;
      }
  }
  ```

---

#### 6.4 Create Checkpoint Handler

**Task**: Add handler class
- [ ] Continue in `<script>` section, add:
  ```javascript
  // ============================================
  // CHECKPOINT HANDLER
  // ============================================

  class CheckpointHandler {
      constructor() {
          this.currentSessionId = null;
      }

      async startWorkflow(userRequest) {
          try {
              console.log('Starting workflow...');

              const response = await fetch('/api/workflow/start', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({request: userRequest})
              });

              if (!response.ok) {
                  throw new Error(`HTTP ${response.status}: ${await response.text()}`);
              }

              const data = await response.json();
              this.currentSessionId = data.session_id;

              // Show parameter checkpoint
              const modal = new CheckpointModal('parameters', data);
              modal.show();

          } catch (error) {
              console.error('Error starting workflow:', error);
              alert(`Error starting workflow: ${error.message}`);
          }
      }

      async approveParameters(sessionId) {
          try {
              console.log('Approving parameters...');

              // Show loading
              const modal = document.getElementById('checkpoint-modal');
              modal.innerHTML = '<div class="modal-content"><h2>Generating script...</h2><p>This may take 30-60 seconds...</p></div>';

              const response = await fetch('/api/checkpoint/parameters', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({
                      session_id: sessionId,
                      action: 'approve'
                  })
              });

              if (!response.ok) {
                  throw new Error(`HTTP ${response.status}`);
              }

              const data = await response.json();
              data.session_id = sessionId;  // Preserve session ID

              // Show script checkpoint
              const scriptModal = new CheckpointModal('script', data);
              scriptModal.show();

          } catch (error) {
              console.error('Error approving parameters:', error);
              alert(`Error: ${error.message}`);
          }
      }

      showModifyPanel() {
          document.getElementById('modify-panel').style.display = 'block';
      }

      async submitModification(sessionId) {
          const feedback = document.getElementById('modification-text').value.trim();

          if (!feedback) {
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
              alert('Preference stored! You can now approve or make more changes.');

              // Close panel
              document.getElementById('modify-panel').style.display = 'none';
              document.getElementById('modification-text').value = '';

          } catch (error) {
              console.error('Error submitting modification:', error);
              alert(`Error: ${error.message}`);
          }
      }

      async approveScript(sessionId) {
          try {
              console.log('Approving script...');

              // Show loading
              const modal = document.getElementById('checkpoint-modal');
              modal.innerHTML = '<div class="modal-content"><h2>Executing simulation...</h2><p>This may take several minutes...</p></div>';

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
              modal.style.display = 'none';

              if (data.status === 'complete') {
                  alert('✅ Simulation completed successfully!');
                  // Refresh page or update UI with results
                  location.reload();
              } else {
                  alert(`Simulation status: ${data.status}\n${data.message || ''}`);
              }

          } catch (error) {
              console.error('Error approving script:', error);
              alert(`Error: ${error.message}`);
          }
      }

      showChangesPanel() {
          document.getElementById('changes-panel').style.display = 'block';
      }

      async submitChanges(sessionId) {
          const feedback = document.getElementById('changes-text').value.trim();

          if (!feedback) {
              alert('Please describe the changes you want');
              return;
          }

          try {
              // Show loading
              const modal = document.getElementById('checkpoint-modal');
              modal.innerHTML = '<div class="modal-content"><h2>Regenerating script...</h2><p>This may take 30-60 seconds...</p></div>';

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
              data.session_id = sessionId;

              // Show updated script
              const scriptModal = new CheckpointModal('script', data);
              scriptModal.show();

          } catch (error) {
              console.error('Error requesting changes:', error);
              alert(`Error: ${error.message}`);
          }
      }

      closeModal() {
          document.getElementById('checkpoint-modal').style.display = 'none';
      }
  }

  // Initialize global handler
  window.checkpointHandler = new CheckpointHandler();
  ```

---

#### 6.5 Integrate with Existing Chat UI

**Task**: Hook up to send button
- [ ] Find your existing chat send button handler
- [ ] Modify it to use the new workflow:
  ```javascript
  // Find this in your existing code:
  // sendButton.addEventListener('click', () => { ... })

  // Replace with:
  sendButton.addEventListener('click', () => {
      const message = chatInput.value.trim();

      if (message) {
          // Use new checkpoint workflow
          window.checkpointHandler.startWorkflow(message);

          // Clear input
          chatInput.value = '';
      }
  });

  // Also handle Enter key
  chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendButton.click();
      }
  });
  ```

**Task**: Test the UI
- [ ] Open the app in browser: `http://localhost:5002`
- [ ] Type: "Run a Cu simulation at 300K"
- [ ] Click send
- [ ] Should see parameter checkpoint modal
- [ ] Click "Approve"
- [ ] Should see script checkpoint modal after generation
- [ ] Test all buttons work

**End of Day 6**: Frontend complete ✓

---

### Day 7: Polish & Deploy

#### 7.1 Add Memory Viewer (Optional)

**Task**: Add endpoint to view memories
- [ ] Add to `app.py`:
  ```python
  @app.route('/api/memories', methods=['GET'])
  def get_memories():
      """Get all memories for viewing"""
      try:
          results = memory.get_all(user_id=USER_ID)
          memories = results.get('results', [])

          return jsonify({
              "total": len(memories),
              "memories": memories
          })
      except Exception as e:
          return jsonify({"error": str(e)}), 500
  ```

**Task**: Add UI to view memories (optional)
- [ ] Add button in index.html to show memories
- [ ] Display in modal or sidebar

---

#### 7.2 Update Configuration

**Task**: Update config.py
- [ ] Add these settings:
  ```python
  # === AGENTIFY SETTINGS ===

  # Iteration limits
  MAX_LAMMPS_GENERATION_ATTEMPTS = 3
  MAX_OVITO_GENERATION_ATTEMPTS = 2

  # Checkpoints (single user - always enabled)
  CHECKPOINT_PARAMETERS_REQUIRED = True
  CHECKPOINT_SCRIPT_REQUIRED = True

  # Memory (using Mem0)
  # Mem0 config is in .mem0/ directory (auto-managed)
  ```

---

#### 7.3 Update .gitignore

**Task**: Add Mem0 directory to .gitignore
- [ ] Open `.gitignore` (or create if doesn't exist)
- [ ] Add:
  ```
  # Mem0 storage
  .mem0/

  # Test scripts
  test_*.py
  ```

---

#### 7.4 Create Documentation

**Task**: Update README.md
- [ ] Add section about new features:
  ```markdown
  ## Agentify Features (Multi-Turn Agents + Memory)

  ### Iterative Agent Collaboration
  - Generator and Reviewer agents work together iteratively
  - Up to 3 attempts to produce correct scripts
  - Agents learn from mistakes and improve

  ### Agent Memory (Mem0)
  - System remembers your preferences
  - Learns from successful patterns
  - Improves over time based on your feedback

  ### User Checkpoints
  - ✋ Review parameters before generation
  - ✋ Review scripts before execution
  - Request changes at any step
  - Feedback stored for future runs

  ### Installation

  ```bash
  pip install dspy-ai mem0ai
  ```

  ### Usage

  1. Start a workflow: Enter your request in the chat
  2. Review parameters: Approve or modify proposed settings
  3. Review script: Approve to run, or request changes
  4. System learns from your feedback for next time!

  ### Memory Storage

  Memories are stored in `.mem0/` directory (auto-created).
  To reset memories: `rm -rf .mem0/`
  ```

**Task**: Create CHANGELOG.md
- [ ] Document the changes:
  ```markdown
  # Changelog

  ## [Agentify v4 LEAN] - 2025-11-16

  ### Added
  - Iterative agent collaboration (Generator ↔ Reviewer)
  - Agent memory with Mem0 (learns from user feedback)
  - User checkpoints for parameter and script approval
  - Request changes workflow
  - Memory-informed parameter proposals

  ### Dependencies
  - dspy-ai >= 2.5.0
  - mem0ai >= 0.1.0

  ### Changed
  - Workflow now requires user approval at checkpoints
  - Scripts are reviewed and improved iteratively
  - System learns preferences over time
  ```

---

#### 7.5 Final Testing

**Task**: Complete system test
- [ ] Clear all memories: `rm -rf .mem0/`
- [ ] Restart server
- [ ] Run workflow 1: "Run Cu at 300K"
  - [ ] Approve parameters
  - [ ] Modify something ("add comments")
  - [ ] Approve script
  - [ ] Verify simulation runs
- [ ] Run workflow 2: "Run Cu at 500K"
  - [ ] Verify memories shown from run 1
  - [ ] Verify preferences applied
  - [ ] Complete workflow
- [ ] Run workflow 3: "Run Al at 300K"
  - [ ] Should be fast (system learned)

**Task**: Performance check
- [ ] Verify iteration doesn't take > 2 minutes
- [ ] Verify UI is responsive
- [ ] Check memory file size: `du -sh .mem0/`
- [ ] Should be < 50MB

---

#### 7.6 Cleanup

**Task**: Remove test files
- [ ] Delete test scripts:
  ```bash
  rm test_dspy.py test_mem0.py test_iteration.py test_full_workflow.py test_memory_learning.py test_request_changes.py
  ```
- [ ] Or move to `tests/` directory if you want to keep them

**Task**: Final commit
- [ ] Review all changes:
  ```bash
  git status
  git diff
  ```
- [ ] Stage files:
  ```bash
  git add prompts.py agents.py app.py index.html config.py requirements.txt README.md
  ```
- [ ] Commit:
  ```bash
  git commit -m "Implement agentify v4 LEAN: iterative agents + Mem0 memory + checkpoints"
  ```
- [ ] Push:
  ```bash
  git push origin claude/agentify-multi-turn-plan-015TBZuSZD3GtccmWrbAjs7n
  ```

**End of Day 7**: COMPLETE! ✓

---

## Implementation Complete!

### What You Built

✅ **Iterative Agent Collaboration**
- Generator ↔ Reviewer loops (up to 3 attempts)
- Automatic error correction
- Full conversation tracking

✅ **Agent Memory with Mem0**
- Learns from user feedback
- Stores preferences and corrections
- Retrieves relevant memories automatically
- Improves over time

✅ **User Checkpoints**
- Parameter approval
- Script review
- Request changes workflow
- Feedback stored in memory

### Code Summary

| File | Changes | Purpose |
|------|---------|---------|
| `prompts.py` | +150 lines (NEW) | DSPy signatures |
| `agents.py` | +100 lines | Iterative modules with Mem0 |
| `app.py` | +100 lines | Checkpoint endpoints |
| `index.html` | +200 lines | Checkpoint UI |
| `config.py` | +10 lines | Settings |
| **Total** | **~350 lines** | Complete implementation |

### Dependencies Added

```
dspy-ai>=2.5.0
mem0ai>=0.1.0
```

### Success Metrics

- [ ] Scripts pass within 1-2 attempts (90%+)
- [ ] Memories stored and retrieved correctly
- [ ] Second run shows improvement from first
- [ ] User can approve/modify at checkpoints
- [ ] System learns over time

### Next Steps

1. **Use it**: Run several simulations to build up memory
2. **Monitor**: Watch how the system improves
3. **Tune**: Adjust max_attempts if needed
4. **Extend**: Add OVITO iteration if desired

**Congratulations! You've implemented a learning, iterative agent system!** 🎉
