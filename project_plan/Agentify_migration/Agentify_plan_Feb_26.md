# ChatMD Agentify Enhancement Plan
## True Agentic Workflow with Planning, Tool Selection, and Confidence Scoring

**Date**: 2026-02-02
**Base**: IMPLEMENTATION_PLAN_V4_LEAN_FINAL.md
**Status**: Enhancement Proposal
**Timeline**: +3-4 days on top of base plan

---

## Overview

This document enhances the V4 LEAN FINAL plan with capabilities needed for a **true agentic workflow**:

1. **Planning & Verification Phase** - Verify feasibility before generation
2. **Dynamic Tool Selection** - Agents choose tools based on context
3. **Confidence Scoring** - Agents express uncertainty
4. **Intelligent Error Recovery** - Adaptive retry strategies
5. **Session Persistence** - Survive server restarts
6. **Semantic Memory Enhancement** - Better Mem0 utilization

These enhancements transform the system from a "smart pipeline" into a genuine agent that reasons, plans, and adapts.

---

## Enhancement 1: Planning & Verification Phase

### Problem
The current plan goes directly from Orchestrator → Generator without verifying the proposed parameters are feasible. This leads to:
- Wasted generation attempts on impossible simulations
- No early warning for users about potential issues
- Generator discovering problems that should have been caught earlier

### Solution: Add PlanVerifier Agent

**New DSPy Signature** (add to `prompts.py`):

```python
class VerifySimulationPlan(dspy.Signature):
    """Verify proposed simulation parameters are physically reasonable and executable.

    Check that:
    1. KIM model exists and supports the requested element
    2. Temperature/pressure are physically reasonable
    3. Simulation duration and timestep are compatible
    4. Required properties can be computed
    5. System size is appropriate for the physics

    OUTPUT FORMAT:
    - "VERIFICATION: PASS" if plan is feasible
    - "VERIFICATION: FAIL" with specific issues
    - "VERIFICATION: WARNING" for non-critical concerns
    """

    proposed_parameters: str = dspy.InputField(
        desc="Simulation parameters from orchestrator"
    )

    kim_registry: str = dspy.InputField(
        desc="Available KIM models (from registry)"
    )

    element_properties: str = dspy.InputField(
        desc="Known element properties (lattice constant, structure, melting point)"
    )

    verification_result: str = dspy.OutputField(
        desc="VERIFICATION: PASS/FAIL/WARNING with explanation"
    )

    issues: list[str] = dspy.OutputField(
        desc="List of specific issues or warnings"
    )

    suggestions: list[str] = dspy.OutputField(
        desc="Suggested fixes for any issues found"
    )

    confidence: float = dspy.OutputField(
        desc="0.0-1.0 confidence that this simulation will succeed"
    )
```

**New Module** (add to `agents.py`):

```python
class PlanVerifier(dspy.Module):
    """
    Verifies simulation plans before generation.

    Checks:
    - KIM model availability and element compatibility
    - Physical reasonableness of parameters
    - Potential issues that would cause simulation failure
    """

    def __init__(self):
        super().__init__()
        self.verifier = dspy.ChainOfThought(VerifySimulationPlan)

    def forward(self, parameters: str) -> dspy.Prediction:
        # Load KIM registry
        from kim_utils import load_kim_registry, get_element_properties

        kim_registry = load_kim_registry()

        # Extract element from parameters (simple parsing)
        element = self._extract_element(parameters)
        element_props = get_element_properties(element) if element else "Unknown element"

        # Run verification
        result = self.verifier(
            proposed_parameters=parameters,
            kim_registry=kim_registry,
            element_properties=str(element_props)
        )

        return dspy.Prediction(
            verification_result=result.verification_result,
            issues=result.issues,
            suggestions=result.suggestions,
            confidence=result.confidence,
            is_feasible="PASS" in result.verification_result or "WARNING" in result.verification_result
        )

    def _extract_element(self, parameters: str) -> str:
        """Extract element symbol from parameters string"""
        import re
        # Look for "Element: Cu" or similar patterns
        match = re.search(r'Element:\s*(\w+)', parameters, re.IGNORECASE)
        if match:
            return match.group(1)
        # Look for common element symbols
        elements = ['Cu', 'Al', 'Ni', 'Fe', 'Au', 'Ag', 'Pt', 'Ti', 'Mg', 'Zn']
        for elem in elements:
            if elem in parameters:
                return elem
        return None
```

**Updated Workflow**:

```
User Request
    ↓
Orchestrator (propose parameters)
    ↓
✋ CHECKPOINT 0: Plan Verification (NEW)
    ↓ (if FAIL → show issues, ask user to modify request)
    ↓ (if WARNING → show warnings, let user proceed or modify)
    ↓ (if PASS → continue)
✋ CHECKPOINT 1: Parameter Approval
    ↓
Generator → Reviewer Loop
    ↓
✋ CHECKPOINT 2: Script Approval
    ↓
Execution
```

**New Endpoint** (add to `app.py`):

```python
@app.route('/api/workflow/start', methods=['POST'])
def start_workflow():
    """Start workflow with verification"""
    user_request = request.json.get('request')

    # Step 1: Orchestrator proposes parameters
    proposal = orchestrator.execute(user_request)

    # Step 2: Verify the plan (NEW)
    verifier = PlanVerifier()
    verification = verifier(parameters=proposal)

    # Step 3: Create session with verification info
    session_id = str(uuid.uuid4())
    workflow_sessions[session_id] = {
        "request": user_request,
        "stage": "verification",
        "proposal": proposal,
        "verification": {
            "result": verification.verification_result,
            "issues": verification.issues,
            "suggestions": verification.suggestions,
            "confidence": verification.confidence,
            "is_feasible": verification.is_feasible
        }
    }

    # Step 4: Return with verification info
    return jsonify({
        "session_id": session_id,
        "checkpoint": "verification",
        "proposal": proposal,
        "verification": {
            "result": verification.verification_result,
            "confidence": verification.confidence,
            "issues": verification.issues,
            "suggestions": verification.suggestions
        }
    })
```

---

## Enhancement 2: Dynamic Tool Selection

### Problem
Current agents have fixed capabilities. The Orchestrator always proposes parameters, Generator always generates scripts. But a true agent should:
- Choose the right tool for the task
- Decide whether to query external resources
- Know when to ask for clarification vs. proceed

### Solution: Tool Registry and Selection Agent

**Tool Registry** (new file `tool_registry.py`):

```python
"""
tool_registry.py - Dynamic tool registry for agent use
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import dspy

@dataclass
class Tool:
    """Represents an available tool"""
    name: str
    description: str
    input_schema: str
    output_schema: str
    function: Callable
    requires_approval: bool = False

class ToolRegistry:
    """Registry of available tools for agents"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools"""

        # KIM Registry Query
        self.register(Tool(
            name="query_kim_registry",
            description="Search for KIM potentials by element or name. Use when you need to find available interatomic potentials.",
            input_schema="element: str (e.g., 'Cu', 'Al')",
            output_schema="list of matching KIM model names with download status",
            function=self._query_kim_registry,
            requires_approval=False
        ))

        # OpenKIM Search
        self.register(Tool(
            name="search_openkim",
            description="Search OpenKIM.org for potentials not in local registry. Use when local registry has no matches.",
            input_schema="element: str, model_type: str (optional)",
            output_schema="list of available models from OpenKIM.org",
            function=self._search_openkim,
            requires_approval=False
        ))

        # Element Properties
        self.register(Tool(
            name="get_element_properties",
            description="Get physical properties of an element (lattice constant, structure, melting point). Use for parameter validation.",
            input_schema="element: str",
            output_schema="dict with lattice_constant, structure, melting_point, atomic_mass",
            function=self._get_element_properties,
            requires_approval=False
        ))

        # Generate LAMMPS Script
        self.register(Tool(
            name="generate_lammps_script",
            description="Generate a complete LAMMPS input script from parameters. Use after parameters are verified and approved.",
            input_schema="parameters: str, user_feedback: str (optional)",
            output_schema="LAMMPS script string",
            function=self._generate_lammps_script,
            requires_approval=True  # Requires checkpoint
        ))

        # Run Simulation
        self.register(Tool(
            name="run_simulation",
            description="Execute a LAMMPS simulation. Only use after script is approved by user.",
            input_schema="script: str",
            output_schema="simulation results (thermo data, trajectories)",
            function=self._run_simulation,
            requires_approval=True  # Requires checkpoint
        ))

        # Generate Visualization
        self.register(Tool(
            name="generate_visualization",
            description="Generate OVITO visualization script for trajectory. Use when user requests specific visualization.",
            input_schema="visualization_request: str, available_properties: list",
            output_schema="OVITO Python script",
            function=self._generate_visualization,
            requires_approval=False
        ))

    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> str:
        """Get formatted list of available tools for prompts"""
        lines = ["Available Tools:"]
        for name, tool in self.tools.items():
            lines.append(f"\n- {name}: {tool.description}")
            lines.append(f"  Input: {tool.input_schema}")
            lines.append(f"  Output: {tool.output_schema}")
            if tool.requires_approval:
                lines.append(f"  (Requires user approval)")
        return "\n".join(lines)

    # Tool implementations
    def _query_kim_registry(self, element: str) -> List[Dict]:
        from kim_utils import query_local_registry
        return query_local_registry(element)

    def _search_openkim(self, element: str, model_type: str = None) -> List[Dict]:
        from kim_utils import query_openkim_models
        return query_openkim_models(element)

    def _get_element_properties(self, element: str) -> Dict:
        from kim_utils import get_element_properties
        return get_element_properties(element)

    def _generate_lammps_script(self, parameters: str, user_feedback: str = "") -> str:
        # Delegates to IterativeLammpsGeneration
        result = lammps_generator(parameters=parameters, user_feedback=user_feedback)
        return result.script

    def _run_simulation(self, script: str) -> Dict:
        from tools import run_lammps_simulation
        return run_lammps_simulation(script)

    def _generate_visualization(self, request: str, properties: List[str]) -> str:
        from ovito_script_generator import OVITOScriptGenerator
        generator = OVITOScriptGenerator()
        return generator.generate(request, properties)

# Global registry
tool_registry = ToolRegistry()
```

**Tool Selection Signature** (add to `prompts.py`):

```python
class SelectTools(dspy.Signature):
    """Select which tools to use for a given task.

    Analyze the user's request and current context to determine:
    1. Which tools are needed to fulfill the request
    2. In what order they should be called
    3. What information is missing that requires tool use

    DECISION GUIDELINES:
    - Use query_kim_registry BEFORE proposing a KIM model
    - Use get_element_properties to validate temperature/structure choices
    - Use generate_lammps_script only AFTER parameters are approved
    - Use run_simulation only AFTER script is approved

    Be conservative - only select tools that are truly needed.
    """

    user_request: str = dspy.InputField(desc="The user's simulation request")

    current_context: str = dspy.InputField(
        desc="Current workflow state (parameters proposed, script generated, etc.)"
    )

    available_tools: str = dspy.InputField(
        desc="List of available tools with descriptions"
    )

    selected_tools: list[str] = dspy.OutputField(
        desc="Ordered list of tool names to use"
    )

    reasoning: str = dspy.OutputField(
        desc="Explanation of why these tools were selected"
    )

    missing_info: list[str] = dspy.OutputField(
        desc="Information that should be gathered before proceeding"
    )
```

**Smart Orchestrator with Tool Selection** (update in `agents.py`):

```python
class SmartOrchestrator(dspy.Module):
    """
    Orchestrator that dynamically selects and uses tools.

    Unlike fixed orchestrator, this agent:
    - Queries KIM registry before proposing models
    - Validates element properties before setting parameters
    - Adapts its approach based on available information
    """

    def __init__(self, tool_registry: ToolRegistry):
        super().__init__()
        self.tool_registry = tool_registry
        self.tool_selector = dspy.ChainOfThought(SelectTools)
        self.parameter_proposer = dspy.ChainOfThought(AnalyzeRequest)

    def forward(self, request: str, memories: str = "") -> dspy.Prediction:
        # Step 1: Determine which tools to use
        tool_selection = self.tool_selector(
            user_request=request,
            current_context="Initial request analysis",
            available_tools=self.tool_registry.list_tools()
        )

        # Step 2: Execute selected tools to gather info
        gathered_info = {}
        for tool_name in tool_selection.selected_tools:
            tool = self.tool_registry.get_tool(tool_name)

            if tool and not tool.requires_approval:
                # Execute information-gathering tools automatically
                if tool_name == "query_kim_registry":
                    element = self._extract_element(request)
                    if element:
                        gathered_info['kim_models'] = tool.function(element)

                elif tool_name == "get_element_properties":
                    element = self._extract_element(request)
                    if element:
                        gathered_info['element_properties'] = tool.function(element)

        # Step 3: Propose parameters with gathered information
        context = self._format_gathered_info(gathered_info)

        proposal = self.parameter_proposer(
            request=request,
            kim_models=context.get('kim_models', 'No models queried'),
            memories=memories
        )

        return dspy.Prediction(
            proposal=proposal.proposal,
            tools_used=tool_selection.selected_tools,
            reasoning=tool_selection.reasoning,
            gathered_info=gathered_info
        )

    def _extract_element(self, text: str) -> str:
        # Same as PlanVerifier
        import re
        match = re.search(r'Element:\s*(\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        elements = ['Cu', 'Al', 'Ni', 'Fe', 'Au', 'Ag', 'Pt', 'Ti', 'Mg', 'Zn']
        for elem in elements:
            if elem.lower() in text.lower():
                return elem
        return None

    def _format_gathered_info(self, info: Dict) -> Dict:
        formatted = {}
        if 'kim_models' in info:
            models = info['kim_models']
            formatted['kim_models'] = "\n".join([
                f"- {m['name']} ({'downloaded' if m.get('downloaded') else 'available'})"
                for m in models[:10]  # Top 10
            ])
        if 'element_properties' in info:
            props = info['element_properties']
            formatted['element_properties'] = f"""
            Lattice constant: {props.get('lattice_constant', 'unknown')} A
            Structure: {props.get('structure', 'unknown')}
            Melting point: {props.get('melting_point', 'unknown')} K
            """
        return formatted
```

---

## Enhancement 3: Confidence Scoring

### Problem
Agents currently output binary PASS/FAIL. Users don't know:
- How confident the agent is in its output
- Whether the output is borderline or clearly correct
- When to trust the agent vs. manually verify

### Solution: Add Confidence to All Agent Outputs

**Updated Signatures with Confidence**:

```python
# Update GenerateLAMMPSScript
class GenerateLAMMPSScript(dspy.Signature):
    """...(existing docstring)..."""

    # ... existing fields ...

    script: str = dspy.OutputField(desc="Complete executable LAMMPS script")

    reasoning: str = dspy.OutputField(desc="Explanation of key decisions")

    confidence: float = dspy.OutputField(
        desc="""Confidence score 0.0-1.0:
        - 0.9-1.0: Very confident, standard simulation
        - 0.7-0.9: Confident, some minor uncertainties
        - 0.5-0.7: Moderate confidence, user should review carefully
        - 0.3-0.5: Low confidence, significant uncertainties
        - 0.0-0.3: Very low confidence, likely needs user input
        """
    )

    uncertainties: list[str] = dspy.OutputField(
        desc="Specific aspects the agent is uncertain about"
    )


# Update ReviewLAMMPSScript
class ReviewLAMMPSScript(dspy.Signature):
    """...(existing docstring)..."""

    # ... existing fields ...

    review: str = dspy.OutputField(desc="REVIEW: PASS/FAIL with explanation")

    issues: list[str] = dspy.OutputField(desc="List of specific issues")

    confidence: float = dspy.OutputField(
        desc="""Confidence in review accuracy 0.0-1.0:
        - High (0.9+): Clear pass or obvious critical issues
        - Medium (0.7-0.9): Minor concerns or edge cases
        - Low (<0.7): Uncertain about correctness, recommend manual review
        """
    )

    review_coverage: float = dspy.OutputField(
        desc="0.0-1.0 estimate of how thoroughly the script was reviewed"
    )
```

**Confidence-Aware Iteration** (update `IterativeLammpsGeneration`):

```python
class IterativeLammpsGeneration(dspy.Module):
    """Iterative generation with confidence-aware decisions"""

    def __init__(self, max_attempts=3, min_confidence=0.7):
        super().__init__()
        self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
        self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
        self.max_attempts = max_attempts
        self.min_confidence = min_confidence  # NEW: minimum acceptable confidence

    def forward(self, parameters: str, user_feedback: str = ""):
        # ... memory retrieval code ...

        for attempt in range(1, self.max_attempts + 1):
            # Generate with confidence
            gen_result = self.generator(
                parameters=parameters,
                memories=memories_text,
                user_feedback=user_feedback,
                feedback=feedback
            )

            script = gen_result.script
            gen_confidence = getattr(gen_result, 'confidence', 0.8)
            uncertainties = getattr(gen_result, 'uncertainties', [])

            # Review with confidence
            review_result = self.reviewer(
                script=script,
                parameters=parameters
            )

            review_confidence = getattr(review_result, 'confidence', 0.8)

            # Track in conversation
            conversation.append({
                "role": "generator",
                "attempt": attempt,
                "content": script,
                "confidence": gen_confidence,
                "uncertainties": uncertainties
            })

            conversation.append({
                "role": "reviewer",
                "attempt": attempt,
                "content": review_result.review,
                "confidence": review_confidence,
                "issues": review_result.issues
            })

            # Decision logic with confidence
            if "REVIEW: PASS" in review_result.review:
                # Check if confidence is sufficient
                combined_confidence = (gen_confidence + review_confidence) / 2

                if combined_confidence >= self.min_confidence:
                    return dspy.Prediction(
                        script=script,
                        review=review_result.review,
                        attempts=attempt,
                        conversation=conversation,
                        success=True,
                        confidence=combined_confidence,
                        uncertainties=uncertainties,
                        needs_manual_review=False
                    )
                else:
                    # Pass but low confidence - flag for manual review
                    return dspy.Prediction(
                        script=script,
                        review=review_result.review,
                        attempts=attempt,
                        conversation=conversation,
                        success=True,
                        confidence=combined_confidence,
                        uncertainties=uncertainties,
                        needs_manual_review=True,
                        manual_review_reason=f"Low confidence ({combined_confidence:.2f})"
                    )

            # Build feedback for next iteration
            feedback = f"Attempt {attempt} failed. Issues: {', '.join(review_result.issues)}"

        # Failed after max attempts
        return dspy.Prediction(
            script=script,
            review=review_result.review,
            attempts=self.max_attempts,
            conversation=conversation,
            success=False,
            confidence=0.0,
            uncertainties=["Failed after maximum attempts"],
            needs_manual_review=True
        )
```

**UI Updates for Confidence Display** (add to `index.html`):

```javascript
// In CheckpointModal.renderScript()
renderScript() {
    const confidence = this.data.confidence || 0;
    const needsReview = this.data.needs_manual_review || false;
    const uncertainties = this.data.uncertainties || [];

    const confidenceColor = confidence >= 0.8 ? '#4caf50' :
                           confidence >= 0.6 ? '#ff9800' : '#f44336';
    const confidenceLabel = confidence >= 0.8 ? 'High' :
                           confidence >= 0.6 ? 'Medium' : 'Low';

    return `
        <div class="modal-content">
            <h2>Review Generated Script</h2>

            <!-- Confidence indicator -->
            <div class="confidence-box" style="border-left: 4px solid ${confidenceColor};">
                <strong>Agent Confidence: ${confidenceLabel} (${(confidence * 100).toFixed(0)}%)</strong>
                ${needsReview ? '<span class="warning-badge">Manual Review Recommended</span>' : ''}
            </div>

            ${uncertainties.length > 0 ? `
                <div class="uncertainties-box">
                    <strong>Uncertainties:</strong>
                    <ul>
                        ${uncertainties.map(u => `<li>${u}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}

            <!-- Rest of script display -->
            ...
        </div>
    `;
}
```

---

## Enhancement 4: Intelligent Error Recovery

### Problem
Current plan uses fixed 3 attempts with same strategy. But different errors need different responses:
- Syntax errors → Minor fix
- Wrong KIM model → Major restructure
- Timeout → Simplify simulation
- User feedback → Targeted changes

### Solution: Error Classification and Adaptive Retry

**Error Classifier** (add to `agents.py`):

```python
from enum import Enum
from dataclasses import dataclass

class ErrorSeverity(Enum):
    CRITICAL = "critical"      # Cannot proceed without major changes
    RECOVERABLE = "recoverable" # Can fix with targeted modifications
    WARNING = "warning"         # Non-blocking, informational

class ErrorCategory(Enum):
    KIM_MODEL = "kim_model"           # Wrong or missing KIM model
    SYNTAX = "syntax"                  # LAMMPS syntax error
    PHYSICS = "physics"                # Physically unreasonable parameters
    MISSING_COMMAND = "missing"        # Required command missing
    ORDER = "order"                    # Commands in wrong order
    PARAMETER_MISMATCH = "mismatch"   # Inconsistent parameters
    RESOURCE = "resource"              # Memory/time constraints

@dataclass
class ClassifiedError:
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    suggested_fix: str
    retry_strategy: str  # "modify", "restructure", "simplify", "ask_user"

class ErrorClassifier:
    """Classifies errors and determines retry strategy"""

    def classify(self, issues: list[str], review_text: str) -> list[ClassifiedError]:
        """Classify each issue and determine appropriate response"""
        classified = []

        for issue in issues:
            issue_lower = issue.lower()

            # KIM model issues
            if any(kw in issue_lower for kw in ['kim', 'potential', 'pair_style']):
                classified.append(ClassifiedError(
                    category=ErrorCategory.KIM_MODEL,
                    severity=ErrorSeverity.CRITICAL,
                    description=issue,
                    suggested_fix="Use correct KIM model from registry",
                    retry_strategy="restructure"
                ))

            # Syntax issues
            elif any(kw in issue_lower for kw in ['syntax', 'command', 'undefined']):
                classified.append(ClassifiedError(
                    category=ErrorCategory.SYNTAX,
                    severity=ErrorSeverity.RECOVERABLE,
                    description=issue,
                    suggested_fix="Fix LAMMPS syntax",
                    retry_strategy="modify"
                ))

            # Order issues
            elif any(kw in issue_lower for kw in ['order', 'before', 'after', 'sequence']):
                classified.append(ClassifiedError(
                    category=ErrorCategory.ORDER,
                    severity=ErrorSeverity.RECOVERABLE,
                    description=issue,
                    suggested_fix="Reorder commands correctly",
                    retry_strategy="modify"
                ))

            # Physics issues
            elif any(kw in issue_lower for kw in ['temperature', 'pressure', 'unstable', 'melting']):
                classified.append(ClassifiedError(
                    category=ErrorCategory.PHYSICS,
                    severity=ErrorSeverity.CRITICAL,
                    description=issue,
                    suggested_fix="Adjust physical parameters",
                    retry_strategy="ask_user"
                ))

            # Missing command
            elif any(kw in issue_lower for kw in ['missing', 'required', 'must have']):
                classified.append(ClassifiedError(
                    category=ErrorCategory.MISSING_COMMAND,
                    severity=ErrorSeverity.RECOVERABLE,
                    description=issue,
                    suggested_fix="Add missing command",
                    retry_strategy="modify"
                ))

            # Default
            else:
                classified.append(ClassifiedError(
                    category=ErrorCategory.SYNTAX,
                    severity=ErrorSeverity.RECOVERABLE,
                    description=issue,
                    suggested_fix="Review and fix issue",
                    retry_strategy="modify"
                ))

        return classified

    def determine_retry_strategy(self, errors: list[ClassifiedError]) -> dict:
        """Determine overall retry strategy based on classified errors"""

        # Check for critical errors that need user input
        critical_physics = [e for e in errors
                          if e.severity == ErrorSeverity.CRITICAL
                          and e.category == ErrorCategory.PHYSICS]
        if critical_physics:
            return {
                "strategy": "ask_user",
                "reason": "Physical parameters may be incorrect",
                "issues": [e.description for e in critical_physics],
                "max_auto_retries": 0
            }

        # Check for KIM model issues
        kim_errors = [e for e in errors if e.category == ErrorCategory.KIM_MODEL]
        if kim_errors:
            return {
                "strategy": "restructure",
                "reason": "KIM model configuration needs restructuring",
                "issues": [e.description for e in kim_errors],
                "max_auto_retries": 2
            }

        # Recoverable errors - standard retry
        recoverable = [e for e in errors if e.severity == ErrorSeverity.RECOVERABLE]
        if recoverable:
            return {
                "strategy": "modify",
                "reason": "Minor fixes needed",
                "issues": [e.description for e in recoverable],
                "max_auto_retries": 3
            }

        return {
            "strategy": "proceed",
            "reason": "No blocking errors",
            "issues": [],
            "max_auto_retries": 3
        }
```

**Adaptive Iteration Module**:

```python
class AdaptiveLammpsGeneration(dspy.Module):
    """
    Iterative generation with adaptive retry strategies.

    Different from basic iteration:
    - Classifies errors to determine best approach
    - Adjusts strategy based on error type
    - Knows when to ask user vs. retry automatically
    - Uses different prompts for different error types
    """

    def __init__(self, max_attempts=3):
        super().__init__()
        self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
        self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
        self.restructure_generator = dspy.ChainOfThought(RestructureLAMMPSScript)  # NEW
        self.error_classifier = ErrorClassifier()
        self.max_attempts = max_attempts

    def forward(self, parameters: str, user_feedback: str = ""):
        # ... memory retrieval ...

        conversation = []
        attempt = 0
        current_strategy = "modify"

        while attempt < self.max_attempts:
            attempt += 1

            # Choose generator based on strategy
            if current_strategy == "restructure":
                gen_result = self.restructure_generator(
                    parameters=parameters,
                    memories=memories_text,
                    previous_issues=feedback,
                    restructure_focus="KIM model configuration"
                )
            else:
                gen_result = self.generator(
                    parameters=parameters,
                    memories=memories_text,
                    user_feedback=user_feedback,
                    feedback=feedback
                )

            script = gen_result.script

            # Review
            review_result = self.reviewer(script=script, parameters=parameters)

            # Check success
            if "REVIEW: PASS" in review_result.review:
                return dspy.Prediction(
                    script=script,
                    review=review_result.review,
                    attempts=attempt,
                    success=True,
                    strategy_used=current_strategy
                )

            # Classify errors
            classified_errors = self.error_classifier.classify(
                review_result.issues,
                review_result.review
            )

            retry_info = self.error_classifier.determine_retry_strategy(classified_errors)

            # Check if we should ask user
            if retry_info["strategy"] == "ask_user":
                return dspy.Prediction(
                    script=script,
                    review=review_result.review,
                    attempts=attempt,
                    success=False,
                    needs_user_input=True,
                    user_input_reason=retry_info["reason"],
                    classified_errors=classified_errors
                )

            # Update strategy for next attempt
            current_strategy = retry_info["strategy"]

            # Check if we've exceeded auto-retry limit for this strategy
            if attempt >= retry_info["max_auto_retries"]:
                break

            # Build targeted feedback
            feedback = self._build_targeted_feedback(classified_errors, current_strategy)

        # Failed after max attempts
        return dspy.Prediction(
            script=script,
            review=review_result.review,
            attempts=attempt,
            success=False,
            classified_errors=classified_errors
        )

    def _build_targeted_feedback(self, errors: list[ClassifiedError], strategy: str) -> str:
        """Build feedback string targeted to the error type"""

        if strategy == "restructure":
            return f"""
            RESTRUCTURE REQUIRED: Previous script had fundamental issues.

            Critical problems:
            {chr(10).join(f'- {e.description}' for e in errors if e.severity == ErrorSeverity.CRITICAL)}

            Start fresh with correct structure. Focus on:
            {chr(10).join(f'- {e.suggested_fix}' for e in errors)}
            """

        else:  # modify
            return f"""
            MODIFICATION NEEDED: Fix these specific issues:

            {chr(10).join(f'- {e.description} -> {e.suggested_fix}' for e in errors)}

            Keep the rest of the script intact.
            """
```

---

## Enhancement 5: Session Persistence

### Problem
Current plan uses in-memory dict for sessions. Server restart = lost state.

### Solution: SQLite-based Session Storage

**Session Manager** (new file `session_manager.py`):

```python
"""
session_manager.py - Persistent session storage
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class SessionManager:
    """SQLite-backed session storage"""

    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT 'default',
                    request TEXT,
                    stage TEXT,
                    proposal TEXT,
                    script TEXT,
                    verification TEXT,
                    conversation TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON sessions(user_id, created_at DESC)
            """)

    def create(self, session_id: str, data: Dict[str, Any]) -> None:
        """Create new session"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions
                (session_id, user_id, request, stage, proposal, script,
                 verification, conversation, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                data.get('user_id', 'default'),
                data.get('request', ''),
                data.get('stage', 'initial'),
                data.get('proposal', ''),
                data.get('script', ''),
                json.dumps(data.get('verification', {})),
                json.dumps(data.get('conversation', [])),
                json.dumps(data.get('metadata', {})),
                now,
                now
            ))

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if row:
                return {
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'request': row['request'],
                    'stage': row['stage'],
                    'proposal': row['proposal'],
                    'script': row['script'],
                    'verification': json.loads(row['verification'] or '{}'),
                    'conversation': json.loads(row['conversation'] or '[]'),
                    'metadata': json.loads(row['metadata'] or '{}'),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
            return None

    def update(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session fields"""
        # Build dynamic update query
        fields = []
        values = []

        for key, value in updates.items():
            if key in ['verification', 'conversation', 'metadata']:
                value = json.dumps(value)
            fields.append(f"{key} = ?")
            values.append(value)

        fields.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(session_id)

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?",
                values
            )
            return result.rowcount > 0

    def list_recent(self, user_id: str = 'default', limit: int = 10) -> list:
        """List recent sessions for a user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT session_id, request, stage, created_at, updated_at
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()

            return [dict(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        """Delete a session"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            return result.rowcount > 0

# Global session manager
session_manager = SessionManager()
```

**Update app.py to use SessionManager**:

```python
from session_manager import session_manager

@app.route('/api/workflow/start', methods=['POST'])
def start_workflow():
    # ... existing code ...

    # Create persistent session
    session_manager.create(session_id, {
        'request': user_request,
        'stage': 'verification',
        'proposal': proposal,
        'verification': verification_dict,
        'metadata': {'memories_used': memories_list}
    })

    return jsonify({...})

@app.route('/api/checkpoint/parameters', methods=['POST'])
def checkpoint_parameters():
    session_id = request.json.get('session_id')

    # Get from persistent storage
    session = session_manager.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    # ... process ...

    # Update persistent session
    session_manager.update(session_id, {
        'stage': 'script',
        'script': result.script,
        'conversation': result.conversation
    })

    return jsonify({...})

@app.route('/api/sessions/recent', methods=['GET'])
def list_recent_sessions():
    """List recent sessions for resuming"""
    sessions = session_manager.list_recent(limit=10)
    return jsonify({"sessions": sessions})

@app.route('/api/sessions/<session_id>/resume', methods=['POST'])
def resume_session(session_id):
    """Resume an existing session"""
    session = session_manager.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "session_id": session_id,
        "stage": session['stage'],
        "checkpoint": session['stage'],
        "proposal": session['proposal'],
        "script": session['script']
    })
```

---

## Enhancement 6: Better Mem0 Utilization

### Problem
The FINAL plan uses basic Mem0 search. But Mem0 supports:
- Structured memory with metadata
- Category-based filtering
- Importance scoring

### Solution: Enhanced Memory Integration

**Memory Categories and Structured Storage**:

```python
"""
memory_enhanced.py - Enhanced Mem0 integration
"""

from mem0 import Memory
from typing import List, Dict, Any, Optional
from enum import Enum

class MemoryCategory(Enum):
    PREFERENCE = "preference"           # User preferences (e.g., "prefers detailed comments")
    CORRECTION = "correction"           # User corrections (e.g., "use 5ps not 2ps equilibration")
    SUCCESS_PATTERN = "success"         # Successful simulation patterns
    FAILURE_PATTERN = "failure"         # Failed patterns to avoid
    KIM_MODEL = "kim_model"             # KIM model preferences per element
    VISUALIZATION = "visualization"     # Visualization preferences
    PARAMETER = "parameter"             # Parameter preferences (temp, duration, etc.)

class EnhancedMemory:
    """Enhanced Mem0 wrapper with structured memory management"""

    def __init__(self, user_id: str = "default"):
        self.memory = Memory()
        self.user_id = user_id

    def add_preference(self, content: str, category: MemoryCategory,
                       context: Dict[str, Any] = None) -> str:
        """Add a structured preference memory"""
        messages = [
            {
                "role": "user",
                "content": content,
                "metadata": {
                    "category": category.value,
                    "context": context or {},
                    "type": "preference"
                }
            }
        ]
        result = self.memory.add(messages, user_id=self.user_id)
        return result.get('id', '')

    def add_correction(self, original: str, corrected: str,
                       category: MemoryCategory = MemoryCategory.CORRECTION) -> str:
        """Add a correction memory (learn from user feedback)"""
        messages = [
            {"role": "user", "content": f"I changed: {original}"},
            {"role": "assistant", "content": f"Noted. Will use: {corrected}"},
            {
                "role": "system",
                "content": f"CORRECTION: {corrected}",
                "metadata": {
                    "category": category.value,
                    "type": "correction",
                    "original": original
                }
            }
        ]
        result = self.memory.add(messages, user_id=self.user_id)
        return result.get('id', '')

    def add_success_pattern(self, parameters: str, element: str,
                            notes: str = "") -> str:
        """Record a successful simulation pattern"""
        messages = [
            {
                "role": "system",
                "content": f"SUCCESS: {element} simulation with {parameters}. {notes}",
                "metadata": {
                    "category": MemoryCategory.SUCCESS_PATTERN.value,
                    "type": "success",
                    "element": element
                }
            }
        ]
        result = self.memory.add(messages, user_id=self.user_id)
        return result.get('id', '')

    def add_failure_pattern(self, parameters: str, error: str,
                            element: str = "") -> str:
        """Record a failure pattern to avoid"""
        messages = [
            {
                "role": "system",
                "content": f"FAILURE: {parameters} caused error: {error}",
                "metadata": {
                    "category": MemoryCategory.FAILURE_PATTERN.value,
                    "type": "failure",
                    "element": element,
                    "error": error
                }
            }
        ]
        result = self.memory.add(messages, user_id=self.user_id)
        return result.get('id', '')

    def search_relevant(self, query: str, categories: List[MemoryCategory] = None,
                        limit: int = 10) -> List[Dict]:
        """Search for relevant memories with optional category filter"""

        # Enhance query with category hints if specified
        enhanced_query = query
        if categories:
            category_hints = " ".join([c.value for c in categories])
            enhanced_query = f"{query} [categories: {category_hints}]"

        results = self.memory.search(
            query=enhanced_query,
            user_id=self.user_id,
            limit=limit * 2  # Get more, then filter
        )

        memories = results.get('results', [])

        # Filter by category if specified
        if categories:
            category_values = [c.value for c in categories]
            memories = [
                m for m in memories
                if m.get('metadata', {}).get('category') in category_values
            ]

        return memories[:limit]

    def get_element_preferences(self, element: str) -> Dict[str, Any]:
        """Get all preferences for a specific element"""
        results = self.memory.search(
            query=f"{element} simulation preferences KIM model parameters",
            user_id=self.user_id,
            limit=20
        )

        preferences = {
            'kim_model': None,
            'equilibration': None,
            'timestep': None,
            'comments': None,
            'other': []
        }

        for mem in results.get('results', []):
            content = mem.get('memory', '').lower()

            if 'kim' in content or 'potential' in content:
                preferences['kim_model'] = mem.get('memory')
            elif 'equilibration' in content or 'equilibrate' in content:
                preferences['equilibration'] = mem.get('memory')
            elif 'timestep' in content:
                preferences['timestep'] = mem.get('memory')
            elif 'comment' in content:
                preferences['comments'] = mem.get('memory')
            else:
                preferences['other'].append(mem.get('memory'))

        return preferences

    def format_for_prompt(self, memories: List[Dict], max_length: int = 2000) -> str:
        """Format memories for inclusion in prompts"""
        if not memories:
            return "No relevant memories from past interactions."

        lines = ["Relevant learnings from past interactions:"]
        current_length = len(lines[0])

        # Group by category
        by_category = {}
        for mem in memories:
            category = mem.get('metadata', {}).get('category', 'general')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(mem.get('memory', ''))

        # Format each category
        for category, items in by_category.items():
            header = f"\n{category.upper()}:"
            if current_length + len(header) > max_length:
                break
            lines.append(header)
            current_length += len(header)

            for item in items[:3]:  # Max 3 per category
                line = f"  - {item}"
                if current_length + len(line) > max_length:
                    break
                lines.append(line)
                current_length += len(line)

        return "\n".join(lines)

# Global enhanced memory
enhanced_memory = EnhancedMemory()
```

**Updated Checkpoint Endpoints with Enhanced Memory**:

```python
from memory_enhanced import enhanced_memory, MemoryCategory

@app.route('/api/checkpoint/parameters', methods=['POST'])
def checkpoint_parameters():
    session_id = request.json.get('session_id')
    action = request.json.get('action')

    session = session_manager.get(session_id)

    if action == "approve":
        # Store as success pattern
        element = extract_element(session['proposal'])
        enhanced_memory.add_success_pattern(
            parameters=session['proposal'],
            element=element,
            notes="Parameters approved by user"
        )

        # ... generate script ...

    elif action == "modify":
        feedback = request.json.get('feedback')

        # Determine category from feedback
        category = MemoryCategory.PREFERENCE
        if any(kw in feedback.lower() for kw in ['kim', 'potential', 'model']):
            category = MemoryCategory.KIM_MODEL
        elif any(kw in feedback.lower() for kw in ['equilibration', 'duration', 'time']):
            category = MemoryCategory.PARAMETER

        # Store structured correction
        enhanced_memory.add_correction(
            original=session['proposal'],
            corrected=feedback,
            category=category
        )
```

---

## Pre-Migration Cleanup: Legacy Code Removal

Before implementing the Agentify enhancements, clean up the existing codebase to reduce complexity and eliminate dead code.

### Tier 1: Immediate Removals (~150 lines)

These can be deleted right away with no impact:

| File | What to Remove | Lines | Reason |
|------|----------------|-------|--------|
| `src/kim_utils.py` | `ensure_kim_model_available()` | 389-441 (53 lines) | Marked DEPRECATED in docstring |
| `src/visualization_rendering.py` | Legacy DataCollection path | 88-91 (4 lines) | Never used - all calls pass file paths |
| `src/ovito_executor.py` | `validate_and_sanitize()` | 135-149 (15 lines) | Defined but never called |
| `src/tools.py` | `[TOOLS DEBUG]` print statements | 411-418, 435 (~10 lines) | Debug logging, not needed |
| `src/config.py` | Empty per-agent model configs | 8-11 (4 lines) | All empty strings, use DEFAULT_MODEL |

**Removal Commands:**
```bash
# After backing up, these functions/blocks can be deleted
# kim_utils.py: Delete lines 389-441 (ensure_kim_model_available)
# visualization_rendering.py: Delete lines 88-91 (DataCollection else block)
# ovito_executor.py: Delete lines 135-149 (validate_and_sanitize method)
```

---

### Tier 2: Extract to Shared Utilities (~200 lines saved)

#### 2.1 Registry Context Building (appears 3x in app.py)

**Current locations:**
- `src/app.py:141-166`
- `src/app.py:373-397`
- `src/app.py:629-653`

**Solution:** Add to `src/kim_utils.py`:
```python
def build_registry_context_for_command(user_command: str) -> str:
    """Build KIM registry context string for agent prompts.

    Searches user command for element names and returns formatted
    list of available KIM models for that element.
    """
    import re

    for element in ELEMENT_PROPERTIES.keys():
        if re.search(rf"\b{element.lower()}\b", user_command.lower()):
            potentials = read_registry()
            element_models = [p for p in potentials if p['element'] == element]

            if not element_models:
                return ""

            context = f"\n\nAVAILABLE KIM MODELS FOR {element} (from registry):\n"
            downloaded = [m for m in element_models if m['downloaded']]
            pending = [m for m in element_models if not m['downloaded']]

            if downloaded:
                context += "\nALREADY DOWNLOADED (Ready to use immediately):\n"
                for i, model in enumerate(downloaded, 1):
                    context += f"✅ {i}. {model['model_id']} ({model['type']})\n"

            if pending:
                context += "\nAVAILABLE FOR DOWNLOAD:\n"
                for i, model in enumerate(pending, len(downloaded) + 1):
                    context += f"⬇️ {i}. {model['model_id']} ({model['type']})\n"

            return context

    return ""
```

Then replace all 3 occurrences in `app.py` with:
```python
registry_context = kim_utils.build_registry_context_for_command(user_command)
```

#### 2.2 KIM Regex Patterns (duplicated 3x)

**Current locations:**
- `src/tools.py:72`
- `src/kim_utils.py:122`
- `src/kim_utils.py:198`

**Solution:** Add to `src/config.py`:
```python
# KIM Model ID Patterns
KIM_MODEL_ID_PATTERN = r'([A-Z][A-Za-z0-9_]+__MO_\d+_\d+)'
KIM_INIT_PATTERN = r'kim\s+init\s+([A-Za-z0-9_\-]+(?:__[A-Za-z0-9_\-]+)*)'
```

Then import and use in `tools.py` and `kim_utils.py`.

---

### Tier 3: System Prompts Extraction (~600 lines externalized)

**Current state:** Massive system prompts embedded directly in `__init__` methods.

| Agent | Location | Prompt Lines |
|-------|----------|--------------|
| OrchestratorAgent | `src/agents.py:74-248` | 175 lines |
| LammpsGeneratorAgent | `src/agents.py:320-420` | 101 lines |
| LammpsReviewerAgent | `src/agents.py:429-480` | 52 lines |
| OVITOScriptGenerator | `src/ovito_script_generator.py:23-291` | 269 lines |
| **Total** | | **597 lines** |

**Solution:** Create `src/prompts.py` with all system prompts:

```python
"""
prompts.py - Centralized system prompts for all agents

This file contains all agent system prompts in one place for:
- Easy maintenance and updates
- Reuse in DSPy signatures
- Version control of prompt changes
- Testing prompt variations
"""

# ============================================================================
# KNOWLEDGE BASE - Shared guidance (used by DSPy signatures)
# ============================================================================

class KnowledgeBase:
    """Centralized knowledge for all agents - ensures consistency."""

    KIM_REQUIREMENTS = """..."""
    LAMMPS_BEST_PRACTICES = """..."""
    REVIEW_CRITERIA = """..."""
    # ... (from FINAL plan)

# ============================================================================
# LEGACY PROMPTS - For existing Agent classes (pre-DSPy)
# ============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """
You are a request analyzer for molecular dynamics simulations...
"""  # Move from agents.py:74-248

LAMMPS_GENERATOR_SYSTEM_PROMPT = """
You are an expert LAMMPS script generator...
"""  # Move from agents.py:320-420

LAMMPS_REVIEWER_SYSTEM_PROMPT = """
You are an expert LAMMPS script reviewer...
"""  # Move from agents.py:429-480

OVITO_SCRIPT_SYSTEM_PROMPT = """
You are an expert in OVITO visualization...
"""  # Move from ovito_script_generator.py:23-291

# ============================================================================
# DSPY SIGNATURES - For new agentic workflow
# ============================================================================

# ... (DSPy signature classes from this plan)
```

Then update agent classes:
```python
# In agents.py
from prompts import ORCHESTRATOR_SYSTEM_PROMPT

class OrchestratorAgent(Agent):
    def __init__(self, ...):
        super().__init__(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            ...
        )
```

---

### Tier 4: Endpoint Consolidation (~300 lines saved)

**Current state:** Three endpoints share 80% identical code:

| Endpoint | Location | Purpose |
|----------|----------|---------|
| `/api/llm-stream` | `app.py:113-353` | SSE streaming |
| `/api/llm-command` | `app.py:355-503` | Non-streaming (legacy) |
| `/api/llm-stream-live` | `app.py:601-809` | SSE + WebSocket simulation |

**Duplicated logic across all three:**
- Registry context building (25 lines × 3)
- Phase 1/2 orchestrator logic (50 lines × 3)
- Generator + Reviewer chain (25 lines × 3)
- Execution decision logic (25 lines × 2)

**Solution:** Extract shared workflow to internal function:

```python
def _run_agent_workflow(
    user_command: str,
    conversation_context: list,
    execution_intent: bool,
    streaming_callback: Callable = None,
    websocket_callback: Callable = None
) -> dict:
    """
    Core agent workflow used by all endpoints.

    Returns:
        dict with keys: analysis, lammps_script, review, ovito_script,
                       should_run, thermo_data, frame_images
    """
    # 1. Build registry context
    registry_context = kim_utils.build_registry_context_for_command(user_command)

    # 2. Run orchestrator (Phase 1 or Phase 2)
    orchestrator = OrchestratorAgent(...)
    if streaming_callback:
        analysis = orchestrator.execute_streaming(user_command, streaming_callback)
    else:
        analysis = orchestrator.execute(user_command)

    # 3. Check if ready for generation
    if "SIMULATION PARAMETERS:" not in analysis:
        return {"analysis": analysis, "phase": 1}

    # 4. Generate script
    generator = LammpsGeneratorAgent(...)
    lammps_script = generator.execute(analysis)

    # 5. Review script
    reviewer = LammpsReviewerAgent(...)
    review = reviewer.execute(lammps_script, analysis)

    # 6. Handle visualization
    ovito_script = None
    if needs_visualization(user_command):
        ovito_script = parse_visualization_request(user_command)

    # 7. Execute if approved
    if execution_intent and "APPROVED" in review:
        if websocket_callback:
            result = run_lammps_with_websocket(lammps_script, websocket_callback)
        else:
            result = run_lammps_simulation(lammps_script)
        return {**result, "phase": "complete"}

    return {
        "analysis": analysis,
        "lammps_script": lammps_script,
        "review": review,
        "ovito_script": ovito_script,
        "phase": 2
    }
```

Then each endpoint becomes a thin wrapper:
```python
@app.route('/api/llm-stream')
def llm_stream():
    def generate():
        result = _run_agent_workflow(
            user_command=request.args.get('command'),
            streaming_callback=lambda chunk: f"data: {chunk}\n\n"
        )
        yield f"data: {json.dumps(result)}\n\n"
    return Response(generate(), mimetype='text/event-stream')
```

---

### Tier 5: Architecture Simplification

#### 5.1 Flatten ovito_executor.py

**Current:** `DynamicOVITOExecutor` class with 3 methods, only 2 used.

**Replace with utility functions in `src/visualization_utils.py`:**
```python
"""
visualization_utils.py - OVITO script validation and execution
"""

def validate_ovito_script(script: str) -> None:
    """
    Validate OVITO script for security issues.
    Raises ValueError if script is unsafe.
    """
    # Import whitelist check
    # Dangerous operation detection
    # ... (logic from ovito_executor.py:86-133)

def execute_ovito_script(script: str, pipeline, frame: int = 0) -> Any:
    """
    Execute validated OVITO Python script.
    Returns script execution result.
    """
    validate_ovito_script(script)  # Always validate first
    # ... (logic from ovito_executor.py:43-84)
```

Then delete `src/ovito_executor.py` entirely.

#### 5.2 Cache Agent Instances

**Current:** Agents instantiated per-request in `app.py`.

**Solution:** Create once at startup:
```python
# In app.py, after Flask app creation

# Cache agent instances (reuse across requests)
_orchestrator = None
_generator = None
_reviewer = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from prompts import ORCHESTRATOR_SYSTEM_PROMPT
        _orchestrator = OrchestratorAgent(
            model=config.ORCHESTRATOR_MODEL or config.DEFAULT_MODEL,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT
        )
    return _orchestrator

# Similar for generator, reviewer
```

---

### DSPy Migration: What Becomes Legacy

After implementing DSPy agents, the following become obsolete:

| Component | Current Location | Replaced By |
|-----------|------------------|-------------|
| `Agent` base class | `agents.py:1-70` | DSPy `Module` |
| `OrchestratorAgent` | `agents.py:71-318` | `SmartOrchestrator` |
| `LammpsGeneratorAgent` | `agents.py:319-427` | `IterativeLammpsGeneration` |
| `LammpsReviewerAgent` | `agents.py:428-515` | Integrated into iteration loop |
| Manual OpenAI API calls | `agents.py` (execute methods) | DSPy `ChainOfThought` |
| `OVITOScriptGenerator` class | `ovito_script_generator.py` | DSPy-based generator |

**Post-migration cleanup:**
```python
# agents.py after DSPy migration - REMOVE:
# - class Agent (base class)
# - class OrchestratorAgent
# - class LammpsGeneratorAgent
# - class LammpsReviewerAgent
# - All execute() and execute_streaming() methods

# KEEP:
# - DSPy Module classes (SmartOrchestrator, IterativeLammpsGeneration, etc.)
# - EnhancedMemory integration
# - ErrorClassifier
```

---

### Cleanup Timeline

| Phase | When | Tasks | Lines Removed |
|-------|------|-------|---------------|
| **Pre-migration** | Day 0 | Tier 1 removals (dead code) | ~150 |
| **Pre-migration** | Day 0 | Extract prompts to `prompts.py` | 0 (reorganize) |
| **During migration** | Days 1-7 | Build DSPy alongside old agents | 0 |
| **During migration** | Days 1-7 | Extract shared utilities | ~75 |
| **Post-migration** | Day 8+ | Remove old Agent classes | ~500 |
| **Post-migration** | Day 8+ | Consolidate endpoints | ~300 |
| **Post-migration** | Day 8+ | Flatten ovito_executor | ~100 |
| **Total** | | | **~1,125 lines** |

---

### Summary: Before vs After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **agents.py** | ~515 lines | ~400 lines (DSPy) | -115 |
| **app.py** | ~850 lines | ~500 lines | -350 |
| **ovito_executor.py** | ~150 lines | 0 (deleted) | -150 |
| **ovito_script_generator.py** | ~350 lines | ~100 lines | -250 |
| **Duplicate code blocks** | ~350 lines | 0 | -350 |
| **Total codebase** | ~2,500 lines | ~1,400 lines | **-44%** |

**Net result:** ~1,100 lines removed + 750 lines added (enhancements) = **350 fewer lines** with significantly more capability.

---

## Implementation Timeline

### Phase 0: Pre-Migration Cleanup (Day 0)

| Task | Time | Description |
|------|------|-------------|
| Tier 1 removals | 1 hour | Delete deprecated functions, dead code |
| Extract prompts | 2 hours | Create `prompts.py`, move all system prompts |
| Extract utilities | 1 hour | Add `build_registry_context_for_command()` to kim_utils |
| Add regex constants | 30 min | Add KIM patterns to config.py |
| Test existing functionality | 1 hour | Verify nothing broke |

### Phase 1: Base DSPy Implementation (Days 1-7)

From IMPLEMENTATION_PLAN_V4_LEAN_FINAL.md:
- Day 1: DSPy + Mem0 setup
- Day 2: DSPy Signatures + Knowledge Base
- Day 3: Iterative modules in agents.py
- Day 4: Checkpoint backend (app.py endpoints)
- Day 5: Integration testing
- Day 6: Frontend checkpoint UI
- Day 7: Polish & documentation

### Phase 2: Enhancements (Days 8-11)

| Day | Enhancement | Tasks |
|-----|-------------|-------|
| 8 | Planning & Verification | Add PlanVerifier, update workflow, test |
| 9 | Tool Selection | Implement ToolRegistry, SmartOrchestrator, test |
| 10 | Confidence + Error Recovery | Add confidence fields, ErrorClassifier, AdaptiveGeneration |
| 11 | Persistence + Memory | SessionManager, EnhancedMemory, UI updates, final testing |

### Phase 3: Post-Migration Cleanup (Days 12-13)

| Task | Time | Description |
|------|------|-------------|
| Remove old Agent classes | 2 hours | Delete Agent base, old Orchestrator/Generator/Reviewer |
| Consolidate endpoints | 3 hours | Extract `_run_agent_workflow()`, simplify endpoints |
| Flatten ovito_executor | 1 hour | Replace class with utility functions |
| Delete obsolete files | 30 min | Remove ovito_executor.py if fully replaced |
| Final testing | 2 hours | End-to-end verification |
| Update documentation | 1 hour | Update CLAUDE.md, README |

### Total Timeline

| Phase | Days | Outcome |
|-------|------|---------|
| Cleanup | 0 | Clean starting point |
| Base DSPy | 1-7 | Working agentic system |
| Enhancements | 8-11 | Full capabilities |
| Post-cleanup | 12-13 | Lean codebase |
| **Total** | **~13 days** | **Production-ready** |

---

## Summary of Enhancements

| Enhancement | What It Adds | Lines of Code |
|-------------|--------------|---------------|
| **Planning & Verification** | Pre-generation feasibility check | ~100 |
| **Tool Selection** | Dynamic tool choice by agents | ~200 |
| **Confidence Scoring** | Uncertainty quantification | ~50 |
| **Intelligent Error Recovery** | Adaptive retry strategies | ~150 |
| **Session Persistence** | SQLite session storage | ~100 |
| **Enhanced Memory** | Structured Mem0 usage | ~150 |
| **Total** | | **~750 lines** |

### Final Architecture

```
User Request
    ↓
SmartOrchestrator (with Tool Selection)
    ├── Queries KIM registry
    ├── Gets element properties
    └── Proposes parameters
    ↓
PlanVerifier (NEW)
    ├── Checks feasibility
    ├── Returns confidence score
    └── Flags issues early
    ↓
✋ CHECKPOINT 0: Verification (confidence + issues shown)
    ↓
✋ CHECKPOINT 1: Parameter Approval
    ↓
AdaptiveLammpsGeneration (NEW)
    ├── Generator with confidence
    ├── Reviewer with confidence
    ├── Error classification
    └── Adaptive retry strategy
    ↓
✋ CHECKPOINT 2: Script Approval (confidence shown)
    ↓
Execution
    ↓
EnhancedMemory (structured storage)
```

### Key Benefits

1. **Catches problems early** - Verification phase prevents wasted generation
2. **Smarter tool use** - Agents query resources before making decisions
3. **Transparent uncertainty** - Users know when to trust vs. verify
4. **Faster recovery** - Right strategy for each error type
5. **Survives restarts** - Sessions persist in SQLite
6. **Better learning** - Structured memories improve over time

---

## Next Steps

1. **Implement base plan first** (Days 1-7)
2. **Add enhancements incrementally** (Days 8-11)
3. **Test each enhancement independently**
4. **Measure improvement in success rate and iteration count**

---

## Complete Summary

### What This Plan Delivers

| Category | Items |
|----------|-------|
| **New Capabilities** | Planning/Verification, Tool Selection, Confidence Scoring, Adaptive Error Recovery, Session Persistence, Enhanced Memory |
| **Code Removed** | ~1,125 lines of legacy/duplicate code |
| **Code Added** | ~750 lines of new agentic functionality |
| **Net Change** | ~375 fewer lines with significantly more capability |
| **Architecture** | Clean DSPy-based agents with centralized prompts |

### Files Changed

| File | Action | Notes |
|------|--------|-------|
| `src/prompts.py` | **NEW** | All system prompts + DSPy signatures + KnowledgeBase |
| `src/agents.py` | **REWRITE** | DSPy modules replace Agent classes |
| `src/app.py` | **REFACTOR** | Consolidated endpoints, checkpoint workflow |
| `src/tool_registry.py` | **NEW** | Dynamic tool selection |
| `src/session_manager.py` | **NEW** | SQLite session persistence |
| `src/memory_enhanced.py` | **NEW** | Structured Mem0 integration |
| `src/kim_utils.py` | **MODIFY** | Add utility functions, remove deprecated |
| `src/ovito_executor.py` | **DELETE** | Replaced by utility functions |
| `src/config.py` | **MODIFY** | Add constants, remove empty configs |
| `src/visualization_rendering.py` | **MODIFY** | Remove legacy DataCollection path |
| `src/tools.py` | **MODIFY** | Remove debug prints, use shared patterns |
| `templates/index.html` | **MODIFY** | Checkpoint UI, confidence display |

### Key Metrics

| Metric | Target |
|--------|--------|
| Script generation success rate | 90%+ on first attempt |
| Average attempts needed | < 1.5 |
| User checkpoint approval rate | 85%+ first try (with memory) |
| Session recovery after restart | 100% |
| Memory retrieval relevance | 80%+ useful memories |

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| DSPy compatibility with local LLM | Test on Day 1 before proceeding |
| Mem0 performance | Monitor memory size, add cleanup if needed |
| Breaking existing functionality | Keep old agents as fallback until Day 12 |
| Session DB corruption | Add backup/recovery in SessionManager |

---

**Document Status**: Ready for Implementation
**Created**: 2026-02-02
**Updated**: 2026-02-02 (Added legacy cleanup section)
**Author**: Claude (with user guidance)
