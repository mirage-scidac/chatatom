import re
import config
import kim_utils
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import dspy
from prompts import (
    AnalyzeRequest,
    VerifySimulationPlan,
    GenerateLAMMPSScript,
    ReviewLAMMPSScript,
    ReviewOVITOScript,
    SelectTools,
    ClassifyUserIntent,
    ExtractFeedback,
    ExtractSimulationParameters,
    DetectVisualization,
    GenerateOVITOScript,
    KnowledgeBase,
)
from memory import get_memory_backend, MemoryEntry


# =============================================================================
# DSPy-Based Agentic Modules
# =============================================================================

def configure_dspy_lm():
    """Configure DSPy to use the Argonne API endpoint."""
    lm = dspy.LM(
        model=f'openai/{config.DEFAULT_MODEL}',
        api_base=config.API_BASE_URL,
        api_key=config.API_KEY,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    return lm


@dataclass
class GenerationResult:
    """Result from the iterative LAMMPS generation process."""
    script: str
    review_passed: bool
    attempts: int
    issues: List[str]
    suggestions: List[str]
    conversation: List[Dict[str, str]]
    confidence: float
    dump_filename_pattern: str = ""


class IterativeLammpsGeneration(dspy.Module):
    """
    DSPy module that implements iterative LAMMPS script generation with review.

    This module generates a LAMMPS script, reviews it, and if issues are found,
    iteratively improves it up to max_attempts times.

    Integrates with the memory system to:
    - Retrieve relevant past failures before generation
    - Store new failures and successes for future reference
    """

    def __init__(self, max_attempts: int = 3, use_memory: bool = True):
        super().__init__()
        self.max_attempts = max_attempts
        self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
        self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
        self.use_memory = use_memory
        self._memory = None

    @property
    def memory(self):
        """Lazy-load memory backend to avoid initialization issues."""
        if self._memory is None and self.use_memory:
            try:
                self._memory = get_memory_backend()
            except Exception as e:
                print(f"[MEMORY] Failed to initialize: {e}")
                self.use_memory = False
        return self._memory

    def _get_memory_context(self, material: str, ensemble: str, kim_model: str) -> str:
        """Retrieve relevant memories to inform generation."""
        if not self.use_memory or not self.memory:
            return ""

        try:
            # Search for relevant past failures
            query = f"{material} {ensemble} simulation"
            relevant_failures = self.memory.search(query, category="failure", limit=3)

            if not relevant_failures:
                return ""

            memory_lines = ["Previous issues to avoid:"]
            for m in relevant_failures:
                memory_lines.append(f"- {m.content}")

            return "\n".join(memory_lines)
        except Exception as e:
            print(f"[MEMORY] Search failed: {e}")
            return ""

    def _store_failure(self, material: str, kim_model: str, ensemble: str,
                      issues: List[str], attempt: int):
        """Store a failure in memory for future reference."""
        if not self.use_memory or not self.memory:
            return

        try:
            content = f"Script for {material} ({kim_model}) failed: {'; '.join(issues[:2])}"
            self.memory.add(
                content=content,
                category="failure",
                metadata={
                    "material": material,
                    "kim_model": kim_model,
                    "ensemble": ensemble,
                    "attempt": attempt,
                    "issues": issues[:3]
                }
            )
        except Exception as e:
            print(f"[MEMORY] Failed to store failure: {e}")

    def _store_success(self, material: str, kim_model: str, ensemble: str,
                      temperature: float, attempts: int):
        """Store a successful generation pattern."""
        if not self.use_memory or not self.memory:
            return

        try:
            content = f"Successful {material} {ensemble} simulation at {temperature}K"
            self.memory.add(
                content=content,
                category="success",
                metadata={
                    "material": material,
                    "kim_model": kim_model,
                    "ensemble": ensemble,
                    "temperature": temperature,
                    "attempts_needed": attempts
                }
            )
        except Exception as e:
            print(f"[MEMORY] Failed to store success: {e}")

    def forward(
        self,
        material: str,
        kim_model: str,
        temperature: float,
        ensemble: str,
        duration_ps: float,
        system_size: int = 5,
        thermo_freq: int = 100,
        dump_freq: int = 1000,
        visualization_notes: str = "",
        simulation_details: str = "",
        progress_callback=None
    ) -> GenerationResult:
        """
        Generate and iteratively refine a LAMMPS script.

        Args:
            progress_callback: Optional callable(stage, attempt, max_attempts, **kwargs)
                stage is one of: "generating", "script_generated", "reviewing",
                "review_complete", "done"

        Returns:
            GenerationResult with the final script and metadata
        """
        conversation = []
        current_script = ""
        all_issues = []
        all_suggestions = []

        # Retrieve relevant memories before first generation
        memory_context = self._get_memory_context(material, ensemble, kim_model)
        if memory_context:
            print(f"[MEMORY] Found relevant context for {material} {ensemble}")

        # Assemble KnowledgeBase context for generator and reviewer
        generator_knowledge = "\n".join([
            KnowledgeBase.KIM_MANDATORY_SEQUENCE,
            KnowledgeBase.ATOMIC_MASSES,
            KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE,
            KnowledgeBase.CYCLIC_LOADING_TECHNIQUE,
            KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS,
        ])

        reviewer_knowledge = "\n".join([
            KnowledgeBase.KIM_MANDATORY_SEQUENCE,
            KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE,
            KnowledgeBase.CYCLIC_LOADING_TECHNIQUE,
            KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS,
        ])

        for attempt in range(1, self.max_attempts + 1):
            if progress_callback:
                progress_callback("generating", attempt, self.max_attempts)

            # Generate or regenerate script
            if attempt == 1:
                gen_result = self.generator(
                    material=material,
                    kim_model=kim_model,
                    temperature=temperature,
                    ensemble=ensemble,
                    duration_ps=duration_ps,
                    system_size=system_size,
                    thermo_freq=thermo_freq,
                    dump_freq=dump_freq,
                    visualization_notes=visualization_notes,
                    simulation_details=simulation_details,
                    past_lessons=memory_context,
                    knowledge_context=generator_knowledge,
                    review_feedback="",
                )
            else:
                # Build review feedback with both issues AND suggestions
                feedback_parts = []
                recent_issues = all_issues[-3:]
                recent_suggestions = all_suggestions[-3:]
                if recent_issues:
                    feedback_parts.append("ISSUES TO FIX:\n" + "\n".join(f"- {i}" for i in recent_issues))
                if recent_suggestions:
                    feedback_parts.append("HOW TO FIX:\n" + "\n".join(f"- {s}" for s in recent_suggestions))
                review_feedback = "\n\n".join(feedback_parts)

                gen_result = self.generator(
                    material=material,
                    kim_model=kim_model,
                    temperature=temperature,
                    ensemble=ensemble,
                    duration_ps=duration_ps,
                    system_size=system_size,
                    thermo_freq=thermo_freq,
                    dump_freq=dump_freq,
                    visualization_notes=visualization_notes,
                    simulation_details=simulation_details,
                    past_lessons=memory_context,
                    knowledge_context=generator_knowledge,
                    review_feedback=review_feedback,
                )

            current_script = gen_result.lammps_script
            dump_filename_pattern = getattr(gen_result, 'dump_filename_pattern', '') or ''
            conversation.append({
                "role": "generator",
                "attempt": attempt,
                "content": current_script,
                "explanation": gen_result.explanation
            })

            # Fire script_generated callback so UI can show intermediate scripts
            if progress_callback:
                progress_callback("script_generated", attempt, self.max_attempts,
                                  script=current_script,
                                  explanation=gen_result.explanation)

            if progress_callback:
                progress_callback("reviewing", attempt, self.max_attempts)

            # Review the script with KnowledgeBase context
            review_result = self.reviewer(
                lammps_script=current_script,
                intended_parameters=f"{material} at {temperature}K, {ensemble}, {duration_ps}ps",
                knowledge_context=reviewer_knowledge,
            )

            conversation.append({
                "role": "reviewer",
                "attempt": attempt,
                "is_valid": review_result.is_valid,
                "issues": review_result.issues,
                "suggestions": review_result.suggestions,
                "severity": review_result.severity
            })

            all_issues.extend(review_result.issues)
            all_suggestions.extend(review_result.suggestions)

            if progress_callback:
                progress_callback("review_complete", attempt, self.max_attempts,
                                  is_valid=review_result.is_valid,
                                  issues=review_result.issues,
                                  suggestions=review_result.suggestions,
                                  severity=review_result.severity)

            # If review passes, we're done - store success
            if review_result.is_valid or review_result.severity == "none":
                self._store_success(material, kim_model, ensemble, temperature, attempt)
                return GenerationResult(
                    script=current_script,
                    review_passed=True,
                    attempts=attempt,
                    issues=[],
                    suggestions=review_result.suggestions,
                    conversation=conversation,
                    confidence=0.9 if attempt == 1 else 0.8 - (attempt - 1) * 0.1,
                    dump_filename_pattern=dump_filename_pattern,
                )

            # Store this failure for future reference
            self._store_failure(material, kim_model, ensemble, review_result.issues, attempt)

        # Max attempts reached without passing review
        return GenerationResult(
            script=current_script,
            review_passed=False,
            attempts=self.max_attempts,
            issues=all_issues[-5:],  # Last 5 issues
            suggestions=all_suggestions[-3:],
            conversation=conversation,
            confidence=0.3,
            dump_filename_pattern=dump_filename_pattern,
        )



@dataclass
class AnalysisResult:
    """Result from request analysis."""
    material: str
    temperature: float
    ensemble: str
    duration_ps: float
    kim_model: str
    system_size: int
    needs_clarification: bool
    clarification_question: str
    confidence: float


def detect_precomputed_properties(lammps_script: str) -> Dict[str, str]:
    """Scan LAMMPS script for compute commands and return pre-computed properties.

    Parses the deterministic LAMMPS syntax (``compute ID group style args``)
    to identify properties that are already available in the dump file.

    Returns:
        Dict mapping property name (e.g. 'c_cna') to compute style (e.g. 'cna/atom').
    """
    properties: Dict[str, str] = {}
    for line in lammps_script.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        # Match: compute <ID> <group> <style> [args...]
        match = re.match(r'^compute\s+(\S+)\s+\S+\s+(\S+)', stripped, re.IGNORECASE)
        if match:
            compute_id = match.group(1)
            compute_style = match.group(2)
            properties[f'c_{compute_id}'] = compute_style
    return properties


class SmartOrchestrator(dspy.Module):
    """
    DSPy module for intelligent request analysis and tool selection.

    Analyzes user requests, extracts parameters, and determines
    what tools/actions are needed. Also classifies user intent
    using LLM instead of string matching.
    """

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeRequest)
        self.tool_selector = dspy.ChainOfThought(SelectTools)
        self.intent_classifier = dspy.ChainOfThought(ClassifyUserIntent)
        self.feedback_extractor = dspy.ChainOfThought(ExtractFeedback)
        self.param_extractor = dspy.ChainOfThought(ExtractSimulationParameters)
        self.viz_detector = dspy.Predict(DetectVisualization)

    def analyze_request(
        self,
        user_request: str,
        registry_context: str = ""
    ) -> AnalysisResult:
        """
        Analyze a user request and extract simulation parameters.
        """
        result = self.analyzer(
            user_request=user_request,
            registry_context=registry_context
        )

        return AnalysisResult(
            material=result.material,
            temperature=result.temperature,
            ensemble=result.ensemble,
            duration_ps=result.duration_ps,
            kim_model=result.kim_model,
            system_size=result.system_size,
            needs_clarification=result.needs_clarification,
            clarification_question=result.clarification_question,
            confidence=result.confidence
        )

    def extract_parameters(self, analysis_text: str) -> Dict[str, Any]:
        """Extract structured simulation parameters from orchestrator analysis text.

        Uses the ``ExtractSimulationParameters`` DSPy signature to replace
        brittle ``startswith``/``split(':')`` parsing of free-text LLM output.

        Returns:
            Dict with keys: has_parameters, material, kim_model, temperature,
            ensemble, duration_ps, system_size, visualization_notes.
        """
        result = self.param_extractor(analysis_text=analysis_text)

        # Safe type coercion with sensible defaults
        try:
            temperature = float(result.temperature)
        except (ValueError, TypeError):
            temperature = 300.0

        try:
            duration_ps = float(result.duration_ps)
        except (ValueError, TypeError):
            duration_ps = 10.0

        try:
            system_size = int(result.system_size)
        except (ValueError, TypeError):
            system_size = 5

        ensemble = str(result.ensemble).strip().upper()
        if ensemble not in ('NVE', 'NVT', 'NPT'):
            ensemble = 'NVT'

        return {
            "has_parameters": bool(result.has_parameters),
            "material": str(result.material).strip() if result.material else "",
            "kim_model": str(result.kim_model).strip() if result.kim_model else "",
            "temperature": temperature,
            "ensemble": ensemble,
            "duration_ps": duration_ps,
            "system_size": system_size,
            "visualization_notes": str(result.visualization_notes).strip() if result.visualization_notes else "",
        }

    def select_tools(
        self,
        user_request: str,
        available_tools: List[str],
        conversation_context: str = ""
    ) -> Tuple[List[str], List[str], str]:
        """
        Select appropriate tools for handling a request.

        Returns:
            Tuple of (selected_tools, tool_arguments, reasoning)
        """
        result = self.tool_selector(
            user_request=user_request,
            available_tools=available_tools,
            conversation_context=conversation_context
        )

        return result.selected_tools, result.tool_arguments, result.reasoning

    def classify_intent(
        self,
        user_message: str,
        workflow_stage: str = "initial"
    ) -> Dict[str, Any]:
        """
        Classify user intent using LLM instead of string matching.

        Args:
            user_message: The user's message text
            workflow_stage: Current stage - 'initial', 'parameters_proposed', or 'script_ready'

        Returns:
            Dict with 'intent', 'confidence', and 'modifications' keys
        """
        result = self.intent_classifier(
            user_message=user_message,
            workflow_stage=workflow_stage
        )

        # With Literal type constraint, the LLM should return a valid value.
        # Minimal fallback in case of unexpected output.
        intent = str(result.intent).strip().lower()
        valid_intents = {'execute', 'modify', 'cancel', 'question', 'new_simulation'}
        if intent not in valid_intents:
            print(f"[INTENT] WARNING: unexpected intent '{intent}', defaulting to 'question'")
            intent = 'question'

        try:
            confidence = float(result.confidence)
        except (ValueError, TypeError):
            confidence = 0.5

        return {
            "intent": intent,
            "confidence": confidence,
            "modifications": str(result.extracted_modifications).strip() if result.extracted_modifications else ""
        }

    def detect_visualization(self, user_request: str) -> bool:
        """Use LLM to determine if request contains custom visualization intent."""
        result = self.viz_detector(user_request=user_request)
        return bool(result.is_custom_visualization)

    def process_feedback(self, user_feedback: str, simulation_summary: str) -> Dict[str, Any]:
        """
        Use LLM to extract a structured lesson from user feedback, then store in memory.

        Args:
            user_feedback: The user's feedback about the simulation result
            simulation_summary: Summary of the simulation that was run

        Returns:
            Dict with 'category', 'lesson', 'keywords', and 'stored' keys
        """
        result = self.feedback_extractor(
            user_feedback=user_feedback,
            simulation_summary=simulation_summary
        )

        category = str(result.category).strip().lower()
        lesson = str(result.lesson).strip()
        keywords = str(result.keywords).strip()

        # Normalize to known categories that the retrieval system actually searches for
        KNOWN_CATEGORIES = {
            'lammps_issue', 'visualization_issue', 'success', 'preference',
            'failure', 'visualization_success'
        }
        if category not in KNOWN_CATEGORIES:
            # Infer from content: if it mentions visualization/color/ovito/atoms → visualization_issue
            # Otherwise → lammps_issue (most common)
            feedback_lower = (user_feedback + " " + lesson).lower()
            viz_keywords = ['color', 'orange', 'visual', 'ovito', 'render', 'atom', 'frame', 'display']
            if any(kw in feedback_lower for kw in viz_keywords):
                category = 'visualization_issue'
            else:
                category = 'lammps_issue'
            print(f"[MEMORY] Normalized unknown category to '{category}'")

        # Store in memory
        stored = False
        try:
            memory = get_memory_backend()
            memory.add(
                content=lesson,
                category=category,
                metadata={
                    "keywords": keywords,
                    "simulation": simulation_summary,
                    "user_feedback": user_feedback
                }
            )
            stored = True
            print(f"[MEMORY] Stored feedback lesson: category={category}, lesson={lesson[:80]}...")
        except Exception as e:
            print(f"[MEMORY] Failed to store feedback: {e}")

        return {
            "category": category,
            "lesson": lesson,
            "keywords": keywords,
            "stored": stored
        }


@dataclass
class OVITOReviewResult:
    """Result from OVITO script review."""
    script: str
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    severity: str


class OVITOScriptReview(dspy.Module):
    """
    DSPy module that reviews OVITO Python scripts for correctness.

    Checks for unauthorized imports, proper PythonModifier patterns,
    per-frame guards, and OVITO API best practices.
    """

    def __init__(self):
        super().__init__()
        self.reviewer = dspy.ChainOfThought(ReviewOVITOScript)

    def forward(self, ovito_script: str, visualization_request: str = "") -> OVITOReviewResult:
        ovito_knowledge = "\n".join([
            KnowledgeBase.OVITO_IMPORT_REFERENCE,
            KnowledgeBase.OVITO_PROPERTY_GUARDS,
            KnowledgeBase.CNA_STRUCTURE_MAPPING,
        ])
        review = self.reviewer(
            ovito_script=ovito_script,
            visualization_request=visualization_request,
            knowledge_context=ovito_knowledge,
        )
        return OVITOReviewResult(
            script=ovito_script,
            is_valid=review.is_valid,
            issues=review.issues,
            suggestions=review.suggestions,
            severity=review.severity,
        )


@dataclass
class OVITOGenerationResult:
    """Result from the iterative OVITO generation process."""
    script: Optional[str]       # None if all retries exhausted
    review_passed: bool
    attempts: int
    issues: List[str]
    suggestions: List[str]
    conversation: List[Dict[str, str]]


class OVITOScriptGeneration(dspy.Module):
    """
    DSPy module that generates OVITO Python scripts from visualization requests.

    Replaces the former OVITOScriptGenerator Agent subclass with a structured
    DSPy ChainOfThought pipeline that leverages KnowledgeBase context.
    """

    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(GenerateOVITOScript)

    def forward(self, user_request: str, simulation_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an OVITO Python script from a visualization request.

        Args:
            user_request: Natural language visualization request
            simulation_context: Optional dict with material, temperature, etc.

        Returns:
            Complete OVITO Python script as a string
        """
        # Build context string from simulation_context dict
        context_str = ""
        if simulation_context:
            context_str = f"Material: {simulation_context.get('material', 'Unknown')}"
            if simulation_context.get('natoms'):
                context_str += f", Atoms: {simulation_context['natoms']}"
            if simulation_context.get('temperature'):
                context_str += f", Temperature: {simulation_context['temperature']} K"
            if simulation_context.get('structure'):
                context_str += f", Structure: {simulation_context['structure']}"

            precomputed = simulation_context.get('precomputed_properties', {})
            if precomputed:
                props = ", ".join(f"'{k}' ({v})" for k, v in precomputed.items())
                context_str += f"\nPRE-COMPUTED LAMMPS PROPERTIES (use directly, do not recompute): {props}"

            viz_lessons = simulation_context.get('viz_lessons', '')
            if viz_lessons:
                context_str += f"\n{viz_lessons}"

        # Assemble knowledge from KnowledgeBase
        knowledge = "\n".join([
            KnowledgeBase.OVITO_IMPORT_REFERENCE,
            KnowledgeBase.OVITO_PROPERTY_GUARDS,
            KnowledgeBase.CNA_STRUCTURE_MAPPING,
        ])

        result = self.generator(
            visualization_request=user_request,
            simulation_context=context_str,
            knowledge_context=knowledge,
        )

        script = result.ovito_script

        # Strip markdown fences if present
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0].strip()
        elif "```" in script:
            script = script.split("```")[1].split("```")[0].strip()

        return script


class IterativeOVITOGeneration(dspy.Module):
    """
    DSPy module that implements iterative OVITO script generation with review.

    Generates an OVITO script, reviews it, and if the review finds errors,
    iteratively improves it up to max_attempts times by feeding review issues
    back to the generator.

    Follows the same pattern as IterativeLammpsGeneration.
    """

    def __init__(self, max_attempts: int = 3, use_memory: bool = True):
        super().__init__()
        self.max_attempts = max_attempts
        self.generator = dspy.ChainOfThought(GenerateOVITOScript)
        self.reviewer = dspy.ChainOfThought(ReviewOVITOScript)
        self.use_memory = use_memory
        self._memory = None

    @property
    def memory(self):
        """Lazy-load memory backend to avoid initialization issues."""
        if self._memory is None and self.use_memory:
            try:
                self._memory = get_memory_backend()
            except Exception as e:
                print(f"[MEMORY] Failed to initialize: {e}")
                self.use_memory = False
        return self._memory

    def _get_memory_context(self, visualization_request: str) -> str:
        """Retrieve relevant visualization memories to inform generation."""
        if not self.use_memory or not self.memory:
            return ""

        try:
            relevant = self.memory.search(visualization_request, category="visualization_issue", limit=3)
            if not relevant:
                return ""

            memory_lines = ["Previous visualization issues to avoid:"]
            for m in relevant:
                memory_lines.append(f"- {m.content}")
            return "\n".join(memory_lines)
        except Exception as e:
            print(f"[MEMORY] Search failed: {e}")
            return ""

    def _store_failure(self, visualization_request: str, issues: List[str], attempt: int):
        """Store a visualization failure in memory for future reference."""
        if not self.use_memory or not self.memory:
            return

        try:
            content = f"OVITO script failed: {'; '.join(issues[:2])}"
            self.memory.add(
                content=content,
                category="visualization_issue",
                metadata={
                    "request": visualization_request[:200],
                    "attempt": attempt,
                    "issues": issues[:3]
                }
            )
        except Exception as e:
            print(f"[MEMORY] Failed to store failure: {e}")

    def _store_success(self, visualization_request: str, attempts: int):
        """Store a successful visualization generation."""
        if not self.use_memory or not self.memory:
            return

        try:
            content = f"Successful OVITO script for: {visualization_request[:100]}"
            self.memory.add(
                content=content,
                category="visualization_success",
                metadata={
                    "request": visualization_request[:200],
                    "attempts_needed": attempts
                }
            )
        except Exception as e:
            print(f"[MEMORY] Failed to store success: {e}")

    def _build_context_string(self, simulation_context: Optional[Dict[str, Any]]) -> str:
        """Build context string from simulation_context dict."""
        if not simulation_context:
            return ""

        context_str = f"Material: {simulation_context.get('material', 'Unknown')}"
        if simulation_context.get('natoms'):
            context_str += f", Atoms: {simulation_context['natoms']}"
        if simulation_context.get('temperature'):
            context_str += f", Temperature: {simulation_context['temperature']} K"
        if simulation_context.get('structure'):
            context_str += f", Structure: {simulation_context['structure']}"

        precomputed = simulation_context.get('precomputed_properties', {})
        if precomputed:
            props = ", ".join(f"'{k}' ({v})" for k, v in precomputed.items())
            context_str += f"\nPRE-COMPUTED LAMMPS PROPERTIES (use directly, do not recompute): {props}"

        viz_lessons = simulation_context.get('viz_lessons', '')
        if viz_lessons:
            context_str += f"\n{viz_lessons}"

        return context_str

    def _strip_markdown_fences(self, script: str) -> str:
        """Strip markdown code fences from script if present."""
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0].strip()
        elif "```" in script:
            script = script.split("```")[1].split("```")[0].strip()
        return script

    def forward(
        self,
        visualization_request: str,
        simulation_context: Optional[Dict[str, Any]] = None,
        progress_callback=None
    ) -> OVITOGenerationResult:
        """
        Generate and iteratively refine an OVITO Python script.

        Args:
            visualization_request: Natural language visualization request
            simulation_context: Optional dict with material, temperature, etc.
            progress_callback: Optional callable(stage, attempt, max_attempts, **kwargs)
                stage is one of: "generating", "script_generated", "reviewing",
                "review_complete", "done"

        Returns:
            OVITOGenerationResult with the final script and metadata
        """
        conversation = []
        current_script = ""
        all_issues = []
        all_suggestions = []

        # Build context and knowledge strings
        context_str = self._build_context_string(simulation_context)
        memory_context = self._get_memory_context(visualization_request)
        if memory_context:
            context_str = f"{context_str}\n{memory_context}" if context_str else memory_context
            print(f"[MEMORY] Found relevant visualization context")

        knowledge = "\n".join([
            KnowledgeBase.OVITO_IMPORT_REFERENCE,
            KnowledgeBase.OVITO_PROPERTY_GUARDS,
            KnowledgeBase.CNA_STRUCTURE_MAPPING,
        ])

        ovito_knowledge = "\n".join([
            KnowledgeBase.OVITO_IMPORT_REFERENCE,
            KnowledgeBase.OVITO_PROPERTY_GUARDS,
            KnowledgeBase.CNA_STRUCTURE_MAPPING,
        ])

        for attempt in range(1, self.max_attempts + 1):
            if progress_callback:
                progress_callback("generating", attempt, self.max_attempts)

            # Build review_feedback from previous issues AND suggestions
            review_feedback = ""
            if attempt > 1:
                feedback_parts = []
                recent_issues = all_issues[-3:]
                recent_suggestions = all_suggestions[-3:]
                if recent_issues:
                    feedback_parts.append("ISSUES TO FIX:\n" + "\n".join(f"- {i}" for i in recent_issues))
                if recent_suggestions:
                    feedback_parts.append("HOW TO FIX:\n" + "\n".join(f"- {s}" for s in recent_suggestions))
                review_feedback = "\n\n".join(feedback_parts)

            # Generate script
            gen_result = self.generator(
                visualization_request=visualization_request,
                simulation_context=context_str,
                knowledge_context=knowledge,
                review_feedback=review_feedback,
            )

            current_script = self._strip_markdown_fences(gen_result.ovito_script)

            conversation.append({
                "role": "generator",
                "attempt": attempt,
                "content": current_script,
            })

            if progress_callback:
                progress_callback("script_generated", attempt, self.max_attempts,
                                  script=current_script)

            # Review the script
            if progress_callback:
                progress_callback("reviewing", attempt, self.max_attempts)

            review_result = self.reviewer(
                ovito_script=current_script,
                visualization_request=visualization_request,
                knowledge_context=ovito_knowledge,
            )

            conversation.append({
                "role": "reviewer",
                "attempt": attempt,
                "is_valid": review_result.is_valid,
                "issues": review_result.issues,
                "suggestions": review_result.suggestions,
                "severity": review_result.severity,
            })

            all_issues.extend(review_result.issues)
            all_suggestions.extend(review_result.suggestions)

            if progress_callback:
                progress_callback("review_complete", attempt, self.max_attempts,
                                  is_valid=review_result.is_valid,
                                  issues=review_result.issues,
                                  suggestions=review_result.suggestions,
                                  severity=review_result.severity)

            # Accept if valid or severity is warning/none
            if review_result.is_valid or review_result.severity in ("none", "warning"):
                self._store_success(visualization_request, attempt)
                return OVITOGenerationResult(
                    script=current_script,
                    review_passed=True,
                    attempts=attempt,
                    issues=[],
                    suggestions=review_result.suggestions,
                    conversation=conversation,
                )

            # Error severity — store failure and retry
            self._store_failure(visualization_request, review_result.issues, attempt)

        # Max attempts reached without passing review
        return OVITOGenerationResult(
            script=None,
            review_passed=False,
            attempts=self.max_attempts,
            issues=all_issues[-5:],
            suggestions=all_suggestions[-3:],
            conversation=conversation,
        )
